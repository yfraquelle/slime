import base64
import logging
import pickle
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)

_MAX_IPC_BUCKET_BYTES = 128 * 1024**2
_CPU_BUCKET_PREFIX = "slime_cpu_flattened_bucket:"
_CPU_TENSOR_CHUNK_PREFIX = "slime_cpu_tensor_chunk:"
_MAX_CPU_TENSOR_CHUNK_BYTES = 32 * 1024**2
_CUDA_IPC_DISABLED_FOR_COLOCATE = False


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (sglang engines)
            # have already closed their IPC handles.
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        torch.cuda.ipc_collect()

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
            identical_payload_across_group=self.args.megatron_to_hf_mode == "bridge",
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
    identical_payload_across_group: bool = False,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    if identical_payload_across_group:
        return _send_identical_payload_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=ipc_engine,
            ipc_gather_src=ipc_gather_src,
            ipc_gather_group=ipc_gather_group,
            weight_version=weight_version,
        )

    long_live_tensors = []

    serialized_tensors = []
    for named_tensors in _iter_ipc_named_tensor_buckets(
        hf_named_tensors,
        max_bucket_bytes=_MAX_IPC_BUCKET_BYTES,
    ):
        serialized_buckets, long_lived_tensor = _serialize_flattened_tensor_bucket(
            named_tensors,
            max_cpu_tensor_chunk_bytes=_MAX_CPU_TENSOR_CHUNK_BYTES,
        )
        if long_lived_tensor is not None:
            long_live_tensors.append(long_lived_tensor)
        serialized_tensors.extend(serialized_buckets)

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _send_identical_payload_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    if dist.get_rank() != ipc_gather_src:
        return [], None

    serialized_tensors = []
    long_live_tensors = []
    for named_tensors in _iter_ipc_named_tensor_buckets(
        hf_named_tensors,
        max_bucket_bytes=_MAX_IPC_BUCKET_BYTES,
    ):
        serialized_buckets, long_lived_tensor = _serialize_flattened_tensor_bucket(
            named_tensors,
            max_cpu_tensor_chunk_bytes=_MAX_CPU_TENSOR_CHUNK_BYTES,
        )
        if long_lived_tensor is not None:
            long_live_tensors.append(long_lived_tensor)
        serialized_tensors.extend(serialized_buckets)

    # Bridge export already materializes full HF tensors on every training rank,
    # so in colocate mode per-rank payloads are identical. Reuse the source rank
    # payload for every engine worker instead of serializing the same tensor on
    # all local ranks again.
    engine_group_size = dist.get_world_size(ipc_gather_group)
    refs = []
    for serialized_tensor in serialized_tensors:
        kwargs = {
            "serialized_named_tensors": [serialized_tensor] * engine_group_size,
            "load_format": "flattened_bucket",
            "weight_version": str(weight_version),
        }
        refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _iter_ipc_named_tensor_buckets(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    max_bucket_bytes: int,
):
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        grouped_named_tensors = [named_tensors]
    else:
        tensors_by_dtype = {}
        for name, tensor in named_tensors:
            tensors_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))
        grouped_named_tensors = tensors_by_dtype.values()

    for same_group_named_tensors in grouped_named_tensors:
        current_bucket = []
        current_bucket_bytes = 0
        for name, tensor in same_group_named_tensors:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if current_bucket and current_bucket_bytes + tensor_bytes > max_bucket_bytes:
                yield current_bucket
                current_bucket = []
                current_bucket_bytes = 0
            current_bucket.append((name, tensor))
            current_bucket_bytes += tensor_bytes

        if current_bucket:
            yield current_bucket


def _serialize_flattened_tensor_bucket(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    max_cpu_tensor_chunk_bytes: int,
):
    global _CUDA_IPC_DISABLED_FOR_COLOCATE

    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    metadata = flattened_tensor_bucket.get_metadata()
    flattened_tensor = flattened_tensor_bucket.get_flattened_tensor().detach().contiguous()
    flattened_tensor_data = {
        "flattened_tensor": flattened_tensor,
        "metadata": metadata,
    }

    if flattened_tensor.is_cuda and _CUDA_IPC_DISABLED_FOR_COLOCATE:
        return _serialize_cpu_fallback_bucket(
            named_tensors=named_tensors,
            flattened_tensor=flattened_tensor,
            metadata=metadata,
            max_cpu_tensor_chunk_bytes=max_cpu_tensor_chunk_bytes,
        )

    try:
        serialized = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
        return [serialized], flattened_tensor_data if flattened_tensor.is_cuda else None
    except Exception as exc:
        if not flattened_tensor.is_cuda:
            raise

        _CUDA_IPC_DISABLED_FOR_COLOCATE = True
        logger.warning(
            "Disabling CUDA IPC for subsequent colocate weight buckets after serialization failure. "
            "reason=%r",
            exc,
        )

        return _serialize_cpu_fallback_bucket(
            named_tensors=named_tensors,
            flattened_tensor=flattened_tensor,
            metadata=metadata,
            max_cpu_tensor_chunk_bytes=max_cpu_tensor_chunk_bytes,
            exc=exc,
        )


def _serialize_cpu_fallback_bucket(
    *,
    named_tensors: list[tuple[str, torch.Tensor]],
    flattened_tensor: torch.Tensor,
    metadata,
    max_cpu_tensor_chunk_bytes: int,
    exc: Exception | None = None,
):
    if not flattened_tensor.is_cuda:
        raise RuntimeError("CPU fallback should only be used for CUDA flattened tensors")

    first_name = named_tensors[0][0]
    total_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)
    if exc is not None:
        logger.warning(
            "Falling back to CPU serialization for colocate weight bucket. "
            "first_name=%s tensor_count=%d total_bytes=%d reason=%r",
            first_name,
            len(named_tensors),
            total_bytes,
            exc,
        )

    cpu_flattened_tensor_data = {
        "flattened_tensor": flattened_tensor.cpu(),
        "metadata": metadata,
    }
    if len(named_tensors) == 1 and total_bytes > max_cpu_tensor_chunk_bytes:
        logger.warning(
            "Chunking oversized CPU weight payload for colocate sync. "
            "name=%s total_bytes=%d chunk_bytes=%d",
            first_name,
            total_bytes,
            max_cpu_tensor_chunk_bytes,
        )
        serialized = _serialize_cpu_tensor_chunks(
            name=first_name,
            tensor=named_tensors[0][1].detach().cpu(),
            max_chunk_bytes=max_cpu_tensor_chunk_bytes,
        )
        return serialized, None

    serialized = _serialize_cpu_flattened_tensor_bucket(cpu_flattened_tensor_data)
    return [serialized], None


def _serialize_cpu_flattened_tensor_bucket(flattened_tensor_data: dict[str, Any]) -> str:
    cpu_flattened_tensor = flattened_tensor_data["flattened_tensor"]
    payload = {
        "flattened_tensor_bytes": cpu_flattened_tensor.numpy().tobytes(),
        "metadata": flattened_tensor_data["metadata"],
    }
    serialized_payload = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return _CPU_BUCKET_PREFIX + base64.b64encode(serialized_payload).decode("ascii")


def _serialize_cpu_tensor_chunks(
    *,
    name: str,
    tensor: torch.Tensor,
    max_chunk_bytes: int,
) -> list[str]:
    flattened_tensor = tensor.contiguous().view(torch.uint8)
    tensor_bytes = flattened_tensor.numpy().tobytes()
    total_bytes = len(tensor_bytes)
    serialized_chunks = []

    for chunk_start in range(0, total_bytes, max_chunk_bytes):
        chunk_end = min(chunk_start + max_chunk_bytes, total_bytes)
        payload = {
            "name": name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "total_bytes": total_bytes,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk_bytes": tensor_bytes[chunk_start:chunk_end],
        }
        serialized_payload = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        serialized_chunks.append(
            _CPU_TENSOR_CHUNK_PREFIX + base64.b64encode(serialized_payload).decode("ascii")
        )

    return serialized_chunks
