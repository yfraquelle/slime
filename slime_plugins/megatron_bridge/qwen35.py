"""
Qwen3.5-VL CP alignment patch for megatron.bridge.

This plugin keeps the stock Qwen3.5 bridge/provider registration, but swaps the
model instance produced by the provider with a local subclass that fixes the
vision/text alignment issue under BSHD + context parallelism.
"""

from __future__ import annotations

from types import MethodType
import torch
import torch.distributed as dist
import torch.distributed.nn
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    AllGatherVisionEmbeddings,
    collapse_thw,
    get_vision_cp_data,
    qwen3vl_cp_split,
    reorganize_inputs,
    split_deepstack_embs,
)
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    HAVE_TE,
    Qwen35VLMoEModelProvider,
    Qwen35VLModelProvider,
    Qwen3VLSelfAttention,
    _patch_standard_attention_specs,
)
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
    ExpertMLPDownProjMapping,
    ExpertMLPGateUpProjMapping,
    _align_weight_to_shape,
    extract_expert_number_from_param,
    get_module_and_param_from_name,
)
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams


def _patched_expert_down_proj_hf_to_megatron(
    self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module
) -> torch.Tensor:
    global_expert_number = extract_expert_number_from_param(self.megatron_param)
    expert_weight = hf_weights[global_expert_number] if hf_weights.ndim >= 3 else hf_weights

    normalized_param = self._normalize_expert_param_name(self.megatron_param)
    _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

    # Megatron expert fc2 local weight layout is [input_per_partition, hidden],
    # while HF down_proj uses nn.Linear layout [hidden, input]. So the ETP
    # expansion lands on dim=1 in HF layout rather than target_param.partition_dim.
    full_target_shape = [target_param.shape[0], target_param.shape[1]]
    etp_world_size = mpu.get_expert_tensor_parallel_world_size()
    if etp_world_size > 1:
        full_target_shape[1] *= etp_world_size

    expert_weight = _align_weight_to_shape(expert_weight, torch.Size(full_target_shape), "down_proj")
    return super(ExpertMLPDownProjMapping, self).hf_to_megatron(expert_weight, megatron_module)


def _patched_expert_gate_up_hf_to_megatron(
    self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module
) -> torch.Tensor:
    global_expert_number = extract_expert_number_from_param(self.megatron_param)
    expert_weight = hf_weights[global_expert_number] if hf_weights.ndim >= 3 else hf_weights

    normalized_param = self._normalize_expert_param_name(self.megatron_param)
    _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
    target_shape = target_param.shape
    if target_shape[0] % 2 != 0:
        raise ValueError(f"Expected even fused dim for {self.megatron_param}, got {target_shape}.")

    full_target_shape = list(target_shape)
    partition_dim = getattr(target_param, "partition_dim", -1)
    etp_world_size = mpu.get_expert_tensor_parallel_world_size()
    if partition_dim >= 0 and etp_world_size > 1:
        full_target_shape[partition_dim] *= etp_world_size
    full_target_shape = torch.Size(full_target_shape)
    gate_full_target_shape = (full_target_shape[0] // 2, full_target_shape[1])

    if expert_weight.ndim == 3 and expert_weight.shape[0] == 2:
        gate = _align_weight_to_shape(expert_weight[0], gate_full_target_shape, "gate")
        up = _align_weight_to_shape(expert_weight[1], gate_full_target_shape, "up")
    else:
        expert_weight = _align_weight_to_shape(expert_weight, full_target_shape, "gate_up")
        gate, up = torch.chunk(expert_weight, 2, dim=0)

    return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)


ExpertMLPDownProjMapping.hf_to_megatron = _patched_expert_down_proj_hf_to_megatron
ExpertMLPGateUpProjMapping.hf_to_megatron = _patched_expert_gate_up_hf_to_megatron


def _gather_input_ids_from_cp_bshd(
    input_ids: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size <= 1:
        return input_ids

    assert input_ids.dim() == 2, f"expected BSHD-style input_ids [b, s], got {input_ids.shape}"
    local_seq_len = input_ids.shape[1]
    assert local_seq_len % 2 == 0, f"local cp sequence should be 2 chunks, got {local_seq_len}"
    chunk_size = local_seq_len // 2

    gathered = torch.distributed.nn.all_gather(input_ids, group=cp_group)
    first_half = [rank_input[:, :chunk_size] for rank_input in gathered]
    second_half = [rank_input[:, chunk_size:] for rank_input in gathered][::-1]
    return torch.cat(first_half + second_half, dim=1)


def _select_local_vision_embeds_bshd(
    full_input_ids: torch.Tensor,
    vision_embeds: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    if vision_embeds is None or vision_embeds.shape[0] == 0:
        return vision_embeds

    assert full_input_ids.dim() == 2, f"expected full_input_ids [b, s], got {full_input_ids.shape}"
    batch_size, seq_len = full_input_ids.shape
    assert seq_len % (2 * cp_size) == 0, f"{seq_len=} must be divisible by {2 * cp_size=}"
    chunk_size = seq_len // (2 * cp_size)

    full_flat = full_input_ids.reshape(-1)
    full_mask = (full_flat == image_token_id) | (full_flat == video_token_id)

    rank_mask = torch.zeros_like(full_mask, dtype=torch.bool)
    for batch_idx in range(batch_size):
        offset = batch_idx * seq_len
        first_start = offset + cp_rank * chunk_size
        rank_mask[first_start : first_start + chunk_size] = True

        second_end = offset + seq_len - cp_rank * chunk_size
        rank_mask[second_end - chunk_size : second_end] = True

    local_vision_mask = full_mask & rank_mask
    local_count = int(local_vision_mask.sum().item())
    if local_count == 0:
        return vision_embeds[:0]
    if local_count == vision_embeds.shape[0]:
        return vision_embeds

    embed_indices = full_mask.long().cumsum(0)[local_vision_mask] - 1
    return vision_embeds[embed_indices]


def _thd_to_bshd(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = int(seqlens.max().item())
    batch_size = len(cu_seqlens) - 1
    out = packed.new_zeros(batch_size, max_seq, *packed.shape[2:])
    cu_seqlens_list = cu_seqlens.tolist()
    for i, seq_len in enumerate(seqlens.tolist()):
        start = cu_seqlens_list[i]
        out[i, :seq_len] = packed[0, start : start + seq_len]
    return out


def _bshd_to_thd(unpacked: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = int(cu_seqlens[-1].item())
    out = unpacked.new_zeros(1, total, *unpacked.shape[2:])
    cu_seqlens_list = cu_seqlens.tolist()
    for i, seq_len in enumerate(seqlens.tolist()):
        start = cu_seqlens_list[i]
        out[0, start : start + seq_len] = unpacked[i, :seq_len]
    return out


def _gather_input_ids_from_cp_thd(input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return input_ids

    gathered = torch.distributed.nn.all_gather(input_ids, group=mpu.get_context_parallel_group())
    local_cu_seqlens = (cu_seqlens // cp_size).tolist()
    whole_list = []
    for i in range(len(cu_seqlens) - 1):
        seqlen = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
        chunk_size = seqlen // 2 // cp_size
        whole_list.extend(
            gathered[cp_rank][0, local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
            for cp_rank in range(cp_size)
        )
        whole_list.extend(
            [
                gathered[cp_rank][0, local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                for cp_rank in range(cp_size)
            ][::-1]
        )
    return torch.cat(whole_list).unsqueeze(0)


def _select_local_vision_embeds_thd(
    full_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    vision_embeds: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    if vision_embeds is None or vision_embeds.shape[0] == 0:
        return vision_embeds

    full_flat = full_input_ids[0]
    full_mask = (full_flat == image_token_id) | (full_flat == video_token_id)
    rank_mask = torch.zeros_like(full_mask, dtype=torch.bool)

    for i in range(len(cu_seqlens) - 1):
        seq_start = int(cu_seqlens[i].item())
        seqlen = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
        chunk_size = seqlen // (2 * cp_size)
        first_start = seq_start + cp_rank * chunk_size
        rank_mask[first_start : first_start + chunk_size] = True

        second_end = seq_start + seqlen - cp_rank * chunk_size
        rank_mask[second_end - chunk_size : second_end] = True

    local_vision_mask = full_mask & rank_mask
    local_count = int(local_vision_mask.sum().item())
    if local_count == 0:
        return vision_embeds[:0]
    if local_count == vision_embeds.shape[0]:
        return vision_embeds

    embed_indices = full_mask.long().cumsum(0)[local_vision_mask] - 1
    return vision_embeds[embed_indices]


def _select_local_position_ids_thd(
    full_position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    if cp_size <= 1:
        return full_position_ids

    whole_list = []
    for i in range(len(cu_seqlens) - 1):
        seq_start = int(cu_seqlens[i].item())
        seqlen = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
        chunk_size = seqlen // (2 * cp_size)
        first_start = seq_start + cp_rank * chunk_size
        second_end = seq_start + seqlen - cp_rank * chunk_size
        whole_list.append(full_position_ids[:, :, first_start : first_start + chunk_size])
        whole_list.append(full_position_ids[:, :, second_end - chunk_size : second_end])
    return torch.cat(whole_list, dim=2)


def _select_local_position_ids_from_bshd(
    position_ids_bshd: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    if cp_size <= 1:
        return _bshd_to_thd(position_ids_bshd.permute(1, 2, 0), cu_seqlens).permute(2, 0, 1)

    whole_list = []
    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    for i, seq_len in enumerate(seq_lens):
        chunk_size = seq_len // (2 * cp_size)
        first_start = cp_rank * chunk_size
        second_end = seq_len - cp_rank * chunk_size
        whole_list.append(position_ids_bshd[:, i : i + 1, first_start : first_start + chunk_size])
        whole_list.append(position_ids_bshd[:, i : i + 1, second_end - chunk_size : second_end])
    return torch.cat(whole_list, dim=2)


class SlimeQwen35VLModel(Qwen3VLModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        cp_img_num: list[int] = None,
        images_padded: list[bool] = None,
        inference_context: object | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del inference_context, runtime_gather_output

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1

        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )

        vision_grid_thw = None
        vision_data = None
        vision_mask = None
        deepstack_feature_lists = None
        position_ids = None
        full_input_ids = None

        assert inference_params is None, "not support inference"
        torch.cuda.nvtx.range_push("SlimeQwen35VLModel.forward.pre_process")

        if self.pre_process:
            vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                square_merge_size=self.square_merge_size,
            )

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                seqlen_on_cp_ranks = None
                if cp_size > 1 and self.config.vision_dp_when_cp:
                    # Reduce duplicated vision compute by reusing the stock vision CP split,
                    # while still applying our own input_ids-based local selection after the
                    # embeds are gathered back.
                    if cp_img_num is None:
                        assert images_padded is None
                        vision_data, vision_grid_thw, cp_img_num, images_padded = qwen3vl_cp_split(
                            cp_size,
                            vision_data,
                            vision_grid_thw,
                        )
                    vision_data, vision_grid_thw, seqlen_on_cp_ranks = get_vision_cp_data(
                        vision_data,
                        vision_grid_thw,
                        self.square_merge_size,
                        cp_img_num,
                        images_padded,
                        cp_rank,
                        cp_size,
                    )
                    vision_grid_thw = collapse_thw(vision_grid_thw)

                if vision_data.shape[0] > 0:
                    vision_embeds, deepstack_feature_lists = self.vision_model(
                        hidden_states=vision_data,
                        grid_thw=vision_grid_thw,
                    )
                else:
                    hidden_size = self.language_model.config.hidden_size
                    vision_embeds = torch.zeros(
                        (0, hidden_size),
                        device=vision_data.device,
                        dtype=torch.bfloat16,
                    )
                    deepstack_feature_lists = [
                        torch.zeros((0, hidden_size), device=vision_data.device, dtype=torch.bfloat16)
                        for _ in self.vision_model.config.deepstack_visual_indexes
                    ]

                if cp_size > 1 and self.config.vision_dp_when_cp:
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds,
                        seqlen_on_cp_ranks,
                        self.pg_collection.cp,
                    )
                    if deepstack_feature_lists is not None:
                        deepstack_feature_lists = [
                            AllGatherVisionEmbeddings.apply(
                                deepstack_visual_embed,
                                seqlen_on_cp_ranks,
                                self.pg_collection.cp,
                            )
                            for deepstack_visual_embed in deepstack_feature_lists
                        ]

                    if packed_seq_params is None:
                        full_input_ids = _gather_input_ids_from_cp_bshd(input_ids, self.pg_collection.cp)
                        vision_embeds = _select_local_vision_embeds_bshd(
                            full_input_ids,
                            vision_embeds,
                            cp_rank,
                            cp_size,
                            self.image_token_id,
                            self.video_token_id,
                        )
                        if deepstack_feature_lists is not None:
                            deepstack_feature_lists = [
                                _select_local_vision_embeds_bshd(
                                    full_input_ids,
                                    deepstack_visual_embed,
                                    cp_rank,
                                    cp_size,
                                    self.image_token_id,
                                    self.video_token_id,
                                )
                                for deepstack_visual_embed in deepstack_feature_lists
                            ]
                    else:
                        full_input_ids = _gather_input_ids_from_cp_thd(input_ids, cu_seqlens)
                        vision_embeds = _select_local_vision_embeds_thd(
                            full_input_ids,
                            cu_seqlens,
                            vision_embeds,
                            cp_rank,
                            cp_size,
                            self.image_token_id,
                            self.video_token_id,
                        )
                        if deepstack_feature_lists is not None:
                            deepstack_feature_lists = [
                                _select_local_vision_embeds_thd(
                                    full_input_ids,
                                    cu_seqlens,
                                    deepstack_visual_embed,
                                    cp_rank,
                                    cp_size,
                                    self.image_token_id,
                                    self.video_token_id,
                                )
                                for deepstack_visual_embed in deepstack_feature_lists
                            ]

                local_vision_count = int(vision_mask.sum().item()) if vision_mask is not None else 0
                embed_count = int(vision_embeds.shape[0]) if vision_embeds is not None else 0
                # print(
                #     f"[Rank {rank}] cp_rank={cp_rank} qwen35_cp_align "
                #     f"input_ids={tuple(input_ids.shape)} "
                #     f"vision_mask={local_vision_count} "
                #     f"vision_embeds={embed_count}"
                # )

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            )

            if vision_embeds is not None:
                if vision_mask is not None and vision_mask.any():
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous().clone()
                    local_vision_count = int(vision_mask.sum().item())
                    embed_count = int(vision_embeds.shape[0])
                    assert local_vision_count == embed_count, (
                        f"local vision/token mismatch on rank={rank} cp_rank={cp_rank}: "
                        f"{local_vision_count=} {embed_count=} "
                        f"{tuple(input_ids.shape)=} {tuple(vision_data.shape) if vision_data is not None else None=}"
                    )
                    combined_embeddings[vision_mask] = vision_embeds
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        visual_pos_masks = vision_mask
        deepstack_visual_embeds = deepstack_feature_lists
        if self.config.sequence_parallel or cp_size > 1:
            # input_ids/get_batch are already CP-local in slime's BSHD path, and the
            # vision embeds above are selected to the same local zigzag shard.
            # Only keep the TP/SP split here; applying CP splitting again would use a
            # local-length mask as if it were global and break deepstack partitioning.
            deepstack_cp_size = 1 if cp_size > 1 else cp_size
            deepstack_cp_rank = 0 if deepstack_cp_size == 1 else self.pg_collection.cp.rank()
            visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=self.pg_collection.tp.size(),
                tp_rank=self.pg_collection.tp.rank(),
                cp_size=deepstack_cp_size,
                cp_rank=deepstack_cp_rank,
                sequence_parallel=self.config.sequence_parallel,
            )

        if position_ids is None:
            if packed_seq_params is None:
                rope_input_ids = input_ids
                if cp_size > 1:
                    if full_input_ids is None:
                        full_input_ids = _gather_input_ids_from_cp_bshd(input_ids, self.pg_collection.cp)
                    rope_input_ids = full_input_ids
                position_ids, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    rope_input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
            else:
                if cp_size > 1:
                    if full_input_ids is None:
                        full_input_ids = _gather_input_ids_from_cp_thd(input_ids, cu_seqlens)
                else:
                    full_input_ids = input_ids
                rope_input_ids = _thd_to_bshd(full_input_ids, cu_seqlens)
                rope_attention_mask = torch.zeros_like(rope_input_ids, dtype=rope_input_ids.dtype)
                seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                for i, seq_len in enumerate(seq_lens.tolist()):
                    rope_attention_mask[i, :seq_len] = 1
                position_ids_bshd, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    rope_input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=rope_attention_mask,
                )
                position_ids = _select_local_position_ids_from_bshd(
                    position_ids_bshd.contiguous(),
                    cu_seqlens,
                    cp_rank,
                    cp_size,
                ).contiguous()
                del rope_input_ids, rope_attention_mask, position_ids_bshd, seq_lens
                attention_mask = None
                self.language_model.rotary_pos_emb.is_thd_format = True

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("SlimeQwen35VLModel.forward.language_model")

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        torch.cuda.nvtx.range_pop()
        return output


def _patched_qwen35_provide(self, pre_process=None, post_process=None, vp_stage=None) -> SlimeQwen35VLModel:
    language_transformer_config = self
    hf_vision_config = self.vision_config
    block_spec = get_gpt_decoder_block_spec(
        language_transformer_config,
        vp_stage=vp_stage,
        use_transformer_engine=HAVE_TE,
    )
    _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)

    model = Qwen3VLModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=block_spec,
        vision_transformer_config=hf_vision_config,
        pre_process=pre_process,
        post_process=post_process,
    )
    model.forward = MethodType(SlimeQwen35VLModel.forward, model)

    if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
        model.freeze(
            freeze_language_model=self.freeze_language_model,
            freeze_vision_model=self.freeze_vision_model,
            freeze_vision_projection=self.freeze_vision_projection,
        )

    return model


_ORIG_QWEN35VL_FINALIZE = Qwen35VLModelProvider.finalize
_ORIG_QWEN35VLMOE_FINALIZE = Qwen35VLMoEModelProvider.finalize


def _patched_qwen35_finalize(self) -> None:
    self.vision_dp_when_cp = True
    self.batch_invariant_mode = False
    _ORIG_QWEN35VL_FINALIZE(self)


def _patched_qwen35_moe_finalize(self) -> None:
    self.vision_dp_when_cp = True
    self.batch_invariant_mode = False
    _ORIG_QWEN35VLMOE_FINALIZE(self)


Qwen35VLModelProvider.finalize = _patched_qwen35_finalize
Qwen35VLMoEModelProvider.finalize = _patched_qwen35_moe_finalize
Qwen35VLModelProvider.provide = _patched_qwen35_provide
Qwen35VLMoEModelProvider.provide = _patched_qwen35_provide
