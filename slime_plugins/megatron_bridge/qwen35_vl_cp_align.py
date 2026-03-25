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
    preprocess_packed_seqs,
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
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams


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
        del cp_img_num, images_padded, inference_context, runtime_gather_output

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1

        if packed_seq_params is not None:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                **kwargs,
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
            ).clone()

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if vision_mask is not None and vision_mask.any():
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
            deepstack_cp_size = 1 if (packed_seq_params is None and cp_size > 1) else cp_size
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


Qwen35VLModelProvider.provide = _patched_qwen35_provide
Qwen35VLMoEModelProvider.provide = _patched_qwen35_provide
