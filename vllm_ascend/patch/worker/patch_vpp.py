#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
VPP (Virtual Pipeline Parallelism) model patches for DeepSeek V2/V3.

Patches DeepseekV2Model and DeepseekV2ForCausalLM to support V-shaped
fold-back layer assignment.  The current virtual stage is read from the
global state set by ``set_virtual_pipeline_parallel_rank``.
"""
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    _get_llama_4_scaling,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.distributed.parallel_state import (
    get_virtual_pipeline_parallel_rank,
    get_virtual_pipeline_parallel_size,
)
from vllm_ascend.distributed.vpp_utils import (
    is_vpp_first_stage,
    is_vpp_last_stage,
    make_vpp_layers,
)


# ---- Runtime helpers (called during forward) ----

def vpp_is_first_stage() -> bool:
    """Check whether the current virtual stage is the global first stage."""
    vp_size = get_virtual_pipeline_parallel_size()
    if vp_size <= 1:
        return get_pp_group().is_first_rank
    pp_rank = get_pp_group().rank_in_group
    vp_stage = get_virtual_pipeline_parallel_rank()
    return is_vpp_first_stage(pp_rank, vp_stage)


def vpp_is_last_stage() -> bool:
    """Check whether the current virtual stage is the global last stage."""
    vp_size = get_virtual_pipeline_parallel_size()
    if vp_size <= 1:
        return get_pp_group().is_last_rank
    pp_rank = get_pp_group().rank_in_group
    pp_size = get_pp_group().world_size
    vp_stage = get_virtual_pipeline_parallel_rank()
    return is_vpp_last_stage(pp_rank, pp_size, vp_stage, vp_size)


# ---- Construction helpers (called during __init__) ----

def vpp_rank_has_embedding(pp_rank: int, pp_size: int, vp_size: int) -> bool:
    """Rank 0 always has the embedding (first stage is always rank 0)."""
    if vp_size <= 1:
        return get_pp_group().is_first_rank
    return pp_rank == 0


def vpp_rank_has_norm(pp_rank: int, pp_size: int, vp_size: int) -> bool:
    """The rank hosting the last virtual stage has the norm.

    Even vp_size -> last sweep backward -> rank 0.
    Odd  vp_size -> last sweep forward  -> rank pp_size-1.
    """
    if vp_size <= 1:
        return get_pp_group().is_last_rank
    if vp_size % 2 == 0:
        return pp_rank == 0
    else:
        return pp_rank == pp_size - 1


def _get_vp_size() -> int:
    from vllm_ascend.ascend_config import get_ascend_config
    try:
        return get_ascend_config().virtual_pipeline_parallel_size
    except RuntimeError:
        return 1


def _get_custom_layer_ranges_for_rank() -> list[tuple[int, int]] | None:
    """Return the manually specified layer ranges for this PP rank, or None."""
    from vllm_ascend.ascend_config import get_ascend_config
    try:
        all_ranges = get_ascend_config().vpp_layer_ranges
    except RuntimeError:
        return None
    if all_ranges is None:
        return None
    pp_rank = get_pp_group().rank_in_group
    return all_ranges[pp_rank]


# ---- Patched DeepseekV2Model methods ----

_original_dsv2_model_init = DeepseekV2Model.__init__


def _vpp_dsv2_model_init(self, *, vllm_config, prefix=""):
    """Patched __init__ that uses VPP layer assignment when enabled."""
    vp_size = _get_vp_size()

    if vp_size <= 1:
        _original_dsv2_model_init(
            self, vllm_config=vllm_config, prefix=prefix)
        return

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    from vllm.platforms import current_platform

    super(DeepseekV2Model, self).__init__()

    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config

    self.config = config
    self.device = current_platform.device_type
    self.vocab_size = config.vocab_size

    self.is_v32 = hasattr(config, "index_topk")
    if self.is_v32:
        topk_tokens = config.index_topk
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )
    else:
        topk_indices_buffer = None

    pp_rank = get_pp_group().rank_in_group
    pp_size = get_pp_group().world_size

    has_embed = vpp_rank_has_embedding(pp_rank, pp_size, vp_size)
    has_norm = vpp_rank_has_norm(pp_rank, pp_size, vp_size)

    if has_embed:
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
    else:
        self.embed_tokens = PPMissingLayer()

    custom_ranges = _get_custom_layer_ranges_for_rank()
    self.vpp_layer_ranges, self.layers = make_vpp_layers(
        config.num_hidden_layers,
        lambda pfx: DeepseekV2DecoderLayer(
            vllm_config, pfx, topk_indices_buffer=topk_indices_buffer),
        f"{prefix}.layers",
        vp_size,
        custom_layer_ranges=custom_ranges,
    )
    self.start_layer = self.vpp_layer_ranges[0][0]
    self.end_layer = self.vpp_layer_ranges[-1][1]

    if has_norm:
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    else:
        self.norm = PPMissingLayer()

    self.make_empty_intermediate_tensors = (
        make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size))


def _vpp_dsv2_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
):
    """Patched forward that respects the current VPP virtual stage."""
    if not hasattr(self, "vpp_layer_ranges"):
        is_first = get_pp_group().is_first_rank
        is_last = get_pp_group().is_last_rank
        start, end = self.start_layer, self.end_layer
    else:
        is_first = vpp_is_first_stage()
        is_last = vpp_is_last_stage()
        vp_stage = get_virtual_pipeline_parallel_rank()
        start, end = self.vpp_layer_ranges[vp_stage]

    if is_first:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=(
                llama_4_scaling_config["original_max_position_embeddings"]),
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    for layer in islice(self.layers, start, end):
        hidden_states, residual = layer(
            positions, hidden_states, residual, llama_4_scaling)

    if not is_last:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual})

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


# ---- Patched DeepseekV2ForCausalLM ----

_original_dsv2_causal_lm_init = DeepseekV2ForCausalLM.__init__


def _vpp_dsv2_causal_lm_init(self, *, vllm_config, prefix=""):
    """Patched __init__ that places lm_head based on VPP topology."""
    vp_size = _get_vp_size()

    if vp_size <= 1:
        _original_dsv2_causal_lm_init(
            self, vllm_config=vllm_config, prefix=prefix)
        return

    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    super(DeepseekV2ForCausalLM, self).__init__()
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    self.config = config
    self.quant_config = quant_config

    qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
    self.use_mha = config.model_type == "deepseek" or all(
        dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))
    if self.use_mha:
        self.packed_modules_mapping["qkv_proj"] = [
            "q_proj", "k_proj", "v_proj"]

    self.fuse_qkv_a_proj = (
        hasattr(config, "q_lora_rank") and config.q_lora_rank is not None)
    if self.fuse_qkv_a_proj:
        self.packed_modules_mapping["fused_qkv_a_proj"] = [
            "q_a_proj", "kv_a_proj_with_mqa"]

    self.model = self.model_cls(
        vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

    pp_rank = get_pp_group().rank_in_group
    pp_size = get_pp_group().world_size
    has_output = vpp_rank_has_norm(pp_rank, pp_size, vp_size)

    if has_output:
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
    else:
        self.lm_head = PPMissingLayer()

    self.logits_processor = LogitsProcessor(config.vocab_size)
    self.make_empty_intermediate_tensors = (
        self.model.make_empty_intermediate_tensors)

    self.num_moe_layers = (
        self.config.num_hidden_layers - self.config.first_k_dense_replace)
    self.set_moe_parameters()


# ---- Apply patches ----

DeepseekV2Model.__init__ = _vpp_dsv2_model_init
DeepseekV2Model.forward = _vpp_dsv2_model_forward
DeepseekV2ForCausalLM.__init__ = _vpp_dsv2_causal_lm_init
