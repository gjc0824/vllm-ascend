# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend model adapters for layered prefill execution."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
import torch.nn as nn

from vllm.model_executor.models.layered_prefill import (
    LayeredPrefillFrontier,
    LayeredPrefillModelAdapter,
    StandardDecoderLayeredPrefillAdapter,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.core.layered_prefill import LayeredForwardOutput


def _get_model_type(model: nn.Module) -> str | None:
    for candidate in (model, getattr(model, "model", None)):
        config = getattr(candidate, "config", None)
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str):
            return model_type
    return None


class DeepseekV4LayeredPrefillAdapter(LayeredPrefillModelAdapter):
    """Layer-range transitions for DeepSeek-V4 hyper-connections."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if _get_model_type(model) != "deepseek_v4":
            raise TypeError(
                f"Model {type(model).__name__} is not a DeepSeek-V4 model"
            )
        required = (
            "hc_mult",
            "hc_head",
            "hc_head_fn",
            "hc_head_scale",
            "hc_head_base",
        )
        missing = [name for name in required if not hasattr(self.backbone, name)]
        if missing:
            raise TypeError(
                f"DeepSeek-V4 backbone is missing layered state hooks: {missing}"
            )

    def _prepare_initial_state(
        self,
        *,
        input_ids: torch.Tensor | None,
        frontier: LayeredPrefillFrontier | None,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> LayeredPrefillFrontier:
        hidden_states, residual = super()._prepare_initial_state(
            input_ids=input_ids,
            frontier=frontier,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        if (
            self.start_layer == 0
            and frontier is None
            and intermediate_tensors is None
        ):
            hidden_states = hidden_states.unsqueeze(1).repeat(
                1, self.backbone.hc_mult, 1
            )
        return hidden_states, residual

    def _forward_layer(
        self,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> LayeredPrefillFrontier:
        if input_ids is None:
            raise ValueError("DeepSeek-V4 layered prefill requires input_ids")
        output = layer(
            positions,
            hidden_states,
            residual,
            None,
            input_ids=input_ids,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                f"DeepSeek-V4 layer {type(layer).__name__} must return "
                "(hidden_states, residual)"
            )
        # Every DeepSeek-V4 layer recreates residual from its hidden input, so
        # it is not part of the cross-group frontier.
        return output[0], None

    def _finalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> LayeredPrefillFrontier:
        del residual
        mtp_hidden_buffer = getattr(self.backbone, "_mtp_hidden_buffer", None)
        if mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        hidden_states = self.backbone.hc_head(
            hidden_states,
            self.backbone.hc_head_fn,
            self.backbone.hc_head_scale,
            self.backbone.hc_head_base,
        )
        hidden_states = self.backbone.norm(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states, None

    def to_intermediate_tensors(
        self, output: LayeredForwardOutput
    ) -> IntermediateTensors:
        return IntermediateTensors({"hidden_states": output.hidden_states})


# Keep model-specific behavior in this module instead of model forward files.
# The mapping is immutable so adapter selection has no runtime global state.
_SPECIALIZED_ADAPTERS: Mapping[
    str, type[LayeredPrefillModelAdapter]
] = MappingProxyType(
    {"deepseek_v4": DeepseekV4LayeredPrefillAdapter}
)


def create_layered_prefill_model_adapter(
    model: nn.Module,
) -> LayeredPrefillModelAdapter:
    """Select a non-invasive layer executor for a loaded model."""

    model_type = _get_model_type(model)
    adapter_type = _SPECIALIZED_ADAPTERS.get(model_type or "")
    if adapter_type is not None:
        return adapter_type(model)
    return StandardDecoderLayeredPrefillAdapter(model)
