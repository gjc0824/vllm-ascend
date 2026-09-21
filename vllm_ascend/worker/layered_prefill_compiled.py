# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Runner-owned layer ranges using the configured vLLM compilation backend."""

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile


@support_torch_compile(
    dynamic_arg_dims={
        "hidden_states": 0,
        "residual": 0,
        "positions": -1,
        "input_ids": 0,
    }
)
class LayeredPrefillCompiledGroup(nn.Module):
    """Share existing layers, without copying weights or model-specific code."""

    def __init__(self, *, adapter, start, end, vllm_config):
        super().__init__()
        self.adapter = adapter
        self.layers = nn.ModuleList(adapter.layers[start:end])

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ):
        for layer in self.layers:
            hidden_states, residual = self.adapter._forward_layer(layer, positions, hidden_states, residual, input_ids)
        return hidden_states, residual
