# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import torch
import torch.nn as nn
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.layered_prefill import (
    DeepseekV4LayeredPrefillAdapter,
    create_layered_prefill_model_adapter,
)


class _HyperConnectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_ids_seen = None

    def forward(
        self,
        positions,
        hidden_states,
        residual,
        llama_4_scaling,
        input_ids=None,
    ):
        del positions, residual, llama_4_scaling
        self.input_ids_seen = input_ids
        residual = hidden_states.clone()
        return hidden_states + input_ids[:, None, None], residual


class _DeepseekV4Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="deepseek_v4", num_hidden_layers=2
        )
        self.start_layer = 0
        self.end_layer = 2
        self.hc_mult = 2
        self.layers = nn.ModuleList(
            [_HyperConnectionLayer(), _HyperConnectionLayer()]
        )
        self.norm = nn.Identity()
        self.hc_head_fn = None
        self.hc_head_scale = None
        self.hc_head_base = None
        self._mtp_hidden_buffer = None

    @staticmethod
    def embed_input_ids(input_ids):
        return input_ids.float().unsqueeze(-1)

    @staticmethod
    def hc_head(hidden_states, hc_fn, hc_scale, hc_base):
        del hc_fn, hc_scale, hc_base
        return hidden_states.sum(dim=1)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    batch_size,
                    self.hc_mult,
                    1,
                    dtype=dtype,
                    device=device,
                )
            }
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                None,
                input_ids=input_ids,
            )
        return self.norm(
            self.hc_head(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
            )
        )


class _DeepseekV4ForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DeepseekV4Backbone()
        self.config = self.model.config
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)


def test_deepseek_v4_adapter_matches_hyper_connection_forward():
    model = _DeepseekV4ForCausalLM()
    adapter = create_layered_prefill_model_adapter(model)
    input_ids = torch.tensor([2, 4])
    positions = torch.arange(2)

    first = adapter.forward(
        input_ids=input_ids,
        positions=positions,
        layer_start=0,
        layer_end=1,
    )
    second = adapter.forward(
        input_ids=input_ids,
        positions=positions,
        layer_start=1,
        layer_end=2,
        frontier=(first.hidden_states, first.residual),
    )

    assert isinstance(adapter, DeepseekV4LayeredPrefillAdapter)
    assert first.hidden_states.shape == (2, 2, 1)
    assert first.residual is None
    assert model.model.layers[1].input_ids_seen is input_ids
    torch.testing.assert_close(second.hidden_states, model(input_ids, positions))


def test_deepseek_v4_adapter_preserves_hidden_only_pp_schema():
    model = _DeepseekV4ForCausalLM()
    adapter = DeepseekV4LayeredPrefillAdapter(model)

    hidden_states, residual = adapter.make_transport_frontier(
        3, torch.float32, torch.device("cpu")
    )
    assert hidden_states.shape == (3, 2, 1)
    assert residual is None

    output = adapter.forward(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.arange(3),
        layer_start=0,
        layer_end=1,
    )
    intermediate_tensors = adapter.to_intermediate_tensors(output)
    assert set(intermediate_tensors.tensors) == {"hidden_states"}


def test_factory_uses_standard_adapter_for_common_decoder_contract():
    from vllm.model_executor.models.layered_prefill import (
        StandardDecoderLayeredPrefillAdapter,
    )

    class StandardLayer(nn.Module):
        def forward(self, positions, hidden_states, residual):
            return hidden_states, residual

    class StandardNorm(nn.Module):
        def forward(self, hidden_states, residual):
            return hidden_states, residual

    model = _DeepseekV4ForCausalLM()
    model.config.model_type = "common_decoder"
    model.model.layers = nn.ModuleList([StandardLayer(), StandardLayer()])
    model.model.norm = StandardNorm()
    adapter = create_layered_prefill_model_adapter(model)

    assert isinstance(adapter, StandardDecoderLayeredPrefillAdapter)
