"""Unit tests for Layered Prefill V2 MoE layer-offset helpers."""

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def test_layered_prefill_moe_layer_offset_counts_prior_moe_layers():
    # Layer names follow vLLM's extract_layer_index convention.
    layers = [
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
        "model.layers.2.mlp.experts",
        "model.layers.5.mlp.experts",
        "model.layers.6.mlp.experts",
    ]
    assert NPUModelRunner._layered_prefill_moe_layer_offset(layers, 0) == 0
    assert NPUModelRunner._layered_prefill_moe_layer_offset(layers, 2) == 2
    assert NPUModelRunner._layered_prefill_moe_layer_offset(layers, 5) == 3
    assert NPUModelRunner._layered_prefill_moe_layer_offset(layers, 6) == 4
    assert NPUModelRunner._layered_prefill_moe_layer_offset(layers, 99) == 5


def test_layered_prefill_moe_layer_offset_empty():
    assert NPUModelRunner._layered_prefill_moe_layer_offset([], 12) == 0
