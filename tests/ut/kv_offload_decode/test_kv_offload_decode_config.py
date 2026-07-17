"""Unit tests for KVOffloadDecodeConfig validation gates."""
from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_config import KVOffloadDecodeConfig


def _vllm_config(
    enforce_eager: bool = True,
    index_topk: int = 3,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_DECODE_ONLY,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            hf_text_config=SimpleNamespace(index_topk=index_topk),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def test_disabled_config_skips_validation():
    config = KVOffloadDecodeConfig(_vllm_config(enforce_eager=False), {"enabled": False})
    assert config.enabled is False


def test_enabled_config_defaults():
    config = KVOffloadDecodeConfig(_vllm_config(), {"enabled": True})
    assert config.enabled is True
    assert config.topk_buffer_size == 4096
    assert config.topk == 3


def test_enabled_config_allows_full_decode_only_graph():
    config = KVOffloadDecodeConfig(
        _vllm_config(enforce_eager=False), {"enabled": True}
    )
    assert config.enabled is True


def test_enabled_config_rejects_other_graph_modes():
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        KVOffloadDecodeConfig(
            _vllm_config(
                enforce_eager=False,
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            ),
            {"enabled": True},
        )


def test_enabled_config_rejects_topk_buffer_below_topk():
    with pytest.raises(ValueError, match="topk_buffer_size"):
        KVOffloadDecodeConfig(_vllm_config(), {"enabled": True, "topk_buffer_size": 2})
