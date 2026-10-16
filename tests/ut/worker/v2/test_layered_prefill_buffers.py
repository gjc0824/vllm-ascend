"""Layered Prefill × Model Runner V2 — buffer isolation (milestone V1)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_BUFFER_TENSOR_ATTRS = (
    "input_ids",
    "positions",
    "is_padding",
    "query_start_loc",
    "seq_lens",
    "dcp_local_seq_lens",
    "seq_lens_cpu",
)


def _tensor_data_ptrs(buffers: AscendInputBuffers) -> dict[str, int]:
    return {name: getattr(buffers, name).data_ptr() for name in _BUFFER_TENSOR_ATTRS}


def test_ascend_input_buffers_storage_does_not_overlap():
    device = torch.device("cpu")
    main = AscendInputBuffers(max_num_reqs=4, max_num_tokens=32, device=device)
    p = AscendInputBuffers(max_num_reqs=4, max_num_tokens=32, device=device)

    main_ptrs = set(_tensor_data_ptrs(main).values())
    p_ptrs = set(_tensor_data_ptrs(p).values())
    assert main_ptrs.isdisjoint(p_ptrs)


def test_layered_p_buffers_swap_restores_and_isolates_writes():
    """Swap CM must restore main buffers; P writes must not touch main storage."""
    device = torch.device("cpu")
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_buffers = AscendInputBuffers(max_num_reqs=4, max_num_tokens=32, device=device)
    runner._layered_input_buffers = AscendInputBuffers(max_num_reqs=4, max_num_tokens=32, device=device)

    main = runner.input_buffers
    main.input_ids[:8] = torch.arange(8, dtype=torch.int32)
    main.positions[:8] = torch.arange(8, dtype=torch.int64)
    main.query_start_loc[:3] = torch.tensor([0, 4, 8], dtype=torch.int32)
    main.seq_lens[:2] = torch.tensor([4, 4], dtype=torch.int32)
    main.seq_lens_np[:2] = [4, 4]

    snapshot = {
        "input_ids": main.input_ids[:8].clone(),
        "positions": main.positions[:8].clone(),
        "query_start_loc": main.query_start_loc[:3].clone(),
        "seq_lens": main.seq_lens[:2].clone(),
        "seq_lens_np": main.seq_lens_np[:2].copy(),
    }
    main_ptrs_before = _tensor_data_ptrs(main)

    with runner._layered_p_buffers():
        assert runner.input_buffers is runner._layered_input_buffers
        runner.input_buffers.input_ids[:8] = 99
        runner.input_buffers.positions[:8] = -1
        runner.input_buffers.query_start_loc[:3] = 7
        runner.input_buffers.seq_lens[:2] = 1
        runner.input_buffers.seq_lens_np[:2] = [1, 1]

    assert runner.input_buffers is main
    assert torch.equal(main.input_ids[:8], snapshot["input_ids"])
    assert torch.equal(main.positions[:8], snapshot["positions"])
    assert torch.equal(main.query_start_loc[:3], snapshot["query_start_loc"])
    assert torch.equal(main.seq_lens[:2], snapshot["seq_lens"])
    assert (main.seq_lens_np[:2] == snapshot["seq_lens_np"]).all()
    assert _tensor_data_ptrs(main) == main_ptrs_before


def test_execute_model_fails_closed_when_layered_plan_present():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._layered_prefill_v2_ready = False
    scheduler_output = SimpleNamespace(layered_prefill_plan=object())

    with pytest.raises(NotImplementedError, match="not ready"):
        NPUModelRunner.execute_model(runner, scheduler_output)


def test_layered_p_buffers_requires_allocation():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_buffers = MagicMock()
    runner._layered_input_buffers = None
    with pytest.raises(RuntimeError, match="P buffers were not allocated"), runner._layered_p_buffers():
        pass
