"""Unit tests for Layered Prefill V2 sampling / frontier lifecycle (V3)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.core.layered_prefill import LayeredFrontier, LayeredPrefillStateStore
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import ExecuteModelState

from vllm_ascend.worker.v2.layered_prefill import (
    LayeredV2ExecuteModelState,
    detach_execute_model_state,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _empty_cached() -> CachedRequestData:
    return CachedRequestData.make_empty()


def _scheduler_output(num_scheduled_tokens: dict[str, int]) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=_empty_cached(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_merge_layered_v2_outputs_preserves_scheduler_order():
    d_out = ModelRunnerOutput(
        req_ids=["d0"],
        req_id_to_index={"d0": 0},
        sampled_token_ids=[[11]],
    )
    p_out = ModelRunnerOutput(
        req_ids=["p0"],
        req_id_to_index={"p0": 0},
        sampled_token_ids=[[22]],
    )
    merged = NPUModelRunner._merge_layered_v2_outputs(
        _scheduler_output({"p0": 8, "d0": 1}),
        [d_out, p_out],
    )
    assert merged.req_ids == ["p0", "d0"]
    assert merged.sampled_token_ids == [[22], [11]]


def test_merge_intermediate_p_emits_empty_sampled_tokens():
    d_out = ModelRunnerOutput(
        req_ids=["d0"],
        req_id_to_index={"d0": 0},
        sampled_token_ids=[[7]],
    )
    p_out = ModelRunnerOutput(
        req_ids=["p0"],
        req_id_to_index={"p0": 0},
        sampled_token_ids=[[]],
    )
    merged = NPUModelRunner._merge_layered_v2_outputs(
        _scheduler_output({"d0": 1, "p0": 8}),
        [d_out, p_out],
    )
    assert merged.sampled_token_ids == [[7], []]


@pytest.mark.parametrize("sample_p", [False, True])
@pytest.mark.parametrize("prefill_first", [False, True])
def test_merge_logprobs_aligns_final_and_unsampled_prefill(sample_p, prefill_first):
    d_out = ModelRunnerOutput(
        req_ids=["d1", "d0"],
        req_id_to_index={"d1": 0, "d0": 1},
        sampled_token_ids=[[11], [12]],
        logprobs=LogprobsLists(
            np.array([[11, 21], [12, 22]]),
            np.array([[-0.1, -1.1], [-0.2, -1.2]], dtype=np.float32),
            np.array([1, 2]),
        ),
    )
    p_out = ModelRunnerOutput(
        req_ids=["p"],
        req_id_to_index={"p": 0},
        sampled_token_ids=[[13]] if sample_p else [[]],
        logprobs=(
            LogprobsLists(np.array([[13, 23, 33]]), np.array([[-0.3, -1.3, -2.3]]), np.array([3])) if sample_p else None
        ),
    )
    order = ["p", "d0", "d1"] if prefill_first else ["d0", "d1", "p"]
    merged = NPUModelRunner._merge_layered_v2_outputs(_scheduler_output(dict.fromkeys(order, 1)), [d_out, p_out])
    assert merged.req_ids == order
    expected = {"d0": (12, -0.2, 2), "d1": (11, -0.1, 1), "p": (13, -0.3, 3)}
    for row, req_id in enumerate(order):
        count = 0 if req_id == "p" and not sample_p else 1
        result = merged.logprobs.slice_request(row, count)
        assert result.logprobs.shape[0] == count
        if count:
            token, value, rank = expected[req_id]
            assert result.logprob_token_ids[0, 0] == token
            assert result.logprobs[0, 0] == pytest.approx(value)
            assert result.sampled_token_ranks[0] == rank
    # A narrower Decode top-k must not inherit Prefill's extra candidates.
    if sample_p:
        result = merged.logprobs.slice_request(order.index("d0"), 1)
        assert np.isneginf(result.logprobs[0, 2])


def test_merge_logprobs_keeps_offsets_when_one_sample_does_not_request_logprobs():
    outputs = [
        ModelRunnerOutput(req_ids=["d"], req_id_to_index={"d": 0}, sampled_token_ids=[[7]]),
        ModelRunnerOutput(
            req_ids=["p"],
            req_id_to_index={"p": 0},
            sampled_token_ids=[[8]],
            logprobs=LogprobsLists(np.array([[8]]), np.array([[-0.5]]), np.array([1])),
        ),
    ]
    merged = NPUModelRunner._merge_layered_v2_outputs(_scheduler_output({"d": 1, "p": 4}), outputs)
    assert merged.logprobs.cu_num_generated_tokens == [0, 1, 2]
    assert merged.logprobs.slice_request(1, 1).logprob_token_ids.tolist() == [[8]]


def test_frontier_store_clears_finished_and_preempted():
    store = LayeredPrefillStateStore()
    hs = torch.zeros(2, 4)
    store.put(
        LayeredFrontier(
            req_id="a",
            group_id=1,
            query_len=2,
            hidden_states=hs,
            residual=None,
        )
    )
    store.put(
        LayeredFrontier(
            req_id="b",
            group_id=2,
            query_len=2,
            hidden_states=hs.clone(),
            residual=None,
        )
    )
    store.clear_many({"a"})
    assert store.get("a") is None
    assert store.get("b") is not None
    store.clear_many(["b"])
    assert store.get("b") is None


def test_detach_keeps_d_hidden_alive_after_alias_write():
    hidden = torch.arange(4, dtype=torch.float32)
    state = ExecuteModelState(
        input_batch=object(),  # type: ignore[arg-type]
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=hidden,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    detached = detach_execute_model_state(state)
    hidden.fill_(-1)
    assert detached.hidden_states is not None
    assert detached.hidden_states.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_layered_v2_state_sample_p_flag():
    state = LayeredV2ExecuteModelState(
        scheduler_output=_scheduler_output({"p0": 4}),
        d_state=None,
        p_state=MagicMock(),
        sample_p=False,
    )
    assert state.sample_p is False
    assert isinstance(state.scheduler_output, SchedulerOutput)


def test_pad_for_sequence_parallelism_rounds_up_when_dsa_cp(monkeypatch):
    from types import SimpleNamespace

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=8))
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.enable_dsa_cp", lambda: True)
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.enable_sp", lambda *args, **kwargs: False)
    assert runner._pad_for_sequence_parallelism(4100) == 4104
    assert runner._pad_for_sequence_parallelism(4104) == 4104


def test_pad_for_sequence_parallelism_noop_without_dsa_cp(monkeypatch):
    from types import SimpleNamespace

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=8))
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.enable_dsa_cp", lambda: False)
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.enable_sp", lambda *args, **kwargs: False)
    assert runner._pad_for_sequence_parallelism(4100) == 4100


def test_prepare_attn_splits_actual_and_padded_input(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    captured: dict = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_states.default.build_attn_metadata",
        fake_build_attn_metadata,
    )
    state = AscendModelState.__new__(AscendModelState)
    state.max_model_len = 8192
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=4100,
        num_tokens_after_padding=4104,
        query_start_loc_np=np.array([0, 4100], dtype=np.int32),
        query_start_loc=object(),
        is_prefilling_np=np.array([True]),
        num_scheduled_tokens=np.array([4100], dtype=np.int32),
        seq_lens=object(),
        seq_lens_np=np.array([4100], dtype=np.int32),
        positions=object(),
        attn_state=None,
        dcp_local_seq_lens=None,
    )
    metadata = state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=object(),
        attn_groups=[],
        kv_cache_config=object(),
        num_input_tokens=4104,
    )
    assert metadata == {"ok": True}
    assert captured["num_actual_tokens"] == 4100
    assert captured["num_input_tokens"] == 4104
    assert captured["num_tokens"] == 4104
