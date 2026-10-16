"""Unit tests for Layered Prefill V2 scheduler-output subset helpers."""

from types import SimpleNamespace

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_ascend.worker.v2.layered_prefill import (
    detach_execute_model_state,
    split_d_p_req_ids,
    subset_scheduler_output,
)


def _empty_cached() -> CachedRequestData:
    return CachedRequestData.make_empty()


def test_subset_scheduler_output_strips_one_time_updates():
    base = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=_empty_cached(),
        num_scheduled_tokens={"d0": 1, "p0": 8},
        total_num_scheduled_tokens=9,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={"done"},
        free_encoder_mm_hashes=["h"],
        preempted_req_ids={"pre"},
        new_block_ids_to_zero=[1, 2],
        layered_prefill_plan=object(),
    )
    cropped = subset_scheduler_output(
        base,
        ["p0"],
        layered_plan="plan",
        include_one_time_updates=False,
    )
    assert cropped.num_scheduled_tokens == {"p0": 8}
    assert cropped.total_num_scheduled_tokens == 8
    assert cropped.finished_req_ids == set()
    assert cropped.preempted_req_ids == set()
    assert cropped.free_encoder_mm_hashes == []
    assert cropped.new_block_ids_to_zero is None
    assert cropped.layered_prefill_plan == "plan"

    with_updates = subset_scheduler_output(
        base,
        ["d0"],
        layered_plan=None,
        include_one_time_updates=True,
    )
    assert with_updates.finished_req_ids == {"done"}
    assert with_updates.layered_prefill_plan is None


def test_split_d_p_req_ids_orders_decode_and_prefill():
    plan = SimpleNamespace(prefill_req_ids=["p0"])
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"d0": 1, "p0": 16, "d1": 1})
    d_ids, p_ids = split_d_p_req_ids(scheduler_output, plan)
    assert p_ids == ["p0"]
    assert d_ids == ["d0", "d1"]


def test_detach_execute_model_state_clones_hidden_tensors():
    import torch
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState

    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    aux = [torch.ones(2, 3), torch.zeros(2, 3)]
    state = ExecuteModelState(
        input_batch=object(),  # type: ignore[arg-type]
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=hidden,
        aux_hidden_states=aux,
        finished_req_ids=set(),
    )
    detached = detach_execute_model_state(state)
    assert detached.hidden_states is not hidden
    assert torch.equal(detached.hidden_states, hidden)
    assert detached.aux_hidden_states is not aux
    assert detached.aux_hidden_states[0] is not aux[0]
    hidden.fill_(7)
    aux[0].fill_(9)
    assert detached.hidden_states[0, 0].item() == 0
    assert detached.aux_hidden_states[0][0, 0].item() == 1
