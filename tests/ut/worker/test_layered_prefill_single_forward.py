# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.layered_prefill import LayeredPrefillStateStore

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.layered_prefill_single_forward import (
    LayeredPrefillBatch,
    LayeredPrefillSingleForward,
)


class Adapter:
    start_layer, end_layer = 0, 4
    layers = tuple(range(4))

    def __init__(self, hyper=False):
        self.calls = []
        self.hyper = hyper

    def _validate_range(self, start, end):
        assert 0 <= start < end <= self.end_layer

    def _prepare_initial_state(self, *, input_ids, **kwargs):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 3)
        if self.hyper:
            hidden = hidden.unsqueeze(1).repeat(1, 2, 1)
        return hidden, None

    def _forward_layer(self, layer, positions, hidden, residual, input_ids):
        self.calls.append((layer, input_ids.clone()))
        # Also exercise hash-routing IDs and token positions for resumed P rows.
        shape = (-1, *([1] * (hidden.ndim - 1)))
        value = hidden + (layer + 1) + input_ids.reshape(shape) + positions.reshape(shape)
        return value, None if self.hyper else hidden.clone()

    def _finalize(self, hidden, residual):
        return (hidden.mean(1) if self.hyper else hidden + residual), None


def runner_for(req_ids=("d0", "p", "d1"), lengths=(1, 5, 1), hyper=False):
    starts = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    runner = NS(
        input_batch=NS(req_ids=list(req_ids)),
        query_start_loc=NS(cpu=starts),
        discard_request_mask=NS(np=np.zeros(len(req_ids), dtype=bool)),
        device=torch.device("cpu"),
        layered_prefill_state=LayeredPrefillStateStore(),
        layered_prefill_model_adapter=Adapter(hyper),
        _pad_for_sequence_parallelism=lambda n: ((n + 3) // 4) * 4,
    )

    def restore(snapshot):
        indices, count, mask = snapshot
        runner.discard_request_mask.np[:] = mask
        runner.discarded = indices

    runner._restore_layered_sampling_masks = restore
    return runner


def plan(group, final=False, query=5):
    return NS(
        prefill_req_ids=("p",),
        query_tokens={"p": query},
        group_id=group,
        group_start=group * 2,
        group_end=group * 2 + 2,
        is_sampling_step=final,
        is_final_group=final,
    )


@pytest.mark.parametrize("hyper", [False, True])
@pytest.mark.parametrize("d_reqs", [0, 1, 3])
@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("graph_enabled", [False, True])
def test_one_layer_call_per_step_and_full_forward_equivalence(monkeypatch, hyper, d_reqs, groups, graph_enabled):
    """Compaction preserves D/P results, padding and both residual contracts."""
    monkeypatch.setattr(LayeredPrefillBatch, "_context", lambda *a, **k: nullcontext())
    reqs = tuple([f"d{i}" for i in range(d_reqs)] + ["p"])
    lengths = tuple([1] * d_reqs + [5])
    runner = runner_for(reqs, lengths, hyper)
    executor = LayeredPrefillSingleForward(runner)
    if graph_enabled:
        runner.compilation_config = NS(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
        runner.model_config = NS(enforce_eager=False)
        runner.ascend_config = NS(scheduler_config=NS(layered_prefill_config=NS(single_forward_decode_graph=True)))

        def graph_forward(start, end, state, input_ids, positions):
            for idx in range(start, end):
                state = runner.layered_prefill_model_adapter._forward_layer(idx, positions, *state, input_ids)
            return state

        executor.decode_graph_cache = NS(forward=graph_forward)
    size = runner._pad_for_sequence_parallelism(sum(lengths))
    ids = torch.arange(1, size + 1)
    positions = torch.arange(size)
    reference = Adapter(hyper)
    state = reference._prepare_initial_state(input_ids=ids)
    for idx in reference.layers:
        state = reference._forward_layer(idx, positions, *state, ids)
    expected, _ = reference._finalize(*state)
    for group in range(groups):
        runner.layered_prefill_model_adapter.calls.clear()
        step_plan = plan(group, final=group == groups - 1)
        step_plan.num_groups = groups
        step_plan.group_start = group * (4 // groups)
        step_plan.group_end = (group + 1) * (4 // groups)
        batch = executor.prepare(step_plan, size)
        actual = batch.forward(ids, positions, None, {})
        calls = runner.layered_prefill_model_adapter.calls
        assert [c[0] for c in calls] == (
            list(range(4)) if d_reqs else list(range(step_plan.group_start, step_plan.group_end))
        )
        if d_reqs:
            torch.testing.assert_close(actual[:d_reqs], expected[:d_reqs])
        if group < groups - 1:
            frontier = runner.layered_prefill_state.get("p")
            assert frontier.hidden_states.shape[0] == 5
            assert runner.discard_request_mask.np[-1]
        else:
            torch.testing.assert_close(actual[d_reqs : d_reqs + 5], expected[d_reqs : d_reqs + 5])
            assert runner.layered_prefill_state.get("p") is None


def test_compact_common_metadata_owns_selected_rows_and_invalid_padding():
    runner = runner_for()
    batch = LayeredPrefillSingleForward(runner).prepare(plan(0), 8)
    seq = torch.tensor([101, 205, 301], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=runner.query_start_loc.cpu,
        query_start_loc_cpu=runner.query_start_loc.cpu,
        seq_lens=seq,
        _seq_lens_cpu=seq,
        seq_lens_cpu=seq,
        num_computed_tokens_cpu=seq - torch.tensor([1, 5, 1]),
        num_reqs=3,
        num_actual_tokens=7,
        num_input_tokens=8,
        max_query_len=5,
        max_seq_len=301,
        block_table_tensor=torch.arange(12).reshape(3, 4),
        slot_mapping=torch.arange(8),
        positions=torch.arange(100, 108),
        positions_cpu=torch.arange(100, 108),
        is_prefilling=torch.tensor([False, True, False]),
        actual_seq_lengths_q=[1, 6, 7],
        attn_state=AscendAttentionState.ChunkedPrefill,
        group_len=torch.arange(3),
        group_key_idx=torch.arange(3),
        group_key_cache_idx=torch.arange(3),
    )
    compact = batch.compact_metadata(common)
    assert compact.num_reqs == 4
    assert compact.num_actual_tokens == 2
    assert compact.num_input_tokens == 4
    assert compact.seq_lens.tolist() == [101, 301, 0, 0]
    assert compact.query_start_loc.tolist() == [0, 1, 2, 3, 4]
    assert compact.slot_mapping.tolist() == [0, 6, -1, -1]
    assert compact.positions.tolist() == [100, 106, 0, 0]
    assert compact.attn_state == AscendAttentionState.DecodeOnly
    assert compact.block_table_tensor[:2].tolist() == common.block_table_tensor[[0, 2]].tolist()
    compact.seq_lens.fill_(0)
    assert common.seq_lens.tolist() == [101, 205, 301]


def test_decode_builder_is_cached_but_never_aliases_mixed_workspace():
    runner = NS(vllm_config=object(), device=torch.device("cpu"))

    class Builder:
        def __init__(self, kv_cache_spec, **kwargs):
            self.kv_cache_spec = kv_cache_spec
            self.buffer = torch.zeros(2)

    original = Builder(object())
    executor = LayeredPrefillSingleForward(runner)
    group = NS(layer_names=["layer0"])
    decode = executor.builder(0, 0, group, original)
    assert decode is executor.builder(0, 0, group, original)
    decode.buffer.fill_(1)
    assert original.buffer.tolist() == [0, 0]


def test_resumed_group_requires_matching_frontier(monkeypatch):
    monkeypatch.setattr(LayeredPrefillBatch, "_context", lambda *a, **k: nullcontext())
    batch = LayeredPrefillSingleForward(runner_for()).prepare(plan(1, True), 8)
    with pytest.raises(RuntimeError, match="frontier"):
        batch.forward(torch.arange(8), torch.arange(8), None, {})


def test_reuses_frontier_storage_without_aliasing_model_workspace():
    batch = LayeredPrefillSingleForward(runner_for()).prepare(plan(0), 8)
    value = torch.arange(15).reshape(5, 3).float()
    batch._save_frontier((value, None))
    stored = batch.runner.layered_prefill_state.get("p").hidden_states
    assert stored.data_ptr() != value.data_ptr()
    batch._save_frontier((value + 1, None))
    assert batch.runner.layered_prefill_state.get("p").hidden_states.data_ptr() == stored.data_ptr()
    torch.testing.assert_close(stored, value + 1)


@pytest.mark.parametrize("enabled", [True, False])
def test_single_forward_config_parsers_agree(enabled):
    from vllm.v1.core.layered_prefill import LayeredPrefillConfig

    from vllm_ascend.ascend_config import SchedulerConfig

    additional = {"scheduler_config": {"layered_prefill_config": {"single_forward": enabled}}}
    assert LayeredPrefillConfig.from_vllm_config(NS(additional_config=additional)).single_forward == enabled
    assert SchedulerConfig(additional, balance_env_value=False).layered_prefill_config.single_forward == enabled


@pytest.mark.parametrize(
    "field,value",
    [("data_parallel_size", 2), ("pipeline_parallel_size", 2), ("enable_dbo", True)],
)
def test_unsupported_parallel_modes_keep_existing_path(field, value):
    runner = NS(
        parallel_config=NS(data_parallel_size=1, pipeline_parallel_size=1, enable_dbo=False),
        dcp_size=1,
        use_dcp=False,
        speculative_config=None,
        use_async_scheduling=False,
        supports_mm_inputs=False,
        is_pooling_model=False,
        sparse_kv_offload_enabled=False,
        dynamic_eplb=False,
        use_aux_hidden_state_outputs=False,
        cache_config=NS(mamba_cache_mode="none"),
        max_num_reqs=64,
        _pad_for_sequence_parallelism=lambda n: ((n + 3) // 4) * 4,
    )
    assert LayeredPrefillSingleForward.supported(runner)
    setattr(runner.parallel_config, field, value)
    assert not LayeredPrefillSingleForward.supported(runner)


@dataclass
class _WrappedMetadata:
    values: object
    host_length: int


def test_decode_graph_snapshot_keeps_aliases_and_backend_wrappers_private():
    from vllm_ascend.worker.layered_prefill_decode_graph import (
        graph_signature,
        refresh_tree,
        snapshot_tree,
    )

    # Backends can wrap shared tensor leaves in ordinary Python objects.
    tensor = torch.arange(4).float()
    wrapper = NS(data={"rope": (tensor, tensor)}, index=0)
    source = {"layer0": _WrappedMetadata(wrapper, 4)}
    source["layer1"] = source["layer0"]
    target = snapshot_tree(source)
    assert target["layer0"] is target["layer1"]
    copied = target["layer0"].values.data["rope"]
    assert copied[0] is copied[1]
    assert copied[0].data_ptr() != tensor.data_ptr()
    before = graph_signature(source)
    tensor.add_(10)
    refresh_tree(target, source)
    torch.testing.assert_close(copied[0], tensor)
    # CPU tensor values are host-side constants and must change the signature.
    assert graph_signature(source) != before
    source["layer0"].host_length = 5
    assert graph_signature(source) != graph_signature(target)


def test_decode_graph_refuses_opaque_metadata_instead_of_replaying_it():
    from vllm_ascend.worker.layered_prefill_decode_graph import graph_signature

    with pytest.raises(TypeError, match="Unsupported"):
        graph_signature({"metadata": object()})


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("field", ["single_forward_decode_graph", "single_forward_compile"])
def test_optimization_switch_has_matching_scheduler_and_worker_values(enabled, field):
    from vllm.v1.core.layered_prefill import LayeredPrefillConfig

    from vllm_ascend.ascend_config import SchedulerConfig

    additional = {"scheduler_config": {"layered_prefill_config": {field: enabled}}}
    assert (
        getattr(
            LayeredPrefillConfig.from_vllm_config(NS(additional_config=additional)),
            field,
        )
        == enabled
    )
    assert getattr(SchedulerConfig(additional, False).layered_prefill_config, field) == enabled


def test_graph_metadata_borrows_only_decode_storage_not_mixed_builder_storage():
    from vllm_ascend.worker.layered_prefill_decode_graph import (
        refresh_tree,
        snapshot_tree,
    )

    mixed = torch.arange(6).float()
    decode = mixed[[0, 5]]
    source = NS(tensor=decode)
    target = snapshot_tree(source, clone_tensors=False)
    assert target is not source
    assert target.tensor is decode
    address = target.tensor.data_ptr()
    refresh_tree(target, NS(tensor=torch.tensor([10.0, 20.0])))
    assert target.tensor.data_ptr() == address
    torch.testing.assert_close(target.tensor, torch.tensor([10.0, 20.0]))
    torch.testing.assert_close(mixed, torch.arange(6).float())


def test_final_group_flag_must_agree_with_model_depth(monkeypatch):
    monkeypatch.setattr(LayeredPrefillBatch, "_context", lambda *a, **k: nullcontext())
    batch = LayeredPrefillSingleForward(runner_for()).prepare(plan(0, final=True), 8)
    with pytest.raises(ValueError, match="final-group"):
        batch.forward(torch.arange(8), torch.arange(8), None, {})


@pytest.mark.parametrize("prefix", [False, True])
def test_metadata_buffers_are_reused_refreshed_and_isolated_by_kv_group(prefix):
    runner = runner_for(("d0", "d1", "p"), (1, 1, 5)) if prefix else runner_for()
    executor = LayeredPrefillSingleForward(runner)
    batch = executor.prepare(plan(0), 8)
    value = torch.arange(12).reshape(3, 4)
    result = batch._compact_rows(value, (0, "blocks"))
    address = result.data_ptr()
    torch.testing.assert_close(result[:2], value[batch.d_requests])
    assert result[2:].count_nonzero() == 0
    other = batch._compact_rows(value + 100, (1, "blocks"))
    assert other.data_ptr() != address
    again = executor.prepare(plan(0), 8)._compact_rows(value + 10, (0, "blocks"))
    assert again.data_ptr() == address
    torch.testing.assert_close(again[:2], (value + 10)[batch.d_requests])
    torch.testing.assert_close(other[:2], (value + 100)[batch.d_requests])
    torch.testing.assert_close(value, torch.arange(12).reshape(3, 4))


@pytest.mark.parametrize("hyper", [False, True])
@pytest.mark.parametrize("p_start", [0, 1, 2])
def test_join_initializes_every_real_and_padding_row(hyper, p_start):
    shape = (2, 3) if hyper else (3,)
    p = torch.full((5, *shape), 7.0)
    d = torch.full((4, *shape), 11.0)
    indices = torch.tensor([i for i in range(7) if not p_start <= i < p_start + 5])
    mixed, residual = LayeredPrefillBatch._join((d, None), (p, None), p_start, p_start + 5, indices, 8)
    torch.testing.assert_close(mixed[p_start : p_start + 5], p)
    torch.testing.assert_close(mixed[indices], d[:2])
    assert mixed[7:].count_nonzero() == 0
    assert residual is None


def test_compile_groups_cached_by_range_without_extra_live_forward(monkeypatch):
    import sys

    from vllm.config import CompilationMode

    from vllm_ascend.worker import layered_prefill_single_forward as module

    calls = []
    created = []

    class Compiled:
        def __init__(self, *, adapter, start, end, vllm_config):
            self.range = (start, end)
            self.vllm_config = vllm_config
            created.append(self.range)

        def __call__(self, hidden, residual, positions, input_ids):
            calls.append(self.range)
            return hidden + self.range[0] + 1, residual

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.layered_prefill_compiled",
        NS(LayeredPrefillCompiledGroup=Compiled),
    )
    monkeypatch.setattr(module, "set_current_vllm_config", lambda c: nullcontext())
    context = NS(moe_comm_type="allgather")
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    runner = runner_for()
    runner.compilation_config = NS(mode=CompilationMode.VLLM_COMPILE)
    runner.model_config = NS(enforce_eager=False)
    runner.vllm_config = NS(
        additional_config={"original": True},
        compilation_config=NS(cache_dir="full-model-cache", traced_files={"original"}),
    )
    runner.ascend_config = NS(scheduler_config=NS(layered_prefill_config=NS(single_forward_compile=True)))
    executor = LayeredPrefillSingleForward(runner)
    state = (torch.zeros(1), None)
    for start in (0, 2, 0):
        result, _ = executor.forward_mixed(start, start + 2, state, None, None)
        torch.testing.assert_close(result, torch.tensor([start + 1.0]))
    assert calls == [(0, 2), (2, 4), (0, 2)]
    assert created == [(0, 2), (2, 4)]
    context.moe_comm_type = "mc2"
    executor.forward_mixed(0, 2, state, None, None)
    assert created == [(0, 2), (2, 4), (0, 2)]
    configs = [group.vllm_config for group in executor.compiled_groups.values()]
    assert len({c.additional_config["layered_prefill_compile_key"] for c in configs}) == 3
    assert runner.vllm_config.additional_config == {"original": True}
    assert runner.vllm_config.compilation_config.cache_dir == "full-model-cache"
    configs[0].compilation_config.traced_files.add("layer-group")
    assert runner.vllm_config.compilation_config.traced_files == {"original"}
