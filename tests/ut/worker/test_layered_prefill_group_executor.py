# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for the shared MRV2 layer-range execution cache."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend.worker import layered_prefill_decode_graph as graphs
from vllm_ascend.worker import layered_prefill_single_forward as execution


def runner_for(*, eager=False, compile_enabled=True, graph_enabled=True):
    calls = []

    def layer(index, positions, hidden, residual, ids):
        calls.append(index)
        return hidden + index + 1, residual

    return NS(
        device=torch.device("cpu"),
        tp_rank=0,
        calls=calls,
        layered_prefill_model_adapter=NS(start_layer=0, end_layer=4, layers=list(range(4)), _forward_layer=layer),
        model_config=NS(enforce_eager=eager),
        compilation_config=NS(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        ascend_config=NS(
            scheduler_config=NS(
                layered_prefill_config=NS(
                    single_forward_compile=compile_enabled,
                    single_forward_decode_graph=graph_enabled,
                )
            )
        ),
        vllm_config=NS(
            additional_config={"original": True},
            compilation_config=NS(cache_dir="full-model-cache", traced_files={"original"}),
        ),
        _set_layered_prefill_moe_layer_offset=lambda _: None,
    )


@pytest.mark.parametrize("eager,enabled", [(True, True), (False, False)])
def test_disabled_compilation_executes_only_requested_layers(eager, enabled):
    runner = runner_for(eager=eager, compile_enabled=enabled)
    hidden, residual = execution.LayeredPrefillGroupExecutor(runner).forward_mixed(
        1, 3, (torch.zeros(2), None), None, None
    )
    assert runner.calls == [1, 2]
    torch.testing.assert_close(hidden, torch.full((2,), 5.0))
    assert residual is None


def test_compile_cache_separates_range_communication_and_residual(monkeypatch):
    created, calls = [], []

    class Compiled:
        def __init__(self, *, adapter, start, end, vllm_config):
            self.vllm_config = vllm_config
            self.range = (start, end)
            created.append(self)

        def __call__(self, hidden, residual, positions, ids):
            calls.append(self.range)
            return hidden + 1, residual

    monkeypatch.setitem(
        sys.modules, "vllm_ascend.worker.layered_prefill_compiled", NS(LayeredPrefillCompiledGroup=Compiled)
    )
    import vllm.compilation.backends as backends

    monkeypatch.setattr(backends, "set_model_tag", lambda _: nullcontext())
    monkeypatch.setattr(execution, "set_current_vllm_config", lambda _: nullcontext())
    context = NS(moe_comm_type="allgather")
    monkeypatch.setattr(execution, "get_forward_context", lambda: context)
    runner = runner_for()
    executor = execution.LayeredPrefillGroupExecutor(runner)
    for start in (0, 2, 0):
        executor.forward_mixed(start, start + 2, (torch.zeros(2), None), None, None)
    assert len(created) == 2
    context.moe_comm_type = "mc2"
    executor.forward_mixed(0, 2, (torch.zeros(2), None), None, None)
    executor.forward_mixed(0, 2, (torch.zeros(2), torch.zeros(2)), None, None)
    assert len(created) == 4
    assert len(calls) == 5  # No extra warmup on live KV caches.
    keys = [group.vllm_config.additional_config["layered_prefill_compile_key"] for group in created]
    assert len(set(keys)) == 4
    created[0].vllm_config.compilation_config.traced_files.add("group")
    assert runner.vllm_config.additional_config == {"original": True}
    assert runner.vllm_config.compilation_config.traced_files == {"original"}
    assert runner.vllm_config.compilation_config.cache_dir == "full-model-cache"


def test_decode_graph_ranges_do_not_skip_partial_groups():
    runner = runner_for()
    executor = execution.LayeredPrefillGroupExecutor(runner)
    ranges = []

    def forward(start, end, state, ids, positions):
        ranges.append((start, end))
        return state

    executor.decode_graph_cache = NS(forward=forward)
    executor.forward_decode(NS(num_groups=2), 1, 4, (torch.zeros(2), None), None, None)
    assert ranges == [(1, 2), (2, 4)]


@pytest.mark.parametrize(
    "partition,stage_start,stage_end,start,end",
    [
        ("22,21", 0, 22, 11, 22),
        ("22,21", 0, 22, 0, 11),
        ("22,21", 22, 43, 33, 43),
        ("22,21", 22, 43, 22, 33),
        ("20,23", 0, 20, 10, 20),
        ("20,23", 0, 20, 0, 10),
        ("20,23", 20, 43, 32, 43),
        ("20,23", 20, 43, 20, 32),
    ],
)
def test_pp2_g4_decode_uses_one_aligned_graph_per_half_stage(
    monkeypatch, partition, stage_start, stage_end, start, end
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    runner = runner_for()
    runner.parallel_config = NS(pipeline_parallel_size=2)
    adapter = runner.layered_prefill_model_adapter
    adapter.start_layer, adapter.end_layer = stage_start, stage_end
    adapter.num_hidden_layers = 43
    executor = execution.LayeredPrefillGroupExecutor(runner)
    ranges, visited = [], []

    def forward(first, last, state, ids, positions, ranges=ranges, visited=visited):
        ranges.append((first, last))
        visited.extend(range(first, last))
        hidden, residual = state
        return hidden + sum(range(first, last)), residual

    executor.decode_graph_cache = NS(forward=forward)
    hidden, residual = executor.forward_decode(NS(num_groups=4), start, end, (torch.zeros(2), None), None, None)
    assert ranges == [(start, end)]
    assert visited == list(range(start, end))
    torch.testing.assert_close(hidden, torch.full((2,), float(sum(range(start, end)))))
    assert residual is None


@pytest.mark.parametrize("num_groups", [2, 3, 4, 5, 8, 16, 43])
@pytest.mark.parametrize("partition", ["22,21", "20,23"])
def test_pp2_all_group_counts_decode_prefix_and_suffix_visit_each_layer_once(monkeypatch, num_groups, partition):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    boundary = int(partition.split(",")[0])
    groups = execution.make_pp_aligned_layer_group_ranges(43, num_groups, 2)
    for active in groups:
        stage_start, stage_end = (0, boundary) if active.end <= boundary else (boundary, 43)
        runner = runner_for()
        runner.parallel_config = NS(pipeline_parallel_size=2)
        adapter = runner.layered_prefill_model_adapter
        adapter.start_layer, adapter.end_layer = stage_start, stage_end
        adapter.num_hidden_layers = 43
        executor = execution.LayeredPrefillGroupExecutor(runner)
        ranges, visited = [], []

        def forward(first, last, state, ids, positions, ranges=ranges, visited=visited):
            ranges.append((first, last))
            visited.extend(range(first, last))
            hidden, residual = state
            return hidden + sum(range(first, last)), residual

        executor.decode_graph_cache = NS(forward=forward)
        state = (torch.zeros(2), None)
        for start, end in ((stage_start, active.start), (active.end, stage_end)):
            state = executor.forward_decode(NS(num_groups=num_groups), start, end, state, None, None)
        expected = [g for g in groups if stage_start <= g.start and g.end <= stage_end and g != active]
        assert ranges == [(g.start, g.end) for g in expected]
        assert visited == [layer for g in expected for layer in range(g.start, g.end)]
        assert sorted(visited + list(range(active.start, active.end))) == list(range(stage_start, stage_end))
        torch.testing.assert_close(state[0], torch.full((2,), float(sum(visited))))


@pytest.mark.parametrize("pp_size,num_groups", [(None, 4), (1, 4), (4, 4)])
def test_other_decode_envelopes_keep_existing_local_boundaries(monkeypatch, pp_size, num_groups):
    runner = runner_for()
    if pp_size is not None:
        runner.parallel_config = NS(pipeline_parallel_size=pp_size)
    runner.layered_prefill_model_adapter.end_layer = 22
    # Legacy mocks need neither global layer count nor a PP config.
    assert not hasattr(runner.layered_prefill_model_adapter, "num_hidden_layers")

    def unexpected_pp_ranges(*args):
        pytest.fail("Only PP2 may use the aligned Decode boundaries")

    monkeypatch.setattr(execution, "make_pp_aligned_layer_group_ranges", unexpected_pp_ranges)
    executor = execution.LayeredPrefillGroupExecutor(runner)
    ranges = []

    def forward(start, end, state, ids, positions):
        ranges.append((start, end))
        return state

    executor.decode_graph_cache = NS(forward=forward)
    executor.forward_decode(NS(num_groups=num_groups), 11, 22, (torch.zeros(2), None), None, None)
    assert ranges == ([(11, 22)] if num_groups == 2 else [(11, 12), (12, 17), (17, 22)])


def test_snapshot_keeps_aliases_but_isolates_wrappers_and_tensor_storage():
    tensor = torch.arange(4)
    source = {"first": NS(value=tensor), "second": tensor}
    target = graphs.snapshot_tree(source)
    assert target["first"] is not source["first"]
    assert target["first"].value is target["second"]
    assert target["second"].data_ptr() != tensor.data_ptr()
    tensor.add_(10)
    graphs.refresh_tree(target, source)
    torch.testing.assert_close(target["second"], tensor)
    with pytest.raises(TypeError, match="Unsupported"):
        graphs.graph_signature(object())


def test_decode_capture_restores_and_refreshes_v2_context(monkeypatch):
    runner = runner_for()
    stream = NS(wait_stream=lambda _: None)
    replayed = []
    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            graph_pool_handle=lambda: object(),
            Stream=lambda **_: stream,
            current_stream=lambda: stream,
            NPUGraph=lambda: NS(replay=lambda: replayed.append(True)),
            graph=lambda *_, **kwargs: nullcontext(),
        ),
        raising=False,
    )
    offloader = NS(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
    monkeypatch.setattr(graphs, "get_offloader", lambda: offloader)
    context = NS(
        attn_metadata={"layer": NS(length=2)},
        slot_mapping={"layer": torch.tensor([7, 9])},
        is_padding=torch.tensor([False, False]),
        capturing=False,
        moe_comm_type="allgather",
        num_tokens=2,
    )
    monkeypatch.setattr(graphs, "get_forward_context", lambda: context)
    cache = graphs.LayeredDecodeGraphCache(runner)
    old_slots, old_padding = context.slot_mapping, context.is_padding
    cache.forward(0, 2, (torch.zeros(2), None), torch.zeros(2), torch.zeros(2))
    assert context.slot_mapping is old_slots
    assert context.is_padding is old_padding
    assert context.capturing is False
    assert runner.calls == [0, 1]
    # New buffers with the same host contract must reuse captured addresses.
    context.slot_mapping = {"layer": old_slots["layer"].clone()}
    context.is_padding = old_padding.clone()
    cache.forward(0, 2, (torch.zeros(2), None), torch.zeros(2), torch.zeros(2))
    assert len(cache.entries) == 1
    assert runner.calls == [0, 1]
    assert len(replayed) == 2
    assert context.slot_mapping is not old_slots


def test_decode_capture_exception_restores_context(monkeypatch):
    runner = runner_for()
    stream = NS(wait_stream=lambda _: None)
    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            graph_pool_handle=lambda: object(),
            Stream=lambda **_: stream,
            current_stream=lambda: stream,
            NPUGraph=lambda: object(),
            graph=lambda *_, **kwargs: nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setattr(graphs, "get_offloader", lambda: NS(sync_prev_onload=lambda: None))
    context = NS(
        attn_metadata={}, slot_mapping={}, is_padding=torch.zeros(1), capturing=False, moe_comm_type="allgather"
    )
    original = dict(vars(context))
    monkeypatch.setattr(graphs, "get_forward_context", lambda: context)

    def fail(*args):
        raise RuntimeError("capture failed")

    runner.layered_prefill_model_adapter._forward_layer = fail
    cache = graphs.LayeredDecodeGraphCache(runner)
    with pytest.raises(RuntimeError, match="capture failed"):
        cache.forward(0, 1, (torch.zeros(1), None), None, None)
    assert vars(context).keys() == original.keys()
    for name, value in original.items():
        assert getattr(context, name) is value
