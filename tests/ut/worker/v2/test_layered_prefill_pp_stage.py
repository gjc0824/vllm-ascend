# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tensor contracts for the PP owner mixed-stage execution path."""

import ast
import re
import runpy
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

PP2_RANGES = runpy.run_path(str(Path(__file__).with_name("test_layered_prefill_nonowner.py")))["PP2_RANGES"]


@pytest.fixture
def implementation():
    base = runpy.run_path(str(Path(__file__).with_name("test_layered_prefill_single_forward.py")))
    single = base["implementation"].__wrapped__()
    source = Path(__file__).parents[4] / "vllm_ascend/worker/v2/layered_prefill_pp_stage.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

    class Intermediate:
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, key):
            return self.tensors[key]

    dispatch_calls = []

    def dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return NS(num_tokens=8), None

    namespace = dict(
        torch=torch,
        nullcontext=nullcontext,
        SimpleNamespace=NS,
        IntermediateTensors=Intermediate,
        ExecuteModelState=NS,
        LayeredV2ExecuteModelState=NS,
        LayeredV2SingleForward=single.LayeredV2SingleForward,
        get_pp_group=lambda: NS(rank_in_group=0),
        dispatch_cg_and_sync_dp=dispatch,
        get_uniform_token_count=lambda reqs, tokens, query: query,
        sort_batch_req_ids=lambda tokens, query: list(tokens),
        subset_scheduler_output=lambda schedule, ids, **kwargs: NS(
            num_scheduled_tokens={req: schedule.num_scheduled_tokens[req] for req in ids},
            finished_req_ids=schedule.finished_req_ids,
        ),
    )
    exec(compile(tree, str(source), "exec"), namespace)
    transport_tree = ast.parse(source.with_name("layered_prefill_nonowner.py").read_text(encoding="utf-8"))
    transport_tree.body = [
        n
        for n in transport_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "can_compact_internal_prefill_payload"
    ]
    exec(compile(transport_tree, str(source), "exec"), namespace)
    # Use the real wire packer: only the transport wrapper is replaced.
    runner_source = source.with_name("model_runner.py")
    runner_tree = ast.parse(runner_source.read_text(encoding="utf-8"))
    runner_class = next(n for n in runner_tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    packer = next(
        n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "_combine_layered_pp_intermediate"
    )
    packer.decorator_list = []
    trimmer = next(
        n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "_layered_nonfused_pp_payload"
    )
    exec(
        compile(
            ast.Module(body=[packer, trimmer], type_ignores=[]),
            str(runner_source),
            "exec",
        ),
        dict(namespace, torch=torch),
        namespace,
    )
    return NS(base=base, single=single, dispatch_calls=dispatch_calls, **namespace)


def stage_runner(impl, last, hc_mult=1):
    runner = impl.base["supported_runner"]()
    runner.parallel_config.pipeline_parallel_size = 2
    runner.is_first_pp_rank = not last
    runner.is_last_pp_rank = last
    runner._single_forward = True
    runner.vllm_config = NS(kv_transfer_config=None)
    runner.decode_query_len = 1
    runner.dp_size = 1
    runner.dp_rank = 0
    runner.cudagraph_manager = object()
    runner._pad_for_sequence_parallelism = lambda n: (n + 1) // 2 * 2
    runner.layered_prefill_state = impl.base["Store"]()
    calls = []
    initial_calls = []

    def initial(**kwargs):
        initial_calls.append(kwargs)
        if kwargs["frontier"] is not None:
            return kwargs["frontier"]
        intermediate = kwargs["intermediate_tensors"]
        if intermediate is not None:
            return intermediate["hidden_states"], intermediate.tensors.get("residual")
        hidden = kwargs["input_ids"].float().unsqueeze(1)
        if hc_mult > 1:
            hidden = hidden.unsqueeze(1).repeat(1, hc_mult, 1)
        return hidden, None

    def layers(start, end, state, ids, positions):
        hidden, residual = state
        for index in range(start, end):
            calls.append(index)
            hidden = hidden * (index + 2) + positions.reshape((-1,) + (1,) * (hidden.ndim - 1))
        runner.kernel_output = hidden
        return hidden, residual

    adapter = NS(
        start_layer=4 if last else 0,
        end_layer=8 if last else 4,
        num_hidden_layers=8,
        backbone=NS(),
        _prepare_initial_state=initial,
        _finalize=lambda h, r: (h * 10, r),
    )
    runner.layered_prefill_model_adapter = adapter
    runner._layered_group_executor = NS(forward_mixed=layers)
    views = impl.single.LayeredV2SingleForward(runner, runner._layered_group_executor)

    def unexpected_decode_metadata():
        raise AssertionError("The PP mixed stage must not build Decode backend metadata")

    views._private_groups = unexpected_decode_metadata
    views.context = lambda *args: nullcontext()
    runner.kv_cache_config = object()
    runner._layered_single_forward_executor = views
    runner._prepare_layered_p_activation = lambda *args, **kwargs: (
        ["p"],
        (torch.full((3, hc_mult, 1) if hc_mult > 1 else (3, 1), 5.0), None) if last else None,
        None,
        None,
        "frontier" if last else "embed",
    )
    runner._record_layered_activation = lambda *args: None
    runner._synchronize_layered_compute = lambda: None
    runner._layered_p_buffers = nullcontext
    runner._strip_lifecycle_fields = lambda value: value
    runner.kv_connector = NS(pre_forward=lambda _: None)
    runner._combine_layered_pp_intermediate = impl._combine_layered_pp_intermediate
    runner._layered_pp_group_owner = lambda _: int(last)
    runner._layered_nonfused_pp_payload = lambda plan, payload: impl._layered_nonfused_pp_payload(runner, plan, payload)
    counters = {}
    runner._layered_counters = lambda: counters
    plan = NS(
        prefill_req_ids=["p"],
        query_tokens={"p": 3},
        num_groups=2,
        group_id=int(last),
        group_start=adapter.start_layer,
        group_end=adapter.end_layer,
        is_final_group=last,
        is_sampling_step=last,
    )
    return runner, plan, calls, initial_calls


@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize(
    "order",
    [["d0", "d1", "d2", "p"], ["p", "d0", "d1", "d2"], ["d0", "p", "d1", "d2"]],
)
def test_stage_layers_once_match_separate_rows_with_native_wire_padding(implementation, last, order):
    impl = implementation
    runner, plan, calls, initial_calls = stage_runner(impl, last)
    if order.index("p") in (0, len(order) - 1):

        def unexpected_indices(*args):
            pytest.fail("Contiguous D/P rows must not allocate device index tensors")

        runner._layered_single_forward_executor.buffers.indices = unexpected_indices
    prepared = impl.base["prepared"](order, p_tokens=3, pad_to=2)
    incoming = impl.IntermediateTensors({"hidden_states": torch.full((8, 1), 2.0)}) if last else None
    stage = impl.LayeredV2PPStageForward(runner)
    wire = stage.decode_wire_tokens(3)
    decode, d_output, p_output = stage.forward_stage(prepared, plan, incoming, None, wire)
    assert decode.input_batch.num_tokens_after_padding == 4
    assert d_output[0].shape == (8, 1)
    assert p_output[0].shape == (4, 1)
    assert calls == list(range(plan.group_start, plan.group_end))
    p_index = order.index("p")
    start, end = prepared.input_batch.query_start_loc_np[p_index : p_index + 2]
    d_positions = prepared.input_batch.positions[decode.rows]
    p_positions = prepared.input_batch.positions[start:end]
    expected_d = torch.full((3, 1), 2.0) if last else prepared.input_batch.input_ids[decode.rows].float()[:, None]
    expected_p = torch.full((3, 1), 5.0) if last else prepared.input_batch.input_ids[start:end].float()[:, None]
    for index in calls:
        expected_d = expected_d * (index + 2) + d_positions[:, None]
        expected_p = expected_p * (index + 2) + p_positions[:, None]
    if last:
        expected_d *= 10
        expected_p *= 10
    assert torch.equal(d_output[0][:3], expected_d)
    assert torch.equal(p_output[0][:3], expected_p)
    assert not d_output[0][3:].any()
    assert not p_output[0][3:].any()
    assert impl.dispatch_calls[0][1] == dict(need_eager=False, num_active_loras=0)
    assert (initial_calls[0]["intermediate_tensors"] is not None) == last
    runner.kernel_output.fill_(-1)
    assert torch.equal(d_output[0][:3], expected_d)
    assert torch.equal(p_output[0][:3], expected_p)
    if not last:
        frontier = runner.layered_prefill_state.get("p")
        assert frontier.group_id == 1 and frontier.query_len == 3
        p_output[0].zero_()
        assert torch.equal(frontier.hidden_states, expected_p)
    else:
        assert runner.layered_prefill_state.get("p") is None


@pytest.mark.parametrize("last", [False, True])
def test_stage_execute_preserves_d_p_sampling_states_and_wire(implementation, last):
    impl = implementation
    runner, plan, calls, _ = stage_runner(impl, last)
    mixed = impl.base["prepared"](["d0", "d1", "d2", "p"], p_tokens=3, pad_to=2)
    p_only = impl.base["prepared"](["p"], p_tokens=3, pad_to=2)
    preparations = []

    def prepare(schedule, *, build_attention_metadata=True):
        ids = list(schedule.num_scheduled_tokens)
        preparations.append((ids, build_attention_metadata))
        return p_only if ids == ["p"] else mixed

    runner._prepare_layered_batch = prepare
    schedule = NS(num_scheduled_tokens={"d0": 1, "d1": 1, "d2": 1, "p": 3}, finished_req_ids=set())
    incoming = impl.IntermediateTensors({"hidden_states": torch.full((8, 1), 2.0)}) if last else None
    result = impl.LayeredV2PPStageForward(runner).execute(schedule, plan, ["d0", "d1", "d2"], incoming, None)
    state = runner.execute_model_state
    assert state.scheduler_output is schedule
    assert state.d_state.input_batch.req_ids == ["d0", "d1", "d2"]
    assert state.d_state.attn_metadata is None
    assert state.d_state.slot_mappings_by_layer is None
    assert state.p_state.input_batch.req_ids == ["p"]
    assert state.sample_p == last
    assert state.single_forward
    assert preparations == [(["d0", "d1", "d2", "p"], True)] + ([(["p"], False)] if last else [])
    assert runner._layered_counters()["pp_stage_mixed_steps"] == 1
    if last:
        assert result is None
        assert state.d_state.hidden_states.shape == (4, 1)
        assert state.p_state.hidden_states.shape == (4, 1)
        assert state.d_state.hidden_states.data_ptr() != state.p_state.hidden_states.data_ptr()
    else:
        assert int(result["layered_pp_d_rows"]) == 8
        assert int(result["layered_pp_p_rows"]) == 4
        assert result["hidden_states"].shape == (12, 1)
        assert state.d_state.hidden_states is None
        assert state.p_state.hidden_states is None


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "pipeline_parallel_size",
        "async_scheduling",
        "speculative_config",
        "_fuse_mixed_batch",
        "_same_layer_batch",
        "_single_forward",
        "num_groups",
        "group_start",
        "p_query",
        "d_query",
        "kv_transfer_config",
    ],
)
def test_stage_unsupported_envelopes_fall_back(implementation, field):
    impl = implementation
    runner, plan, _, _ = stage_runner(impl, False)
    schedule = NS(num_scheduled_tokens={"d0": 1, "p": 3})

    def supported():
        return impl.LayeredV2PPStageForward.supported(runner, plan, ["d0"], schedule)

    assert supported()
    if field.endswith("parallel_size"):
        setattr(runner.parallel_config, field, 4)
    elif field == "async_scheduling":
        runner.scheduler_config.async_scheduling = True
    elif field == "num_groups":
        plan.num_groups = 1
    elif field == "group_start":
        plan.group_start = -1
    elif field == "p_query":
        plan.query_tokens["p"] = 1
    elif field == "d_query":
        schedule.num_scheduled_tokens["d0"] = 2
    elif field == "kv_transfer_config":
        runner.vllm_config.kv_transfer_config = object()
    else:
        setattr(runner, field, field != "_single_forward")
    assert not supported()


def test_stage_rejects_short_native_decode_activation(implementation):
    impl = implementation
    runner, plan, _, _ = stage_runner(impl, True)
    prepared = impl.base["prepared"](["d0", "d1", "d2", "p"], p_tokens=3, pad_to=2)
    incoming = impl.IntermediateTensors({"hidden_states": torch.ones(4, 1)})
    with pytest.raises(RuntimeError, match="native wire width"):
        impl.LayeredV2PPStageForward(runner).forward_stage(prepared, plan, incoming, None, 8)


def test_stage_transport_keeps_hyper_connection_axes(implementation):
    impl = implementation
    runner, plan, _, _ = stage_runner(impl, False, hc_mult=4)
    prepared = impl.base["prepared"](["d0", "d1", "d2", "p"], p_tokens=3, pad_to=2)
    _, d_output, p_output = impl.LayeredV2PPStageForward(runner).forward_stage(prepared, plan, None, None, 8)
    assert d_output[0].shape == (8, 4, 1)
    assert p_output[0].shape == (4, 4, 1)
    frontier = runner.layered_prefill_state.get("p")
    assert frontier.hidden_states.shape == (3, 4, 1)
    assert torch.equal(frontier.hidden_states[:, 0], frontier.hidden_states[:, 3])


def test_final_group_of_nonfinal_chunk_samples_only_decode_and_clears_frontier(
    implementation,
):
    impl = implementation
    runner, plan, _, _ = stage_runner(impl, True)
    plan.is_sampling_step = False
    runner.layered_prefill_state.put(NS(req_id="p", hidden_states=torch.ones(3, 1), residual=None))
    prepared = impl.base["prepared"](["d0", "d1", "d2", "p"], p_tokens=3, pad_to=2)
    preparations = []

    def prepare(schedule):
        preparations.append(list(schedule.num_scheduled_tokens))
        return prepared

    runner._prepare_layered_batch = prepare
    schedule = NS(num_scheduled_tokens={"d0": 1, "d1": 1, "d2": 1, "p": 3}, finished_req_ids=set())
    incoming = impl.IntermediateTensors({"hidden_states": torch.full((8, 1), 2.0)})
    impl.LayeredV2PPStageForward(runner).execute(schedule, plan, ["d0", "d1", "d2"], incoming, None)
    state = runner.execute_model_state
    assert not state.sample_p
    assert state.d_state.hidden_states.shape == (4, 1)
    assert state.p_state.hidden_states is None
    assert state.p_state.input_batch.req_ids == ["p"]
    assert runner.layered_prefill_state.get("p") is None
    assert preparations == [["d0", "d1", "d2", "p"]]


@pytest.mark.parametrize("profiling", [False, True])
def test_owner_profile_labels_preserve_legacy_request_identity(implementation, monkeypatch, profiling):
    impl = implementation
    runner, plan, calls, _ = stage_runner(impl, False)
    runner.vllm_config.profiler_config = NS(profiler="torch" if profiling else None)
    prepared = impl.base["prepared"](["d0", "d1", "d2", "p"], p_tokens=3, pad_to=2)
    runner._prepare_layered_batch = lambda _: prepared
    schedule = NS(
        num_scheduled_tokens={"d0": 1, "d1": 1, "d2": 1, "p": 3},
        finished_req_ids=set(),
        ring_step_plan=NS(step_id=17),
    )
    marks = []

    @contextmanager
    def record(label):
        assert profiling, "Disabled profiling must not call record_function"
        marks.append(("enter", label, len(calls)))
        yield
        marks.append(("exit", label, len(calls)))

    monkeypatch.setattr(torch.profiler, "record_function", record)
    output = impl.LayeredV2PPStageForward(runner).execute(schedule, plan, ["d0", "d1", "d2"], None, None)
    assert calls == [0, 1, 2, 3]
    assert int(output["layered_pp_d_rows"]) == 8
    if not profiling:
        assert marks == []
        return
    label = "layered_pp_prefill/step=17/group=0/layers=0:4/P=p"
    assert marks == [
        ("enter", label, 0),
        ("enter", "layered_pp_owner_mixed", 0),
        ("enter", "layered_pp_mixed_group", 0),
        ("exit", "layered_pp_mixed_group", 4),
        ("exit", "layered_pp_owner_mixed", 4),
        ("exit", label, 4),
    ]
    # Same fullmatch contract as tools/pp_trace_overlap.py.
    match = re.fullmatch(
        r"layered_pp_prefill/step=(-?\d+)/group=(\d+)/layers=(\d+):(\d+)/P=(.*)",
        label,
    )
    assert match.groups() == ("17", "0", "0", "4", "p")


@pytest.mark.parametrize(
    "num_groups,group",
    [(count, group) for count in PP2_RANGES for group in range(count)],
)
@pytest.mark.parametrize("order", [["d0", "d1", "d2", "p"], ["d0", "p", "d1", "d2"]])
def test_partial_stage_keeps_complete_decode_and_unfinalized_p_frontier(implementation, num_groups, group, order):
    impl = implementation
    group_start, group_end = PP2_RANGES[num_groups][group]
    last = group_start >= 22
    runner, plan, _, _ = stage_runner(impl, last, hc_mult=4)
    adapter = runner.layered_prefill_model_adapter
    adapter.start_layer, adapter.end_layer = (22, 43) if last else (0, 22)
    adapter.num_hidden_layers = 43
    plan.num_groups = num_groups
    plan.group_id = group
    plan.group_start, plan.group_end = group_start, group_end
    plan.is_final_group = plan.is_sampling_step = group + 1 == num_groups
    views = runner._layered_single_forward_executor
    views._private_groups = lambda: []
    prepared = impl.base["prepared"](order, p_tokens=3, pad_to=2)
    batch = prepared.input_batch
    p_index = order.index("p")
    p_start, p_end = map(int, batch.query_start_loc_np[p_index : p_index + 2])
    d_rows = [int(batch.query_start_loc_np[i]) for i, req in enumerate(order) if req != "p"]
    p_hidden = torch.full((3, 4, 1), 5.0)
    p_residual = torch.full((3, 4, 1), 7.0)
    runner._prepare_layered_p_activation = lambda *args, **kwargs: (
        ["p"],
        (p_hidden, p_residual) if group else None,
        None,
        None,
        "frontier" if group else "embed",
    )
    calls = []
    contexts = []

    @contextmanager
    def context(active, start):
        contexts.append((active, start))
        yield

    views.context = context

    def layers(kind, start, end, state, ids, positions):
        calls.append((kind, start, end))
        hidden, residual = state
        assert hidden.shape[0] == positions.shape[0]
        for index in range(start, end):
            hidden = hidden + index + 1 + positions[:, None, None]
            residual = hidden.clone() if residual is None else residual + hidden
        return hidden, residual

    runner._layered_group_executor.forward_mixed = lambda *args: layers("mixed", *args)
    runner._layered_group_executor.forward_decode = lambda _, *args: layers("decode", *args)
    incoming = (
        impl.IntermediateTensors(
            {
                "hidden_states": torch.full((8, 4, 1), 2.0),
                "residual": torch.full((8, 4, 1), 3.0),
            }
        )
        if last
        else None
    )
    schedule = NS(num_scheduled_tokens={req: 3 if req == "p" else 1 for req in order})
    assert impl.LayeredV2PPStageForward.supported(runner, plan, ["d0", "d1", "d2"], schedule)
    decode, d_output, p_output = impl.LayeredV2PPStageForward(runner).forward_stage(prepared, plan, incoming, None, 8)
    expected_calls = []
    if plan.group_start > adapter.start_layer:
        expected_calls.append(("decode", adapter.start_layer, plan.group_start))
    expected_calls.append(("mixed", plan.group_start, plan.group_end))
    if plan.group_end < adapter.end_layer:
        expected_calls.append(("decode", plan.group_end, adapter.end_layer))
    assert calls == expected_calls
    assert [layer for _, start, end in calls for layer in range(start, end)] == (
        list(range(adapter.start_layer, adapter.end_layer))
    )
    assert [start for _, start in contexts] == [start for _, start, _ in calls]
    assert all(
        active is (prepared if kind == "mixed" else decode) for (active, _), (kind, _, _) in zip(contexts, calls)
    )
    assert (decode.attn_metadata is not None) == ((group_start, group_end) != (adapter.start_layer, adapter.end_layer))
    assert decode.input_batch.num_tokens_after_padding == 4

    def expected(state, positions, start, end, finalize):
        hidden, residual = state
        for index in range(start, end):
            hidden = hidden + index + 1 + positions[:, None, None]
            residual = hidden.clone() if residual is None else residual + hidden
        return (hidden * 10 if finalize else hidden), residual

    initial_d = (
        (torch.full((3, 4, 1), 2.0), torch.full((3, 4, 1), 3.0))
        if last
        else (batch.input_ids[d_rows].float()[:, None, None].repeat(1, 4, 1), None)
    )
    initial_p = (
        (p_hidden, p_residual)
        if group
        else (
            batch.input_ids[p_start:p_end].float()[:, None, None].repeat(1, 4, 1),
            None,
        )
    )
    expected_d = expected(initial_d, batch.positions[d_rows], adapter.start_layer, adapter.end_layer, last)
    expected_p = expected(
        initial_p,
        batch.positions[p_start:p_end],
        plan.group_start,
        plan.group_end,
        group + 1 == num_groups,
    )
    for actual, reference in zip(d_output, expected_d):
        torch.testing.assert_close(actual[:3], reference)
        assert actual.shape == (8, 4, 1)
        assert not actual[3:].any()
    for actual, reference in zip(p_output, expected_p):
        torch.testing.assert_close(actual[:3], reference)
        assert actual.shape == (4, 4, 1)
        assert not actual[3:].any()
    frontier = runner.layered_prefill_state.get("p")
    if group + 1 == num_groups:
        assert frontier is None
    else:
        assert frontier.group_id == group + 1 and frontier.query_len == 3
        # A subsequent buffer reuse must not overwrite the paused activation.
        for value in p_output:
            value.fill_(-1)
        torch.testing.assert_close(frontier.hidden_states, expected_p[0])
        torch.testing.assert_close(frontier.residual, expected_p[1])


@pytest.mark.parametrize("start,end,final", [(-1, 2, False), (2, 5, False), (2, 2, False), (0, 2, True)])
def test_partial_stage_rejects_foreign_or_inconsistent_ranges(implementation, start, end, final):
    runner, plan, _, _ = stage_runner(implementation, False)
    plan.num_groups = 4
    plan.group_start, plan.group_end, plan.is_final_group = start, end, final
    schedule = NS(num_scheduled_tokens={"d0": 1, "p": 3})
    assert not implementation.LayeredV2PPStageForward.supported(runner, plan, ["d0"], schedule)


@pytest.mark.parametrize(
    "num_groups,group",
    [(count, group) for count in PP2_RANGES for group in range(count)],
)
@pytest.mark.parametrize("final_chunk", [False, True])
@pytest.mark.parametrize("compact_group0", [False, True])
def test_partial_stage_execute_keeps_wire_and_samples_p_only_at_final_chunk(
    implementation, num_groups, group, final_chunk, compact_group0
):
    impl = implementation
    group_start, group_end = PP2_RANGES[num_groups][group]
    last = group_start >= 22
    runner, plan, calls, _ = stage_runner(impl, last)
    adapter = runner.layered_prefill_model_adapter
    adapter.start_layer, adapter.end_layer = (22, 43) if last else (0, 22)
    adapter.num_hidden_layers = 43
    plan.num_groups, plan.group_id = num_groups, group
    plan.group_start, plan.group_end = group_start, group_end
    plan.is_final_group = group + 1 == num_groups
    plan.is_sampling_step = group + 1 == num_groups and final_chunk
    if compact_group0:
        runner.ascend_config = NS(
            scheduler_config=NS(
                layered_prefill_config=NS(
                    enabled=True,
                    single_forward=True,
                    fuse_mixed_batch=False,
                    same_layer_batch=False,
                )
            )
        )
        runner._layered_next_group_owner = lambda _: (
            int(PP2_RANGES[num_groups][group + 1][0] >= 22) if group + 1 < num_groups else None
        )
    runner._layered_single_forward_executor._private_groups = lambda: []
    runner._layered_group_executor.forward_decode = lambda _, *args: runner._layered_group_executor.forward_mixed(*args)
    mixed = impl.base["prepared"](["d0", "d1", "d2", "p"], p_tokens=3, pad_to=2)
    p_only = impl.base["prepared"](["p"], p_tokens=3, pad_to=2)
    preparations = []

    def prepare(schedule, *, build_attention_metadata=True):
        ids = list(schedule.num_scheduled_tokens)
        preparations.append((ids, build_attention_metadata))
        return p_only if ids == ["p"] else mixed

    runner._prepare_layered_batch = prepare
    schedule = NS(
        num_scheduled_tokens={"d0": 1, "d1": 1, "d2": 1, "p": 3},
        finished_req_ids={"finished"},
    )
    incoming = impl.IntermediateTensors({"hidden_states": torch.full((8, 1), 2.0)}) if last else None
    result = impl.LayeredV2PPStageForward(runner).execute(schedule, plan, ["d0", "d1", "d2"], incoming, None)
    state = runner.execute_model_state
    assert calls == list(range(adapter.start_layer, adapter.end_layer))
    assert state.sample_p == plan.is_sampling_step
    assert state.d_state.input_batch.req_ids == ["d0", "d1", "d2"]
    assert state.p_state.input_batch.req_ids == ["p"]
    assert state.d_state.finished_req_ids == state.p_state.finished_req_ids == {"finished"}
    assert (state.d_state.attn_metadata is not None) == (
        (group_start, group_end) != (adapter.start_layer, adapter.end_layer)
    )
    assert preparations == [(["d0", "d1", "d2", "p"], True)] + ([(["p"], False)] if plan.is_sampling_step else [])
    if last:
        assert result is None
        assert state.d_state.hidden_states.shape == (4, 1)
        assert (state.p_state.hidden_states is not None) == plan.is_sampling_step
    else:
        assert int(result["layered_pp_d_rows"]) == 8
        assert int(result["layered_pp_p_rows"]) == (1 if compact_group0 and group_end < 22 else 4)
        assert state.d_state.hidden_states is None
    frontier = runner.layered_prefill_state.get("p")
    assert (frontier is None) == plan.is_final_group
    if frontier is not None:
        assert frontier.group_id == group + 1


@pytest.mark.parametrize("num_groups", [4, 8, 16])
@pytest.mark.parametrize("mode", ["tail", "head"])
def test_boundary_cross_stage_is_one_interval_with_complete_decode(implementation, num_groups, mode):
    """Both ranks execute the same global plan; only its local intersection mixes P."""
    impl = implementation
    layout = PP2_RANGES[num_groups]
    handoff = next(i for i, (start, _) in enumerate(layout) if start == 22)
    first = 0 if mode == "tail" else handoff - 1
    stop = handoff + 1 if mode == "tail" else num_groups
    plan = NS(
        prefill_req_ids=["p"],
        query_tokens={"p": 3},
        num_groups=num_groups,
        group_id=first,
        resume_group_id=stop,
        group_start=layout[first][0],
        group_end=layout[stop - 1][1],
        is_final_group=stop == num_groups,
        is_sampling_step=stop == num_groups,
        boundary_mode=mode,
        cohort_id=73,
    )
    p_wire = None
    d_wire = None
    p_layers = []
    all_runners = []
    finalized = []
    expected_p = None
    for last in (False, True):
        runner, _, _, _ = stage_runner(impl, last)
        all_runners.append(runner)
        adapter = runner.layered_prefill_model_adapter
        adapter.start_layer, adapter.end_layer = (22, 43) if last else (0, 22)
        adapter.num_hidden_layers = 43
        runner._layered_next_group_owner = lambda _: None if plan.is_final_group else 1
        runner._layered_pp_group_owner = lambda _: -1
        views = runner._layered_single_forward_executor
        views._private_groups = lambda: []
        prepared = impl.base["prepared"](["d0", "p", "d1", "d2"], p_tokens=3, pad_to=2)
        batch = prepared.input_batch
        p_start, p_end = map(int, batch.query_start_loc_np[1:3])
        if not last:
            expected_p = batch.input_ids[p_start:p_end].float()[:, None] if first == 0 else torch.full((3, 1), 5.0)
        incoming_p = p_wire
        runner._prepare_layered_p_activation = lambda *args, last=last, incoming_p=incoming_p, **kwargs: (
            ["p"],
            None if last or first == 0 else (torch.full((3, 1), 5.0), None),
            None,
            incoming_p if last else None,
            "pp_recv" if last else "frontier",
        )
        stage_layers = []

        def layers(kind, start, end, state, ids, positions, stage_layers=stage_layers):
            stage_layers.extend(range(start, end))
            if kind == "mixed":
                p_layers.extend(range(start, end))
            return state[0] + sum(range(start + 1, end + 1)), state[1]

        runner._layered_group_executor.forward_mixed = lambda *args: layers("mixed", *args)
        runner._layered_group_executor.forward_decode = lambda _, *args: layers("decode", *args)

        def finalize(hidden, residual, last=last):
            finalized.append(last)
            return hidden * 10, residual

        adapter._finalize = finalize
        schedule = NS(num_scheduled_tokens={"d0": 1, "p": 3, "d1": 1, "d2": 1})
        assert impl.LayeredV2PPStageForward.supported(runner, plan, ["d0", "d1", "d2"], schedule)
        _, d_output, p_output = impl.LayeredV2PPStageForward(runner).forward_stage(
            prepared, plan, d_wire, incoming_p, 8
        )
        assert stage_layers == list(range(adapter.start_layer, adapter.end_layer))
        local_start = max(plan.group_start, adapter.start_layer)
        local_end = min(plan.group_end, adapter.end_layer)
        expected_p = expected_p + sum(range(local_start + 1, local_end + 1))
        if last and plan.is_final_group:
            expected_p *= 10
        torch.testing.assert_close(p_output[0][:3], expected_p)
        p_wire = impl.IntermediateTensors({"hidden_states": p_output[0].clone()})
        d_wire = impl.IntermediateTensors({"hidden_states": d_output[0].clone()})
        if not last:
            assert runner.layered_prefill_state.get("p") is None
            # A cross-stage plan always sends all P rows, never the one-row fence.
            assert runner._layered_nonfused_pp_payload(plan, p_wire)["hidden_states"].shape[0] == 4
    assert p_layers == list(range(plan.group_start, plan.group_end))
    assert finalized == [True]
    frontier = all_runners[1].layered_prefill_state.get("p")
    if plan.is_final_group:
        assert frontier is None
    else:
        assert frontier.group_id == stop and frontier.cohort_id == 73
        torch.testing.assert_close(frontier.hidden_states, expected_p)
