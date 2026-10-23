# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-only execution contracts, independent of the installed NPU runtime.

Load the production class definitions with narrow backend stand-ins; tensor
selection, row layout, frontier ownership and execution are the real code.
"""

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import make_dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


@pytest.fixture
def implementation():
    source = Path(__file__).parents[4] / "vllm_ascend/worker/v2/layered_prefill_single_forward.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(
        torch=torch,
        np=np,
        replace=replace,
        SimpleNamespace=NS,
        contextmanager=contextmanager,
        LayeredFrontier=NS,
        AscendAttentionState=NS(DecodeOnly="decode"),
        build_attn_metadata=lambda **kw: kw,
        build_slot_mappings_by_layer=lambda slots, config: {"layer": slots[0]},
    )
    exec(compile(tree, str(source), "exec"), namespace)
    return NS(**namespace)


def prepared(order, p_tokens=3, pad_to=4):
    queries = p_tokens if isinstance(p_tokens, dict) else {"p": p_tokens}
    lengths = np.array([queries.get(name, 1) for name in order], dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    tokens = int(starts[-1])
    padded = (tokens + pad_to - 1) // pad_to * pad_to
    count = len(order)
    values = dict(
        req_ids=order,
        num_reqs=count,
        num_reqs_after_padding=count,
        idx_mapping=torch.arange(count, dtype=torch.int32),
        idx_mapping_np=np.arange(count),
        expanded_idx_mapping=torch.arange(count, dtype=torch.int32),
        expanded_local_pos=torch.zeros(count, dtype=torch.int32),
        seq_lens=torch.arange(count, dtype=torch.int32) + 10,
        seq_lens_np=np.arange(count, dtype=np.int32) + 10,
        seq_lens_cpu_upper_bound=torch.arange(count) + 10,
        num_scheduled_tokens=lengths,
        num_computed_tokens_np=np.arange(count),
        prefill_len_np=np.arange(count),
        num_computed_prefill_tokens_np=np.arange(count),
        is_prefilling_np=np.array([x in queries for x in order]),
        max_seq_len_np=None,
        prompt_lens=None,
        dcp_local_seq_lens=None,
        input_ids=torch.arange(padded, dtype=torch.int32) + 1,
        positions=torch.arange(padded, dtype=torch.int64),
        is_padding=torch.arange(padded) >= tokens,
        num_tokens=tokens,
        num_tokens_after_padding=padded,
        query_start_loc=torch.from_numpy(starts),
        query_start_loc_np=starts,
        logits_indices=torch.from_numpy(starts[1:] - 1),
        cu_num_logits=torch.arange(count + 1),
        cu_num_logits_np=np.arange(count + 1),
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        attn_state="mixed",
    )
    batch = make_dataclass("Batch", [(name, object) for name in values])(**values)
    return NS(
        input_batch=batch,
        block_tables=(torch.arange(count * 2).reshape(count, 2),),
        slot_mappings=torch.arange(padded).reshape(1, padded),
        attn_metadata={"mixed": True},
        slot_mappings_by_layer={"layer": torch.arange(padded)},
    )


class Store:
    def __init__(self):
        self.entries = {}

    def get(self, name):
        return self.entries.get(name)

    def put(self, frontier):
        self.entries[frontier.req_id] = frontier

    def clear(self, name):
        self.entries.pop(name, None)


def executor(implementation, residual=False):
    calls = []

    def initial(**kw):
        hidden = kw["input_ids"].float().unsqueeze(-1)
        return hidden, None

    def run(start, end, state, ids, positions):
        hidden, res = state
        for layer in range(start, end):
            calls.append(layer)
            hidden = hidden * (layer + 2) + positions[:, None]
            if residual:
                res = hidden.clone() if res is None else res + hidden
        return hidden, res

    adapter = NS(
        start_layer=0,
        end_layer=6,
        _validate_range=lambda a, b: None,
        _prepare_initial_state=initial,
        _finalize=lambda hidden, res: (hidden, res),
    )
    runner = NS(
        layered_prefill_model_adapter=adapter,
        layered_prefill_state=Store(),
        _pad_for_sequence_parallelism=lambda n: (n + 3) // 4 * 4,
        kv_cache_config=object(),
    )
    group = NS(forward_mixed=run, forward_decode=lambda plan, *args: run(*args))
    result = implementation.LayeredV2SingleForward(runner, group)
    result._private_groups = lambda: []
    result.context = lambda active, start: nullcontext()
    return result, calls


def plan(group, sample=True, p_tokens=3):
    return NS(
        prefill_req_ids=["p"],
        query_tokens={"p": p_tokens},
        group_id=group,
        group_start=2 * group,
        group_end=2 * group + 2,
        num_groups=3,
        is_final_group=group == 2,
        is_sampling_step=group == 2 and sample,
    )


@pytest.mark.parametrize("order", [["d0", "d1", "p"], ["p", "d0", "d1"], ["d0", "p", "d1"]])
@pytest.mark.parametrize("residual", [False, True])
def test_each_layer_once_matches_complete_decode_and_final_prefill(implementation, order, residual):
    execution, calls = executor(implementation, residual)
    data = prepared(order)
    batch = data.input_batch
    expected = batch.input_ids.float().unsqueeze(-1)
    for layer in range(6):
        expected = expected * (layer + 2) + batch.positions[:, None]
    previous_pointer = None
    for group in range(3):
        calls.clear()
        output, decode = execution.forward(data, plan(group))
        assert calls == list(range(6))
        if group < 2:
            torch.testing.assert_close(output[: len(decode.rows)], expected[decode.rows])
            frontier = execution.runner.layered_prefill_state.get("p")
            assert frontier.hidden_states.shape[0] == 3
            if previous_pointer is not None:
                assert frontier.hidden_states.data_ptr() == previous_pointer
            previous_pointer = frontier.hidden_states.data_ptr()
        else:
            torch.testing.assert_close(output[: batch.num_tokens], expected[: batch.num_tokens])
            assert execution.runner.layered_prefill_state.get("p") is None


def test_compact_metadata_has_private_storage_and_initialized_padding(implementation):
    execution, _ = executor(implementation)
    data = prepared(["d0", "p", "d1"])
    source_slots = data.slot_mappings.clone()
    decode, _, _ = execution.prepare_decode(data, plan(0))
    metadata = decode.attn_metadata
    assert metadata["num_reqs"] == 4
    assert metadata["num_actual_tokens"] == 2
    assert metadata["query_start_loc_cpu"].tolist() == [0, 1, 2, 3, 4]
    assert metadata["seq_lens"].tolist() == [10, 12, 0, 0]
    assert metadata["slot_mappings"].tolist() == [[0, 4, -1, -1]]
    assert decode.input_batch.positions.tolist() == [0, 4, 0, 0]
    assert decode.input_batch.is_padding.tolist() == [False, False, True, True]
    metadata["slot_mappings"].fill_(99)
    metadata["block_tables"][0].fill_(99)
    torch.testing.assert_close(data.slot_mappings, source_slots)
    assert data.block_tables[0].tolist() == [[0, 1], [2, 3], [4, 5]]


def test_sampling_only_compaction_preserves_batch_without_backend_access(implementation):
    full_execution, _ = executor(implementation)
    sampling_execution, _ = executor(implementation)
    data = prepared(["d0", "p", "d1"])
    full, start, end = full_execution.prepare_decode(data, plan(0))

    def unexpected_backend_access():
        pytest.fail("Sampling-only compaction must not build backend metadata")

    sampling_execution._private_groups = unexpected_backend_access
    # A sampling-only view needs no block tables or slot mappings at all.
    compact, p_start, p_end = sampling_execution.prepare_decode(
        NS(input_batch=data.input_batch), plan(0), build_attention_metadata=False
    )
    assert (p_start, p_end, compact.rows) == (start, end, full.rows)
    assert compact.attn_metadata is None
    assert compact.slot_mappings_by_layer is None
    for name, expected in vars(full.input_batch).items():
        actual = getattr(compact.input_batch, name)
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected, name
    data.input_batch.input_ids.fill_(-1)
    data.input_batch.idx_mapping.fill_(-1)
    assert compact.input_batch.input_ids.tolist() == [1, 5, 0, 0]
    assert compact.input_batch.idx_mapping.tolist() == [0, 2]
    assert sampling_execution.decode_groups is None
    assert "slot_mappings" not in sampling_execution.buffers.tensors


def test_nonfinal_chunk_returns_only_decode_logits(implementation):
    execution, _ = executor(implementation)
    data = prepared(["p", "d"])
    for group in range(3):
        hidden, decode = execution.forward(data, plan(group, sample=False))
    assert hidden.shape[0] == decode.input_batch.num_tokens_after_padding == 4
    assert decode.input_batch.req_ids == ["d"]
    assert execution.runner.layered_prefill_state.get("p") is None


def test_p_only_and_invalid_frontier(implementation):
    execution, calls = executor(implementation)
    data = prepared(["p"])
    with pytest.raises(RuntimeError, match="frontier"):
        execution.forward(data, plan(1))
    for group in range(3):
        calls.clear()
        hidden, decode = execution.forward(data, plan(group))
        assert decode is None
        assert calls == [2 * group, 2 * group + 1]
        assert (hidden is not None) == (group == 2)


def test_decode_output_storage_is_reused_without_aliasing_source(implementation):
    from collections import namedtuple

    state_type = namedtuple("State", "hidden_states aux_hidden_states")
    buffers = implementation.LayeredV2Buffers()
    original = torch.arange(6).reshape(2, 3)
    saved = buffers.detach_state(state_type(original, [original]))
    pointer = saved.hidden_states.data_ptr()
    original.fill_(99)
    assert saved.hidden_states[0, 0] == 0
    updated = buffers.detach_state(state_type(original, [original]))
    assert updated.hidden_states.data_ptr() == pointer
    assert updated.hidden_states[0, 0] == 99


@pytest.mark.parametrize("rows", [[2, 3, 4], range(2, 5)])
def test_contiguous_offset_selection_copies_to_owned_padded_storage(implementation, rows):
    buffers = implementation.LayeredV2Buffers()
    buffers.indices = lambda *_: pytest.fail("contiguous rows must not build a device index")
    source = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    expected = source[2:5].clone()
    selected = buffers.select("rows", source, rows, padded=5, fill=-3)

    assert selected.shape == (5, 2, 2)
    assert selected.dtype == source.dtype
    assert selected.device == source.device
    torch.testing.assert_close(selected[:3], expected)
    assert torch.all(selected[3:] == -3)
    assert selected.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    source[2:5].fill_(99)
    torch.testing.assert_close(selected[:3], expected)
    selected[0].fill_(-1)
    assert torch.all(source[2] == 99)


def test_singleton_range_with_nonunit_step_uses_contiguous_path(implementation):
    buffers = implementation.LayeredV2Buffers()
    buffers.indices = lambda *_: pytest.fail("a single row is a contiguous selection regardless of range step")
    source = torch.arange(12).reshape(4, 3)

    selected = buffers.select("row", source, range(2, 3, 7))

    torch.testing.assert_close(selected, source[2:3])


def test_empty_selection_initializes_padding_without_device_index(implementation):
    buffers = implementation.LayeredV2Buffers()
    buffers.indices = lambda *_: pytest.fail("empty rows must not build a device index")
    source = torch.arange(12, dtype=torch.int32).reshape(4, 3)

    padded = buffers.select("padded", source, [], padded=2, fill=7)
    empty = buffers.select("empty", source, [])

    assert padded.shape == (2, 3)
    assert padded.dtype == source.dtype
    assert padded.device == source.device
    assert torch.all(padded == 7)
    assert empty.shape == (0, 3)


def test_discrete_selection_keeps_index_select_fallback(implementation):
    buffers = implementation.LayeredV2Buffers()
    source = torch.arange(18, dtype=torch.float64).reshape(6, 3)
    calls = []
    original_indices = buffers.indices

    def indices(rows, device):
        calls.append((tuple(rows), device))
        return original_indices(rows, device)

    buffers.indices = indices
    selected = buffers.select("rows", source, [4, 1, 5])

    assert calls == [((4, 1, 5), source.device)]
    torch.testing.assert_close(selected, source[[4, 1, 5]])
    assert selected.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_uniform_query_indices_reuse_storage(implementation):
    buffers = implementation.LayeredV2Buffers()
    first = buffers.arange(9, torch.int32, "cpu")
    second = buffers.arange(5, torch.int32, "cpu")
    assert first.data_ptr() == second.data_ptr()
    assert second.tolist() == [0, 1, 2, 3, 4]


def supported_runner():
    return NS(
        parallel_config=NS(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=NS(runner_type="generate"),
        scheduler_config=NS(async_scheduling=False),
        speculative_config=None,
        cache_config=NS(),
        lora_config=None,
        layered_prefill_model_adapter=NS(backbone=NS()),
        max_num_reqs=8,
        _pad_for_sequence_parallelism=lambda n: n,
    )


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "speculative_config",
        "lora_config",
    ],
)
def test_single_forward_and_separate_compilation_reject_unsupported_envelopes(implementation, field):
    runner = supported_runner()
    assert implementation.LayeredV2SingleForward.supported(runner)
    if field.endswith("parallel_size"):
        setattr(runner.parallel_config, field, 2)
    else:
        setattr(runner, field, object())
    assert not implementation.LayeredV2SingleForward.supported(runner)
    # Check the production separated-P selection uses that same guard.
    source = Path(__file__).parents[4] / "vllm_ascend/worker/v2/model_runner.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    choice = next(
        node.value
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "use_group_executor" for target in node.targets)
    )
    runner._layered_group_executor = object()
    assert eval(
        compile(ast.Expression(choice), str(source), "eval"),
        dict(self=runner, LayeredV2SingleForward=implementation.LayeredV2SingleForward),
    ) == (field == "pipeline_parallel_size")


@pytest.mark.parametrize("pp_size", [1, 2])
def test_async_scheduling_keeps_compiled_layer_groups_available(implementation, pp_size):
    runner = supported_runner()
    runner.use_async_scheduling = True
    runner.scheduler_config.async_scheduling = True
    runner.parallel_config.pipeline_parallel_size = pp_size
    assert implementation.LayeredV2SingleForward.supported(runner, allow_pipeline_parallel=True)


def test_compact_decode_host_lengths_survive_next_step_preparation(implementation):
    execution, _ = executor(implementation)
    first = prepared(["d", "p"])
    decode, _, _ = execution.prepare_decode(first, plan(0))
    lengths = decode.input_batch.seq_lens_np.copy()
    upper_bound = decode.input_batch.seq_lens_cpu_upper_bound.clone()
    second = prepared(["d", "p"])
    second.input_batch.seq_lens_np[:] = 99
    second.input_batch.seq_lens_cpu_upper_bound.fill_(99)
    execution.prepare_decode(second, plan(0))
    np.testing.assert_array_equal(decode.input_batch.seq_lens_np, lengths)
    torch.testing.assert_close(decode.input_batch.seq_lens_cpu_upper_bound, upper_bound)


@pytest.mark.parametrize("group", [0, 1, 2])
def test_one_token_prefill_falls_back_before_compact_decode_overwrites_rope(implementation, group):
    execution, _ = executor(implementation)
    tail_plan = plan(group, p_tokens=1)
    assert not execution.supports_plan(tail_plan)
    # The ordinary split path remains supported; only mixed traversal is
    # excluded, including a one-token final chunk after a long prompt.
    runner = supported_runner()
    runner._single_forward = True
    assert execution.supported(runner)
    source = Path(__file__).parents[4] / "vllm_ascend/worker/v2/model_runner.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    choice = next(
        node.value
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "single_forward" for target in node.targets)
    )
    assert not eval(
        compile(ast.Expression(choice), str(source), "eval"),
        dict(
            self=runner,
            layered_plan=tail_plan,
            p_req_ids=["p"],
            LayeredV2SingleForward=implementation.LayeredV2SingleForward,
        ),
    )
    execution.prepare_decode = lambda *args: pytest.fail("must not rebuild compact Decode RoPE metadata")
    with pytest.raises(ValueError, match="more than one query token"):
        execution.forward(prepared(["d", "p"], p_tokens=1), tail_plan)
    assert execution.supports_plan(plan(group, p_tokens=2))


@pytest.mark.parametrize("has_decode,sample_p", [(True, False), (False, False), (True, True)])
def test_single_forward_sampling_has_one_connector_pair(implementation, has_decode, sample_p):
    source = Path(__file__).parents[4] / "vllm_ascend/worker/v2/model_runner.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    runner_class = next(
        node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    methods = [
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_execute_layered_single_forward", "_sample_layered_v2")
    ]
    for method in methods:
        method.decorator_list = []
    runner_class.body = methods
    runner_class.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    events = []

    class Output(NS):
        pass

    class Parent:
        def sample_tokens(self, grammar_output):
            state = self.execute_model_state
            events.append(("sample", list(state.input_batch.req_ids)))
            self.kv_connector.post_forward(state.finished_req_ids)
            return Output(req_ids=state.input_batch.req_ids)

    namespace = dict(
        Parent=Parent,
        SimpleNamespace=NS,
        ExecuteModelState=NS,
        LayeredV2ExecuteModelState=NS,
        ModelRunnerOutput=Output,
        AsyncModelRunnerOutput=type("Async", (), {}),
        EMPTY_MODEL_RUNNER_OUTPUT=Output(),
    )
    module = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    runner = namespace["NPUModelRunner"]()
    data = prepared(["d", "p"] if has_decode else ["p"])
    decode = NS(input_batch=NS(req_ids=["d"]), attn_metadata={}, slot_mappings_by_layer={}) if has_decode else None
    runner._prepare_layered_batch = lambda scheduler: data
    runner._strip_lifecycle_fields = lambda scheduler: scheduler
    runner._layered_single_forward_executor = NS(forward=lambda data, plan: (torch.ones(4, 1), decode))
    counters = {}
    runner._layered_counters = lambda: counters
    runner.kv_connector = NS(
        pre_forward=lambda s: events.append("pre"),
        post_forward=lambda s: events.append("post"),
    )
    runner._layered_pp_handler_capture = lambda: nullcontext()
    runner._merge_layered_v2_outputs = lambda scheduler, outputs: outputs
    runner._complete_layered_output = lambda output, state: output
    scheduler = NS(finished_req_ids=set())
    runner._execute_layered_single_forward(scheduler, NS(is_sampling_step=sample_p), ["p"])
    state = runner.execute_model_state
    result = runner._sample_layered_v2(None, state)
    assert result is not None
    assert events.count("pre") == events.count("post") == 1
    sampled = [event[1] for event in events if isinstance(event, tuple)]
    assert sampled == ([["d", "p"]] if sample_p else [["d"]] if has_decode else [])
