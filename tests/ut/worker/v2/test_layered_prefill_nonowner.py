# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU behavioral checks for the metadata-free non-owner P path."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[4]


PP2_RANGES = {
    2: ((0, 22), (22, 43)),
    3: ((0, 11), (11, 22), (22, 43)),
    4: ((0, 11), (11, 22), (22, 33), (33, 43)),
    5: ((0, 8), (8, 15), (15, 22), (22, 33), (33, 43)),
    8: ((0, 6), (6, 12), (12, 17), (17, 22), (22, 28), (28, 33), (33, 38), (38, 43)),
    16: (
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        (15, 18),
        (18, 20),
        (20, 22),
        (22, 25),
        (25, 28),
        (28, 31),
        (31, 34),
        (34, 37),
        (37, 39),
        (39, 41),
        (41, 43),
    ),
}


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def impl():
    root = ROOT / "vllm_ascend/worker/v2"
    namespace = dict(
        np=np,
        torch=torch,
        Sequence=tuple,
        LayeredPrefillStateStore=object,
        LayeredFrontier=NS,
        SimpleNamespace=NS,
        ExecuteModelState=NS,
        LayeredForwardOutput=lambda hidden, residual, final: NS(
            hidden_states=hidden, residual=residual, is_final_layer=final
        ),
        LayeredV2SingleForward=NS(
            supports_plan=lambda plan: len(plan.prefill_req_ids) == 1
            and all(plan.query_tokens[x] > 1 for x in plan.prefill_req_ids),
            supported=lambda runner, **kw: runner.eligible,
        ),
    )
    load_functions(
        root / "layered_prefill.py",
        {"store_req_frontiers", "_persist_frontier_rows"},
        namespace,
    )
    load_functions(
        root / "layered_prefill_nonowner.py",
        {
            "run_nonowner_prefill",
            "can_compact_internal_prefill_payload",
            "run_internal_prefill_fence",
        },
        namespace,
    )
    return NS(**namespace)


class Store:
    def __init__(self):
        self.rows = {}

    def pop(self, key):
        return self.rows.pop(key, None)

    def put(self, value):
        self.rows[value.req_id] = value

    def clear(self, key):
        self.rows.pop(key, None)


def setup_case(rank, *, query=3, sample=False, num_groups=2, group_id=None):
    if group_id is None:
        group_id = 1 - rank
    ranges = PP2_RANGES[num_groups]
    group_start, group_end = ranges[group_id]
    plan = NS(
        group_id=group_id,
        num_groups=num_groups,
        group_start=group_start,
        group_end=group_end,
        is_final_group=group_id + 1 == num_groups,
        is_sampling_step=sample,
        prefill_req_ids=("p",),
        query_tokens={"p": query},
    )
    padded = (query + 1) // 2 * 2
    received = NS(
        tensors={
            "hidden_states": torch.arange(padded * 2).reshape(padded, 2).float(),
            "residual": torch.ones(padded, 2),
        }
    )
    batch = NS(req_ids=["p"], num_reqs=1, idx_mapping_np=np.array([7]))
    runner = NS(
        parallel_config=NS(pipeline_parallel_size=2),
        eligible=True,
        vllm_config=NS(kv_transfer_config=None),
        is_first_pp_rank=rank == 0,
        model_config=NS(dtype=torch.float32),
        device="cpu",
        layered_prefill_state=Store(),
        _layered_pp_group_owner=lambda p: int(p.group_start >= 22),
        _layered_next_group_owner=lambda p: (
            int(ranges[p.group_id + 1][0] >= 22) if p.group_id + 1 < p.num_groups else None
        ),
        _pad_for_sequence_parallelism=lambda q: (q + 1) // 2 * 2,
        _prepare_layered_batch=Mock(return_value=NS(input_batch=batch)),
        _prepare_layered_p_activation=Mock(
            return_value=(
                ["p"],
                None,
                None,
                received,
                "pp_recv",
            )
        ),
        _record_layered_activation=Mock(),
        kv_connector=NS(pre_forward=Mock()),
        layered_prefill_model_adapter=NS(
            start_layer=0 if rank == 0 else 22,
            end_layer=22 if rank == 0 else 43,
            _prepare_initial_state=lambda **kw: (
                kw["intermediate_tensors"].tensors["hidden_states"],
                kw["intermediate_tensors"].tensors.get("residual"),
            ),
            make_transport_frontier=Mock(
                side_effect=lambda n, dtype, device: (
                    torch.zeros(n, 2, dtype=dtype, device=device),
                    None,
                )
            ),
        ),
    )
    schedule = NS(
        num_scheduled_tokens={"p": query},
        total_num_scheduled_tokens=query,
        finished_req_ids=set(),
    )
    return runner, schedule, plan, received, batch


@pytest.mark.parametrize(
    "num_groups,group_id",
    [(count, group) for count in PP2_RANGES for group in range(count)],
)
@pytest.mark.parametrize("rank", (0, 1))
def test_pp2_groups_route_every_owner_and_nonowner_rank(
    impl,
    rank,
    num_groups,
    group_id,
):
    owner = int(PP2_RANGES[num_groups][group_id][0] >= 22)
    sample = group_id + 1 == num_groups
    runner, schedule, plan, received, batch = setup_case(
        rank,
        num_groups=num_groups,
        group_id=group_id,
        sample=sample,
    )
    if rank == owner or rank == 0:
        runner.layered_prefill_state.rows["p"] = object()

    result = impl.run_nonowner_prefill(runner, schedule, plan, received)

    if rank == owner:
        assert result is None
        runner.kv_connector.pre_forward.assert_not_called()
        assert runner.layered_prefill_state.rows["p"] is not None
        return

    state, output = result
    assert output.is_final_layer is sample
    assert state.attn_metadata is None
    assert state.slot_mappings_by_layer is None
    runner.kv_connector.pre_forward.assert_called_once_with(schedule)
    if rank == 1:
        saved = runner.layered_prefill_state.rows["p"]
        assert saved.group_id == group_id + 1
        assert saved.query_len == plan.query_tokens["p"]
        runner._prepare_layered_batch.assert_not_called()
        runner._prepare_layered_p_activation.assert_called_once()
    else:
        assert "p" not in runner.layered_prefill_state.rows
        assert output.hidden_states.shape[0] == 0
        runner._prepare_layered_p_activation.assert_not_called()
        if sample:
            assert state.input_batch is batch
            runner._prepare_layered_batch.assert_called_once_with(
                schedule,
                build_attention_metadata=False,
            )
        else:
            runner._prepare_layered_batch.assert_not_called()


@pytest.mark.parametrize("query", [2, 3, 4095, 4096])
@pytest.mark.parametrize("residual", [False, True])
def test_next_owner_stores_only_logical_owned_frontier_without_metadata(impl, query, residual):
    runner, schedule, plan, received, _ = setup_case(1, query=query)
    if not residual:
        received.tensors.pop("residual")
    state, output = impl.run_nonowner_prefill(runner, schedule, plan, received)
    saved = runner.layered_prefill_state.rows["p"]
    expected = received.tensors["hidden_states"][:query].clone()
    assert saved.group_id == 1 and saved.query_len == query
    assert torch.equal(saved.hidden_states, expected)
    assert saved.hidden_states.data_ptr() != received.tensors["hidden_states"].data_ptr()
    assert (saved.residual is not None) is residual
    received.tensors["hidden_states"].zero_()
    assert torch.equal(saved.hidden_states, expected)
    runner._prepare_layered_batch.assert_not_called()
    runner.kv_connector.pre_forward.assert_called_once_with(schedule)
    assert state.input_batch.req_ids == ["p"] and state.hidden_states is None
    assert state.attn_metadata is None and state.slot_mappings_by_layer is None
    assert output.is_final_layer is False


@pytest.mark.parametrize("sample", [False, True])
def test_preowner_clears_frontier_and_preserves_zero_row_wire_and_sampler_batch(impl, sample):
    runner, schedule, plan, _, batch = setup_case(0, sample=sample)
    runner.layered_prefill_state.rows["p"] = object()
    runner._single_forward = True  # Owner-stage mixing must not disable this path.
    state, output = impl.run_nonowner_prefill(runner, schedule, plan, None)
    assert "p" not in runner.layered_prefill_state.rows
    assert output.is_final_layer and output.hidden_states.shape[0] == 0
    assert state.hidden_states is None
    runner._prepare_layered_p_activation.assert_not_called()
    if sample:
        runner._prepare_layered_batch.assert_called_once_with(schedule, build_attention_metadata=False)
        assert state.input_batch is batch
    else:
        runner._prepare_layered_batch.assert_not_called()
    runner.kv_connector.pre_forward.assert_called_once_with(schedule)


@pytest.mark.parametrize(
    "change",
    [
        "pp",
        "invalid_groups",
        "final_flag",
        "fused",
        "same_layer",
        "unsupported",
        "connector",
        "encoder",
        "one_token",
        "owner",
        "full_range",
        "crossing_range",
    ],
)
def test_unsupported_envelopes_leave_original_path_untouched(impl, change):
    runner, schedule, plan, received, _ = setup_case(1)
    if change == "pp":
        runner.parallel_config.pipeline_parallel_size = 4
    elif change == "invalid_groups":
        plan.num_groups = 1
    elif change == "final_flag":
        plan.is_final_group = True
    elif change == "fused":
        runner._fuse_mixed_batch = True
    elif change == "same_layer":
        runner._same_layer_batch = True
    elif change == "unsupported":
        runner.eligible = False
    elif change == "connector":
        runner.vllm_config.kv_transfer_config = object()
    elif change == "encoder":
        runner.is_encoder_decoder = True
    elif change == "one_token":
        plan.query_tokens["p"] = 1
    elif change == "owner":
        runner._layered_pp_group_owner = lambda _: 1
    elif change == "full_range":
        runner._layered_pp_group_owner = lambda _: -1
    elif change == "crossing_range":
        plan.group_end = runner.layered_prefill_model_adapter.start_layer + 1
    assert impl.run_nonowner_prefill(runner, schedule, plan, received) is None
    runner._prepare_layered_batch.assert_not_called()
    runner.kv_connector.pre_forward.assert_not_called()
    assert not runner.layered_prefill_state.rows


@pytest.mark.parametrize("corrupt", ["schedule", "rows", "residual_rows", "sample"])
def test_inconsistent_nonowner_schedule_fails_closed(impl, corrupt):
    runner, schedule, plan, received, _ = setup_case(1)
    if corrupt == "schedule":
        schedule.num_scheduled_tokens["p"] = 2
    elif corrupt == "rows":
        received.tensors["hidden_states"] = torch.zeros(2, 2)
    elif corrupt == "residual_rows":
        received.tensors["residual"] = torch.zeros(2, 2)
    else:
        plan.is_sampling_step = True
    with pytest.raises(RuntimeError):
        impl.run_nonowner_prefill(runner, schedule, plan, received)
    assert not runner.layered_prefill_state.rows


@dataclass
class Descriptor:
    num_tokens: int
    cg_mode: str = "NONE"


def runner_methods(names, namespace):
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["NPUModelRunner"]


@pytest.mark.parametrize("num_groups", PP2_RANGES)
@pytest.mark.parametrize("rank", (0, 1))
def test_group_owner_and_next_owner_use_actual_partitions_on_both_ranks(num_groups, rank):
    ranges = PP2_RANGES[num_groups]
    make_ranges = Mock(return_value=tuple(NS(start=start, end=end) for start, end in ranges))
    namespace = dict(
        get_pp_group=lambda: NS(rank_in_group=rank, world_size=2),
        get_pp_indices=lambda layers, stage, pp: ((0, 22), (22, 43))[stage],
        make_pp_aligned_layer_group_ranges=make_ranges,
    )
    cls = runner_methods(
        {"_layered_pp_group_owner", "_layered_next_group_owner", "_layered_pp_partitions"},
        namespace,
    )
    runner = cls()
    runner.parallel_config = NS(pipeline_parallel_size=2)
    runner.model_config = NS(num_hidden_layers=43)
    runner._num_hidden_layers = lambda: 43
    for _ in range(2):
        for group_id, (start, end) in enumerate(ranges):
            plan = NS(num_groups=num_groups, group_id=group_id, group_start=start, group_end=end)
            assert runner._layered_pp_group_owner(plan) == int(start >= 22)
            assert runner._layered_next_group_owner(plan) == (
                int(ranges[group_id + 1][0] >= 22) if group_id + 1 < num_groups else None
            )
    make_ranges.assert_called_once_with(43, num_groups, 2)
    with pytest.raises(RuntimeError, match="not aligned"):
        runner._layered_pp_group_owner(NS(group_start=21, group_end=23))


@pytest.mark.parametrize("build_metadata", [False, True])
@pytest.mark.parametrize("query", [3, 4])
def test_sampling_only_prepare_retains_input_padding_without_backend_work(build_metadata, query):
    namespace = dict(
        SimpleNamespace=NS,
        replace=replace,
        CUDAGraphMode=NS(NONE="NONE"),
        vllm_version_is=lambda _: True,
        get_uniform_token_count=lambda *a: None,
        dispatch_cg_and_sync_dp=lambda *a, **kw: (Descriptor(query), None),
        build_slot_mappings_by_layer=Mock(return_value="slots_by_layer"),
    )
    runner = runner_methods({"_prepare_layered_batch"}, namespace)()
    padded = (query + 1) // 2 * 2
    batch = NS(num_tokens_after_padding=padded, num_tokens=query)
    runner.cudagraph_manager = None
    runner.dp_size = 1
    runner.dp_rank = 0
    runner._pad_for_sequence_parallelism = lambda _: padded
    runner.prepare_inputs = Mock(return_value=batch)
    runner.input_buffers = NS(positions=torch.full((padded,), 99))
    runner.prepare_attn = Mock(return_value=("blocks", "slots"))
    runner.model_state = NS(preprocess_state=Mock(), prepare_attn=Mock(return_value="attention"))
    runner.kv_cache_config = object()
    runner.req_states = NS(num_computed_tokens=NS(gpu=object()))
    runner.attn_groups = object()
    schedule = NS(num_scheduled_tokens={"p": query}, total_num_scheduled_tokens=query)
    result = runner._prepare_layered_batch(schedule, build_attention_metadata=build_metadata)
    assert result.input_batch is batch
    assert runner.input_buffers.positions[:query].tolist() == [99] * query
    assert runner.input_buffers.positions[query:].tolist() == [0] * (padded - query)
    assert runner.prepare_attn.call_count == int(build_metadata)
    assert runner.model_state.preprocess_state.call_count == int(build_metadata)
    assert runner.model_state.prepare_attn.call_count == int(build_metadata)
    assert result.attn_metadata == ("attention" if build_metadata else None)


def test_runner_routes_nonowner_before_full_preparation(impl):
    runner, schedule, plan, received, _ = setup_case(1)
    namespace = dict(
        SchedulerOutput=NS,
        IntermediateTensors=NS,
        ExecuteModelState=NS,
        LayeredForwardOutput=NS,
        run_nonowner_prefill=impl.run_nonowner_prefill,
        run_internal_prefill_fence=impl.run_internal_prefill_fence,
    )
    cls = runner_methods({"_run_layered_prefill_subbatch"}, namespace)
    live = cls()
    live.__dict__.update(vars(runner))
    live._layered_nonowner_fastpath = True
    state, output = live._run_layered_prefill_subbatch(schedule, plan, received)
    assert state.input_batch.req_ids == ["p"] and not output.is_final_layer
    live._prepare_layered_batch.assert_not_called()


def enable_group0_compaction(runner):
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
    runner.scheduler_config = NS(async_scheduling=False)


@pytest.mark.parametrize("d_rows", [0, 8])
@pytest.mark.parametrize("local_fastpath", [False, True])
@pytest.mark.parametrize("single_forward", [False, True])
@pytest.mark.parametrize(
    "num_groups,group_id",
    [(count, group) for count, ranges in PP2_RANGES.items() for group, (_, end) in enumerate(ranges) if end < 22],
)
def test_group0_one_row_fence_pairs_sender_and_receiver_without_local_capability_gate(
    impl, d_rows, local_fastpath, single_forward, num_groups, group_id
):
    class Intermediate:
        def __init__(self, tensors):
            self.tensors = tensors

    namespace = dict(
        torch=torch,
        SchedulerOutput=NS,
        IntermediateTensors=Intermediate,
        ExecuteModelState=NS,
        LayeredForwardOutput=NS,
        get_pp_group=lambda: NS(rank_in_group=0),
        can_compact_internal_prefill_payload=impl.can_compact_internal_prefill_payload,
        run_internal_prefill_fence=impl.run_internal_prefill_fence,
        run_nonowner_prefill=impl.run_nonowner_prefill,
    )
    cls = runner_methods(
        {
            "_layered_nonfused_pp_payload",
            "_combine_layered_pp_intermediate",
            "_split_layered_pp_intermediate",
            "_layered_pp_row_count",
            "_run_layered_prefill_subbatch",
        },
        namespace,
    )
    sender, _, plan, _, _ = setup_case(0, num_groups=num_groups, group_id=group_id)
    receiver, schedule, _, _, _ = setup_case(1, num_groups=num_groups, group_id=group_id)
    for runner in (sender, receiver):
        enable_group0_compaction(runner)
        runner.ascend_config.scheduler_config.layered_prefill_config.single_forward = single_forward
        runner._single_forward = single_forward
    receiver.eligible = False  # Rank-local adapter/attention fallback cannot alter the wire.
    receiver._layered_nonowner_fastpath = local_fastpath
    live_sender, live_receiver = cls(), cls()
    live_sender.__dict__.update(vars(sender))
    live_receiver.__dict__.update(vars(receiver))
    # Exercise the DSV4 hyper-connection axis and both activation components.
    p = Intermediate({"hidden_states": torch.ones(4, 4, 2), "residual": torch.full((4, 4, 2), 3.0)})
    d = (
        Intermediate({key: torch.arange(d_rows * 8).reshape(d_rows, 4, 2).float() for key in p.tensors})
        if d_rows
        else None
    )
    trimmed = live_sender._layered_nonfused_pp_payload(plan, p)
    packed = live_sender._combine_layered_pp_intermediate(d, trimmed)
    assert packed.tensors["layered_pp_d_rows"] == d_rows
    assert packed.tensors["layered_pp_p_rows"] == 1
    d_received, p_received = live_receiver._split_layered_pp_intermediate(packed)
    for key, value in packed.tensors.items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (d_rows + 1, 4, 2)
            assert value.numel() > 0  # P-only also retains a device transfer.
            torch.testing.assert_close(p_received.tensors[key], p.tensors[key][:1])
            if d is not None:
                torch.testing.assert_close(d_received.tensors[key], d.tensors[key])
    live_receiver.layered_prefill_state.rows["other"] = object()
    state, output = live_receiver._run_layered_prefill_subbatch(schedule, plan, p_received)
    assert state.input_batch.req_ids == ["p"] and state.hidden_states is None
    assert output.hidden_states.shape == (0, 4, 2) and not output.is_final_layer
    assert set(live_receiver.layered_prefill_state.rows) == {"other"}
    live_receiver._prepare_layered_batch.assert_not_called()
    live_receiver._prepare_layered_p_activation.assert_not_called()
    live_receiver.kv_connector.pre_forward.assert_called_once_with(schedule)


@pytest.mark.parametrize(
    "num_groups,group_id",
    [(count, group) for count in PP2_RANGES for group in range(count)],
)
def test_every_group_transports_only_the_frontier_needed_by_next_stage(impl, num_groups, group_id):
    class Intermediate:
        def __init__(self, tensors):
            self.tensors = tensors

    namespace = dict(
        IntermediateTensors=Intermediate,
        get_pp_group=lambda: NS(rank_in_group=0),
        can_compact_internal_prefill_payload=impl.can_compact_internal_prefill_payload,
    )
    cls = runner_methods({"_layered_nonfused_pp_payload"}, namespace)
    sender, _, plan, received, _ = setup_case(0, num_groups=num_groups, group_id=group_id)
    enable_group0_compaction(sender)
    live = cls()
    live.__dict__.update(vars(sender))
    result = live._layered_nonfused_pp_payload(plan, received)
    expected_rows = 0 if plan.group_start >= 22 else (4 if plan.group_end == 22 else 1)
    assert all(value.shape[0] == expected_rows for value in result.tensors.values())
    for rank in (0, 1):
        runner, _, _, _, _ = setup_case(rank, num_groups=num_groups, group_id=group_id)
        enable_group0_compaction(runner)
        assert impl.can_compact_internal_prefill_payload(runner, plan) == (expected_rows == 1)


def test_internal_fence_waits_for_async_activation_before_completing_lifecycle(impl):
    runner, schedule, plan, received, _ = setup_case(1, num_groups=16, group_id=6)
    enable_group0_compaction(runner)
    waited = []

    class AsyncIntermediate:
        @property
        def tensors(self):
            waited.append(True)
            return {key: value[:1] for key, value in received.tensors.items()}

    def pre_forward(_):
        assert waited == [True]

    runner.kv_connector.pre_forward = Mock(side_effect=pre_forward)
    state, output = impl.run_internal_prefill_fence(runner, schedule, plan, AsyncIntermediate())
    assert waited == [True]
    assert state.input_batch.req_ids == ["p"]
    assert output.hidden_states.shape[0] == 0
    assert not runner.layered_prefill_state.rows


@pytest.mark.parametrize(
    "blocked",
    [
        "disabled",
        "fuse",
        "same",
        "async",
        "pp",
        "dp",
        "dbo",
        "spec",
        "kv",
        "group",
        "query",
        "next_owner",
        "relay",
    ],
)
def test_group0_compaction_shared_gate_falls_back_on_both_ranks(impl, blocked):
    for rank in (0, 1):
        runner, schedule, plan, received, _ = setup_case(rank, num_groups=4, group_id=0)
        enable_group0_compaction(runner)
        config = runner.ascend_config.scheduler_config.layered_prefill_config
        if blocked == "disabled":
            config.enabled = False
        elif blocked in ("fuse", "same"):
            setattr(
                config,
                "fuse_mixed_batch" if blocked == "fuse" else "same_layer_batch",
                True,
            )
        elif blocked == "async":
            runner.scheduler_config.async_scheduling = True
        elif blocked in ("pp", "dp"):
            setattr(
                runner.parallel_config,
                "pipeline_parallel_size" if blocked == "pp" else "data_parallel_size",
                3,
            )
        elif blocked == "dbo":
            runner.parallel_config.enable_dbo = True
        elif blocked == "spec":
            runner.speculative_config = object()
        elif blocked == "kv":
            runner.vllm_config.kv_transfer_config = object()
        elif blocked == "relay":
            runner.ascend_config.scheduler_config.layered_prefill_extensions = NS(skip_relay_stages=True)
        elif blocked == "group":
            plan.group_id = 1
        elif blocked == "query":
            plan.query_tokens["p"] = 1
        else:
            runner._layered_next_group_owner = lambda _: 1
        assert not impl.can_compact_internal_prefill_payload(runner, plan)
        assert impl.run_internal_prefill_fence(runner, schedule, plan, received) is None


@pytest.mark.parametrize("bad_rows", [0, 4])
def test_group0_compaction_requires_fence_and_g1_still_owns_frontier(impl, bad_rows):
    runner, schedule, plan, received, _ = setup_case(1, num_groups=4, group_id=0)
    enable_group0_compaction(runner)
    invalid = NS(tensors={key: value[:bad_rows] for key, value in received.tensors.items()})
    with pytest.raises(RuntimeError, match="one-row"):
        impl.run_internal_prefill_fence(runner, schedule, plan, invalid)
    plan.group_id, plan.group_start, plan.group_end = 1, 11, 22
    assert impl.run_internal_prefill_fence(runner, schedule, plan, received) is None
    state, output = impl.run_nonowner_prefill(runner, schedule, plan, received)
    saved = runner.layered_prefill_state.rows["p"]
    assert saved.group_id == 2 and saved.query_len == 3
    expected = received.tensors["hidden_states"][:3].clone()
    received.tensors["hidden_states"].zero_()
    torch.testing.assert_close(saved.hidden_states, expected)
