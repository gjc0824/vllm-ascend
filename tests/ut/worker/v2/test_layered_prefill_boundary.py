# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Boundary routing/epoch contracts, runnable without importing a device runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).parents[4] / "vllm_ascend"


def test_mixed_final_and_partial_chunks_keep_native_per_request_sample_mask():
    source = ROOT.parent.parent / "vllm/vllm/v1/worker/gpu/pp_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_need_sampled_mask"
    ]
    namespace = {"np": np, "InputBatch": NS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    batch = NS(
        num_computed_tokens_np=np.array([20, 0, 4096]),
        prefill_len_np=np.array([16, 16384, 16384]),
        max_seq_len_np=np.array([100, 16400, 16400]),
        num_scheduled_tokens=np.array([1, 16384, 3071]),
    )
    mask = namespace["compute_need_sampled_mask"](batch)
    assert mask.tolist() == [True, True, False]


@pytest.mark.parametrize("boundary", [False, True])
def test_multi_prefill_split_is_limited_to_boundary_plans(boundary):
    source = ROOT / "worker/v2/layered_prefill.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "split_d_p_req_ids"]
    namespace = {"SchedulerOutput": NS, "LayeredPrefillPlan": NS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    schedule = NS(num_scheduled_tokens={"long": 16384, "d": 1, "partial": 3071})
    p = NS(prefill_req_ids=("long", "partial"), boundary_mode="head" if boundary else None)
    if boundary:
        assert namespace["split_d_p_req_ids"](schedule, p) == (["d"], ["long", "partial"])
    else:
        with pytest.raises(RuntimeError, match="exactly one"):
            namespace["split_d_p_req_ids"](schedule, p)


@pytest.fixture
def routing():
    source = ROOT / "worker/v2/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    names = {"_prepare_boundary_p_activation", "_layered_relay_stage_action", "_layered_pp_group_owner"}
    nodes = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "concat_req_frontiers": lambda store, ids, **kw: (tuple(ids), kw),
        "pad_activation_rows": lambda hidden, residual, padded: (("p",), padded),
        "get_pp_group": lambda: NS(world_size=2),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    source = ROOT / "worker/v2/layered_prefill_nonowner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "can_compact_internal_prefill_payload"
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return NS(**namespace)


def plan(start, end, *, group=0, query=3):
    return NS(
        boundary_mode="head",
        group_id=group,
        group_start=start,
        group_end=end,
        num_groups=4,
        prefill_req_ids=["p"],
        query_tokens={"p": query},
        cohort_id=42,
        is_final_group=end == 8,
        is_sampling_step=end == 8,
    )


def runner(last=False, frontier=None):
    return NS(
        layered_prefill_model_adapter=NS(
            start_layer=4 if last else 0,
            end_layer=8 if last else 4,
            make_transport_frontier=lambda *args: "placeholder",
        ),
        model_config=NS(dtype="float", num_hidden_layers=8),
        device="cpu",
        layered_prefill_state={"p": frontier} if frontier is not None else {},
        _layered_pp_partitions=lambda *_: [(0, 4), (4, 8)],
    )


@pytest.mark.parametrize("query", [1, 3])
@pytest.mark.parametrize(
    "start,end,last,source",
    [
        (0, 6, False, "embed"),
        (0, 6, True, "pp_recv"),
        (2, 8, False, "frontier"),
        (2, 8, True, "pp_recv"),
        (4, 6, False, "transport_frontier"),
        (4, 6, True, "frontier"),
        (2, 4, True, "pp_recv"),
    ],
)
def test_boundary_entry_uses_interval_entry_or_this_step_wire(routing, query, start, end, last, source):
    p = plan(start, end, group=start // 2, query=query)
    stored = (
        NS(group_id=p.group_id, query_len=query, cohort_id=42, hidden_states=object(), residual=None)
        if source == "frontier"
        else None
    )
    r = runner(last, stored)
    received = object()
    result = routing._prepare_boundary_p_activation(r, p, ["p"], query, "embeds", received)
    assert result[-1] == source
    if source == "pp_recv":
        assert result[3] is received and result[1] is None
    elif source == "frontier":
        assert result[1][0] == ("p",) and result[3] is None
    elif source == "transport_frontier":
        assert result[1] == "placeholder" and result[3] is None


@pytest.mark.parametrize("field,value", [("cohort_id", 41), ("group_id", 0), ("query_len", 2)])
def test_boundary_rejects_stale_chunk_or_wrong_progress(routing, field, value):
    stored = NS(group_id=1, query_len=3, cohort_id=42)
    setattr(stored, field, value)
    with pytest.raises(RuntimeError, match="mismatched boundary frontier"):
        routing._prepare_boundary_p_activation(runner(frontier=stored), plan(2, 8, group=1), ["p"], 3, None, None)


def test_boundary_entry_rejects_previous_chunk_leftover(routing):
    with pytest.raises(RuntimeError, match="chunk entry"):
        routing._prepare_boundary_p_activation(runner(frontier=object()), plan(0, 6), ["p"], 3, None, None)


def test_cross_stage_cannot_substitute_old_local_frontier_for_wire(routing):
    with pytest.raises(RuntimeError, match="this step"):
        routing._prepare_boundary_p_activation(runner(True, object()), plan(2, 8), ["p"], 3, None, None)


@pytest.mark.parametrize("start,end,owner", [(0, 2, 0), (0, 6, -1), (2, 8, -1), (4, 6, 1)])
def test_boundary_owner_accepts_cross_stage_range(routing, start, end, owner):
    assert routing._layered_pp_group_owner(runner(), plan(start, end)) == owner


def test_legacy_cross_stage_range_still_rejected(routing):
    p = plan(0, 6)
    p.boundary_mode = None
    with pytest.raises(RuntimeError, match="not aligned"):
        routing._layered_pp_group_owner(runner(), p)


def test_p_only_boundary_disables_unfenced_relay(routing):
    # In particular, skip_relay_stages=True must not bypass rank0's completion fence.
    assert routing._layered_relay_stage_action(NS(_skip_relay_stages=True), plan(0, 2)) is None


@pytest.mark.parametrize("query", [1, 3])
@pytest.mark.parametrize(
    "start,end,owner,next_owner,compact",
    [
        (0, 2, 0, 0, True),
        (2, 4, 0, 1, False),
        (0, 6, -1, 1, False),
        (2, 8, -1, None, False),
        (4, 6, 1, 1, False),
    ],
)
def test_boundary_wire_compacts_only_stage_internal_steps(routing, query, start, end, owner, next_owner, compact):
    config = NS(enabled=True, fuse_mixed_batch=False, same_layer_batch=False)
    r = NS(
        parallel_config=NS(pipeline_parallel_size=2, data_parallel_size=1),
        ascend_config=NS(
            scheduler_config=NS(layered_prefill_config=config, layered_prefill_extensions=NS(skip_relay_stages=True))
        ),
        scheduler_config=NS(async_scheduling=False),
        speculative_config=None,
        vllm_config=NS(kv_transfer_config=None),
        _layered_pp_group_owner=lambda _: owner,
        _layered_next_group_owner=lambda _: next_owner,
    )
    assert routing.can_compact_internal_prefill_payload(r, plan(start, end, query=query)) == compact


@pytest.fixture
def validate_config():
    source = ROOT / "ascend_config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    node = next(
        n for n in init.body if isinstance(n, ast.If) and "boundary_config.boundary_policy" in ast.unparse(n.test)
    )
    code = compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec")

    def validate(config, policy="pp_transition"):
        exec(code, {"boundary_config": NS(boundary_policy=policy), "vllm_config": config})

    return validate


def valid_config():
    return NS(
        use_v2_model_runner=True,
        parallel_config=NS(pipeline_parallel_size=2, data_parallel_size=1),
        cache_config=NS(enable_prefix_caching=False),
        scheduler_config=NS(async_scheduling=False),
        speculative_config=None,
        kv_transfer_config=None,
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "use_v2_model_runner", False),
        (None, "speculative_config", object()),
        (None, "kv_transfer_config", object()),
        ("parallel_config", "pipeline_parallel_size", 1),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("cache_config", "enable_prefix_caching", True),
    ],
)
def test_boundary_configuration_fails_closed(validate_config, section, key, value):
    config = valid_config()
    validate_config(config)
    setattr(getattr(config, section) if section else config, key, value)
    with pytest.raises(ValueError, match="requires MRV2"):
        validate_config(config)
    validate_config(config, policy="off")


def test_boundary_configuration_accepts_async_scheduling(validate_config):
    config = valid_config()
    config.scheduler_config.async_scheduling = True
    validate_config(config)


@pytest.fixture
def execute_native():
    source = ROOT / "worker/v2/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    execute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    execute.decorator_list = []
    namespace = {
        "SchedulerOutput": object,
        "IntermediateTensors": object,
        "vllm_version_is": lambda _: True,
        "super": lambda: NS(execute_model=lambda *args, **kwargs: "native"),
    }
    exec(compile(ast.Module(body=[execute], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["execute_model"]


@pytest.mark.parametrize("field", ["finished_req_ids", "preempted_req_ids"])
@pytest.mark.parametrize(
    "enabled,dummy,profile",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_native_dispatch_cleans_boundary_frontier_after_abort_or_preempt(
    routing, execute_native, field, enabled, dummy, profile
):
    class Store(dict):
        def clear_many(self, ids):
            for rid in ids:
                self.pop(rid, None)

    r = runner(frontier=NS(group_id=1, query_len=3, cohort_id=41))
    r.layered_prefill_state = Store(r.layered_prefill_state)
    r._boundary_prefill_enabled = enabled
    schedule = NS(layered_prefill_plan=None, **{field: {"p"}})
    assert execute_native(r, schedule, dummy_run=dummy, is_profile=profile) == "native"
    if enabled and not dummy and not profile:
        assert r.layered_prefill_state.get("p") is None
        # A restarted/reused request with a new epoch must embed successfully.
        assert routing._prepare_boundary_p_activation(r, plan(0, 6), ["p"], 3, None, None)[-1] == "embed"
    else:
        assert r.layered_prefill_state["p"].cohort_id == 41
