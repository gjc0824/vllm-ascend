# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared graph lifecycle with the actual MRV2 runner field contract.

Load the production executor/cache with fake device primitives so these tests
need neither torch nor a distributed runtime. The mock intentionally has no
MRV1-only tp_rank, input_batch, use_dcp, or query_start_loc fields.
"""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).parents[4] / "vllm_ascend" / "worker"


class Tensor:
    shape = (2,)
    dtype = "float32"
    device = NS(type="npu")

    def __init__(self, value):
        self.value = value

    def stride(self):
        return (1,)

    def clone(self):
        return Tensor(self.value)

    def data_ptr(self):
        return id(self)

    def copy_(self, other):
        self.value = other.value


@pytest.fixture
def runtime(monkeypatch):
    source = ROOT / "layered_prefill_decode_graph.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # Preserve production stdlib imports, all helpers and both cache classes.
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(n.name == "torch" for n in node.names))
        and not (isinstance(node, ast.ImportFrom) and node.module.startswith("vllm"))
    ]
    module = ModuleType("_shared_decode_graph_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    events = []
    stream = NS(wait_stream=lambda _: None)
    module.torch = NS(
        Tensor=Tensor,
        npu=NS(
            graph_pool_handle=lambda: object(),
            Stream=lambda **_: stream,
            current_stream=lambda: stream,
            NPUGraph=lambda: NS(replay=lambda: events.append("replay")),
            graph=lambda *_, **__: nullcontext(),
        ),
    )
    context = NS(
        attn_metadata={"layer": NS(length=2)},
        slot_mapping=Tensor(7),
        is_padding=Tensor(False),
        capturing=False,
        moe_comm_type="allgather",
        num_tokens=2,
    )
    module.get_forward_context = lambda: context
    module.get_offloader = lambda: NS(
        sync_prev_onload=lambda: events.append("sync"),
        join_after_forward=lambda: events.append("join"),
    )
    module.logger = NS(info=lambda *args: events.append("log"), warning_once=lambda *args: None)
    exec(compile(tree, str(source), "exec"), module.__dict__)

    source = ROOT / "layered_prefill_single_forward.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayeredPrefillGroupExecutor")
    module.CUDAGraphMode = NS(NONE="none")
    module.make_pp_aligned_layer_group_ranges = lambda layers, groups, pp: [
        NS(start=i * layers // groups, end=(i + 1) * layers // groups) for i in range(groups)
    ]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), module.__dict__)
    return module, context, events


@pytest.mark.parametrize("groups", [4, 8, 16])
@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("version,rank", [("v1", 0), ("v1", 1), ("v2", 0), ("v2", 1)])
def test_capture_and_replay_without_v1_only_runner_fields(runtime, groups, stage, version, rank):
    module, context, events = runtime
    lookups = []

    def rank_lookup():
        assert version == "v2", "MRV1 must keep using its cached TP rank"
        lookups.append(rank)
        return rank

    module.get_tensor_model_parallel_rank = rank_lookup
    calls, offsets = [], []

    def layer(index, positions, hidden, residual, ids):
        assert context.capturing
        calls.append(index)
        return hidden, residual

    runner = NS(
        device="npu",
        model_config=NS(enforce_eager=False),
        compilation_config=NS(cudagraph_mode="full_decode_only"),
        parallel_config=NS(pipeline_parallel_size=2),
        ascend_config=NS(scheduler_config=NS(layered_prefill_config=NS(single_forward_decode_graph=True))),
        layered_prefill_model_adapter=NS(
            num_hidden_layers=32,
            start_layer=stage * 16,
            end_layer=(stage + 1) * 16,
            layers=list(range(32)),
            _forward_layer=layer,
        ),
        _set_layered_prefill_moe_layer_offset=offsets.append,
    )
    if version == "v1":
        runner.tp_rank = rank
    else:
        assert not hasattr(runner, "tp_rank")
    executor = module.LayeredPrefillGroupExecutor(runner)
    originals = (context.attn_metadata, context.slot_mapping, context.is_padding)
    for value in (1, 9):
        result = executor.forward_decode(
            NS(num_groups=groups), stage * 16, (stage + 1) * 16, (Tensor(value), None), None, Tensor(0)
        )
        assert result[0].value == value
        assert not context.capturing
        assert all(a is b for a, b in zip(originals, (context.attn_metadata, context.slot_mapping, context.is_padding)))
    # One capture per local group, then exactly one replay per group per call.
    assert calls == list(range(stage * 16, (stage + 1) * 16))
    assert len(executor.decode_graph_cache.entries) == groups // 2
    assert events.count("replay") == groups
    assert events.count("sync") == events.count("join") == groups // 2
    assert events.count("log") == (groups // 2 if rank == 0 else 0)
    assert lookups == ([rank] * (groups // 2) if version == "v2" else [])
    assert offsets[-1] == (stage + 1) * 16
