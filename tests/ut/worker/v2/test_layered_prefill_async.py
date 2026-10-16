# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for device completion and mutable H2D source ownership."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).parents[4]


@pytest.fixture
def implementation():
    helper_path = ROOT / "vllm_ascend/worker/v2/layered_prefill_async.py"
    helper = ast.parse(helper_path.read_text(encoding="utf-8"))
    helper.body = [n for n in helper.body if not isinstance(n, ast.ImportFrom)]
    namespace = dict(AsyncModelRunnerOutput=object, ModelRunnerOutput=NS)
    exec(compile(helper, str(helper_path), "exec"), namespace)
    runner_path = helper_path.with_name("model_runner.py")
    runner = next(
        n
        for n in ast.parse(runner_path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner"
    )
    wanted = {
        "_synchronize_layered_compute",
        "_layered_async_compute_enabled",
        "_num_hidden_layers",
        "layered_prefill_snapshot",
        "_complete_layered_output",
        "_sample_layered_v2",
    }
    runner.body = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    runner.bases = []
    for method in runner.body:
        method.decorator_list = []

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    runner = RemoveImports().visit(runner)
    config = NS(uva=False)
    # Load the lightweight env registry without importing the NPU package.
    env_path = ROOT / "vllm_ascend/envs.py"
    import importlib.util

    env_spec = importlib.util.spec_from_file_location("layered_test_envs", env_path)
    envs_ascend = importlib.util.module_from_spec(env_spec)
    env_spec.loader.exec_module(envs_ascend)
    namespace.update(
        envs_ascend=envs_ascend,
        is_uva_available=lambda: config.uva,
        LayeredV2ExecuteModelState=NS,
        AsyncOutput=type("Async", (), {}),
        EMPTY_MODEL_RUNNER_OUTPUT=NS(),
    )
    exec(compile(ast.Module(body=[runner], type_ignores=[]), str(runner_path), "exec"), namespace)
    return NS(config=config, namespace=namespace, **namespace)


class Stream:
    def __init__(self):
        self.done = False
        self.waited = []

    def wait_stream(self, other):
        self.waited.append(other)

    def synchronize(self):
        for stream in self.waited:
            stream.synchronize()
        self.done = True


class Event:
    def record(self, stream):
        self.stream = stream

    def synchronize(self):
        if not self.stream.done:
            raise RuntimeError("device work still pending")


def make_runner(impl, pp_size=2):
    compute, update, broadcast = Stream(), Stream(), Stream()
    impl.namespace["torch"] = NS(
        npu=NS(
            current_stream=lambda: compute,
            Event=Event,
            synchronize=lambda: [s.synchronize() for s in (compute, update, broadcast)],
        )
    )
    runner = impl.NPUModelRunner()
    runner.parallel_config = NS(pipeline_parallel_size=pp_size)
    runner.model_config = NS(num_hidden_layers=43)
    runner._layered_compute_num_groups = 2
    runner.update_stream = update
    runner.is_last_pp_rank = True
    runner._layered_counters = lambda: {"execute_steps": 0, "groups": [], "pp_slots": []}
    return runner, compute, update, broadcast


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("uva", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16, 43])
def test_compute_boundary_preserves_default_and_uva_fences(
    implementation,
    monkeypatch,
    enabled,
    uva,
    pp_size,
    groups,
):
    monkeypatch.setenv("VLLM_LAYERED_ASYNC_COMPUTE", "1" if enabled else "0")
    implementation.config.uva = uva
    runner, compute, update, broadcast = make_runner(implementation, pp_size)
    runner._layered_compute_num_groups = groups
    runner._synchronize_layered_compute()
    deferred = enabled and not uva and pp_size == 2
    assert runner.layered_prefill_snapshot()["async_compute_enabled"] is deferred
    assert compute.done is not deferred
    assert update.done is not deferred
    assert broadcast.done is (pp_size == 1)
    if pp_size > 1:
        assert compute.waited == [update]


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16, 43])
def test_empty_prefill_output_cannot_retire_before_device_completion(implementation, monkeypatch, groups):
    monkeypatch.setenv("VLLM_LAYERED_ASYNC_COMPUTE", "1")
    runner, compute, update, broadcast = make_runner(implementation)
    runner._layered_compute_num_groups = groups
    runner._synchronize_layered_compute()
    state = NS(d_state=None, sample_p=False)
    output = NS(req_ids=["prompt"], sampled_token_ids=[[]])
    pending = runner._complete_layered_output(output, state)
    assert isinstance(pending, implementation.LayeredPrefillAsyncOutput)
    assert pending.keepalive is state
    with pytest.raises(RuntimeError, match="still pending"):
        pending.get_output()
    assert pending.keepalive is state
    compute.synchronize()
    assert pending.get_output() is output
    assert pending.keepalive is None
    assert not broadcast.done


def test_unsampled_prefill_sampling_path_returns_completion_wrapper(implementation, monkeypatch):
    monkeypatch.setenv("VLLM_LAYERED_ASYNC_COMPUTE", "1")
    runner, compute, *_ = make_runner(implementation)
    runner._layered_pp_handler_capture = lambda: nullcontext()
    runner.kv_connector = NS(post_forward=lambda finished: None)
    state = NS(
        d_state=None,
        sample_p=False,
        single_forward=False,
        p_state=NS(input_batch=NS(req_ids=["p"]), finished_req_ids=set()),
    )
    pending = runner._sample_layered_v2(None, state)
    assert isinstance(pending, implementation.LayeredPrefillAsyncOutput)
    with pytest.raises(RuntimeError, match="still pending"):
        pending.get_output()
    compute.synchronize()
    output = pending.get_output()
    assert output.req_ids == ["p"]
    assert output.sampled_token_ids == [[]]


@pytest.mark.parametrize(
    "groups,flags",
    [
        (0, {}),
        (1, {}),
        (44, {}),
    ]
    + [
        (groups, flags)
        for groups in (2, 3, 4, 5, 8, 16, 43)
        for flags in (
            {"_skip_relay_stages": True},
            {"_fuse_mixed_batch": True},
            {"_same_layer_batch": True},
            {"use_async_scheduling": True},
            {"scheduler_config": NS(async_scheduling=True)},
            {"dp_size": 2},
            {"vllm_config": NS(kv_transfer_config=object())},
        )
    ],
)
def test_unvalidated_layouts_keep_host_completion(implementation, monkeypatch, groups, flags):
    monkeypatch.setenv("VLLM_LAYERED_ASYNC_COMPUTE", "1")
    runner, compute, *_ = make_runner(implementation)
    runner._layered_compute_num_groups = groups
    for name, value in flags.items():
        setattr(runner, name, value)
    runner._synchronize_layered_compute()
    assert compute.done
    assert runner.layered_prefill_snapshot()["async_compute_enabled"] is False


@pytest.mark.parametrize("has_decode,sample_p,last", [(True, False, True), (False, True, True), (False, False, False)])
@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16, 43])
def test_sampling_and_nonlast_outputs_keep_existing_completion_contract(
    implementation,
    monkeypatch,
    has_decode,
    sample_p,
    last,
    groups,
):
    monkeypatch.setenv("VLLM_LAYERED_ASYNC_COMPUTE", "1")
    runner, *_ = make_runner(implementation)
    runner._layered_compute_num_groups = groups
    runner.is_last_pp_rank = last
    output = object()
    state = NS(d_state=object() if has_decode else None, sample_p=sample_p)
    assert runner._complete_layered_output(output, state) is output


def test_uva_prefix_dma_owns_snapshot_when_cpu_pool_slot_is_reused(monkeypatch):
    path = ROOT / "vllm_ascend/patch/worker/patch_v2/patch_uva.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UvaBufferWrapper")
    namespace = dict(torch=torch, Sequence=tuple, is_uva_available=lambda: False)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    wrapper = namespace["UvaBufferWrapper"].__new__(namespace["UvaBufferWrapper"])
    wrapper._cpu = torch.tensor([11, 22, 33], dtype=torch.int32)
    wrapper._modified_indices = {0, 1}
    queued = []

    class Device:
        def __getitem__(self, index):
            return self

        def copy_(self, source, non_blocking):
            queued.append((source, non_blocking))
            return self

    wrapper._uva = Device()
    real_empty_like = torch.empty_like
    allocations = []

    def cpu_empty_like(value, *, pin_memory):
        allocations.append(pin_memory)
        return real_empty_like(value)  # CPU-only test host has no pinned allocator.

    monkeypatch.setattr(torch, "empty_like", cpu_empty_like)
    _ = wrapper.uva
    # Reuse the same CPU slot before either simulated DMA reads its source.
    wrapper._cpu[:2].fill_(99)
    wrapper._modified_indices.update((0, 1))
    _ = wrapper.uva
    wrapper._cpu[:2].zero_()
    assert [source.tolist() for source, _ in queued] == [[11, 22], [99, 99]]
    assert all(non_blocking for _, non_blocking in queued)
    assert allocations == [True, True]
    assert not wrapper._modified_indices
