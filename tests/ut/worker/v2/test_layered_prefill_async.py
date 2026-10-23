# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for device completion and mutable H2D source ownership."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[4]


@pytest.fixture
def implementation():
    helper_path = ROOT / "vllm_ascend/worker/v2/layered_prefill_async.py"
    helper = ast.parse(helper_path.read_text(encoding="utf-8"))
    helper.body = [n for n in helper.body if not isinstance(n, ast.ImportFrom)]
    namespace = dict(AsyncModelRunnerOutput=type("AsyncModelRunnerOutput", (), {}), ModelRunnerOutput=NS)
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
        "_layered_relay_stage_action",
        "_execute_layered_step",
    }
    runner.body = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    runner.bases = [ast.Name(id="BaseRunner", ctx=ast.Load())]
    for method in runner.body:
        method.decorator_list = []

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    runner = RemoveImports().visit(runner)
    config = NS(uva=False)

    class BaseRunner:
        def sample_tokens(self, grammar_output):
            return self.sample_outputs.pop(0)

    # Load the lightweight env registry without importing the NPU package.
    env_path = ROOT / "vllm_ascend/envs.py"
    import importlib.util

    env_spec = importlib.util.spec_from_file_location("layered_test_envs", env_path)
    envs_ascend = importlib.util.module_from_spec(env_spec)
    env_spec.loader.exec_module(envs_ascend)
    namespace.update(
        BaseRunner=BaseRunner,
        envs_ascend=envs_ascend,
        is_uva_available=lambda: config.uva,
        LayeredV2ExecuteModelState=NS,
        SchedulerOutput=NS,
        IntermediateTensors=NS,
        AsyncOutput=type("Async", (), {}),
        EMPTY_MODEL_RUNNER_OUTPUT=NS(),
    )
    module = ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[]))
    exec(compile(module, str(runner_path), "exec"), namespace)
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


def test_async_scheduling_rejects_mutable_mapped_host_buffers(implementation):
    runner, *_ = make_runner(implementation)
    runner.use_async_scheduling = True
    implementation.config.uva = True
    with pytest.raises(ValueError, match="requires staged H2D buffers"):
        runner._layered_async_compute_enabled()


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


def test_merged_output_defers_both_sample_copies_and_unsampled_prefill(implementation):
    """Returning a step must neither resolve samples nor retire later P work."""
    calls = []

    class Sample(implementation.AsyncModelRunnerOutput):
        def get_output(self):
            calls.append("sample")
            return NS(req_ids=["d"], sampled_token_ids=[[7]])

    compute = Stream()
    event = Event()
    event.record(compute)
    keepalive = object()
    prefill = NS(req_ids=["p"], sampled_token_ids=[[]])

    def merge(outputs):
        calls.append("merge")
        return NS(sampled_token_ids=[row for output in outputs for row in output.sampled_token_ids])

    pending = implementation.LayeredPrefillMergedAsyncOutput(
        [Sample(), prefill], merge, event=event, keepalive=keepalive
    )
    assert calls == []
    with pytest.raises(RuntimeError, match="still pending"):
        pending.get_output()
    assert calls == []
    assert pending.keepalive is keepalive
    compute.synchronize()
    output = pending.get_output()
    assert output.sampled_token_ids == [[7], []]
    assert calls == ["sample", "merge"]
    assert pending.keepalive is None
    assert pending.outputs == []
    assert pending.get_output() is output
    assert calls == ["sample", "merge"]


def test_merged_output_keeps_ownership_if_a_child_copy_fails(implementation):
    class FailedSample(implementation.AsyncModelRunnerOutput):
        def get_output(self):
            raise RuntimeError("copy failed")

    child = FailedSample()
    keepalive = object()
    pending = implementation.LayeredPrefillMergedAsyncOutput([child], lambda outputs: outputs[0], keepalive=keepalive)
    with pytest.raises(RuntimeError, match="copy failed"):
        pending.get_output()
    assert pending.outputs == [child]
    assert pending.keepalive is keepalive


@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("flag", ["runner", "config"])
def test_async_scheduler_enqueues_without_waiting_for_compute(implementation, monkeypatch, pp_size, flag):
    monkeypatch.setenv("VLLM_LAYERED_ASYNC_COMPUTE", "0")
    runner, compute, update, broadcast = make_runner(implementation, pp_size)
    if flag == "runner":
        runner.use_async_scheduling = True
    else:
        runner.scheduler_config = NS(async_scheduling=True)
    runner._synchronize_layered_compute()
    assert compute.waited == [update]
    assert not compute.done and not update.done and not broadcast.done


def test_runner_returns_async_samples_before_device_completion(implementation):
    runner, compute, *_ = make_runner(implementation)
    calls = []

    class Sample(implementation.AsyncModelRunnerOutput):
        def get_output(self):
            calls.append("sample")
            return NS(req_ids=["d"], sampled_token_ids=[[7]])

    runner.sample_outputs = [Sample()]
    runner._layered_pp_handler_capture = lambda: nullcontext()
    runner.kv_connector = NS(post_forward=lambda finished: None)
    runner._merge_layered_v2_outputs = lambda scheduler, outputs: NS(
        req_ids=["d", "p"], sampled_token_ids=[row for output in outputs for row in output.sampled_token_ids]
    )
    state = NS(
        scheduler_output=NS(),
        d_state=NS(input_batch=NS(req_ids=["d"])),
        p_state=NS(input_batch=NS(req_ids=["p"]), finished_req_ids=set()),
        sample_p=False,
        single_forward=False,
    )
    pending = runner._sample_layered_v2(None, state)
    assert isinstance(pending, implementation.LayeredPrefillMergedAsyncOutput)
    assert calls == []
    assert runner.execute_model_state is None
    with pytest.raises(RuntimeError, match="still pending"):
        pending.get_output()
    assert calls == []
    compute.synchronize()
    assert pending.get_output().sampled_token_ids == [[7], []]
    assert calls == ["sample"]


@pytest.mark.parametrize("sampling_prefill", [False, True])
def test_single_async_sample_keeps_original_request_order_and_logprobs(implementation, sampling_prefill):
    runner, compute, *_ = make_runner(implementation)
    logprobs = object()
    original = NS(req_ids=["a", "b"], sampled_token_ids=[[7], [8]], logprobs=logprobs)
    calls = []

    class Sample(implementation.AsyncModelRunnerOutput):
        def get_output(self):
            calls.append("sample")
            return original

    def unexpected_merge(*args):
        raise AssertionError("A single sample must retain its original row alignment")

    runner.sample_outputs = [Sample()]
    runner._layered_pp_handler_capture = lambda: nullcontext()
    runner._merge_layered_v2_outputs = unexpected_merge
    sampled_state = NS(input_batch=NS(req_ids=["a", "b"]))
    state = NS(
        scheduler_output=NS(num_scheduled_tokens={"b": 1, "a": 1}),
        d_state=None if sampling_prefill else sampled_state,
        p_state=sampled_state if sampling_prefill else None,
        sample_p=sampling_prefill,
        single_forward=False,
    )
    pending = runner._sample_layered_v2(None, state)
    assert calls == []
    compute.synchronize()
    output = pending.get_output()
    assert output is original
    assert output.req_ids == ["a", "b"]
    assert output.logprobs is logprobs
    assert calls == ["sample"]


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("rank,synchronous_action", [(0, "empty"), (1, None), (2, None), (3, "forward")])
def test_async_relay_preserves_common_lifecycle_while_sync_keeps_stage_shortcuts(
    implementation, async_scheduling, rank, synchronous_action
):
    runner, *_ = make_runner(implementation, pp_size=4)
    runner.use_async_scheduling = async_scheduling
    runner._skip_relay_stages = True
    runner.dp_size = 1
    runner.vllm_config = NS(kv_transfer_config=None)
    runner._layered_pp_group_owner = lambda plan: 1
    runner._layered_next_group_owner = lambda plan: 2
    implementation.namespace["get_pp_group"] = lambda: NS(rank_in_group=rank)
    plan = NS(is_sampling_step=False, is_final_group=False)
    assert runner._layered_relay_stage_action(plan) == (None if async_scheduling else synchronous_action)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_async_mixed_step_does_not_overwrite_native_decode_device_cursor(implementation, async_scheduling, pp_size):
    """Execute the lifecycle and cursor updates before a stand-in forward."""
    runner, *_ = make_runner(implementation, pp_size=pp_size)
    runner.use_async_scheduling = async_scheduling
    runner.is_first_pp_rank = True
    runner.layered_prefill_model_adapter = object()
    runner.pp_handler = None
    runner.layered_prefill_state = NS(clear_many=lambda ids: None)
    runner._single_forward = True
    runner._record_layered_step = lambda *args, **kwargs: None
    updates = []
    runner._apply_scheduler_lifecycle = lambda schedule: updates.append(("lifecycle", None))
    runner._sync_ascend_num_computed_tokens_cpu = lambda ids: updates.append(("host", list(ids)))
    runner._sync_gpu_num_computed_tokens = lambda ids: updates.append(("device", list(ids)))
    sentinel = object()
    runner._execute_layered_single_forward = lambda *args: sentinel
    implementation.namespace.update(
        split_d_p_req_ids=lambda schedule, plan: (["d"], ["p"]),
        LayeredV2SingleForward=NS(supports_plan=lambda plan: True, supported=lambda runner: True),
        LayeredV2PPStageForward=NS(supported=lambda *args: False),
    )
    schedule = NS(num_scheduled_tokens={"d": 1, "p": 8})
    plan = NS(num_groups=2)
    assert runner._execute_layered_step(schedule, plan) is sentinel
    assert updates == [
        ("lifecycle", None),
        ("host", ["d", "p"]),
        ("device", ["p"] if async_scheduling else ["d", "p"]),
    ]
    assert not runner._layered_skip_pp_decode_update


@pytest.mark.parametrize("api", [0, 1])
def test_queued_prepare_inputs_owns_cpu_attention_lengths(api):
    """Run both real prepare_inputs APIs while a prior host metadata view lives."""
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    methods = [n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == "prepare_inputs"]
    method = methods[api]

    def copy_to_device(value, out=None, device=None):
        value = torch.as_tensor(value)
        return value.clone() if out is None else out.copy_(value)

    namespace = dict(
        torch=torch,
        np=np,
        SchedulerOutput=NS,
        BatchExecutionDescriptor=NS,
        BatchReqState=NS,
        AscendInputBatch=NS,
        CUDAGraphMode=NS(FULL="full"),
        sort_batch_req_ids=lambda counts, *args: list(counts),
        build_attn_state=lambda *args: None,
        async_copy_to_gpu=copy_to_device,
        prepare_pos_seq_lens=lambda *args: None,
        combine_sampled_and_draft_tokens=lambda *args: torch.tensor([0]),
        update_cos_sin=lambda positions: None,
        vllm_model_runner=NS(pcp=NS(maybe_partition_pcp_batch=lambda manager, batch: batch)),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    buffers = NS(
        seq_lens_np=np.zeros(4, dtype=np.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(6, dtype=torch.int32),
        input_ids=torch.zeros(4, dtype=torch.int64),
        positions=torch.zeros(4, dtype=torch.int64),
        is_padding=torch.zeros(4, dtype=torch.bool),
    )
    state = NS(
        req_id_to_index={"d": 0},
        prefill_len=NS(np=np.array([1]), gpu=torch.tensor([1])),
        num_computed_prefill_tokens=np.array([1]),
        num_computed_tokens_np=np.array([4]),
        num_computed_tokens=NS(gpu=torch.tensor([4])),
        last_sampled_tokens=torch.tensor([7]),
        draft_tokens=None,
    )
    runner = NS(
        input_buffers=buffers,
        req_states=state,
        device="cpu",
        max_num_reqs=4,
        decode_query_len=1,
        vllm_config=NS(),
        model_config=NS(rswa_window=None),
        model_state=NS(num_new_sampled_tokens_per_step=1),
        use_pp=False,
        pcp_manager=None,
        eplb=NS(set_batch_phase=lambda value: None),
        _reset_input_padding=lambda *args: None,
    )
    runner._update_seq_lens_cpu = lambda schedule, ids: buffers.seq_lens_np.__setitem__(
        0, state.num_computed_tokens_np[0] + 1
    )
    schedule = NS(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"d": 1},
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
    )
    descriptor = NS(num_tokens=1, num_reqs=1, cg_mode="none")
    args = (schedule, descriptor) if api == 0 else (schedule, NS(), descriptor)
    first = namespace["prepare_inputs"](runner, *args)
    # This is the same zero-copy view used by build_attn_metadata.
    pending_host_metadata = torch.from_numpy(first.seq_lens_np)
    state.num_computed_tokens_np[0] = 12
    second = namespace["prepare_inputs"](runner, *args)
    assert second.seq_lens_np[0] == 13
    assert first.seq_lens_np[0] == 5
    assert pending_host_metadata.tolist() == [5, 0, 0, 0]
    assert not np.shares_memory(first.seq_lens_np, buffers.seq_lens_np)


@pytest.mark.parametrize("pp_size", [1, 2, 3, 4])
@pytest.mark.parametrize("single_forward", [False, True])
@pytest.mark.parametrize("has_decode", [False, True])
@pytest.mark.parametrize("sample_p", [False, True])
def test_ordinary_async_steps_queue_before_output_resolution(
    implementation, pp_size, single_forward, has_decode, sample_p
):
    runner, compute, *_ = make_runner(implementation, pp_size)
    runner.use_async_scheduling = True
    runner._layered_pp_handler_capture = lambda: nullcontext()
    runner.kv_connector = NS(post_forward=lambda finished: None)
    resolved = []

    class Sample(implementation.AsyncModelRunnerOutput):
        def __init__(self, ids, token):
            self.ids, self.token = ids, token

        def get_output(self):
            resolved.append(tuple(self.ids))
            return NS(req_ids=self.ids, sampled_token_ids=[[self.token] for _ in self.ids])

    def merge(schedule, outputs):
        rows = {req: tokens for out in outputs for req, tokens in zip(out.req_ids, out.sampled_token_ids)}
        return NS(
            req_ids=list(schedule.num_scheduled_tokens),
            sampled_token_ids=[rows[r] for r in schedule.num_scheduled_tokens],
        )

    runner._merge_layered_v2_outputs = merge
    pending = []
    for step in range(2):
        d_ids = [f"d{step}"] if has_decode else []
        p_ids = [f"p{step}"]
        # PP1 single-forward final sampling uses one combined batch; PP2's
        # mixed stage and PP4's fallback retain separate D/P sampler batches.
        combined = single_forward and pp_size == 1 and sample_p
        sample_ids = [d_ids + p_ids] if combined else ([d_ids] if has_decode else []) + ([p_ids] if sample_p else [])
        runner.sample_outputs = [Sample(ids, 70 + step) for ids in sample_ids]
        state = NS(
            scheduler_output=NS(
                num_scheduled_tokens={req: 1 for req in p_ids + d_ids}, layered_prefill_plan=NS(boundary_policy="off")
            ),
            d_state=NS(input_batch=NS(req_ids=d_ids)) if has_decode and not combined else None,
            p_state=NS(input_batch=NS(req_ids=d_ids + p_ids if combined else p_ids), finished_req_ids=set()),
            sample_p=sample_p,
            single_forward=single_forward and pp_size <= 2,
        )
        pending.append(runner._sample_layered_v2(None, state))
        assert not runner.sample_outputs
        assert runner.execute_model_state is None
        assert not compute.done
        assert resolved == []
    # Retiring the first result cannot consume or overwrite the second result.
    compute.synchronize()
    for step, output in enumerate(pending):
        result = output.get_output()
        rows = dict(zip(result.req_ids, result.sampled_token_ids))
        assert rows[f"p{step}"] == ([70 + step] if sample_p else [])
        if has_decode:
            assert rows[f"d{step}"] == [70 + step]
