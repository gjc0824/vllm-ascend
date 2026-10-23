# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run the actual MRV2/Ascend constructors with CPU-only hardware stand-ins."""

import ast
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).parents[4]


def _constructor_class(path, name, base, extra_methods=()):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.bases = [ast.Name(id=base, ctx=ast.Load())]
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("__init__", *extra_methods)
    ]
    # Imported NPU platform probes are supplied by the test, like allocations.
    class HardwareImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    return HardwareImports().visit(cls)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("layered_enabled", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("boundary_policy", ["off", "pp_transition"])
def test_real_mrv2_constructor_initializes_layered_async_contract(
    async_scheduling, layered_enabled, pp_size, boundary_policy
):
    # Prefer the paired checkout; installed vllm is the normal CI alternative.
    base_path = ROOT.parent / "vllm/vllm/v1/worker/gpu/model_runner.py"
    if not base_path.exists():
        spec = importlib.util.find_spec("vllm")
        assert spec is not None and spec.submodule_search_locations
        base_path = Path(next(iter(spec.submodule_search_locations))) / "v1/worker/gpu/model_runner.py"

    mode = NS(has_full_cudagraphs=lambda: False)
    config = NS(
        model_config=NS(
            dtype=torch.float32,
            get_vocab_size=lambda: 32,
            max_model_len=64,
            is_encoder_decoder=False,
            runner_type="generate",
            enforce_eager=True,
        ),
        cache_config=NS(cache_dtype="auto"),
        compilation_config=NS(cudagraph_mode=mode, mode="eager"),
        lora_config=None,
        load_config=NS(),
        parallel_config=NS(
            pipeline_parallel_size=pp_size,
            data_parallel_size=1,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            enable_eplb=False,
        ),
        scheduler_config=NS(
            async_scheduling=async_scheduling,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
        speculative_config=None,
        observability_config=NS(),
        max_concurrent_batches=pp_size + int(async_scheduling),
        num_speculative_tokens=0,
    )
    ascend_config = NS(
        eplb_config=NS(load_collection_phase="all"),
        scheduler_config=NS(
            layered_prefill_config=NS(enabled=layered_enabled, boundary_policy=boundary_policy),
        ),
    )
    allocations = []

    def allocation(*args, **kwargs):
        result = NS(**kwargs)
        allocations.append(result)
        return result

    def no_op(*args, **kwargs):
        return None

    fake_torch = NS(
        cuda=NS(Stream=allocation),
        npu=NS(Stream=allocation, Event=allocation),
        empty=lambda *args, **kwargs: torch.empty(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}),
        int32=torch.int32,
    )
    namespace = dict(
        torch=fake_torch,
        LoRAModelRunnerMixin=object,
        get_pp_group=lambda: NS(is_first_rank=True, is_last_rank=pp_size == 1),
        set_default_max_concurrency=no_op,
        MULTIMODAL_REGISTRY=NS(supports_multimodal_inputs=lambda config: False),
        NO_OP_KV_CONNECTOR=NS(),
        get_ascend_config=lambda: ascend_config,
        torch_cuda_wrapper=nullcontext,
        CompilationMode=NS(VLLM_COMPILE="compile"),
        CUDAGraphMode=NS(NONE=mode),
        set_potential_max_tokens=no_op,
        set_cos_and_sin=no_op,
        set_mc2_tokens_capacity=no_op,
        set_mc2_mask=no_op,
        empty_layered_prefill_counters=lambda: {},
        envs_ascend=NS(VLLM_ASCEND_LAYERED_ASYNC_COMPUTE=False),
        is_uva_available=lambda: False,
    )
    for name in (
        "DraftTokensHandler",
        "RequestState",
        "InputBuffers",
        "PPHandler",
        "LoraState",
        "EPLBController",
        "AscendEPLBController",
        "AscendRequestState",
        "AscendInputBuffers",
        "LayeredPrefillGroupExecutor",
        "LayeredV2SingleForward",
        "LayeredV2Buffers",
        "LayeredPrefillStateStore",
    ):
        namespace[name] = allocation
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            _constructor_class(base_path, "GPUModelRunner", "LoRAModelRunnerMixin"),
            _constructor_class(
                ROOT / "vllm_ascend/worker/v2/model_runner.py",
                "NPUModelRunner",
                "GPUModelRunner",
                ("_layered_async_compute_enabled",),
            ),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(base_path), "exec"), namespace)
    runner = namespace["NPUModelRunner"](config, torch.device("cpu"))
    assert runner.use_async_scheduling is async_scheduling
    assert runner.scheduler_config is config.scheduler_config
    assert runner._layered_prefill_enabled is layered_enabled
    assert runner._boundary_prefill_enabled is (layered_enabled and boundary_policy == "pp_transition")
    assert (runner._layered_input_buffers is not None) is layered_enabled
    assert runner._layered_async_compute_enabled() is async_scheduling
    assert any(runner.input_buffers is buffer for buffer in allocations)
