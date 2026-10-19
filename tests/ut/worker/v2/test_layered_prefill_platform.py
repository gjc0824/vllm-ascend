# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Execute production config gates without importing torch or allocating devices."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).parents[4] / "vllm_ascend"


@pytest.fixture
def validate():
    source = ROOT / "platform.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    check = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_check_ascend_config"
    )
    branch = next(
        node
        for node in check.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "layered_prefill_config.enabled"
    )
    platform_code = compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec")
    source = ROOT / "ascend_config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    boundary = next(
        node
        for node in init.body
        if isinstance(node, ast.If) and "boundary_config.boundary_policy" in ast.unparse(node.test)
    )
    boundary_code = compile(ast.Module(body=[boundary], type_ignores=[]), str(source), "exec")

    def run(vllm, ascend):
        namespace = {
            "vllm_config": vllm,
            "ascend_config": ascend,
            "scheduler_extension_config": ascend.scheduler_config,
            "layered_prefill_config": ascend.scheduler_config.layered_prefill_config,
            "boundary_config": ascend.scheduler_config.layered_prefill_config,
            "enable_sp": lambda config: getattr(config, "test_sequence_parallel", False),
        }
        # check_and_update_config constructs AscendConfig before its platform
        # gate. Exercise that ordering so opening MRV2 cannot bypass boundary's
        # independently narrower support envelope.
        exec(boundary_code, namespace)
        exec(platform_code, namespace)

    return run


def configuration(*, v2=True, pp=2, policy="off", groups=4):
    vllm = NS(
        use_v2_model_runner=v2,
        parallel_config=NS(pipeline_parallel_size=pp, data_parallel_size=1),
        cache_config=NS(enable_prefix_caching=False),
        model_config=NS(hf_text_config=NS(model_type="deepseek_v4", num_hidden_layers=43), enforce_eager=False),
        scheduler_config=NS(async_scheduling=False),
        kv_transfer_config=None,
        speculative_config=None,
        compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY"), mode=NS(name="VLLM_COMPILE")),
    )
    layered = NS(enabled=True, boundary_policy=policy, require_eager=False, allowed_num_groups=(groups,))
    ascend = NS(
        scheduler_config=NS(
            layered_prefill_config=layered,
            profiling_chunk_config=NS(enabled=False),
            enable_balance_scheduling=False,
            batch_job_sched_config=NS(enabled=False),
            short_request_first_config=NS(enabled=False),
            dyntra_lb_config=NS(enabled=False),
            recompute_scheduler_enable=False,
        )
    )
    return vllm, ascend


@pytest.mark.parametrize("groups", [4, 8, 16])
@pytest.mark.parametrize(
    "v2,pp,policy",
    [
        (False, 1, "off"),
        (False, 2, "off"),
        (True, 1, "off"),
        (True, 2, "off"),
        (True, 2, "pp_transition"),
    ],
)
def test_both_runners_admit_supported_layered_configs(validate, groups, v2, pp, policy):
    validate(*configuration(v2=v2, pp=pp, policy=policy, groups=groups))


@pytest.mark.parametrize(
    "owner,path,value,match",
    [
        ("vllm", "parallel_config.pipeline_parallel_size", 0, "must be positive"),
        ("vllm", "parallel_config.pipeline_parallel_size", 44, "model layer count"),
        ("vllm", "parallel_config.data_parallel_size", 2, "data_parallel_size"),
        ("vllm", "parallel_config.prefill_context_parallel_size", 2, "context parallelism"),
        ("vllm", "parallel_config.decode_context_parallel_size", 2, "context parallelism"),
        ("vllm", "parallel_config.enable_dbo", True, "DBO"),
        ("vllm", "scheduler_config.async_scheduling", True, "async_scheduling"),
        ("vllm", "test_sequence_parallel", True, "sequence-parallel"),
        ("vllm", "parallel_config.enable_eplb", True, "upstream EPLB"),
        ("vllm", "cache_config.kv_offloading_size", 1, "KV offloading"),
        ("vllm", "cache_config.mamba_cache_mode", "all", "hybrid KV"),
        ("vllm", "kv_transfer_config", object(), "connector-free"),
        ("vllm", "lora_config", object(), "LoRA"),
        ("vllm", "model_config.is_encoder_decoder", True, "decoder-only"),
        ("vllm", "model_config.is_multimodal_model", True, "decoder-only"),
        ("vllm", "model_config.enable_return_routed_experts", True, "routed-expert"),
        ("vllm", "compilation_config.cudagraph_mode", NS(name="FULL"), "cudagraph_mode"),
        ("vllm", "compilation_config.mode", NS(name="STOCK_TORCH_COMPILE"), "VLLM_COMPILE"),
        ("ascend", "enable_prefill_mc2", True, "enable_prefill_mc2"),
        ("ascend", "enable_fused_mc2", 1, "MC2/fused MC2"),
        ("ascend", "multistream_overlap_shared_expert", True, "shared-expert overlap"),
        ("ascend", "scheduler_config.profiling_chunk_config.enabled", True, "profiling_chunk_config"),
        ("ascend", "scheduler_config.enable_balance_scheduling", True, "balance scheduling"),
        ("ascend", "scheduler_config.batch_job_sched_config.enabled", True, "batch-job"),
        ("ascend", "scheduler_config.short_request_first_config.enabled", True, "short-request"),
        ("ascend", "scheduler_config.dyntra_lb_config.enabled", True, "DyntraLB"),
        ("ascend", "scheduler_config.recompute_scheduler_enable", True, "recompute"),
        ("ascend", "scheduler_config.layered_prefill_config.require_eager", True, "enforce_eager"),
    ],
)
@pytest.mark.parametrize("v2", [False, True])
def test_shared_capability_guards_still_reject(validate, owner, path, value, match, v2):
    vllm, ascend = configuration(v2=v2)
    obj = vllm if owner == "vllm" else ascend
    fields = path.split(".")
    for field in fields[:-1]:
        obj = getattr(obj, field)
    setattr(obj, fields[-1], value)
    with pytest.raises(ValueError, match=match):
        validate(vllm, ascend)


@pytest.mark.parametrize("method", ["mtp", "dspark", "eagle"])
def test_v2_does_not_inherit_v1_speculation_exception(validate, method):
    vllm, ascend = configuration(v2=True, pp=1)
    vllm.speculative_config = NS(method=method)
    with pytest.raises(ValueError, match="Model Runner V2.*speculative decoding"):
        validate(vllm, ascend)


def test_v1_mtp_exception_is_preserved(validate):
    vllm, ascend = configuration(v2=False, pp=1)
    vllm.speculative_config = NS(method="mtp")
    validate(vllm, ascend)


@pytest.mark.parametrize("v2,pp", [(False, 2), (True, 1), (True, 3)])
def test_boundary_still_requires_v2_and_exactly_two_stages(validate, v2, pp):
    with pytest.raises(ValueError, match="requires MRV2"):
        validate(*configuration(v2=v2, pp=pp, policy="pp_transition"))


def test_disabled_layered_has_no_new_restrictions(validate):
    vllm, ascend = configuration()
    ascend.scheduler_config.layered_prefill_config.enabled = False
    vllm.speculative_config = object()
    vllm.scheduler_config.async_scheduling = True
    vllm.parallel_config.data_parallel_size = 2
    validate(vllm, ascend)


@pytest.mark.parametrize("v2", [False, True])
def test_eager_none_remains_supported(validate, v2):
    vllm, ascend = configuration(v2=v2)
    vllm.model_config.enforce_eager = True
    ascend.scheduler_config.layered_prefill_config.require_eager = True
    vllm.compilation_config.cudagraph_mode = NS(name="NONE")
    vllm.compilation_config.mode = NS(name="NONE")
    validate(vllm, ascend)
