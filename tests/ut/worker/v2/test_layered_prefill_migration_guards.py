# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fail-closed MRV2 mode selection and backward-compatible async opt-in."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[4]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fuse_mixed", [False, True])
@pytest.mark.parametrize("same_layer", [False, True])
def test_runner_rejects_excluded_modes_only_when_enabled(enabled, fuse_mixed, same_layer):
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    init = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    guard = next(
        node
        for node in init.body
        if isinstance(node, ast.If)
        and "fuse_mixed_batch" in ast.unparse(node.test)
        and "same_layer_batch" in ast.unparse(node.test)
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
    namespace = {
        "self": SimpleNamespace(_layered_prefill_enabled=enabled),
        "layered_cfg": SimpleNamespace(fuse_mixed_batch=fuse_mixed, same_layer_batch=same_layer),
    }
    if enabled and (fuse_mixed or same_layer):
        with pytest.raises(NotImplementedError, match="fuse_mixed_batch or same_layer_batch"):
            exec(code, namespace)
    else:
        exec(code, namespace)


@pytest.mark.parametrize(
    "current,legacy,expected",
    [(None, None, False), (None, "1", True), ("0", "1", False), ("1", "0", True)],
)
def test_async_opt_in_uses_current_name_then_legacy_fallback(monkeypatch, current, legacy, expected):
    for name, value in (
        ("VLLM_ASCEND_LAYERED_ASYNC_COMPUTE", current),
        ("VLLM_LAYERED_ASYNC_COMPUTE", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    path = ROOT / "vllm_ascend/envs.py"
    spec = importlib.util.spec_from_file_location("layered_guard_test_envs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VLLM_ASCEND_LAYERED_ASYNC_COMPUTE is expected
