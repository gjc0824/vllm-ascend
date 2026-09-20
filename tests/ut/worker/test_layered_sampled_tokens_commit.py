# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NPUModelRunner._commit_layered_sampled_tokens."""

from types import SimpleNamespace

import pytest
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _make_runner(requests: dict) -> NPUModelRunner:
    runner = object.__new__(NPUModelRunner)
    runner.requests = requests
    runner.input_batch = SimpleNamespace(req_ids=list(requests))
    return runner


def _output(req_id: str, req_index: int, sampled_token_ids: list[list[int]]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: req_index},
        sampled_token_ids=sampled_token_ids,
    )


def test_commit_replaces_placeholder_tail():
    request = SimpleNamespace(output_token_ids=[5, 6, -1])
    runner = _make_runner({"req": request})

    runner._commit_layered_sampled_tokens(_output("req", 0, [[42]]))

    assert request.output_token_ids == [5, 6, 42]


def test_commit_leaves_real_tokens_untouched():
    request = SimpleNamespace(output_token_ids=[5, 6])
    runner = _make_runner({"req": request})

    runner._commit_layered_sampled_tokens(_output("req", 0, [[42]]))

    assert request.output_token_ids == [5, 6]


def test_commit_partial_sampled_ids_keeps_remaining_placeholders():
    request = SimpleNamespace(output_token_ids=[-1, -1])
    runner = _make_runner({"req": request})

    runner._commit_layered_sampled_tokens(_output("req", 0, [[42]]))

    assert request.output_token_ids == [42, -1]


def test_commit_skips_rows_without_sampled_tokens():
    request = SimpleNamespace(output_token_ids=[5, -1])
    runner = _make_runner({"req": request})

    # Empty sampled row and unknown request index must not corrupt state.
    runner._commit_layered_sampled_tokens(_output("req", 0, [[]]))
    assert request.output_token_ids == [5, -1]

    runner._commit_layered_sampled_tokens(
        ModelRunnerOutput(
            req_ids=["other"],
            req_id_to_index={"other": 0},
            sampled_token_ids=[[9]],
        )
    )
    assert request.output_token_ids == [5, -1]


@pytest.mark.parametrize("requests", [{}, {"req": None}])
def test_commit_tolerates_missing_request_state(requests):
    runner = _make_runner(requests)

    runner._commit_layered_sampled_tokens(_output("req", 0, [[42]]))
