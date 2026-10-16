# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Completion ownership for layered steps which do not invoke the sampler."""

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


class LayeredPrefillAsyncOutput(AsyncModelRunnerOutput):
    """Retire an empty-token step only after its compute event completes."""

    def __init__(self, output: ModelRunnerOutput, event, keepalive=None):
        self.output = output
        self.event = event
        self.keepalive = keepalive

    def get_output(self) -> ModelRunnerOutput:
        self.event.synchronize()
        self.keepalive = None
        return self.output
