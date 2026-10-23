# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Deferred output materialization and completion ownership for layered steps."""

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


class LayeredPrefillMergedAsyncOutput(AsyncModelRunnerOutput):
    """Resolve D/P output copies only when the executor consumes this step.

    A Decode copy event need not cover the subsequent Prefill work. ``event``
    must cover the entire step, including unsampled Prefill and postprocessing.
    Keep both child outputs and their execution state alive until completion.
    """

    def __init__(self, outputs, merge, *, event=None, keepalive=None):
        self.outputs = list(outputs)
        self.merge = merge
        self.event = event
        self.keepalive = keepalive
        self.output = None

    def get_output(self) -> ModelRunnerOutput:
        if self.output is not None:
            return self.output
        if self.event is not None:
            self.event.synchronize()
        resolved = [
            output.get_output() if isinstance(output, AsyncModelRunnerOutput) else output
            for output in self.outputs
        ]
        output = self.merge(resolved)
        # Do not release ownership on a failed copy or merge.
        self.output = output
        self.outputs = []
        self.merge = None
        self.keepalive = None
        return output
