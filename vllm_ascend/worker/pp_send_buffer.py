#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from dataclasses import dataclass, field

import torch
from vllm.distributed.parallel_state import Handle


def _get_max_slots() -> int:
    value = os.getenv("VLLM_ASCEND_PP_SEND_BUFFER_SLOTS", "2")
    try:
        return max(1, int(value))
    except ValueError:
        return 2


@dataclass
class _PPSendBufferSlot:
    slot_id: int
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    handles: list[Handle] = field(default_factory=list)
    in_use: bool = False
    busy_since: int = 0

    def is_completed(self) -> bool:
        if not self.in_use:
            return True
        if not self.handles:
            return False
        return all(handle.is_completed() for handle in self.handles)

    def wait(self) -> None:
        for handle in self.handles:
            handle.wait()
        self.release()

    def release(self) -> None:
        self.handles = []
        self.in_use = False

    def stage_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        buffer = self.buffers.get(name)
        if self._needs_new_buffer(buffer, tensor):
            buffer = torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            self.buffers[name] = buffer

        staged = self._view_for_shape(buffer, tensor.shape)
        staged.copy_(tensor, non_blocking=True)
        return staged

    @staticmethod
    def _needs_new_buffer(
        buffer: torch.Tensor | None,
        tensor: torch.Tensor,
    ) -> bool:
        if buffer is None:
            return True
        if buffer.dtype != tensor.dtype or buffer.device != tensor.device:
            return True
        if buffer.dim() != tensor.dim():
            return True
        if tensor.dim() == 0:
            return buffer.shape != tensor.shape
        return (
            buffer.shape[0] < tensor.shape[0]
            or tuple(buffer.shape[1:]) != tuple(tensor.shape[1:])
        )

    @staticmethod
    def _view_for_shape(
        buffer: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        if len(shape) == 0:
            return buffer
        return buffer[: shape[0], ...]


class PPAsyncSendBufferPool:
    """Owns NPU staging buffers used as PP async-send source tensors.

    Model forward outputs can be graph-owned or otherwise reused by the next
    forward. Async PP send must therefore send from independent storage whose
    lifetime is tied to the send handles.
    """

    def __init__(self, max_slots: int | None = None) -> None:
        self.max_slots = _get_max_slots() if max_slots is None else max(1, max_slots)
        self.slots: list[_PPSendBufferSlot] = []
        self._pending_slot: _PPSendBufferSlot | None = None
        self._stage_seq = 0
        self.num_waits = 0
        self.num_allocated_slots = 0

    def stage(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._pending_slot is not None:
            raise RuntimeError("Previous PP send buffer slot was not attached.")

        self.release_completed()
        slot = self._get_available_slot()
        slot.in_use = True
        slot.busy_since = self._stage_seq
        self._stage_seq += 1
        slot.handles = []
        self._pending_slot = slot
        return {
            name: slot.stage_tensor(name, tensor)
            for name, tensor in tensors.items()
        }

    def attach_handles(self, handles: list[Handle]) -> None:
        slot = self._pending_slot
        if slot is None:
            return
        self._pending_slot = None
        if handles:
            slot.handles = list(handles)
        else:
            slot.release()

    def abort_pending(self) -> None:
        slot = self._pending_slot
        self._pending_slot = None
        if slot is not None and not slot.handles:
            slot.release()

    def release_completed(self) -> None:
        for slot in self.slots:
            if slot.in_use and slot.is_completed():
                slot.release()

    def wait_all(self) -> None:
        if self._pending_slot is not None:
            self.abort_pending()
        for slot in self.slots:
            if slot.in_use:
                slot.wait()

    def debug_state(self) -> dict[str, object]:
        return {
            "max_slots": self.max_slots,
            "num_slots": len(self.slots),
            "num_waits": self.num_waits,
            "slots": [
                {
                    "id": slot.slot_id,
                    "in_use": slot.in_use,
                    "busy_since": slot.busy_since,
                    "handles": len(slot.handles),
                    "completed": slot.is_completed() if slot.in_use else True,
                }
                for slot in self.slots
            ],
        }

    def _get_available_slot(self) -> _PPSendBufferSlot:
        for slot in self.slots:
            if not slot.in_use:
                return slot

        if len(self.slots) < self.max_slots:
            slot = _PPSendBufferSlot(slot_id=len(self.slots))
            self.slots.append(slot)
            self.num_allocated_slots += 1
            return slot

        slot = min(self.slots, key=lambda candidate: candidate.busy_since)
        slot.wait()
        self.num_waits += 1
        return slot
