#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""
Virtual Pipeline Parallelism (VPP) utilities.

Implements a V-shaped fold-back layer assignment topology:
- Even virtual stages flow forward  (rank 0 -> rank N-1)
- Odd  virtual stages flow backward (rank N-1 -> rank 0)

Example with pp_size=2, vp_size=2, 8 layers (2 per chunk):
  Rank 0: chunks [0, 3] -> layers [0,1] + [6,7]  (first + last)
  Rank 1: chunks [1, 2] -> layers [2,3] + [4,5]  (middle, fold point)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


def get_vpp_indices(
    num_hidden_layers: int,
    pp_rank: int,
    pp_size: int,
    vp_size: int,
) -> list[tuple[int, int]]:
    """Return layer ranges for each virtual stage on a given PP rank.

    Uses V-shaped fold-back assignment:
      even vp: chunk_idx = vp * pp_size + pp_rank        (forward)
      odd  vp: chunk_idx = (vp+1) * pp_size - 1 - pp_rank (backward)

    Returns:
        List of (start_layer, end_layer) tuples, one per virtual stage.
        Layer indices are 0-based, end is exclusive.

    Examples:
        >>> get_vpp_indices(8, 0, 2, 2)
        [(0, 2), (6, 8)]
        >>> get_vpp_indices(8, 1, 2, 2)
        [(2, 4), (4, 6)]
        >>> get_vpp_indices(16, 0, 4, 2)
        [(0, 2), (14, 16)]
        >>> get_vpp_indices(16, 3, 4, 2)
        [(6, 8), (8, 10)]
    """
    total_chunks = pp_size * vp_size
    assert num_hidden_layers % total_chunks == 0, (
        f"num_hidden_layers ({num_hidden_layers}) must be divisible by "
        f"pp_size * vp_size ({pp_size} * {vp_size} = {total_chunks})"
    )
    layers_per_chunk = num_hidden_layers // total_chunks
    ranges: list[tuple[int, int]] = []
    for vp in range(vp_size):
        if vp % 2 == 0:
            chunk_idx = vp * pp_size + pp_rank
        else:
            chunk_idx = (vp + 1) * pp_size - 1 - pp_rank
        start = chunk_idx * layers_per_chunk
        end = start + layers_per_chunk
        ranges.append((start, end))
    return ranges


def is_vpp_first_stage(pp_rank: int, vp_stage: int) -> bool:
    """Whether this is the very first stage in the VPP pipeline."""
    return vp_stage == 0 and pp_rank == 0


def is_vpp_last_stage(
    pp_rank: int,
    pp_size: int,
    vp_stage: int,
    vp_size: int,
) -> bool:
    """Whether this is the very last stage in the VPP pipeline.

    For even vp_size the last sweep is backward, ending at rank 0.
    For odd  vp_size the last sweep is forward,  ending at rank pp_size-1.
    """
    if vp_stage != vp_size - 1:
        return False
    if vp_size % 2 == 0:
        return pp_rank == 0
    else:
        return pp_rank == pp_size - 1


def is_vpp_fold_point(
    pp_rank: int,
    pp_size: int,
    vp_stage: int,
    vp_size: int,
) -> bool:
    """Whether this rank is a fold point after the given virtual stage.

    At a fold point the same GPU continues to the next virtual stage
    without any inter-rank communication.
    """
    if vp_stage >= vp_size - 1:
        return False
    is_forward = (vp_stage % 2 == 0)
    if is_forward:
        return pp_rank == pp_size - 1
    else:
        return pp_rank == 0


@dataclass
class VPPCommInfo:
    need_recv: bool
    recv_src: int
    need_send: bool
    send_dst: int


def get_vpp_comm_info(
    pp_rank: int,
    pp_size: int,
    vp_stage: int,
    vp_size: int,
) -> VPPCommInfo:
    """Compute send/recv info for a given rank and virtual stage.

    Returns a VPPCommInfo with:
      - need_recv / recv_src: whether to recv and from which PP rank
      - need_send / send_dst: whether to send and to which PP rank
    """
    is_forward = (vp_stage % 2 == 0)

    # --- Recv logic ---
    is_first = is_vpp_first_stage(pp_rank, vp_stage)
    is_fold_from_prev = False
    if vp_stage > 0:
        prev_forward = ((vp_stage - 1) % 2 == 0)
        if prev_forward and pp_rank == pp_size - 1:
            is_fold_from_prev = True
        elif not prev_forward and pp_rank == 0:
            is_fold_from_prev = True

    need_recv = not is_first and not is_fold_from_prev
    recv_src = (pp_rank - 1) if is_forward else (pp_rank + 1)

    # --- Send logic ---
    is_last = is_vpp_last_stage(pp_rank, pp_size, vp_stage, vp_size)
    is_fold_to_next = is_vpp_fold_point(pp_rank, pp_size, vp_stage, vp_size)
    need_send = not is_last and not is_fold_to_next
    send_dst = (pp_rank + 1) if is_forward else (pp_rank - 1)

    return VPPCommInfo(
        need_recv=need_recv,
        recv_src=recv_src,
        need_send=need_send,
        send_dst=send_dst,
    )


# ---------------------------------------------------------------------------
# Model-level VPP utilities
# ---------------------------------------------------------------------------

LayerFn = Callable[..., "nn.Module"]


def make_vpp_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
    vp_size: int,
) -> tuple[list[tuple[int, int]], "nn.ModuleList"]:
    """Build a ModuleList with VPP V-shaped fold-back layer assignment.

    Layers not owned by this rank are replaced with ``PPMissingLayer``.

    Returns:
        (layer_ranges, modules) where *layer_ranges* is a list of
        ``(start, end)`` for each virtual stage on this rank.
    """
    import torch
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.models.utils import (PPMissingLayer,
                                                   maybe_offload_to_cpu)

    pp_rank = get_pp_group().rank_in_group
    pp_size = get_pp_group().world_size
    layer_ranges = get_vpp_indices(
        num_hidden_layers, pp_rank, pp_size, vp_size
    )

    local_indices: set[int] = set()
    for start, end in layer_ranges:
        local_indices.update(range(start, end))

    modules = torch.nn.ModuleList(
        [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{i}"))
            if i in local_indices
            else PPMissingLayer()
            for i in range(num_hidden_layers)
        ]
    )
    return layer_ranges, modules
