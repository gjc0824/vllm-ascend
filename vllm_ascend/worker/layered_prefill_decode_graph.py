# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exact-shape graphs for the inactive-P (Decode-only) layer groups.

These record the ordinary eager operators, not attention's full-model graph
update handles. Mutable inputs/metadata have stable, private storage. Shapes
and every host-side scalar are in the key; changing host metadata falls back
to another entry rather than replaying a stale attention contract.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.offloader.base import get_offloader

MAX_DECODE_GRAPH_ENTRIES = 512
MAX_DECODE_GRAPH_VIEWS = 64


def _members(value):
    if is_dataclass(value) and not isinstance(value, type):
        return ((f.name, getattr(value, f.name)) for f in fields(value))
    if hasattr(value, "__dict__") and not callable(value):
        return vars(value).items()
    raise TypeError(f"Unsupported Decode graph metadata: {type(value).__name__}")


def graph_signature(value, memo=None):
    """Pure host signature, including aliases, without reading device values."""
    if memo is None:
        memo = {}
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return value
    if id(value) in memo:
        return ("ref", memo[id(value)])
    memo[id(value)] = len(memo)
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            value.device.type,
            repr(value.tolist()) if value.device.type == "cpu" else None,
        )
    if isinstance(value, dict):
        return tuple((k, graph_signature(v, memo)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(graph_signature(v, memo) for v in value))
    return (
        type(value).__module__,
        type(value).__qualname__,
        tuple((k, graph_signature(v, memo)) for k, v in _members(value)),
    )


def snapshot_tree(value, memo=None, *, clone_tensors=True):
    """Copy wrappers; clone activations or retain runner-owned metadata storage."""
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone() if clone_tensors else value
    elif isinstance(value, dict):
        result = {k: snapshot_tree(v, memo, clone_tensors=clone_tensors) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        items = [snapshot_tree(v, memo, clone_tensors=clone_tensors) for v in value]
        result = tuple(items) if isinstance(value, tuple) else items
    elif value is None or isinstance(value, (str, int, float, bool, Enum)):
        result = value
    else:
        result = copy(value)
        for name, child in _members(value):
            object.__setattr__(result, name, snapshot_tree(child, memo, clone_tensors=clone_tensors))
    memo[id(value)] = result
    return result


def refresh_tree(target, source, seen=None):
    """Refresh data only; all captured addresses and scalar contracts stay fixed."""
    if seen is None:
        seen = set()
    if id(target) in seen:
        return
    seen.add(id(target))
    if isinstance(source, torch.Tensor):
        if target.data_ptr() != source.data_ptr():
            target.copy_(source)
    elif isinstance(source, dict):
        for name, value in source.items():
            refresh_tree(target[name], value, seen)
    elif isinstance(source, (tuple, list)):
        for dst, src in zip(target, source):
            refresh_tree(dst, src, seen)
    elif source is not None and not isinstance(source, (str, int, float, bool, Enum)):
        for name, child in _members(source):
            refresh_tree(getattr(target, name), child, seen)


@dataclass
class DecodeGraphView:
    metadata: Any
    source: Any


@dataclass
class DecodeGraphEntry:
    inputs: Any
    metadata: Any
    graph: Any
    output: Any


class LayeredDecodeGraphCache:
    """Bounded runner-local cache; DP/PP remain on the existing fallback path."""

    def __init__(self, runner):
        self.runner = runner
        self.entries: dict[Any, DecodeGraphEntry] = {}
        self.views: dict[Any, DecodeGraphView] = {}
        self.pool = torch.npu.graph_pool_handle()
        self.stream = torch.npu.Stream(device=runner.device)

    def forward(self, start, end, state, input_ids, positions):
        runner = self.runner
        context = get_forward_context()
        adapter = runner.layered_prefill_model_adapter
        inputs = (state, input_ids, positions)
        metadata = context.attn_metadata

        def run(values):
            active, ids, pos = values
            runner._set_layered_prefill_moe_layer_offset(start)
            for idx in range(start, end):
                active = adapter._forward_layer(adapter.layers[idx], pos, *active, ids)
            return active

        try:
            metadata_key = graph_signature(metadata)
            signature = (graph_signature(inputs), metadata_key)
        except TypeError as exc:
            logger.warning_once("Layered Decode graph fallback: %s", exc)
            return run(inputs)
        key = (start, end, str(context.moe_comm_type), signature)
        entry = self.entries.get(key)
        view = self.views.get(metadata_key)
        if entry is None and (
            len(self.entries) >= MAX_DECODE_GRAPH_ENTRIES
            or (view is None and len(self.views) >= MAX_DECODE_GRAPH_VIEWS)
        ):
            logger.warning_once("Layered Decode graph cache is full; using eager execution for new shapes")
            return run(inputs)

        if view is None:
            # Metadata comes from runner-owned, separate Decode builders.
            # Retain their tensor storage (including large immutable tables)
            # rather than duplicating it for every Decode shape. Strong refs
            # keep ephemeral selected rows alive; refresh copies only inputs
            # whose addresses changed. Python wrappers stay private to graphs.
            view = DecodeGraphView(snapshot_tree(metadata, clone_tensors=False), metadata)
            self.views[metadata_key] = view
        elif view.source is not metadata:
            refresh_tree(view.metadata, metadata)
            view.source = metadata

        if entry is None:
            static_inputs = snapshot_tree(inputs)
            static_metadata = view.metadata
            context.attn_metadata = static_metadata
            was_capturing = context.capturing
            try:
                current = torch.npu.current_stream()
                self.stream.wait_stream(current)
                graph = torch.npu.NPUGraph()
                # The operators have already been initialized by runner warmup.
                # Capture only records work: deliberately no warmup forward on
                # live KV/state caches and exactly one replay for this step.
                get_offloader().sync_prev_onload()
                context.capturing = True
                with torch.npu.graph(graph, pool=self.pool, stream=self.stream):
                    output = run(static_inputs)
                    get_offloader().join_after_forward()
                current.wait_stream(self.stream)
                entry = DecodeGraphEntry(static_inputs, static_metadata, graph, output)
                self.entries[key] = entry
                if runner.tp_rank == 0:
                    logger.info(
                        "Layered Decode graph captured layers=[%d,%d) tokens=%d entries=%d moe=%s",
                        start,
                        end,
                        context.num_tokens,
                        len(self.entries),
                        context.moe_comm_type,
                    )
            finally:
                context.capturing = was_capturing
                context.attn_metadata = metadata
        else:
            refresh_tree(entry.inputs, inputs)
        entry.graph.replay()
        runner._set_layered_prefill_moe_layer_offset(end)
        return entry.output
