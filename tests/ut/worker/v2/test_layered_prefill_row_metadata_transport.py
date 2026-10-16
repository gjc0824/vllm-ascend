"""Exercise the real PP metadata packing and transport control paths.

Only the network send/receive primitives are replaced by an in-process FIFO.
Production metadata pickling, tensor selection, lazy waits, and D/P unpacking
still execute.
"""

from __future__ import annotations

import ast
import pickle
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ASCEND = Path(__file__).parents[4]
CORE = ASCEND.parent / "vllm" / "vllm"
TensorMetadata = namedtuple("TensorMetadata", "device dtype size")


def load(path, namespace, *, classes=(), functions=(), methods=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            if methods is not None:
                node.bases = []
                node.decorator_list = []
                node.body = [
                    child for child in node.body if isinstance(child, ast.FunctionDef) and child.name in methods
                ]
            nodes.append(node)
    nodes.insert(
        0,
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        ),
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture
def implementation():
    ns = {
        "torch": torch,
        "dataclass": dataclass,
        "pickle": pickle,
        "TensorMetadata": TensorMetadata,
    }
    load(CORE / "sequence.py", ns, classes=("IntermediateTensors",))
    load(
        CORE / "v1/worker/gpu_worker.py",
        ns,
        classes=("AsyncIntermediateTensors",),
    )
    load(
        CORE / "distributed/parallel_state.py",
        ns,
        classes=("GroupCoordinator",),
        functions=("_split_tensor_dict",),
        methods={
            "send_object",
            "recv_object",
            "isend_tensor_dict",
            "irecv_tensor_dict",
            "_should_use_all_gather",
        },
    )
    ns["LayeredForwardOutput"] = lambda hidden, residual, final: NS(
        hidden_states=hidden,
        residual=residual,
    )
    load(
        ASCEND / "vllm_ascend/worker/v2/model_runner.py",
        ns,
        classes=("NPUModelRunner",),
        methods={
            "_combine_layered_pp_intermediate",
            "_split_layered_pp_intermediate",
            "_layered_pp_row_count",
            "_clone_intermediate_tensors",
            "_layered_pre_owner_placeholder",
        },
    )
    return NS(**ns)


class Handle:
    def __init__(self, action=lambda: None):
        self.action = action
        self.waits = 0

    def wait(self):
        self.waits += 1
        self.action()


class Wire:
    def __init__(self, monkeypatch):
        self.blocking = deque()
        self.async_queue = deque()
        self.calls = []
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "send", self.send)
        monkeypatch.setattr(torch.distributed, "recv", self.recv)
        monkeypatch.setattr(torch.distributed, "isend", self.isend)
        monkeypatch.setattr(torch.distributed, "irecv", self.irecv)

    def send(self, tensor, *, dst, group):
        self.calls.append(("send", tuple(tensor.shape)))
        self.blocking.append(tensor.clone())

    def recv(self, tensor, *, src, group):
        self.calls.append(("recv", tuple(tensor.shape)))
        tensor.copy_(self.blocking.popleft())
        return src

    def isend(self, tensor, *, dst, group):
        self.calls.append(("isend", tuple(tensor.shape)))
        self.async_queue.append(tensor.clone())
        return Handle()

    def irecv(self, tensor, *, src, group):
        self.calls.append(("irecv", tuple(tensor.shape)))
        value = self.async_queue.popleft()
        return Handle(lambda: tensor.copy_(value))


def group(impl, rank):
    value = impl.GroupCoordinator()
    value.world_size = 2
    value.rank_in_group = rank
    value.ranks = [0, 1]
    value.use_cpu_custom_send_recv = False
    value.cpu_group = "cpu"
    value.device_group = "device"
    return value


def round_trip(impl, packed, gather=None):
    sender, receiver = group(impl, 0), group(impl, 1)
    sent = sender.isend_tensor_dict(packed.tensors, all_gather_group=gather)
    values, received, callbacks = receiver.irecv_tensor_dict(all_gather_group=gather)
    result = impl.AsyncIntermediateTensors(values, received, callbacks)
    return result, sent, received


@pytest.fixture
def row_metadata_payloads(implementation):
    hidden = implementation.IntermediateTensors(
        {
            "hidden_states": torch.arange(24).reshape(6, 4).float(),
            "residual": torch.ones(6, 4),
        }
    )
    packed = implementation.NPUModelRunner._combine_layered_pp_intermediate(
        None,
        hidden,
    )
    legacy = implementation.IntermediateTensors(dict(packed.tensors))
    legacy.tensors["layered_pp_d_rows"] = torch.tensor(packed["layered_pp_d_rows"], dtype=torch.int64)
    legacy.tensors["layered_pp_p_rows"] = torch.tensor(packed["layered_pp_p_rows"], dtype=torch.int64)
    return packed, legacy


@pytest.mark.parametrize("d_rows", [0, 2, 6])
@pytest.mark.parametrize("hc", [False, True])
@pytest.mark.parametrize("gather", [False, True])
def test_int_metadata_roundtrip_with_real_serialization_and_lazy_waits(
    implementation,
    monkeypatch,
    d_rows,
    hc,
    gather,
):
    impl = implementation
    wire = Wire(monkeypatch)
    # Identical halves permit a one-lane fake all-gather; production transport
    # still performs the actual slice/reshape/callback logic.
    hidden = torch.arange(12).reshape(3, 4).repeat(2, 1).float()
    if hc:
        hidden = hidden.reshape(6, 2, 2)
    d = impl.IntermediateTensors({"hidden_states": hidden[:d_rows]})
    p = impl.IntermediateTensors({"hidden_states": hidden[d_rows:]})
    packed = impl.NPUModelRunner._combine_layered_pp_intermediate(d, p)
    assert type(packed["layered_pp_d_rows"]) is int
    assert type(packed["layered_pp_p_rows"]) is int
    tp = (
        NS(
            world_size=2,
            rank_in_group=0,
            all_gather=lambda tensor, dim: torch.cat((tensor, tensor), dim=dim),
        )
        if gather
        else None
    )
    received, sends, recvs = round_trip(impl, packed, tp)
    assert len(sends) == len(recvs) == 1
    assert not object.__getattribute__(received, "_comm_waited")

    actual_d, actual_p = impl.NPUModelRunner._split_layered_pp_intermediate(received)

    assert object.__getattribute__(received, "_comm_waited")
    assert recvs[0].waits == 1
    torch.testing.assert_close(actual_d["hidden_states"], hidden[:d_rows])
    torch.testing.assert_close(actual_p["hidden_states"], hidden[d_rows:])
    clone = impl.NPUModelRunner._clone_intermediate_tensors(received)
    assert clone["layered_pp_d_rows"] == d_rows
    assert clone["hidden_states"].data_ptr() != received["hidden_states"].data_ptr()
    assert recvs[0].waits == 1
    assert sum(name == "send" for name, _ in wire.calls) == 2
    assert sum(name == "isend" for name, _ in wire.calls) == 1
    assert not any(shape == () for _, shape in wire.calls)
    assert not wire.blocking and not wire.async_queue


def test_two_scalar_transfers_removed_and_legacy_wire_still_accepted(
    implementation,
    monkeypatch,
    row_metadata_payloads,
):
    impl = implementation
    wire = Wire(monkeypatch)
    packed, legacy = row_metadata_payloads

    received, sends, recvs = round_trip(impl, legacy)
    assert len(sends) == len(recvs) == 4
    _, legacy_p = impl.NPUModelRunner._split_layered_pp_intermediate(received)
    assert sum(name == "isend" and shape == () for name, shape in wire.calls) == 2

    wire.calls.clear()
    received, sends, recvs = round_trip(impl, packed)
    assert len(sends) == len(recvs) == 2
    _, new_p = impl.NPUModelRunner._split_layered_pp_intermediate(received)
    for name in ("hidden_states", "residual"):
        torch.testing.assert_close(new_p[name], legacy_p[name])
    assert not any(shape == () for _, shape in wire.calls)


def test_empty_placeholder_has_only_object_metadata_and_relay_preserves_ints(
    implementation,
    monkeypatch,
):
    impl = implementation
    wire = Wire(monkeypatch)
    runner = NS(model_config=NS(dtype=torch.float32), device="cpu")
    runner.layered_prefill_model_adapter = NS(
        make_transport_frontier=lambda rows, dtype, device: (
            torch.empty(rows, 2, dtype=dtype),
            None,
        ),
        to_intermediate_tensors=lambda value: impl.IntermediateTensors({"hidden_states": value.hidden_states}),
    )
    packed = impl.NPUModelRunner._layered_pre_owner_placeholder(
        runner,
        with_row_metadata=True,
    )
    received, sends, recvs = round_trip(impl, packed)
    assert not sends and not recvs
    assert received["layered_pp_d_rows"] == received["layered_pp_p_rows"] == 0

    # A relay sends the same packed object without invoking generic slicing.
    relayed, sends, recvs = round_trip(impl, received)
    assert not sends and not recvs
    d, p = impl.NPUModelRunner._split_layered_pp_intermediate(relayed)
    assert d["hidden_states"].shape == p["hidden_states"].shape == (0, 2)
    assert not any(name in ("isend", "irecv") for name, _ in wire.calls)
