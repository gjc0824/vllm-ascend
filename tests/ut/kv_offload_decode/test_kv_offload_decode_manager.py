from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_offload_decode import (
    kv_offload_decode_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.kv_offload_decode_manager import (
    KVOffloadDecodeManager,
)


K_DIM = 2
V_DIM = 1
LAYER_NAME = "model.layers.0.self_attn.attn"


def _make_d2h_manager(*, tp_rank: int = 0, max_tokens: int = 4):
    manager = KVOffloadDecodeManager.__new__(KVOffloadDecodeManager)
    manager.tp_rank = tp_rank
    manager.max_num_tokens = max_tokens
    manager.token_size_bytes_k = K_DIM * torch.float32.itemsize
    manager.token_size_bytes_v = V_DIM * torch.float32.itemsize
    manager.d2h_src_ptrs_npu = torch.empty(max_tokens * 2, dtype=torch.int64)
    manager.d2h_dst_ptrs_npu = torch.empty(max_tokens * 2, dtype=torch.int64)
    manager.d2h_lengths_npu = torch.empty(max_tokens * 2, dtype=torch.int32)
    manager.d2h_size_npu = torch.empty(1, dtype=torch.int32)
    manager.d2h_token_indices_npu = torch.arange(max_tokens, dtype=torch.int64)
    manager._pending_d2h = []
    return manager


def _cpu_caches():
    return torch.zeros(8, K_DIM), torch.zeros(8, V_DIM)


def _capture_sparse_copy(monkeypatch):
    calls = []

    def fake_sparse_copy(src, dst, lengths, size, device):
        count = int(size[0].item())
        calls.append(
            (
                src[:count].clone(),
                dst[:count].clone(),
                lengths[:count].clone(),
                count,
                device,
            )
        )
        return 0

    monkeypatch.setattr(manager_module.offload, "sparse_copy", fake_sparse_copy)
    return calls


def test_decode_offload_builds_d2h_descriptors_and_masks_padding(monkeypatch):
    manager = _make_d2h_manager()
    k_cache_cpu, v_cache_cpu = _cpu_caches()
    k = torch.arange(4, dtype=torch.float32).view(2, 1, 1, K_DIM)
    v = torch.arange(2, dtype=torch.float32).view(2, 1, 1, V_DIM)
    calls = _capture_sparse_copy(monkeypatch)

    manager.offload_new_kv(
        torch.tensor([3, -1]),
        k_cache_cpu,
        v_cache_cpu,
        None,
        None,
        k,
        v,
        capturing=True,
    )

    src, dst, lengths, count, device = calls[0]
    assert count == 4
    assert device.type == "cpu"
    assert src.tolist() == [
        k.data_ptr(),
        k.data_ptr() + manager.token_size_bytes_k,
        v.data_ptr(),
        v.data_ptr() + manager.token_size_bytes_v,
    ]
    assert dst.tolist() == [
        k_cache_cpu.data_ptr() + 3 * manager.token_size_bytes_k,
        k_cache_cpu.data_ptr(),
        v_cache_cpu.data_ptr() + 3 * manager.token_size_bytes_v,
        v_cache_cpu.data_ptr(),
    ]
    assert lengths.tolist() == [manager.token_size_bytes_k, 0, manager.token_size_bytes_v, 0]


def test_prefill_offload_reads_paged_cache_rows_by_slot(monkeypatch):
    manager = _make_d2h_manager()
    k_cache_cpu, v_cache_cpu = _cpu_caches()
    k_cache_npu = torch.zeros_like(k_cache_cpu)
    v_cache_npu = torch.zeros_like(v_cache_cpu)
    calls = _capture_sparse_copy(monkeypatch)

    manager.offload_new_kv(
        torch.tensor([2, 5]),
        k_cache_cpu,
        v_cache_cpu,
        k_cache_npu,
        v_cache_npu,
        None,
        None,
        has_prefill=True,
        capturing=True,
    )

    src, dst, lengths, count, _ = calls[0]
    assert count == 4
    assert src.tolist() == [
        k_cache_npu.data_ptr() + 2 * manager.token_size_bytes_k,
        k_cache_npu.data_ptr() + 5 * manager.token_size_bytes_k,
        v_cache_npu.data_ptr() + 2 * manager.token_size_bytes_v,
        v_cache_npu.data_ptr() + 5 * manager.token_size_bytes_v,
    ]
    assert dst.tolist() == [
        k_cache_cpu.data_ptr() + 2 * manager.token_size_bytes_k,
        k_cache_cpu.data_ptr() + 5 * manager.token_size_bytes_k,
        v_cache_cpu.data_ptr() + 2 * manager.token_size_bytes_v,
        v_cache_cpu.data_ptr() + 5 * manager.token_size_bytes_v,
    ]
    assert lengths.tolist() == [
        manager.token_size_bytes_k,
        manager.token_size_bytes_k,
        manager.token_size_bytes_v,
        manager.token_size_bytes_v,
    ]


def test_non_owner_tp_rank_does_not_submit_d2h(monkeypatch):
    manager = _make_d2h_manager(tp_rank=1)
    calls = _capture_sparse_copy(monkeypatch)

    manager.offload_new_kv(
        torch.tensor([0]),
        None,
        None,
        None,
        None,
        torch.zeros(1, 1, 1, K_DIM),
        torch.zeros(1, 1, 1, V_DIM),
        capturing=True,
    )

    assert calls == []


def test_decode_rows_must_match_slot_mapping(monkeypatch):
    manager = _make_d2h_manager()
    k_cache_cpu, v_cache_cpu = _cpu_caches()
    _capture_sparse_copy(monkeypatch)

    with pytest.raises(ValueError, match="row counts must match"):
        manager.offload_new_kv(
            torch.tensor([0, 1]),
            k_cache_cpu,
            v_cache_cpu,
            None,
            None,
            torch.zeros(1, 1, 1, K_DIM),
            torch.zeros(1, 1, 1, V_DIM),
            capturing=True,
        )


def test_scheduler_step_waits_for_eager_d2h(monkeypatch):
    manager = _make_d2h_manager()
    k_cache_cpu, v_cache_cpu = _cpu_caches()
    _capture_sparse_copy(monkeypatch)
    events = []

    class FakeEvent:
        def record(self, stream):
            events.append(("record", stream))

        def synchronize(self):
            events.append(("synchronize", None))

    monkeypatch.setattr(manager_module.torch_npu.npu, "Event", FakeEvent)
    monkeypatch.setattr(manager_module.torch_npu.npu, "current_stream", lambda: "stream")

    manager.offload_new_kv(
        torch.tensor([0]),
        k_cache_cpu,
        v_cache_cpu,
        None,
        None,
        torch.zeros(1, 1, 1, K_DIM),
        torch.zeros(1, 1, 1, V_DIM),
    )
    assert len(manager._pending_d2h) == 1

    manager.prepare_scheduler_step()

    assert events == [("record", "stream"), ("synchronize", None)]
    assert manager._pending_d2h == []


def test_onload_cpu_callback_keeps_colleague_argument_contract():
    manager = KVOffloadDecodeManager.__new__(KVOffloadDecodeManager)
    manager.topk = 2
    manager.topk_buffer_size = 4
    manager.max_model_len = 16
    manager.lru_workspace_threads = 1
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(barrier_calls=0)
    manager.tp_group.barrier = lambda: setattr(
        manager.tp_group, "barrier_calls", manager.tp_group.barrier_calls + 1
    )
    calls = []
    manager.kv_offload_decode_cpp = SimpleNamespace(
        lru_resident_compact=lambda *args: calls.append(("lru", args)),
        compute_lru_resident_addrs=lambda *args: calls.append(("addr", args)),
    )
    miss_count = torch.zeros(1, dtype=torch.int32)
    miss_tokens = torch.zeros(1, 2, dtype=torch.int32)
    miss_slots = torch.zeros(1, 2, dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    descriptors_i64 = torch.zeros(4, dtype=torch.int64)
    descriptors_i32 = torch.zeros(4, dtype=torch.int32)
    args = (
        1,
        miss_count,
        miss_tokens,
        miss_slots,
        *range(1, 10),
        block_table,
        4,
        8,
        4,
        100,
        200,
        300,
        400,
        *range(10, 15),
        descriptors_i64,
        descriptors_i64,
        descriptors_i32,
        torch.zeros(1, dtype=torch.int32),
        0,
    )

    manager._onload_topk_kv_cpu(args)

    assert [name for name, _ in calls] == ["lru", "addr"]
    assert manager.tp_group.barrier_calls == 1


def test_get_offload_layer_id_rejects_unknown_layer():
    manager = KVOffloadDecodeManager.__new__(KVOffloadDecodeManager)
    manager.offload_layer_names = [LAYER_NAME]
    manager.layer_name_to_offload_id = {LAYER_NAME: 0}

    assert manager._get_offload_layer_id(LAYER_NAME) == 0
    with pytest.raises(KeyError, match="not registered"):
        manager._get_offload_layer_id("model.layers.1.self_attn.attn")
