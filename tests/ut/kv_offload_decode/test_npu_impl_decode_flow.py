"""NPU integration coverage for the SFA backend and colleague manager."""

import gc
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from memfabric_hybrid import offload

import vllm_ascend.ops  # noqa: F401  # pre-existing device_op import-cycle guard
from vllm_ascend.attention import sfa_kv_offload
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_offload_decode import (
    kv_offload_decode_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.kv_offload_decode_manager import (
    KVOffloadDecodeManager,
)


KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
LAYER_NAME = "model.layers.0.self_attn.attn"


def _npu_device() -> torch.device:
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is required")
    import torch_npu

    if not torch_npu.npu.is_available():
        pytest.skip("no NPU device available")
    return torch.device("npu:0")


class _SingleRankGroup:
    def broadcast(self, tensor, src=0):
        return tensor

    def barrier(self):
        return None


def _make_manager(device: torch.device) -> KVOffloadDecodeManager:
    import torch_npu

    manager = KVOffloadDecodeManager.__new__(KVOffloadDecodeManager)
    manager.num_target_layers = 1
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.tp_group = _SingleRankGroup()
    manager.block_size = 4
    manager.topk_buffer_size = 8
    manager.topk = 3
    manager.max_num_reqs = 4
    manager.max_num_tokens = 16
    manager.max_model_len = 64
    manager.max_num_topk_rows = 4
    manager.block_table_cpu = torch.zeros(
        4, 16, dtype=torch.int32, pin_memory=True
    )
    manager.block_table_expanded_cpu = torch.empty(
        4, 16, dtype=torch.int32, pin_memory=True
    )
    manager._pending_d2h = []
    manager._build_cpp()

    config = offload.OffloadConfig()
    config.device_id = torch_npu.npu.current_device()
    config.size = 256 * 1024 * 1024
    config.world_size = 1
    config.rank_id = 0
    result = offload.initialize(config)
    if result not in (None, 0):
        pytest.fail(f"memfabric initialize failed: {result}")

    k_cpu = manager._empty_aligned_cpu_tensor(
        [8, 4, 1, KV_LORA_RANK], torch.bfloat16
    )
    v_cpu = manager._empty_aligned_cpu_tensor(
        [8, 4, 1, QK_ROPE_HEAD_DIM], torch.bfloat16
    )
    topk_k = torch.zeros(
        4, 8, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=device
    )
    topk_v = torch.zeros(
        4, 8, 1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    manager.register_kv_caches(
        {LAYER_NAME: (None, None, k_cpu, v_cpu, topk_k, topk_v)}
    )
    return manager


def _make_impl(device: torch.device) -> AscendSFAKVOffloadImpl:
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.num_kv_heads = 1
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    impl.scale = 1.0
    impl.layer_name = LAYER_NAME
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(KV_LORA_RANK, dtype=torch.bfloat16, device=device),
        variance_epsilon=1e-6,
    )
    impl._current_layer_name = None
    return impl


def _metadata(device: torch.device):
    return SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        req_ids_tensor=torch.tensor([11], dtype=torch.int64, device=device),
        token_to_req=torch.tensor([0], dtype=torch.int32, device=device),
    )


def test_decode_commit_onload_and_graph_replay(monkeypatch):
    import torch_npu

    device = _npu_device()
    manager = _make_manager(device)
    impl = _make_impl(device)
    monkeypatch.setattr(manager_module, "_KV_OFFLOAD_DECODE_MANAGER", manager)
    calls = []

    def fake_attention(
        _impl,
        ql_nope,
        _q_pe,
        kv_cache,
        topk_indices,
        _metadata,
        _query_lens,
        _key_lens,
        **kwargs,
    ):
        calls.append((kv_cache, topk_indices, kwargs["block_table"]))
        return torch.zeros(
            ql_nope.shape[0], 1, KV_LORA_RANK, device=ql_nope.device
        )

    monkeypatch.setattr(
        DeviceOperator,
        "execute_sparse_flash_attention_process",
        staticmethod(fake_attention),
    )

    graph_state = SimpleNamespace(capturing=False)
    monkeypatch.setattr(
        sfa_kv_offload, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_forward_context",
        lambda: SimpleNamespace(
            capturing=graph_state.capturing,
            cudagraph_runtime_mode=None,
        ),
    )

    metadata = _metadata(device)
    kv_no_split = torch.randn(
        1,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cos = torch.randn(
        1, 1, 1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    sin = torch.randn_like(cos)
    slots = torch.tensor([6], dtype=torch.int64, device=device)
    ql_nope = torch.randn(
        1, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        1, 1, 1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    topk_indices = torch.tensor([[[1, 3, 6]]], dtype=torch.int32, device=device)
    query_lens = torch.tensor([1], dtype=torch.int32, device=device)
    key_lens = torch.tensor([7], dtype=torch.int32, device=device)

    # Eager warmup validates compute-only K/V, D2H commit and resident H2D.
    k_pe, k_nope = impl.exec_kv(
        kv_no_split, cos, sin, (None, None, None), slots, metadata
    )
    impl._execute_sparse_flash_attention_process(
        ql_nope,
        q_pe,
        (None, None, None),
        topk_indices,
        metadata,
        query_lens,
        key_lens,
    )
    torch_npu.npu.synchronize()
    k_pool = manager.k_caches_cpu[0].view(-1, KV_LORA_RANK)
    v_pool = manager.v_caches_cpu[0].view(-1, QK_ROPE_HEAD_DIM)
    torch.testing.assert_close(k_pool[6].float(), k_nope[0, 0, 0].cpu().float())
    torch.testing.assert_close(v_pool[6].float(), k_pe[0, 0, 0].cpu().float())
    resident_k, resident_slots, _ = calls[-1]
    slot = int(resident_slots.view(-1)[2].item())
    torch.testing.assert_close(
        resident_k[0].view(1, 8, KV_LORA_RANK)[0, slot].float(),
        k_nope[0, 0, 0].float(),
    )

    # Capture the same end-to-end path, then replay with a new logical token.
    graph = torch.npu.NPUGraph()
    graph_state.capturing = True
    with torch.npu.graph(graph):
        _, captured_k_nope = impl.exec_kv(
            kv_no_split, cos, sin, (None, None, None), slots, metadata
        )
        impl._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            (None, None, None),
            topk_indices,
            metadata,
            query_lens,
            key_lens,
        )
    graph_state.capturing = False
    torch_npu.npu.synchronize()

    kv_no_split.copy_(torch.randn_like(kv_no_split))
    slots.fill_(7)
    topk_indices[0, 0, 2] = 7
    key_lens.fill_(8)
    graph.replay()
    torch_npu.npu.synchronize()

    torch.testing.assert_close(
        k_pool[7].float(), captured_k_nope[0, 0, 0].cpu().float()
    )
    resident_k, resident_slots, _ = calls[-1]
    slot = int(resident_slots.view(-1)[2].item())
    torch.testing.assert_close(
        resident_k[0].view(1, 8, KV_LORA_RANK)[0, slot].float(),
        captured_k_nope[0, 0, 0].float(),
    )

    del graph
    gc.collect()
    torch_npu.npu.synchronize()
    manager.prepare_scheduler_step()
    offload.uninitialize()
