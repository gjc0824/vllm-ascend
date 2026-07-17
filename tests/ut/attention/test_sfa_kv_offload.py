from dataclasses import fields
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.ops  # noqa: F401  # pre-existing circular-import guard for device_op
from vllm_ascend.attention import sfa_kv_offload, sfa_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_kv_offload import (
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)
from vllm_ascend.attention.utils import build_valid_topk_mask
from vllm_ascend.device.device_op import DeviceOperator


def test_backend_selection_returns_offload_classes(monkeypatch):
    monkeypatch.setattr(sfa_v1, "kv_offload_decode_enabled", lambda: True)
    assert AscendSFABackend.get_impl_cls() is AscendSFAKVOffloadImpl
    assert AscendSFABackend.get_builder_cls() is AscendSFAKVOffloadMetadataBuilder


def test_backend_selection_default_unchanged(monkeypatch):
    monkeypatch.setattr(sfa_v1, "kv_offload_decode_enabled", lambda: False)
    monkeypatch.setattr(sfa_v1, "enable_cp", lambda: False)
    monkeypatch.setattr(sfa_v1, "enable_sfa_dcp_replicated_indexer", lambda: False)
    assert AscendSFABackend.get_impl_cls() is AscendSFAImpl
    assert AscendSFABackend.get_builder_cls() is AscendSFAMetadataBuilder


def test_sfa_metadata_carries_offload_request_fields():
    names = {field.name for field in fields(AscendSFAMetadata)}
    assert "req_ids_tensor" in names
    assert "token_to_req" in names


def test_valid_topk_mask_drops_unwritten_tail_slots():
    topk_indices = torch.tensor(
        [[126, 127, 128, -1], [128, 129, 255, 256]],
        dtype=torch.int32,
    )
    seq_len_thresholds = torch.tensor([[128], [130]], dtype=torch.int32)

    mask = build_valid_topk_mask(topk_indices, seq_len_thresholds)

    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
    ]


def test_offload_builder_populates_drafting_metadata(monkeypatch):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(
        AscendSFAKVOffloadMetadataBuilder
    )
    builder.decode_threshold = 4
    metadata = SimpleNamespace()
    req_ids_tensor = torch.tensor([11, 22], dtype=torch.int64)
    token_to_req = torch.tensor([0, 1, 1], dtype=torch.int32)
    common_metadata = SimpleNamespace(
        req_ids_tensor=req_ids_tensor,
        token_to_req=token_to_req,
    )
    monkeypatch.setattr(
        AscendSFAMetadataBuilder,
        "build_for_drafting",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (2, 0, 3, 0),
    )

    result = builder.build_for_drafting(common_metadata, draft_index=1)

    assert result is metadata
    assert result.num_decodes == 2
    assert result.num_prefills == 0
    assert result.num_decode_tokens == 3
    assert result.req_ids_tensor is req_ids_tensor
    assert result.token_to_req is token_to_req


def _make_impl_for_compose() -> AscendSFAKVOffloadImpl:
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.has_indexer = True
    impl.use_sparse_c8_indexer = False
    indexer_k = torch.zeros(2, 4, 1, 8)
    impl.indexer = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=(indexer_k,)))
    impl._indexer_k = indexer_k
    return impl


def test_compose_sfa_kv_cache_unpacks_offload_six_tuple():
    impl = _make_impl_for_compose()
    k_npu = torch.zeros(2, 4, 1, 8)
    v_npu = torch.zeros(2, 4, 1, 4)
    k_cpu = torch.zeros(2, 4, 1, 8)
    v_cpu = torch.zeros(2, 4, 1, 4)
    topk_k = torch.zeros(1, 8, 1, 8)
    topk_v = torch.zeros(1, 8, 1, 4)

    composed = impl._compose_sfa_kv_cache((k_npu, v_npu, k_cpu, v_cpu, topk_k, topk_v))

    assert len(composed) == 3
    assert composed[0] is k_npu
    assert composed[1] is v_npu
    assert composed[2] is impl._indexer_k


def test_compose_sfa_kv_cache_keeps_standard_tuple():
    impl = _make_impl_for_compose()
    k_npu = torch.zeros(2, 4, 1, 8)
    v_npu = torch.zeros(2, 4, 1, 4)

    composed = impl._compose_sfa_kv_cache((k_npu, v_npu))

    assert len(composed) == 3
    assert composed[0] is k_npu
    assert composed[1] is v_npu


def test_decode_exec_kv_commits_immediately(monkeypatch):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl._current_layer_name = None
    k_nope = torch.ones(2, 1, 1, 4)
    k_pe = torch.ones(2, 1, 1, 2)
    impl._compute_kv_only = lambda *_args: (k_nope, k_pe)
    calls = []
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _name: 0,
        tp_rank=0,
        k_caches_cpu=[torch.empty(0)],
        v_caches_cpu=[torch.empty(0)],
        offload_new_kv=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_kv_offload_decode_manager",
        lambda: manager,
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=2,
        num_prefills=0,
    )
    slots = torch.tensor([3, 7])

    result = impl.exec_kv(
        torch.empty(2, 6),
        torch.empty(0),
        torch.empty(0),
        (None, None, None),
        slots,
        metadata,
    )

    assert result[0] is k_pe
    assert result[1] is k_nope
    assert len(calls) == 1
    assert calls[0]["slot_mapping"] is slots
    assert calls[0]["k"] is k_nope
    assert calls[0]["v"] is k_pe
    assert calls[0]["has_prefill"] is False
    assert calls[0]["capturing"] is False


def test_mtp_exec_kv_uses_cpu_only_decode_path(monkeypatch):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl._current_layer_name = None
    k_nope = torch.ones(3, 1, 1, 4)
    k_pe = torch.ones(3, 1, 1, 2)
    impl._compute_kv_only = lambda *_args: (k_nope, k_pe)
    calls = []
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _name: 0,
        tp_rank=0,
        k_caches_cpu=[torch.empty(0)],
        v_caches_cpu=[torch.empty(0)],
        offload_new_kv=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_kv_offload_decode_manager",
        lambda: manager,
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        num_decodes=2,
        num_prefills=0,
    )
    slots = torch.tensor([3, 4, 9])

    result = impl.exec_kv(
        torch.empty(3, 6),
        torch.empty(0),
        torch.empty(0),
        (None, None, None),
        slots,
        metadata,
    )

    assert result[0] is k_pe
    assert result[1] is k_nope
    assert calls[0]["slot_mapping"] is slots
    assert calls[0]["k"] is k_nope
    assert calls[0]["v"] is k_pe
    assert calls[0]["has_prefill"] is False


def test_decode_exec_kv_marks_full_graph_runtime(monkeypatch):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl._current_layer_name = None
    impl._compute_kv_only = lambda *_args: (
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 2),
    )
    calls = []
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _name: 0,
        tp_rank=0,
        k_caches_cpu=[torch.empty(0)],
        v_caches_cpu=[torch.empty(0)],
        offload_new_kv=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_kv_offload_decode_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_forward_context",
        lambda: SimpleNamespace(
            capturing=False,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        ),
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=1,
        num_prefills=0,
    )

    impl.exec_kv(
        torch.empty(1, 6),
        torch.empty(0),
        torch.empty(0),
        (None, None, None),
        torch.tensor([3]),
        metadata,
    )

    assert calls[0]["capturing"] is True


def test_decode_onload_receives_full_graph_runtime_state(monkeypatch):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl._current_layer_name = None
    onload_calls = []
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _name: 0,
        topk_buffer_size=4,
        block_size=2,
        topk_buffers_k=[torch.empty(4, 4, 1, 4)],
        topk_buffers_v=[torch.empty(4, 4, 1, 2)],
        current_slots_npu=torch.zeros(4, 3, dtype=torch.int32),
        resident_block_table_npu=torch.arange(8, dtype=torch.int32).view(4, 2),
        resident_query_lens_npu=torch.arange(1, 5, dtype=torch.int32),
        resident_seq_lens_npu=torch.full((4,), 4, dtype=torch.int32),
        onload_topk_kv=lambda *args, **kwargs: onload_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_kv_offload_decode_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_forward_context",
        lambda: SimpleNamespace(
            capturing=False,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        ),
    )
    monkeypatch.setattr(
        DeviceOperator,
        "execute_sparse_flash_attention_process",
        staticmethod(lambda *_args, **_kwargs: torch.empty(1, 1, 4)),
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        req_ids_tensor=torch.ones(1, dtype=torch.int64),
        token_to_req=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
    )

    impl._execute_sparse_flash_attention_process(
        torch.empty(4, 1, 4),
        torch.empty(4, 1, 1, 2),
        (None, None, None),
        torch.zeros(1, 1, 3, dtype=torch.int32),
        metadata,
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )

    assert onload_calls[0][1]["capturing"] is True


def test_mtp_onload_expands_rows_and_masks_tail_topk(monkeypatch):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl._current_layer_name = None
    onload_calls = []
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _name: 0,
        topk_buffer_size=4,
        block_size=2,
        topk_buffers_k=[torch.empty(3, 4, 1, 4)],
        topk_buffers_v=[torch.empty(3, 4, 1, 2)],
        current_slots_npu=torch.zeros(3, 3, dtype=torch.int32),
        resident_block_table_npu=torch.arange(6, dtype=torch.int32).view(3, 2),
        resident_query_lens_npu=torch.arange(1, 4, dtype=torch.int32),
        resident_seq_lens_npu=torch.full((3,), 4, dtype=torch.int32),
        onload_topk_kv=lambda *args, **kwargs: onload_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sfa_kv_offload,
        "get_kv_offload_decode_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        DeviceOperator,
        "execute_sparse_flash_attention_process",
        staticmethod(lambda *_args, **_kwargs: torch.ones(3, 1, 4)),
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=3,
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        req_ids_tensor=torch.tensor([11, 22], dtype=torch.int64),
        token_to_req=torch.tensor([0, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
    )
    topk_indices = torch.tensor(
        [
            [[0, 2, 3]],
            [[1, 4, 5]],
            [[-1, 3, 6]],
        ],
        dtype=torch.int32,
    )

    output = impl._execute_sparse_flash_attention_process(
        torch.empty(5, 1, 4),
        torch.empty(5, 1, 1, 2),
        (None, None, None),
        topk_indices,
        metadata,
        torch.tensor([1, 3], dtype=torch.int32),
        metadata.seq_lens,
    )

    assert torch.equal(onload_calls[0][0][7], metadata.token_to_req)
    assert onload_calls[0][0][4].tolist() == [
        [0, 2, -1],
        [1, 4, -1],
        [-1, 3, -1],
    ]
    assert onload_calls[0][0][6].tolist() == [11, 22, 22]
    assert output.shape[0] == 5
    assert torch.count_nonzero(output[3:]) == 0
