"""NPU contract tests for memfabric D2H sparse copy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pytest.skip("torch_npu is not installed", allow_module_level=True)

try:
    from memfabric_hybrid import offload
except (ImportError, AttributeError):
    pytest.skip("memfabric_hybrid offload API is not installed", allow_module_level=True)

if not all(hasattr(offload, name) for name in ("initialize", "empty", "sparse_copy", "uninitialize")):
    pytest.skip("memfabric_hybrid offload API is incomplete", allow_module_level=True)


HOST_POOL_BYTES = 256 * 1024 * 1024


@pytest.fixture(scope="module")
def npu_device():
    available = torch.npu.is_available()
    device_count = torch.npu.device_count()
    if not isinstance(available, bool) or not isinstance(device_count, int):
        pytest.skip("NPU runtime is mocked")
    if not available or device_count < 1:
        pytest.skip("NPU is not available")

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    assert offload.initialize(device.index, HOST_POOL_BYTES) == 0
    try:
        yield device
    finally:
        torch.npu.synchronize()
        offload.uninitialize()


def _run_scatter(device: torch.device, num_tokens: int) -> None:
    k_bytes = 32
    v_bytes = 48
    k_source = torch.arange((num_tokens * k_bytes) // 4, dtype=torch.int32, device=device)
    v_source = torch.arange(
        1000,
        1000 + (num_tokens * v_bytes) // 4,
        dtype=torch.int32,
        device=device,
    )
    destination = offload.empty(
        [32 + num_tokens * (k_bytes + v_bytes) + 32],
        dtype=torch.uint8,
        pin_memory=True,
    )
    destination.zero_()
    destination_base = destination.data_ptr()

    source_ptrs = []
    destination_ptrs = []
    lengths = []
    for token_idx in range(num_tokens):
        source_ptrs.append(k_source.data_ptr() + token_idx * k_bytes)
        destination_ptrs.append(destination_base + 32 + token_idx * (k_bytes + v_bytes))
        lengths.append(k_bytes)
    for token_idx in range(num_tokens):
        source_ptrs.append(v_source.data_ptr() + token_idx * v_bytes)
        destination_ptrs.append(
            destination_base + 32 + token_idx * (k_bytes + v_bytes) + k_bytes
        )
        lengths.append(v_bytes)

    ret = offload.sparse_copy(
        torch.tensor(source_ptrs, dtype=torch.int64, device=device),
        torch.tensor(destination_ptrs, dtype=torch.int64, device=device),
        torch.tensor(lengths, dtype=torch.int32, device=device),
        torch.tensor([2 * num_tokens], dtype=torch.int32, device=device),
        device,
    )
    torch.npu.synchronize()

    expected = torch.zeros_like(destination)
    k_host = k_source.cpu().view(torch.uint8)
    v_host = v_source.cpu().view(torch.uint8)
    for token_idx in range(num_tokens):
        destination_offset = 32 + token_idx * (k_bytes + v_bytes)
        expected[destination_offset : destination_offset + k_bytes] = k_host[
            token_idx * k_bytes : (token_idx + 1) * k_bytes
        ]
        expected[destination_offset + k_bytes : destination_offset + k_bytes + v_bytes] = v_host[
            token_idx * v_bytes : (token_idx + 1) * v_bytes
        ]

    assert ret == 0
    assert torch.equal(destination, expected)


@pytest.mark.parametrize("num_tokens", [1, 2, 31, 32, 33])
def test_sparse_copy_d2h_scatter_sizes(npu_device: torch.device, num_tokens: int) -> None:
    _run_scatter(npu_device, num_tokens)


def test_sparse_copy_d2h_bfloat16_payload(npu_device: torch.device) -> None:
    num_tokens = 2
    k_dim = 8
    v_dim = 12
    k_source = torch.arange(num_tokens * k_dim, dtype=torch.bfloat16, device=npu_device)
    v_source = torch.arange(100, 100 + num_tokens * v_dim, dtype=torch.bfloat16, device=npu_device)
    k_bytes = k_dim * 2
    v_bytes = v_dim * 2
    destination = offload.empty([256], dtype=torch.uint8, pin_memory=True)
    destination.zero_()
    base = destination.data_ptr()

    ret = offload.sparse_copy(
        torch.tensor(
            [
                k_source.data_ptr(),
                k_source.data_ptr() + k_bytes,
                v_source.data_ptr(),
                v_source.data_ptr() + v_bytes,
            ],
            dtype=torch.int64,
            device=npu_device,
        ),
        torch.tensor(
            [base + 16, base + 64, base + 112, base + 160],
            dtype=torch.int64,
            device=npu_device,
        ),
        torch.tensor([k_bytes, k_bytes, v_bytes, v_bytes], dtype=torch.int32, device=npu_device),
        torch.tensor([2 * num_tokens], dtype=torch.int32, device=npu_device),
        npu_device,
    )
    torch.npu.synchronize()

    expected = torch.zeros_like(destination)
    k_host = k_source.cpu().view(torch.uint8)
    v_host = v_source.cpu().view(torch.uint8)
    expected[16 : 16 + k_bytes] = k_host[:k_bytes]
    expected[64 : 64 + k_bytes] = k_host[k_bytes : 2 * k_bytes]
    expected[112 : 112 + v_bytes] = v_host[:v_bytes]
    expected[160 : 160 + v_bytes] = v_host[v_bytes : 2 * v_bytes]

    assert ret == 0
    assert torch.equal(destination, expected)


def test_sparse_copy_d2h_preserves_current_stream_order(npu_device: torch.device) -> None:
    host_k = torch.arange(8, dtype=torch.int32)
    host_v = torch.arange(100, 108, dtype=torch.int32)
    k_source = torch.empty_like(host_k, device=npu_device)
    v_source = torch.empty_like(host_v, device=npu_device)
    k_source.copy_(host_k)
    v_source.copy_(host_v)

    destination = offload.empty([128], dtype=torch.uint8, pin_memory=True)
    destination.zero_()
    base = destination.data_ptr()
    token_bytes = host_k.numel() * host_k.element_size()
    ret = offload.sparse_copy(
        torch.tensor(
            [k_source.data_ptr(), v_source.data_ptr()],
            dtype=torch.int64,
            device=npu_device,
        ),
        torch.tensor([base + 16, base + 64], dtype=torch.int64, device=npu_device),
        torch.tensor([token_bytes, token_bytes], dtype=torch.int32, device=npu_device),
        torch.tensor([2], dtype=torch.int32, device=npu_device),
        npu_device,
    )
    torch.npu.synchronize()

    expected = torch.zeros_like(destination)
    expected[16 : 16 + token_bytes] = host_k.view(torch.uint8)
    expected[64 : 64 + token_bytes] = host_v.view(torch.uint8)
    assert ret == 0
    assert torch.equal(destination, expected)


def test_sfa_d2h_descriptor_builder(npu_device: torch.device) -> None:
    from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (
        SFAKVOffloadWorker,
    )

    worker = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    worker.max_num_topk_rows = 4
    worker.block_size = 2
    worker.token_size_bytes_k = 16
    worker.token_size_bytes_v = 24
    worker.cpu_block_table = SimpleNamespace(
        gpu=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=npu_device)
    )
    worker.k_caches_cpu = [offload.empty([8, 2, 1, 8], dtype=torch.bfloat16, pin_memory=True)]
    worker.v_caches_cpu = [offload.empty([8, 2, 1, 12], dtype=torch.bfloat16, pin_memory=True)]
    worker.k_caches_cpu[0].zero_()
    worker.v_caches_cpu[0].zero_()
    worker.d2h_src_ptrs_npu = torch.empty([8], dtype=torch.int64, device=npu_device)
    worker.d2h_dst_ptrs_npu = torch.empty([8], dtype=torch.int64, device=npu_device)
    worker.d2h_lengths_npu = torch.empty([8], dtype=torch.int32, device=npu_device)
    worker.d2h_size_npu = torch.zeros([1], dtype=torch.int32, device=npu_device)

    key_cache = torch.arange(4 * 2 * 8, dtype=torch.bfloat16, device=npu_device).reshape(4, 2, 1, 8)
    value_cache = torch.arange(
        100,
        100 + 4 * 2 * 12,
        dtype=torch.bfloat16,
        device=npu_device,
    ).reshape(4, 2, 1, 12)
    worker._build_d2h_descriptors(
        0,
        2,
        key_cache,
        value_cache,
        torch.tensor([0, 3, -1], dtype=torch.int32, device=npu_device),
        torch.tensor([0, 3, 1], dtype=torch.int64, device=npu_device),
        torch.tensor([0, 1, 0], dtype=torch.int32, device=npu_device),
    )
    torch.npu.synchronize()

    assert worker.d2h_size_npu.cpu().item() == 6
    assert worker.d2h_lengths_npu[:6].cpu().tolist() == [16, 16, 0, 24, 24, 0]
    key_base = worker.k_caches_cpu[0].data_ptr()
    value_base = worker.v_caches_cpu[0].data_ptr()
    assert worker.d2h_dst_ptrs_npu[:3].cpu().tolist() == [key_base + 2 * 16, key_base + 9 * 16, key_base]
    assert worker.d2h_dst_ptrs_npu[3:6].cpu().tolist() == [value_base + 2 * 24, value_base + 9 * 24, value_base]

    assert offload.sparse_copy(
        worker.d2h_src_ptrs_npu,
        worker.d2h_dst_ptrs_npu,
        worker.d2h_lengths_npu,
        worker.d2h_size_npu,
        npu_device,
    ) == 0
    torch.npu.synchronize()
    key_cache_flat = key_cache.view(-1, key_cache.shape[-1])
    value_cache_flat = value_cache.view(-1, value_cache.shape[-1])
    assert torch.equal(worker.k_caches_cpu[0][1, 0, 0], key_cache_flat[0].cpu())
    assert torch.equal(worker.k_caches_cpu[0][4, 1, 0], key_cache_flat[3].cpu())
    assert torch.equal(worker.v_caches_cpu[0][1, 0, 0], value_cache_flat[0].cpu())
    assert torch.equal(worker.v_caches_cpu[0][4, 1, 0], value_cache_flat[3].cpu())

    worker._build_d2h_descriptors(
        0,
        2,
        key_cache,
        value_cache,
        torch.tensor([0, 3, -1], dtype=torch.int32, device=npu_device),
        torch.tensor([0, 3, 1], dtype=torch.int64, device=npu_device),
        None,
    )
    torch.npu.synchronize()
    assert worker.d2h_lengths_npu[:6].cpu().tolist() == [16, 16, 0, 24, 24, 0]

    worker._build_d2h_descriptors(
        0,
        2,
        key_cache,
        value_cache,
        torch.tensor([0], dtype=torch.int32, device=npu_device),
        torch.tensor([4], dtype=torch.int64, device=npu_device),
        torch.tensor([0], dtype=torch.int32, device=npu_device),
    )
    torch.npu.synchronize()
    assert worker.d2h_size_npu.cpu().item() == 2
    assert worker.d2h_lengths_npu[:2].cpu().tolist() == [0, 0]


def test_sparse_copy_d2h_zero_lengths_and_empty_batch(npu_device: torch.device) -> None:
    num_tokens = 3
    k_bytes = 32
    v_bytes = 48
    k_source = torch.arange((num_tokens * k_bytes) // 4, dtype=torch.int32, device=npu_device)
    v_source = torch.arange(1000, 1000 + (num_tokens * v_bytes) // 4, dtype=torch.int32, device=npu_device)
    destination = offload.empty([512], dtype=torch.uint8, pin_memory=True)
    destination.zero_()
    base = destination.data_ptr()

    ret = offload.sparse_copy(
        torch.tensor(
            [
                k_source.data_ptr(),
                k_source.data_ptr() + k_bytes,
                k_source.data_ptr() + 2 * k_bytes,
                v_source.data_ptr(),
                v_source.data_ptr() + v_bytes,
                v_source.data_ptr() + 2 * v_bytes,
            ],
            dtype=torch.int64,
            device=npu_device,
        ),
        torch.tensor(
            [base + 16, base + 80, base + 144, base + 208, base + 272, base + 336],
            dtype=torch.int64,
            device=npu_device,
        ),
        torch.tensor([k_bytes, 0, k_bytes, v_bytes, 0, v_bytes], dtype=torch.int32, device=npu_device),
        torch.tensor([2 * num_tokens], dtype=torch.int32, device=npu_device),
        npu_device,
    )
    torch.npu.synchronize()

    expected = torch.zeros_like(destination)
    k_host = k_source.cpu().view(torch.uint8)
    v_host = v_source.cpu().view(torch.uint8)
    expected[16 : 16 + k_bytes] = k_host[:k_bytes]
    expected[144 : 144 + k_bytes] = k_host[2 * k_bytes : 3 * k_bytes]
    expected[208 : 208 + v_bytes] = v_host[:v_bytes]
    expected[336 : 336 + v_bytes] = v_host[2 * v_bytes : 3 * v_bytes]

    assert ret == 0
    assert torch.equal(destination, expected)

    empty = torch.empty([0], dtype=torch.int64, device=npu_device)
    empty_lengths = torch.empty([0], dtype=torch.int32, device=npu_device)
    assert offload.sparse_copy(
        empty,
        empty,
        empty_lengths,
        torch.tensor([0], dtype=torch.int32, device=npu_device),
        npu_device,
    ) == 0
    torch.npu.synchronize()
