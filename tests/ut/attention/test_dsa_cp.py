# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl


def test_qli_metadata_uses_cpu_lengths_without_device_scalar_reads():
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSACPMetadataBuilder,
    )

    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.compressor_ratio = 4
    builder.common_ratio_to_sas_metadata = {}
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            index_n_heads=64,
            index_head_dim=128,
            index_topk=2048,
        )
    )
    builder.seqused_q = torch.zeros(2, dtype=torch.int32)
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    query_start = torch.tensor([0, 2, 5], dtype=torch.int32)
    seq_lens = torch.tensor([12, 31], dtype=torch.int32)
    query_lens = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.arange(1024, dtype=torch.int32)
    with (
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            return_value=expected,
            create=True,
        ) as metadata_op,
        patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("device scalar read"),
        ),
    ):
        result = builder._build_qli_metadata(
            query_start,
            seq_lens,
            query_lens,
            2,
            max_query_len=3,
            max_seq_lens=31,
        )
        builder._build_qli_metadata(
            query_start,
            seq_lens,
            query_lens,
            2,
            max_query_len=3,
            max_seq_lens=31,
        )
    assert metadata_op.call_count == 1
    assert metadata_op.call_args.kwargs["max_seqlen_q"] == 3
    assert metadata_op.call_args.kwargs["max_seqlen_k"] == 31
    assert torch.equal(result, expected)


class TestAscendDSACPLayerMetadata:
    def test_routes_by_cache_prefix(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        impl.compress_ratio = 4
        impl.swa_cache_layer = SimpleNamespace(prefix="swa_cache")
        impl.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="compressor.state_cache"))
        impl.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="indexer.k_cache"),
            compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="indexer.compressor.state_cache")),
        )
        attention_metadata = object()
        compressor_state_metadata = object()
        indexer_cache_metadata = object()
        indexer_state_metadata = object()
        swa_metadata = object()

        metadata: Any = {
            "layer": attention_metadata,
            "compressor.state_cache": compressor_state_metadata,
            "indexer.k_cache": indexer_cache_metadata,
            "indexer.compressor.state_cache": indexer_state_metadata,
            "swa_cache": swa_metadata,
        }
        layer_metadata = impl._get_layer_metadata("layer", metadata)

        assert layer_metadata.swa is swa_metadata
        assert layer_metadata.compressor_cache is attention_metadata
        assert layer_metadata.compressor_state is compressor_state_metadata
        assert layer_metadata.indexer_cache is indexer_cache_metadata
        assert layer_metadata.indexer_state is indexer_state_metadata

    def test_update_graph_params_is_noop_like_dsa(self):
        # FULL_DECODE_ONLY replay calls this on the impl class.
        AscendDSACPImpl.update_graph_params(None, None, 1)


def test_zero_graph_padding_rows_clears_stale_seq_lens_and_block_table():
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSACPMetadataBuilder,
    )

    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.seq_lens = torch.tensor([17, 4096, 128], dtype=torch.int32)
    builder.seq_lens_cpu = torch.tensor([17, 4096, 128], dtype=torch.int32)
    builder.block_table = torch.tensor([[3, 4], [9, 8], [7, 6]], dtype=torch.int32)
    builder.start_pos_prefill = torch.tensor([16, 99, 50], dtype=torch.int32)

    actual = builder._zero_graph_padding_rows(num_reqs=3, num_reqs_actual=1)

    assert actual == 1
    assert torch.equal(builder.seq_lens, torch.tensor([17, 0, 0], dtype=torch.int32))
    assert torch.equal(builder.seq_lens_cpu, torch.tensor([17, 0, 0], dtype=torch.int32))
    assert torch.count_nonzero(builder.block_table[1:]).item() == 0
    assert torch.equal(builder.start_pos_prefill, torch.tensor([16, 0, 0], dtype=torch.int32))


def test_zero_graph_padding_rows_noop_when_no_pad():
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSACPMetadataBuilder,
    )

    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.seq_lens = torch.tensor([4, 5], dtype=torch.int32)
    builder.seq_lens_cpu = torch.tensor([4, 5], dtype=torch.int32)
    builder.block_table = torch.tensor([[1], [2]], dtype=torch.int32)
    builder.start_pos_prefill = torch.tensor([3, 4], dtype=torch.int32)

    actual = builder._zero_graph_padding_rows(num_reqs=2, num_reqs_actual=2)

    assert actual == 2
    assert torch.equal(builder.seq_lens, torch.tensor([4, 5], dtype=torch.int32))
    assert torch.equal(builder.block_table, torch.tensor([[1], [2]], dtype=torch.int32))
