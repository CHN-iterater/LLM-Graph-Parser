"""Tests for FLOPs estimation with synthetic tensor shapes."""

import pytest
from llm_graph_parser.utils.flops_calculator import estimate_flops
from llm_graph_parser.core.operator_node import TensorMeta


def _tm(shape, dtype="float32"):
    return TensorMeta(shape=shape, dtype=dtype)


class TestFlopsCalculator:
    def test_linear_flops(self):
        flops = estimate_flops("LINEAR", [_tm((4, 768))], [_tm((4, 3072))])
        # 2 * B * K * N = 2 * 4 * 768 * 3072
        assert flops == 2 * 4 * 768 * 3072

    def test_linear_batched(self):
        flops = estimate_flops("LINEAR", [_tm((2, 4, 768))], [_tm((2, 4, 3072))])
        assert flops == 2 * 2 * 4 * 768 * 3072

    def test_gemm(self):
        flops = estimate_flops("GEMM", [_tm((32, 64))], [_tm((32, 128))])
        assert flops == 2 * 32 * 64 * 128

    def test_bmm(self):
        flops = estimate_flops("BMM", [_tm((2, 4, 8, 64)), _tm((2, 4, 64, 128))], [])
        assert flops == 2 * 2 * 8 * 64 * 128

    def test_softmax(self):
        flops = estimate_flops("SOFTMAX", [_tm((4, 16, 128))], [])
        assert flops == 5 * 4 * 16 * 128  # 5 * numel

    def test_layer_norm(self):
        flops = estimate_flops("LAYER_NORM", [_tm((4, 768))], [])
        assert flops == 3 * 4 * 768

    def test_gelu(self):
        flops = estimate_flops("GELU", [_tm((4, 768))], [])
        assert flops == 4 * 768

    def test_add(self):
        flops = estimate_flops("ADD", [_tm((4, 768))], [])
        assert flops == 4 * 768

    def test_attention_decomposed(self):
        """Decomposed attention: (B, H, T, d) format."""
        flops = estimate_flops("ATTENTION", [_tm((2, 8, 128, 64))], [])
        assert flops == 4 * 2 * 8 * 128 * 128 * 64

    def test_attention_fused(self):
        """Fused attention (GroupQueryAttention): (B, T, hidden) format."""
        flops = estimate_flops("ATTENTION", [_tm((2, 128, 2048))], [])
        assert flops == 4 * 2 * 2048 * 128 * 128

    def test_data_movement_zero(self):
        for op in ("RESHAPE", "VIEW", "TRANSPOSE", "SLICE", "CLONE"):
            assert estimate_flops(op, [_tm((4, 768))], []) == 0

    def test_negative_dim(self):
        """-1 dimensions should be treated as 1, not cause negative FLOPs."""
        flops = estimate_flops("LINEAR", [_tm((-1, 768))], [_tm((-1, 3072))])
        assert flops == 2 * 1 * 768 * 3072
        assert flops > 0

    def test_empty_inputs(self):
        assert estimate_flops("LINEAR", [], []) == 0

    def test_unknown_op(self):
        assert estimate_flops("MY_CUSTOM_OP", [_tm((4, 768))], []) == 0
