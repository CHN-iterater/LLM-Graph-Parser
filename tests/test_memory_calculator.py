"""Tests for memory access estimation with synthetic tensor shapes."""

import pytest
from llm_graph_parser.utils.memory_calculator import estimate_memory_bytes
from llm_graph_parser.core.operator_node import TensorMeta


def _tm(shape, dtype="float32"):
    return TensorMeta(shape=shape, dtype=dtype)


class TestMemoryCalculator:
    def test_linear_memory(self):
        mem = estimate_memory_bytes("LINEAR",
            [_tm((4, 768)), _tm((768, 3072)), _tm((3072,))],
            [_tm((4, 3072))])
        assert mem > 0

    def test_data_movement_zero(self):
        for op in ("RESHAPE", "VIEW", "TRANSPOSE"):
            assert estimate_memory_bytes(op, [], []) == 0

    def test_negative_dim(self):
        """-1 dimensions should not cause negative memory."""
        mem = estimate_memory_bytes("LINEAR",
            [_tm((-1, 768)), _tm((768, 3072))],
            [_tm((-1, 3072))])
        assert mem >= 0

    def test_empty_inputs(self):
        assert estimate_memory_bytes("LINEAR", [], []) == 0

    def test_dtype_float32(self):
        mem = estimate_memory_bytes("ADD", [_tm((4, 768))], [_tm((4, 768))])
        expected = 2 * 4 * 768 * 4  # read + write, 4 bytes per float32
        assert mem == expected

    def test_dtype_int8(self):
        mem = estimate_memory_bytes("ADD",
            [TensorMeta((4, 768), "int8")],
            [TensorMeta((4, 768), "int8")])
        expected = 2 * 4 * 768 * 1  # 1 byte per int8
        assert mem == expected

    def test_unknown_op(self):
        mem = estimate_memory_bytes("MY_UNKNOWN_OP",
            [_tm((4, 768))], [_tm((4, 768))])
        assert mem > 0  # Should still estimate from shapes
