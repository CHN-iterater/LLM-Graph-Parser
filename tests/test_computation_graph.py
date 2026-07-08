"""Tests for ComputationGraph: DAG, parallelism, stage stats."""

import pytest
from llm_graph_parser.core.computation_graph import ComputationGraph
from llm_graph_parser.core.operator_node import OperatorNode


def _make_node(op_id, op_type, flops=0, memory=0, stage="unknown", category="other"):
    return OperatorNode(
        op_id=op_id, op_type=op_type, category=category,
        op_name=op_id, flops=flops, memory_bytes=memory,
        stage=stage,
    )


class TestComputationGraph:
    def test_empty_graph(self):
        g = ComputationGraph("empty")
        assert g.num_nodes == 0
        assert g.topo_sort() == []

    def test_add_node(self):
        g = ComputationGraph("test")
        n = _make_node("op_0", "LINEAR")
        g.add_node(n)
        assert g.num_nodes == 1
        assert g.get_node("op_0") is n

    def test_add_edge(self):
        g = ComputationGraph("test")
        a = _make_node("A", "LINEAR")
        b = _make_node("B", "SOFTMAX")
        g.add_node(a)
        g.add_node(b)
        g.add_edge("A", "B")
        assert "B" in a.children
        assert "A" in b.parents

    def test_topo_sort_linear(self):
        g = ComputationGraph("test")
        nodes = [_make_node(f"N{i}", "OP") for i in range(5)]
        for n in nodes:
            g.add_node(n)
        g.add_edge("N0", "N1")
        g.add_edge("N1", "N2")
        g.add_edge("N2", "N3")
        g.add_edge("N3", "N4")
        order = g.topo_sort()
        assert [n.op_id for n in order] == ["N0", "N1", "N2", "N3", "N4"]

    def test_topo_sort_diamond(self):
        g = ComputationGraph("test")
        for nid in ["input", "A", "B", "C", "output"]:
            g.add_node(_make_node(nid, "OP"))
        g.add_edge("input", "A")
        g.add_edge("input", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("C", "output")
        order = [n.op_id for n in g.topo_sort()]
        assert order.index("input") < order.index("A")
        assert order.index("input") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("C")
        assert order.index("C") < order.index("output")

    def test_compute_levels(self):
        g = ComputationGraph("test")
        for nid in ["I", "A", "B", "C", "O"]:
            g.add_node(_make_node(nid, "OP"))
        g.add_edge("I", "A")
        g.add_edge("I", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("C", "O")
        levels = g.compute_levels()
        assert levels["I"] == 0
        assert levels["A"] == 1
        assert levels["B"] == 1
        assert levels["C"] == 2
        assert levels["O"] == 3

    def test_critical_path(self):
        g = ComputationGraph("test")
        g.add_node(_make_node("I", "OP", flops=0))
        g.add_node(_make_node("A", "LINEAR", flops=100))
        g.add_node(_make_node("B", "LINEAR", flops=50))
        g.add_node(_make_node("C", "LINEAR", flops=200))
        g.add_node(_make_node("O", "OP", flops=0))
        g.add_edge("I", "A")
        g.add_edge("I", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("C", "O")

        length, path = g.critical_path("")
        assert length == 4  # I→A→C→O or I→B→C→O

        flops_len, _ = g.critical_path("flops")
        assert 250 <= flops_len <= 350  # weight should be > 0

    def test_parallelism_report(self):
        g = ComputationGraph("test")
        for i in range(5):
            g.add_node(_make_node(f"N{i}", "OP"))
        g.add_edge("N0", "N1")
        g.add_edge("N0", "N2")
        g.add_edge("N1", "N3")
        g.add_edge("N2", "N3")
        g.add_edge("N3", "N4")
        r = g.parallelism_report()
        assert "text" in r
        assert r["max_parallelism"] >= 1
        assert r["critical_path_length"] > 0

    def test_stage_stats(self):
        g = ComputationGraph("test")
        g.add_node(_make_node("A", "LINEAR", flops=100, memory=50, stage="prefill"))
        g.add_node(_make_node("B", "LINEAR", flops=200, memory=30, stage="decode"))
        s = g.get_stage_stats("prefill")
        assert s["num_ops"] == 1
        assert s["total_flops"] == 100

    def test_filter_by_stage(self):
        g = ComputationGraph("test")
        g.add_node(_make_node("A", "LINEAR", stage="prefill"))
        g.add_node(_make_node("B", "LINEAR", stage="decode"))
        pf = g.filter_by_stage("prefill")
        dc = g.filter_by_stage("decode")
        assert pf.num_nodes == 1
        assert dc.num_nodes == 1

    def test_edge_type_intra(self):
        g = ComputationGraph("test")
        a = _make_node("A", "LINEAR")
        a.layer_id = "layer_0"
        b = _make_node("B", "LINEAR")
        b.layer_id = "layer_0"
        g.add_node(a); g.add_node(b)
        g.add_edge("A", "B")
        assert g.get_edge_type("A", "B") == "intra_layer"

    def test_edge_type_cross(self):
        g = ComputationGraph("test")
        a = _make_node("A", "LINEAR")
        a.layer_id = "layer_0"
        b = _make_node("B", "LINEAR")
        b.layer_id = "layer_1"
        g.add_node(a); g.add_node(b)
        g.add_edge("A", "B")
        assert g.get_edge_type("A", "B") == "cross_layer"

    def test_edge_type_unknown(self):
        g = ComputationGraph("test")
        assert g.get_edge_type("NONEXIST", "NONEXIST") == "unknown"

    def test_intra_layer_edges(self):
        g = ComputationGraph("test")
        for nid in ["A", "B", "C"]:
            n = _make_node(nid, "OP")
            n.layer_id = "layer_0"
            g.add_node(n)
        g.add_edge("A", "B"); g.add_edge("B", "C")
        edges = g.get_intra_layer_edges("layer_0")
        assert len(edges) == 2

    def test_cross_layer_edges(self):
        g = ComputationGraph("test")
        a = _make_node("A", "LINEAR"); a.layer_id = "layer_0"
        b = _make_node("B", "LINEAR"); b.layer_id = "layer_1"
        g.add_node(a); g.add_node(b)
        g.add_edge("A", "B")
        edges = g.get_cross_layer_edges()
        assert len(edges) == 1
        assert edges[0][1] == "layer_0"  # parent layer
        assert edges[0][3] == "layer_1"  # child layer

    def test_layer_grouping(self):
        g = ComputationGraph("test")
        a = _make_node("A", "LINEAR")
        a.layer_id = "layer_0"
        b = _make_node("B", "LINEAR")
        b.layer_id = "layer_0"
        g.add_node(a)
        g.add_node(b)
        assert g.get_layers() == ["layer_0"]
        assert len(g.get_layer_nodes("layer_0")) == 2
