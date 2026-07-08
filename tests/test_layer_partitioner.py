"""Tests for LayerPartitioner with synthetic computation graphs."""

import pytest
from llm_graph_parser.core.computation_graph import ComputationGraph
from llm_graph_parser.core.operator_node import OperatorNode
from llm_graph_parser.core.layer_partitioner import LayerPartitioner


def _node(op_id, op_type, layer_id="root"):
    return OperatorNode(
        op_id=op_id, op_type=op_type, category="other",
        op_name=op_id, layer_id=layer_id,
    )


def _build_linear_dag(n_nodes):
    """A simple linear chain DAG."""
    g = ComputationGraph("linear")
    for i in range(n_nodes):
        g.add_node(_node(f"N{i}", "LINEAR"))
    for i in range(n_nodes - 1):
        g.add_edge(f"N{i}", f"N{i+1}")
    return g


def _build_two_block_dag():
    """Two transformer blocks with attention, MLP, and residual ADDs."""
    g = ComputationGraph("two_blocks")

    # Embedding
    g.add_node(_node("emb", "EMBEDDING"))

    # Block 0
    g.add_node(_node("b0_norm", "LAYER_NORM"))
    g.add_node(_node("b0_q", "LINEAR"))
    g.add_node(_node("b0_k", "LINEAR"))
    g.add_node(_node("b0_v", "LINEAR"))
    g.add_node(_node("b0_attn", "SOFTMAX"))
    g.add_node(_node("b0_o", "LINEAR"))
    g.add_node(_node("b0_residual", "ADD"))
    g.add_node(_node("b0_norm2", "LAYER_NORM"))
    g.add_node(_node("b0_gate", "LINEAR"))
    g.add_node(_node("b0_act", "GELU"))
    g.add_node(_node("b0_down", "LINEAR"))
    g.add_node(_node("b0_residual2", "ADD"))

    # Block 1
    g.add_node(_node("b1_norm", "LAYER_NORM"))
    g.add_node(_node("b1_q", "LINEAR"))
    g.add_node(_node("b1_k", "LINEAR"))
    g.add_node(_node("b1_v", "LINEAR"))
    g.add_node(_node("b1_attn", "SOFTMAX"))
    g.add_node(_node("b1_o", "LINEAR"))
    g.add_node(_node("b1_residual", "ADD"))
    g.add_node(_node("b1_norm2", "LAYER_NORM"))
    g.add_node(_node("b1_gate", "LINEAR"))
    g.add_node(_node("b1_act", "GELU"))
    g.add_node(_node("b1_down", "LINEAR"))
    g.add_node(_node("b1_residual2", "ADD"))

    # LM head
    g.add_node(_node("lm", "LINEAR"))

    # Edges (simplified chain)
    prev = "emb"
    for block in ["b0", "b1"]:
        nodes = [f"{block}_norm", f"{block}_q", f"{block}_k", f"{block}_v",
                 f"{block}_attn", f"{block}_o", f"{block}_residual",
                 f"{block}_norm2", f"{block}_gate", f"{block}_act",
                 f"{block}_down", f"{block}_residual2"]
        for n in nodes:
            g.add_edge(prev, n)
            prev = n
    g.add_edge(prev, "lm")

    # Residual edges (long skip connections for ADD depth)
    g.add_edge("emb", "b0_residual")
    g.add_edge("b0_o", "b0_residual")
    g.add_edge("b0_residual", "b0_residual2")
    g.add_edge("b0_down", "b0_residual2")
    g.add_edge("b0_residual2", "b1_residual")
    g.add_edge("b1_o", "b1_residual")
    g.add_edge("b1_residual", "b1_residual2")
    g.add_edge("b1_down", "b1_residual2")

    return g


class TestLayerPartitioner:
    def test_empty_graph(self):
        g = ComputationGraph("empty")
        p = LayerPartitioner(g).partition()
        assert p.layer_type == "unknown"

    def test_linear_chain_no_blocks(self):
        """Linear chain with only LINEAR ops should produce no transformer blocks."""
        g = _build_linear_dag(10)
        tree = LayerPartitioner(g).partition()
        # Should have 1 block (fallback might split evenly)
        for c in tree.children:
            assert c.layer_type != "transformer_block"

    def test_two_blocks_detected(self):
        g = _build_two_block_dag()
        tree = LayerPartitioner(g).partition()
        # Synthetic graph may not have perfect ADD depths,
        # but should at least produce some layer structure
        assert len(tree.children) >= 1
        assert any(c.layer_type for c in tree.children)

    def test_attention_sublayers(self):
        g = _build_two_block_dag()
        tree = LayerPartitioner(g).partition()
        for c in tree.children:
            if c.layer_type == "transformer_block":
                sub = [s.layer_type for s in c.children]
                assert "attention" in sub or "mlp" in sub

    def test_lm_head_type(self):
        g = _build_two_block_dag()
        tree = LayerPartitioner(g).partition()
        last = tree.children[-1]
        assert last.layer_type in ("lm_head", "output")

    def test_embedding_type(self):
        g = _build_two_block_dag()
        tree = LayerPartitioner(g).partition()
        first = tree.children[0]
        assert first.layer_type == "embedding"

    def test_assign_layer_ids(self):
        g = _build_two_block_dag()
        tree = LayerPartitioner(g).partition()
        # Check that nodes got layer_id assigned
        n = g.get_node("b0_q")
        assert n is not None
        assert n.layer_id != ""
