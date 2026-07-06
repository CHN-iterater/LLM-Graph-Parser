"""Prefill / Decode phase splitting logic.

Uses string-based stage identifiers ("prefill", "decode") for
extensibility (future stages like "warmup" can be added easily).
"""

from __future__ import annotations

from .computation_graph import ComputationGraph


class PhaseSplitter:
    """Identify and split computation graph into prefill and decode phases.

    Prefill: processes the entire input prompt in parallel;
             compute-bound, dominated by GEMM and FlashAttention.
    Decode:  generates tokens one at a time;
             memory-bound, dominated by KV Cache access and element-wise ops.
    """

    PREFILL_HEAVY_OPS = {"FLASH_ATTENTION", "ATTENTION", "SOFTMAX", "LINEAR"}
    DECODE_HEAVY_OPS = {"KV_CACHE_READ", "KV_CACHE_WRITE", "LINEAR"}

    def __init__(self, seq_len: int = 0):
        self.seq_len = seq_len

    @staticmethod
    def split_by_sequence(graph: ComputationGraph,
                          prefill_seq_len: int) -> tuple[ComputationGraph, ComputationGraph]:
        """Split based on known sequence length at each phase.

        Typically: the first pass over the full prompt is prefill,
        and each subsequent token generation step is decode.
        """
        prefill = ComputationGraph(f"{graph.model_name}[prefill]")
        decode = ComputationGraph(f"{graph.model_name}[decode]")

        for node in graph.nodes:
            if node.input_tensors:
                first_input = node.input_tensors[0]
                if len(first_input.shape) >= 2 and first_input.shape[1] == prefill_seq_len:
                    node.stage = "prefill"
                elif len(first_input.shape) >= 2 and first_input.shape[1] == 1:
                    node.stage = "decode"

            if node.stage == "prefill":
                prefill.add_node(node)
            elif node.stage == "decode":
                decode.add_node(node)

        return prefill, decode

    @staticmethod
    def split_by_layer_pattern(graph: ComputationGraph) -> tuple[ComputationGraph, ComputationGraph]:
        """Identify phase boundaries by detecting KV Cache operators.

        If the graph contains KV_CACHE_READ or KV_CACHE_WRITE operators,
        those belong to decode phase; the rest is prefill.
        """
        prefill = ComputationGraph(f"{graph.model_name}[prefill]")
        decode = ComputationGraph(f"{graph.model_name}[decode]")

        for node in graph.nodes:
            if node.op_type in ("KV_CACHE_READ", "KV_CACHE_WRITE"):
                node.stage = "decode"
                decode.add_node(node)
            elif node.stage != "decode":
                node.stage = "prefill"
                prefill.add_node(node)

        return prefill, decode

    @staticmethod
    def describe_stages(prefill_graph: ComputationGraph,
                        decode_graph: ComputationGraph) -> dict:
        """Return a descriptive summary of both stages."""
        return {
            "prefill": {
                "num_operators": prefill_graph.num_nodes,
                "total_flops": sum(n.flops for n in prefill_graph.nodes),
                "operator_counts": prefill_graph.get_operator_counts(),
            },
            "decode": {
                "num_operators": decode_graph.num_nodes,
                "total_flops": sum(n.flops for n in decode_graph.nodes),
                "operator_counts": decode_graph.get_operator_counts(),
            },
        }
