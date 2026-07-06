"""Operator-level hooks for fine-grained tracing.

Captures individual operator (call_function) invocations from the
torch.export / torch.fx graph and builds OperatorNode instances.

Uses ``OperatorRegistry`` for extensible operator matching.
"""

from __future__ import annotations
from typing import Callable, Optional

import torch

from llm_graph_parser.core.operator_node import OperatorNode, TensorMeta
from llm_graph_parser.core.computation_graph import ComputationGraph
from llm_graph_parser.core.operator_registry import OperatorRegistry
from llm_graph_parser.parser.tensor_recorder import TensorRecorder


class OperatorHook:
    """Build a ComputationGraph from a torch.export ExportedProgram.

    Iterates over the exported graph nodes, creates OperatorNode instances
    for each call_function node, records tensor metadata, and constructs
    the DAG via node argument dependencies.
    """

    def __init__(self, exported_program,
                 registry: Optional[OperatorRegistry] = None):
        self.exported_program = exported_program
        self.graph_module = exported_program.module()
        self.graph = exported_program.graph
        self._registry = registry or OperatorRegistry.get_default()

    def parse(self, model_name: str = "model",
              layer_id_fn: Optional[Callable] = None) -> ComputationGraph:
        """Parse the exported graph into a ComputationGraph.

        Args:
            model_name: Name for the computation graph.
            layer_id_fn: Optional callable(fx_node) -> str to assign layer IDs.
                         If None, all operators go to "root".

        Returns:
            A ComputationGraph with OperatorNodes and DAG edges.
        """
        comp_graph = ComputationGraph(model_name)
        recorder = TensorRecorder()
        node_map: dict[str, str] = {}  # FX node name -> op_id

        # First pass: create nodes
        for i, fx_node in enumerate(self.graph.nodes):
            if fx_node.op != "call_function":
                continue

            op_id = f"op_{i:04d}"
            op_spec = self._resolve_op(fx_node.target)
            layer_id = layer_id_fn(fx_node) if layer_id_fn else "root"

            node = OperatorNode(
                op_id=op_id,
                op_type=op_spec.name,
                category=op_spec.category,
                op_name=fx_node.name,
                layer_id=layer_id,
                raw_target=str(fx_node.target),
            )

            # Record tensor metadata
            recorder.record_from_node(op_id, fx_node)
            recorder.apply_to_node(node)

            comp_graph.add_node(node)
            node_map[fx_node.name] = op_id

        # Second pass: build edges (data dependencies via args)
        for fx_node in self.graph.nodes:
            if fx_node.name not in node_map:
                continue
            child_id = node_map[fx_node.name]

            for arg in fx_node.args:
                if isinstance(arg, torch.fx.Node) and arg.name in node_map:
                    parent_id = node_map[arg.name]
                    comp_graph.add_edge(parent_id, child_id)

        return comp_graph

    def _resolve_op(self, target):
        """Map FX target to an OperatorSpec via registry."""
        return self._registry.lookup(str(target))
