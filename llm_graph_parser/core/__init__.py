"""Core data structures for computation graph representation.

OperatorNode: Standardized operator node with tensor shapes, FLOPs, DAG edges.
OperatorRegistry: Extensible registry for operator types (plugin-style).
ComputationGraph: DAG construction, topology sort, layer grouping.
PhaseSplitter: Prefill/Decode phase identification.
GraphSerializer: Versioned JSON serialization with schema.

To extend with new operator types::

    from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec
    registry = OperatorRegistry.get_default()
    registry.register(OperatorSpec(name="MY_OP", category="compute", ...))
"""
from .operator_node import OperatorNode, TensorMeta
from .operator_registry import OperatorRegistry, OperatorSpec
from .computation_graph import ComputationGraph
from .phase_splitter import PhaseSplitter
from .serialization import graph_to_dict, graph_to_json, SCHEMA_VERSION

__all__ = [
    "OperatorNode",
    "TensorMeta",
    "OperatorRegistry",
    "OperatorSpec",
    "ComputationGraph",
    "PhaseSplitter",
    "graph_to_dict",
    "graph_to_json",
    "SCHEMA_VERSION",
]
