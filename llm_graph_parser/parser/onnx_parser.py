"""ONNX model parser.

Loads an ``.onnx`` file and converts its computation graph directly
into a standardized ``ComputationGraph`` (DAG).

This is the primary entry point when working with pre-exported models,
bypassing the need to run ``torch.export``.

Usage::

    from llm_graph_parser.parser.onnx_parser import OnnxParser

    parser = OnnxParser()
    graph = parser.parse("path/to/model.onnx", model_name="my_model")
    graph.print_summary()

ONNX graph structure mapping::

    ONNX Node  →  OperatorNode
    ONNX Tensor name  →  DAG edge
    ONNX ValueInfo  →  TensorMeta
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_graph_parser.core.operator_node import OperatorNode, TensorMeta
from llm_graph_parser.core.computation_graph import ComputationGraph
from llm_graph_parser.core.operator_registry import OperatorRegistry

# Mapping from ONNX op_type to our registry operator name.
# Extend this when encountering new ONNX ops.
ONNX_OP_MAP: dict[str, str] = {
    # ---- Compute ----
    "Gemm": "LINEAR",
    "MatMul": "GEMM",
    "MatMulInteger": "GEMM",
    "BatchNormalization": "UNKNOWN",  # not typical in Transformers
    # ---- Attention ----
    "Attention": "ATTENTION",
    "MultiHeadAttention": "ATTENTION",
    "GroupQueryAttention": "ATTENTION",
    # ---- Normalization ----
    "LayerNormalization": "LAYER_NORM",
    "SkipLayerNormalization": "LAYER_NORM",
    "SimplifiedLayerNormalization": "LAYER_NORM",
    "SkipSimplifiedLayerNormalization": "LAYER_NORM",
    # ---- Activation ----
    "Relu": "RELU",
    "Gelu": "GELU",
    "FastGelu": "GELU",
    "Sigmoid": "SIGMOID",
    "Softmax": "SOFTMAX",
    "LogSoftmax": "SOFTMAX",
    # ---- Element-wise ----
    "Add": "ADD",
    "Mul": "MUL",
    "Sub": "SUB",
    "Div": "DIV",
    "Sum": "ADD",
    # ---- Data movement ----
    "Reshape": "RESHAPE",
    "Transpose": "TRANSPOSE",
    "Concat": "CAT",
    "Slice": "SLICE",
    "Split": "SLICE",
    "Squeeze": "RESHAPE",
    "Unsqueeze": "RESHAPE",
    "Expand": "EXPAND",
    "Pad": "UNKNOWN",
    "Tile": "UNKNOWN",
    "Gather": "EMBEDDING",
    "GatherElements": "EMBEDDING",
    "Shape": "UNKNOWN",
    "Size": "UNKNOWN",
    "ConstantOfShape": "UNKNOWN",
    # ---- Embedding ----
    "Embedding": "EMBEDDING",
    # ---- Normalization / div ----
    "ReduceMean": "UNKNOWN",
    # ---- Other ----
    "Dropout": "DROPOUT",
    "Cast": "UNKNOWN",
    "Where": "UNKNOWN",
    "Equal": "UNKNOWN",
    "Less": "UNKNOWN",
    "GreaterOrEqual": "UNKNOWN",
    "LessOrEqual": "UNKNOWN",
    "Greater": "UNKNOWN",
    "Not": "UNKNOWN",
    "And": "UNKNOWN",
    "Or": "UNKNOWN",
    "Loop": "UNKNOWN",
    "If": "UNKNOWN",
    "Identity": "UNKNOWN",
    "Constant": "UNKNOWN",
    "ReduceSum": "UNKNOWN",
    "ReduceProd": "UNKNOWN",
    "ReduceMin": "UNKNOWN",
    "ReduceMax": "UNKNOWN",
    "Erf": "UNKNOWN",
    "Shape": "UNKNOWN",
    "Size": "UNKNOWN",
    "Squeeze": "RESHAPE",
    "Unsqueeze": "RESHAPE",
    "Sqrt": "UNKNOWN",
    "Pow": "UNKNOWN",
    "Neg": "UNKNOWN",
    "Clip": "UNKNOWN",
    "RandomNormal": "UNKNOWN",
    "RandomUniform": "UNKNOWN",
    "TopK": "UNKNOWN",
    "ArgMax": "UNKNOWN",
    "ArgMin": "UNKNOWN",
    "Tile": "UNKNOWN",
}


class OnnxParser:
    """Parse an ONNX model file into a ComputationGraph.

    The parser:
    1. Loads the ``.onnx`` file
    2. Iterates over graph nodes (operators)
    3. Extracts tensor shapes from ``value_info`` and ``initializer``
    4. Constructs the DAG via tensor name matching
    5. Maps ONNX op types to the operator registry

    Args:
        registry: ``OperatorRegistry`` to use. If ``None``, uses the default.
    """

    def __init__(self, registry: Optional[OperatorRegistry] = None):
        self._registry = registry or OperatorRegistry.get_default()
        self._model = None
        self._graph = None
        # tensor name -> TensorMeta
        self._tensor_map: dict[str, TensorMeta] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Load an ONNX model from a file path.

        Args:
            path: Path to the ``.onnx`` file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If ``onnx`` is not installed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX file not found: {path}")

        try:
            import onnx
        except ImportError:
            raise ImportError(
                "onnx package is required. Install with: pip install onnx"
            )

        self._model = onnx.load(str(path), load_external_data=False)
        self._graph = self._model.graph
        self._build_tensor_map()

    def parse(self, model_name: str = "") -> ComputationGraph:
        """Convert the loaded ONNX graph into a ComputationGraph.

        Args:
            model_name: Optional name for the computation graph.

        Returns:
            A ``ComputationGraph`` with ``OperatorNode`` s and DAG edges.

        Raises:
            RuntimeError: If no model has been loaded via ``load()``.
        """
        if self._graph is None:
            raise RuntimeError("No ONNX model loaded. Call load() first.")

        if not model_name:
            model_name = Path(self._model._serial).stem if hasattr(self._model, '_serial') else "onnx_model"

        comp_graph = ComputationGraph(model_name)

        # Track which tensor names each node produces
        producer_map: dict[str, str] = {}  # tensor_name -> op_id

        # First pass: create OperatorNodes
        for i, onnx_node in enumerate(self._graph.node):
            op_id = f"onnx_op_{i:04d}"
            op_spec = self._resolve_op(onnx_node.op_type)

            # Gather input tensor shapes
            input_tensors = [
                self._tensor_map.get(name, TensorMeta((), "unknown"))
                for name in onnx_node.input
            ]
            # Gather output tensor shapes
            output_tensors = [
                self._tensor_map.get(name, TensorMeta((), "unknown"))
                for name in onnx_node.output
            ]

            node = OperatorNode(
                op_id=op_id,
                op_type=op_spec.name,
                category=op_spec.category,
                op_name=onnx_node.name or f"{onnx_node.op_type}_{i}",
                layer_id="root",  # ONNX doesn't have explicit layer info
                raw_target=onnx_node.op_type,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                metadata={
                    "onnx_op_type": onnx_node.op_type,
                    "domain": onnx_node.domain or "",
                    "source": "onnx",
                },
            )

            comp_graph.add_node(node)

            # Record output tensor -> op_id mapping for edge construction
            for out_name in onnx_node.output:
                if out_name:
                    producer_map[out_name] = op_id

        # Second pass: build DAG edges via tensor name dependencies
        for onnx_node in self._graph.node:
            # Find the op_id for this node
            for out_name in onnx_node.output:
                if out_name and out_name in producer_map:
                    child_id = producer_map[out_name]
                    break
            else:
                continue

            # For each input tensor, find which node produced it
            for in_name in onnx_node.input:
                if in_name and in_name in producer_map:
                    parent_id = producer_map[in_name]
                    if parent_id != child_id:
                        comp_graph.add_edge(parent_id, child_id)

        return comp_graph

    def get_onnx_op_types(self) -> set[str]:
        """Return the set of ONNX op types present in the loaded model."""
        if self._graph is None:
            return set()
        return {n.op_type for n in self._graph.node}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tensor_map(self) -> None:
        """Extract tensor shapes/types from the ONNX graph.

        Sources (in priority order):
        1. ``graph.value_info``  — intermediate tensors
        2. ``graph.input``        — model inputs
        3. ``graph.output``       — model outputs
        4. ``graph.initializer``  — constant tensors (weights)
        """
        self._tensor_map.clear()

        def _extract(vinfo) -> tuple[tuple[int, ...], str]:
            """Extract shape and dtype from a ValueInfoProto."""
            shape = tuple(
                d.dim_value if d.dim_value > 0 else -1
                for d in vinfo.type.tensor_type.shape.dim
            )
            dtype = _onnx_dtype_to_str(vinfo.type.tensor_type.elem_type)
            return shape, dtype

        import onnx  # noqa: F811 — already checked in load()

        # value_info (intermediate tensors)
        for vinfo in self._graph.value_info:
            shape, dtype = _extract(vinfo)
            self._tensor_map[vinfo.name] = TensorMeta(shape, dtype)

        # model inputs
        for vinfo in self._graph.input:
            shape, dtype = _extract(vinfo)
            self._tensor_map[vinfo.name] = TensorMeta(shape, dtype)

        # model outputs
        for vinfo in self._graph.output:
            shape, dtype = _extract(vinfo)
            self._tensor_map[vinfo.name] = TensorMeta(shape, dtype)

        # initializers (weights) — shape from numpy array
        for init in self._graph.initializer:
            self._tensor_map[init.name] = TensorMeta(
                shape=tuple(init.dims),
                dtype=_onnx_dtype_to_str(init.data_type),
            )

    def _resolve_op(self, onnx_op_type: str):
        """Map an ONNX op type to an OperatorSpec via the registry.

        First checks ``ONNX_OP_MAP`` for a direct mapping,
        then falls back to the registry's dynamic name resolution
        (never returns UNKNOWN).
        """
        mapped = ONNX_OP_MAP.get(onnx_op_type)
        # If the mapping leads to a real operator, use it
        if mapped and mapped != "UNKNOWN":
            spec = self._registry.get(mapped)
            if spec:
                return spec
        # Fallback: dynamic name resolution from the target string
        return self._registry.lookup(onnx_op_type)


def _onnx_dtype_to_str(elem_type: int) -> str:
    """Map ONNX tensor element type to string."""
    _DTYPE_MAP = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "double",
        12: "uint32",
        13: "uint64",
        14: "complex64",
        15: "complex128",
        16: "bfloat16",
    }
    return _DTYPE_MAP.get(elem_type, "unknown")
