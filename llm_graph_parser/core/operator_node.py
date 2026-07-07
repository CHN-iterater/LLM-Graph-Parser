"""Standardized operator node data structure.

Designed for extensibility:
- ``metadata`` dict for arbitrary custom annotations
- ``hardware_metrics`` dict for device-specific measurements
- Clean separation between structural fields and computed fields
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class LayerNode:
    """A hierarchical layer node in the computation graph.

    Forms a tree: ``Model → Layer → SubLayer → Operator``.
    """
    layer_id: str
    layer_type: str = ""          # "embedding", "transformer_block", "attention",
                                  # "mlp", "lm_head", "norm", "unknown"
    op_ids: list[str] = field(default_factory=list)  # Operators in this layer
    children: list[LayerNode] = field(default_factory=list)
    parent: Optional[LayerNode] = None

    def add_child(self, child: LayerNode) -> None:
        child.parent = self
        self.children.append(child)

    def num_ops(self) -> int:
        return len(self.op_ids) + sum(c.num_ops() for c in self.children)

    def total_flops(self, graph) -> int:
        return sum(graph.get_node(oid).flops for oid in self.op_ids
                   if graph.get_node(oid)) + sum(
            c.total_flops(graph) for c in self.children)

    def print_tree(self, indent: int = 0) -> None:
        prefix = "  " * indent + "├── " if indent > 0 else ""
        n_ops = self.num_ops()
        print(f"{prefix}{self.layer_id} [{self.layer_type}] ({n_ops} ops)")
        for child in self.children:
            child.print_tree(indent + 1)

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "num_ops": self.num_ops(),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class TensorMeta:
    """Metadata for a single tensor."""
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"

    def to_dict(self) -> dict:
        return {"shape": list(self.shape), "dtype": self.dtype, "device": self.device}


@dataclass
class OperatorNode:
    """Standardized operator node in the hierarchical computation graph.

    ``metadata`` is a free-form dict for extensibility.
    ``hardware_metrics`` stores device-specific measured/estimated values.

    Examples of metadata keys (not exhaustive):
        - "source_node": the original FX node name from torch.export
        - "kernel_name": CUDA kernel name if available
        - "cuda_graph_id": for CUDA graph capture scenarios
    """
    op_id: str
    op_type: str  # refers to OperatorSpec.name (e.g. "LINEAR", "ATTENTION")
    op_name: str
    category: str = "other"

    # Hierarchical position
    layer_id: str = ""

    # Tensor information
    input_tensors: list[TensorMeta] = field(default_factory=list)
    output_tensors: list[TensorMeta] = field(default_factory=list)

    # Computed attributes
    flops: int = 0
    memory_bytes: int = 0
    arith_intensity: float = 0.0

    # Inference stage
    stage: str = "unknown"  # "prefill", "decode", or "unknown"

    # DAG edges (store op_ids of parent/child nodes)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    # Raw information
    raw_target: str = ""  # original torch.export target string

    # ---- Extensibility fields ----
    metadata: dict[str, Any] = field(default_factory=dict)
    hardware_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = True) -> dict:
        """Serialize this node to a dictionary."""
        result = {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "category": self.category,
            "op_name": self.op_name,
            "layer_id": self.layer_id,
            "stage": self.stage,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "arith_intensity": round(self.arith_intensity, 3),
            "parents": self.parents,
            "children": self.children,
            "input_tensors": [t.to_dict() for t in self.input_tensors],
            "output_tensors": [t.to_dict() for t in self.output_tensors],
        }
        if include_metadata:
            result["raw_target"] = self.raw_target
            result["metadata"] = self.metadata
            result["hardware_metrics"] = self.hardware_metrics
        return result

    def short_str(self) -> str:
        """Return a concise one-line representation."""
        in_shapes = [str(t.shape) for t in self.input_tensors]
        out_shapes = [str(t.shape) for t in self.output_tensors]
        flops_str = f"{self.flops:,}" if self.flops else "?"
        return (
            f"{self.op_id}: {self.op_type} "
            f"[{', '.join(in_shapes)}] -> [{', '.join(out_shapes)}]"
            f"  FLOPs={flops_str}"
        )
