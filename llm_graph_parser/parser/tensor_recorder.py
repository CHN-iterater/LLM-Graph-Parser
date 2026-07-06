"""Tensor information recorder.

For each operator, records input/output tensor shapes, dtypes, and devices.
"""

from __future__ import annotations
from typing import Optional

import torch

from llm_graph_parser.core.operator_node import TensorMeta, OperatorNode


class TensorRecorder:
    """Record tensor metadata (shape, dtype, device) from captured graph nodes."""

    def __init__(self):
        self._records: dict[str, dict] = {}

    def record_from_node(self, op_id: str, fx_node) -> None:
        """Extract tensor metadata from a torch.fx Node's metadata.

        Stores the result so it can be retrieved later and applied to OperatorNodes.
        """
        meta = getattr(fx_node, "meta", {})
        # torch.export stores tensor metadata in meta['val']
        val = meta.get("val", None)

        input_tensors = []
        output_tensors = []

        if val is not None:
            output_tensors = self._extract_tensor_meta(val)

        # For input tensors, look at the node's args
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node):
                arg_val = getattr(arg, "meta", {}).get("val", None)
                if arg_val is not None:
                    input_tensors.extend(self._extract_tensor_meta(arg_val))

        self._records[op_id] = {
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
        }

    def apply_to_node(self, node: OperatorNode) -> None:
        """Apply recorded tensor metadata to an OperatorNode."""
        record = self._records.get(node.op_id)
        if record:
            node.input_tensors = record["input_tensors"]
            node.output_tensors = record["output_tensors"]

    def get_record(self, op_id: str) -> Optional[dict]:
        return self._records.get(op_id)

    @staticmethod
    def _extract_tensor_meta(val) -> list[TensorMeta]:
        """Extract TensorMeta from a value that may be a Tensor or nested structure."""
        results = []
        if isinstance(val, torch.Tensor):
            results.append(TensorMeta(
                shape=tuple(val.shape),
                dtype=str(val.dtype),
                device=str(val.device),
            ))
        elif isinstance(val, (list, tuple)):
            for v in val:
                results.extend(TensorRecorder._extract_tensor_meta(v))
        elif isinstance(val, dict):
            for v in val.values():
                results.extend(TensorRecorder._extract_tensor_meta(v))
        return results

    @staticmethod
    def format_shape(shape: tuple[int, ...]) -> str:
        """Format a tensor shape for human-readable output."""
        return f"[{','.join(str(s) for s in shape)}]"

    def reset(self) -> None:
        """Clear all recorded data."""
        self._records.clear()
