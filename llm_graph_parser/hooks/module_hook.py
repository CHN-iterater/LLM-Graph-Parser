"""Module-level PyTorch forward hooks.

Register forward hooks on model submodules to trace execution at the module level.
"""

from __future__ import annotations
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from llm_graph_parser.parser.tensor_recorder import TensorRecorder


class ModuleHook:
    """Register forward_pre_hook / forward_hook on model submodules.

    Tracks which modules are executed during a forward pass and records
    their input/output tensor shapes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list = []
        self.execution_order: list[dict] = []
        self.module_stats: dict[str, dict] = {}

    def register_all(self) -> None:
        """Register hooks on every submodule in the model."""
        for name, module in self.model.named_modules():
            if name == "":
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def register_layer_hooks(self, layer_names: list[str]) -> None:
        """Register hooks only on specific layer names."""
        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, module_name: str) -> Callable:
        """Create a forward hook closure for the given module name."""

        def hook(module, input, output):
            in_shape = None
            if input and isinstance(input[0], torch.Tensor):
                in_shape = tuple(input[0].shape)

            out_shape = None
            if isinstance(output, torch.Tensor):
                out_shape = tuple(output.shape)
            elif isinstance(output, (tuple, list)) and output:
                if isinstance(output[0], torch.Tensor):
                    out_shape = tuple(output[0].shape)

            entry = {
                "name": module_name,
                "type": type(module).__name__,
                "input_shape": in_shape,
                "output_shape": out_shape,
            }
            self.execution_order.append(entry)

            # Accumulate stats
            if module_name not in self.module_stats:
                self.module_stats[module_name] = {
                    "type": type(module).__name__,
                    "call_count": 0,
                    "input_shapes": [],
                    "output_shapes": [],
                }
            self.module_stats[module_name]["call_count"] += 1
            if in_shape:
                self.module_stats[module_name]["input_shapes"].append(in_shape)
            if out_shape:
                self.module_stats[module_name]["output_shapes"].append(out_shape)

        return hook

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def print_execution_summary(self) -> None:
        """Print a summary of module execution order and statistics."""
        print("\nModule Execution Order:")
        print("=" * 60)
        for i, entry in enumerate(self.execution_order):
            in_str = TensorRecorder.format_shape(entry["input_shape"]) if entry["input_shape"] else "N/A"
            out_str = TensorRecorder.format_shape(entry["output_shape"]) if entry["output_shape"] else "N/A"
            print(f"  [{i:3d}] {entry['name']:40s} {entry['type']:20s} {in_str} -> {out_str}")

        print("\nModule Call Statistics:")
        print("=" * 60)
        for name, stats in sorted(self.module_stats.items()):
            print(f"  {name:40s} x{stats['call_count']:3d}  ({stats['type']})")
