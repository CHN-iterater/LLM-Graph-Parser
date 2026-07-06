"""Module-level structure parser.

Parses the hierarchical structure of a PyTorch model:
    Model -> Layer -> SubModule
"""

from __future__ import annotations
import torch.nn as nn


class ModuleParser:
    """Parse the module hierarchy of a PyTorch model.

    Walks the nn.Module tree to identify:
    - Transformer layers (repeated blocks)
    - Sub-modules (attention, MLP, etc.)
    - Module types and parameter counts
    """

    # Sub-module names that indicate a transformer block
    LAYER_KEYWORDS = ["layer", "block", "transformer_block"]

    def __init__(self, model: nn.Module):
        self.model = model
        self._hierarchy: dict = {}

    def parse(self) -> dict:
        """Walk the module tree and return the hierarchical structure."""
        self._hierarchy = self._walk(self.model, depth=0)
        return self._hierarchy

    def _walk(self, module: nn.Module, depth: int = 0) -> dict:
        """Recursively walk the module tree."""
        info = {
            "name": "",
            "type": type(module).__name__,
            "depth": depth,
            "children": [],
            "num_params": sum(p.numel() for p in module.parameters()),
            "num_buffers": sum(b.numel() for b in module.buffers()),
        }

        for name, child in module.named_children():
            child_info = self._walk(child, depth + 1)
            child_info["name"] = name
            info["children"].append(child_info)

        return info

    def get_layers(self) -> list[dict]:
        """Identify repeated transformer layers from the hierarchy."""
        layers = []

        def _find_layers(info: dict, path: str = ""):
            if info["depth"] >= 1 and self._is_layer(info):
                layers.append({**info, "path": path})
            for child in info["children"]:
                child_path = f"{path}.{child['name']}" if path else child["name"]
                _find_layers(child, child_path)

        _find_layers(self._hierarchy)
        return layers

    def _is_layer(self, info: dict) -> bool:
        """Heuristic: a module with repeated identical sub-blocks is a layer."""
        name_lower = info.get("name", "").lower()
        for kw in self.LAYER_KEYWORDS:
            if kw in name_lower:
                return True
        # Check if children contain attention-like and MLP-like submodules
        child_types = {c["type"].lower() for c in info.get("children", [])}
        if any("attention" in ct for ct in child_types) and any(
            "mlp" in ct or "linear" in ct or "feedforward" in ct for ct in child_types
        ):
            return True
        return False

    def print_tree(self, info: dict = None, indent: int = 0) -> None:
        """Pretty-print the module hierarchy."""
        if info is None:
            info = self._hierarchy
            print("Model Hierarchy:")
        prefix = "  " * indent + "├── " if indent > 0 else ""
        param_str = f"  [{info['num_params']:,} params]"
        print(f"{prefix}{info['type']}{param_str}")
        for child in info.get("children", []):
            self.print_tree(child, indent + 1)
