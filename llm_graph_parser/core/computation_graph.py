"""Computation graph (DAG) construction and analysis."""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .operator_node import OperatorNode
from .operator_registry import OperatorRegistry
from .serialization import graph_to_json


class ComputationGraph:
    """A hierarchical computation graph representing model inference.

    Structure: Model -> Layer -> Operator -> (Tensor/Kernel)
    Internally stored as a DAG of OperatorNodes connected by edges.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.prompt_text: str = ""      # 输入的自然语言 Prompt
        self.prompt_tokens: int = 0     # 分词后的 token 数
        self._nodes: dict[str, OperatorNode] = {}
        self._layer_map: defaultdict[str, list[str]] = defaultdict(list)

    @property
    def nodes(self) -> list[OperatorNode]:
        return list(self._nodes.values())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def add_node(self, node: OperatorNode) -> None:
        """Add a single operator node to the graph."""
        self._nodes[node.op_id] = node
        if node.layer_id:
            self._layer_map[node.layer_id].append(node.op_id)

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a directed edge: parent -> child (data dependency)."""
        if parent_id in self._nodes and child_id in self._nodes:
            self._nodes[parent_id].children.append(child_id)
            self._nodes[child_id].parents.append(parent_id)

    def get_node(self, op_id: str) -> Optional[OperatorNode]:
        return self._nodes.get(op_id)

    def get_layer_nodes(self, layer_id: str) -> list[OperatorNode]:
        """Get all operator nodes belonging to a given layer."""
        return [self._nodes[oid] for oid in self._layer_map.get(layer_id, [])]

    def get_layers(self) -> list[str]:
        """Get sorted list of layer IDs."""
        return sorted(self._layer_map.keys())

    def get_operator_counts(self) -> dict[str, int]:
        """Count occurrences of each operator type."""
        counts: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            counts[node.op_type] += 1
        return dict(counts)

    def topo_sort(self) -> list[OperatorNode]:
        """Topological sort of the computation graph.

        Returns nodes in execution order (parents before children).
        Uses Kahn's algorithm. Falls back to insertion order if no edges.
        """
        if not self._nodes:
            return []

        in_degree: dict[str, int] = {}
        for nid, node in self._nodes.items():
            in_degree.setdefault(nid, 0)
            for cid in node.children:
                in_degree[cid] = in_degree.get(cid, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            nid = queue.pop(0)
            sorted_nodes.append(self._nodes[nid])
            for cid in self._nodes[nid].children:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    queue.append(cid)

        # If no edges defined, fall back to insertion order
        if len(sorted_nodes) != self.num_nodes:
            return list(self._nodes.values())
        return sorted_nodes

    def print_summary(self) -> None:
        """Print a human-readable summary of the computation graph."""
        print(f"Model: {self.model_name}")
        if self.prompt_text:
            print(f"Prompt: \"{self.prompt_text}\"")
        if self.prompt_tokens:
            print(f"Prompt tokens: {self.prompt_tokens}")
        print(f"Total operator nodes: {self.num_nodes}")
        print()

        layers = self.get_layers()
        print(f"Layers ({len(layers)}):")
        for layer_id in layers:
            layer_nodes = self.get_layer_nodes(layer_id)
            print(f"  {layer_id}: {len(layer_nodes)} ops")

        print()
        print("Operator counts:")
        for op_type, count in sorted(
            self.get_operator_counts().items(), key=lambda x: -x[1]
        ):
            print(f"  {op_type:25s}: {count}")

    def summary_text(self) -> str:
        """Return a complete human-readable summary as text (for saving)."""
        lines = []
        lines.append(f"Model: {self.model_name}")
        if self.prompt_text:
            lines.append(f"Prompt: \"{self.prompt_text}\"")
        if self.prompt_tokens:
            lines.append(f"Prompt tokens: {self.prompt_tokens}")
        lines.append(f"Total operator nodes: {self.num_nodes}")
        lines.append("")

        layers = self.get_layers()
        lines.append(f"Layers ({len(layers)}):")
        for layer_id in layers:
            layer_nodes = self.get_layer_nodes(layer_id)
            lines.append(f"  {layer_id}: {len(layer_nodes)} ops")

        lines.append("")
        lines.append("Operator counts:")
        for op_type, count in sorted(
            self.get_operator_counts().items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {op_type:25s}: {count}")
        return "\n".join(lines)

    def to_stage_graphs(self) -> tuple[ComputationGraph, ComputationGraph]:
        """Split this graph into separate prefill and decode subgraphs."""
        prefill = ComputationGraph(f"{self.model_name}[prefill]")
        decode = ComputationGraph(f"{self.model_name}[decode]")

        for node in self._nodes.values():
            if node.stage == "prefill":
                prefill.add_node(node)
            elif node.stage == "decode":
                decode.add_node(node)

        return prefill, decode

    def save_to_json(self, output_dir: str | Path,
                     registry: OperatorRegistry | None = None,
                     name: str = "") -> Path:
        """Export the graph as a versioned JSON file.

        Uses the standardized serialization format with schema versioning.

        Args:
            output_dir: Output directory.
            registry: Operator registry.
            name: Optional filename stem (e.g. ``"prompt_0"`` →
                   ``prompt_0_graph.json``). Defaults to ``"graph"``.
        """
        if registry is None:
            registry = OperatorRegistry.get_default()
        stem = name or "graph"
        output_path = Path(output_dir) / f"{stem}_graph.json"
        return graph_to_json(self, registry, output_path)

    def save_summary(self, output_dir: str | Path,
                     name: str = "") -> Path:
        """Save the human-readable summary to a text file.

        Args:
            output_dir: Output directory.
            name: Optional filename stem (e.g. ``"prompt_0"`` →
                   ``prompt_0_summary.txt``). Defaults to ``"summary"``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = name or "summary"
        path = output_dir / f"{stem}_summary.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.summary_text())
        return path

    def to_dict(self) -> dict:
        """Export the graph as a serializable dictionary (legacy, prefer save_to_json)."""
        return {
            "model_name": self.model_name,
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }
