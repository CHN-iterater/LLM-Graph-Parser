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

    # ------------------------------------------------------------------
    # Stage (Prefill/Decode) helpers
    # ------------------------------------------------------------------

    def tag_unassigned_as(self, stage: str) -> None:
        """Set stage for all nodes that still have ``"unknown"`` stage."""
        for node in self._nodes.values():
            if node.stage in ("unknown", ""):
                node.stage = stage

    # ------------------------------------------------------------------
    # Stage (Prefill/Decode) analysis
    # ------------------------------------------------------------------

    def filter_by_stage(self, stage: str) -> ComputationGraph:
        """Create a subgraph containing only nodes of the given stage.

        Args:
            stage: ``"prefill"``, ``"decode"``, or others.

        Returns:
            A new ``ComputationGraph`` with matching nodes.
        """
        sub = ComputationGraph(f"{self.model_name}[{stage}]")
        sub.prompt_text = self.prompt_text
        sub.prompt_tokens = self.prompt_tokens
        for node in self._nodes.values():
            if node.stage == stage:
                sub.add_node(node)
        return sub

    def to_stage_graphs(self) -> tuple[ComputationGraph, ComputationGraph]:
        """Split this graph into separate prefill and decode subgraphs."""
        prefill = self.filter_by_stage("prefill")
        decode = self.filter_by_stage("decode")
        return prefill, decode

    def get_stage_stats(self, stage: str) -> dict:
        """Get aggregated statistics for a given inference stage.

        Returns:
            dict with keys: num_ops, total_flops, total_memory_bytes,
            arith_intensity, op_counts, category_flops, category_memory.
        """
        ops = [n for n in self._nodes.values() if n.stage == stage]
        if not ops:
            return {"num_ops": 0, "total_flops": 0, "total_memory_bytes": 0,
                    "arith_intensity": 0.0, "op_counts": {}, "category_flops": {}}

        total_flops = sum(n.flops for n in ops)
        total_memory = sum(n.memory_bytes for n in ops)

        op_counts: dict[str, int] = defaultdict(int)
        category_flops: dict[str, int] = defaultdict(int)
        category_memory: dict[str, int] = defaultdict(int)
        for n in ops:
            op_counts[n.op_type] += 1
            category_flops[n.category] += n.flops
            category_memory[n.category] += n.memory_bytes

        return {
            "num_ops": len(ops),
            "total_flops": total_flops,
            "total_memory_bytes": total_memory,
            "arith_intensity": total_flops / total_memory if total_memory > 0 else 0.0,
            "op_counts": dict(op_counts),
            "category_flops": dict(category_flops),
            "category_memory": dict(category_memory),
        }

    def stage_comparison_text(self, hardware_profile: Optional[dict] = None) -> str:
        """Return formatted comparison between prefill and decode stages.

        Args:
            hardware_profile: Optional dict with "peak_flops" and "memory_bw"
                              (bytes/sec) for roofline analysis.

        Returns:
            Formatted comparison text.
        """
        prefill_stats = self.get_stage_stats("prefill")
        decode_stats = self.get_stage_stats("decode")

        lines = []
        lines.append("=" * 66)
        lines.append("  Phase Comparison: Prefill vs Decode")
        lines.append("=" * 66)

        # ---- Overall ----
        lines.append(f"\n{'Metric':<30s} {'Prefill':>16s} {'Decode':>16s}")
        lines.append("-" * 66)
        pf_ops = prefill_stats["num_ops"]
        dc_ops = decode_stats["num_ops"]
        pf_flops = prefill_stats["total_flops"]
        dc_flops = decode_stats["total_flops"]
        pf_mem = prefill_stats["total_memory_bytes"]
        dc_mem = decode_stats["total_memory_bytes"]
        pf_ai = prefill_stats["arith_intensity"]
        dc_ai = decode_stats["arith_intensity"]

        lines.append(f"{'Operator nodes':<30s} {pf_ops:>16d} {dc_ops:>16d}")
        lines.append(f"{'Total FLOPs':<30s} {pf_flops:>16,} {dc_flops:>16,}")
        lines.append(f"{'Memory bytes':<30s} {pf_mem:>16,} {dc_mem:>16,}")
        lines.append(f"{'Arith intensity (FLOPs/byte)':<30s} {pf_ai:>16.2f} {dc_ai:>16.2f}")

        # Ratio
        total_flops = pf_flops + dc_flops
        if total_flops > 0:
            lines.append(f"\n  FLOPs distribution:   Prefill {pf_flops/total_flops*100:.1f}%"
                         f" / Decode {dc_flops/total_flops*100:.1f}%")

        # ---- Category breakdown ----
        lines.append(f"\n{'Category FLOPs':<30s} {'Prefill':>16s} {'Decode':>16s}")
        lines.append("-" * 66)
        all_cats = set(list(prefill_stats["category_flops"].keys()) +
                       list(decode_stats["category_flops"].keys()))
        for cat in sorted(all_cats):
            pf_cat = prefill_stats["category_flops"].get(cat, 0)
            dc_cat = decode_stats["category_flops"].get(cat, 0)
            lines.append(f"{cat:<30s} {pf_cat:>16,} {dc_cat:>16,}")

        # ---- Roofline analysis ----
        if hardware_profile:
            lines.append("")
            lines.append("")
            lines.append("=" * 66)
            lines.append("  Roofline Analysis")
            lines.append("=" * 66)
            pf_bw = hardware_profile.get("memory_bw", 0)
            pf_fp = hardware_profile.get("peak_flops", 0)
            lines.append(f"  Hardware:             "
                         f"Peak FP = {pf_fp/1e12:.1f} TFLOPS, "
                         f"Memory BW = {pf_bw/1e9:.0f} GB/s")
            ridge = pf_fp / pf_bw if pf_bw > 0 else 0
            lines.append(f"  Ridge point:          "
                         f"{ridge:.1f} FLOPs/byte")

            for label, stats in [("Prefill", prefill_stats), ("Decode", decode_stats)]:
                ai = stats["arith_intensity"]
                ops_flops = stats["total_flops"]
                ops_mem = stats["total_memory_bytes"]
                if ai > 0 and pf_bw > 0:
                    # Compute-bound or memory-bound?
                    bound = "COMPUTE BOUND" if ai >= ridge else "MEMORY BOUND"
                    perf = ai * pf_bw  # achievable performance
                    util = perf / pf_fp * 100 if pf_fp > 0 else 0
                    lines.append(f"\n  {label:<12s} AI={ai:<8.2f}  "
                                 f"{bound:<20s}  "
                                 f"Util={util:.1f}%")
                    if ops_flops > 0 and ops_mem > 0:
                        lines.append(f"             "
                                     f"FLOPs={ops_flops/1e6:.0f}M  "
                                     f"Mem={ops_mem/1e6:.0f}MB  "
                                     f"Batch={self.prompt_tokens}")

        return "\n".join(lines)

    def save_phase_report(self, output_dir: str | Path, name: str = "",
                          hardware_profile: Optional[dict] = None) -> Path:
        """Save the phase comparison report to a text file.

        Args:
            output_dir: Output directory.
            name: Optional filename stem. Defaults to ``"phase"``.
            hardware_profile: Optional hardware spec dict for roofline analysis.

        Returns:
            Path to the saved report file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = name or "phase"
        path = output_dir / f"{stem}_phase_report.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.stage_comparison_text(hardware_profile=hardware_profile))
        return path

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
