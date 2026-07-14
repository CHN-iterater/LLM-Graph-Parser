"""Computation graph (DAG) construction and analysis."""

from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from .operator_node import OperatorNode
from .operator_registry import OperatorRegistry
from .serialization import graph_to_json
from .operator_node import LayerNode as _LayerNode


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
        self._layer_tree: Optional[_LayerNode] = None

    def set_layer_tree(self, tree: _LayerNode) -> None:
        """设置层次化 LayerNode 树，同时更新 _layer_map。"""
        self._layer_tree = tree
        self._layer_map.clear()

        def _walk(node: _LayerNode):
            for oid in node.op_ids:
                self._layer_map[node.layer_id].append(oid)
            for child in node.children:
                _walk(child)
        _walk(tree)

    def get_layer_tree(self) -> Optional[_LayerNode]:
        return self._layer_tree

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

    def get_edge_type(self, parent_id: str, child_id: str) -> str:
        """返回边的类型: ``"intra_layer"``, ``"cross_layer"``, ``"unknown"``."""
        p = self._nodes.get(parent_id)
        c = self._nodes.get(child_id)
        if p is None or c is None:
            return "unknown"
        if p.layer_id and c.layer_id and p.layer_id == c.layer_id:
            return "intra_layer"
        return "cross_layer"

    def get_intra_layer_edges(self, layer_id: str) -> list[tuple[str, str]]:
        """获取层内所有边: [(parent_id, child_id), ...]."""
        edges = []
        for oid in self._layer_map.get(layer_id, []):
            node = self._nodes.get(oid)
            if node:
                for cid in node.children:
                    c = self._nodes.get(cid)
                    if c and c.layer_id == layer_id:
                        edges.append((oid, cid))
        return edges

    def get_cross_layer_edges(self) -> list[tuple[str, str, str, str]]:
        """获取所有跨层边: [(parent_id, p_layer, child_id, c_layer), ...]."""
        edges = []
        for nid, node in self._nodes.items():
            for cid in node.children:
                c = self._nodes.get(cid)
                if c and node.layer_id and c.layer_id and node.layer_id != c.layer_id:
                    edges.append((nid, node.layer_id, cid, c.layer_id))
        return edges

    def get_node(self, op_id: str) -> Optional[OperatorNode]:
        return self._nodes.get(op_id)

    def get_layer_nodes(self, layer_id: str) -> list[OperatorNode]:
        """Get all operator nodes belonging to a given layer."""
        return [self._nodes[oid] for oid in self._layer_map.get(layer_id, [])]

    def get_layers(self) -> list[str]:
        """Get sorted list of layer IDs."""
        return sorted(self._layer_map.keys())

    def get_layer_operator_counts(self, layer_id: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for nid in self._layer_map.get(layer_id, []):
            node = self._nodes.get(nid)
            if node:
                counts[node.op_type] += 1
        return dict(counts)

    def get_layer_stats(self, layer_id: str) -> dict:
        ops = self.get_layer_nodes(layer_id)
        if not ops:
            return {"num_ops": 0, "total_flops": 0, "total_memory_bytes": 0}
        total_flops = sum(n.flops for n in ops)
        total_mem = sum(n.memory_bytes for n in ops)
        counts: dict[str, int] = defaultdict(int)
        for n in ops:
            counts[n.op_type] += 1
        layer_type = ""
        if self._layer_tree:
            def _find(lnode):
                if lnode.layer_id == layer_id:
                    return lnode.layer_type
                for c in lnode.children:
                    r = _find(c)
                    if r:
                        return r
                return ""
            layer_type = _find(self._layer_tree)
        return {"num_ops": len(ops), "total_flops": total_flops,
                "total_memory_bytes": total_mem, "op_counts": dict(counts),
                "layer_type": layer_type}

    def layer_report_text(self) -> str:
        lines = ["=" * 62, "  Layer-Level Breakdown", "=" * 62]
        if self._layer_tree:
            lines.append("")
            lines.append("  Layer hierarchy:")
            lines.append(self._layer_tree_to_text(self._layer_tree, "  "))
        lines.append("")
        lines.append(f"{'Layer':25s} {'Type':20s} {'Ops':>6s} "
                     f"{'FLOPs':>12s} {'Mem(MB)':>10s}")
        lines.append("-" * 75)
        for layer_id in self.get_layers():
            s = self.get_layer_stats(layer_id)
            lt = s['layer_type'] or '?'
            lines.append(f"{layer_id:25s} {lt:20s} {s['num_ops']:6d} "
                         f"{s['total_flops']:>12,} {s['total_memory_bytes']/1e6:>10.2f}")
        return chr(10).join(lines)

    def _layer_tree_to_text(self, node, indent: str) -> str:
        n_ops = node.num_ops()
        label = f"{indent}{node.layer_id} [{node.layer_type}] ({n_ops} ops)"
        for child in node.children:
            label += chr(10) + self._layer_tree_to_text(child, indent + '  ')
        return label

    def print_layer_report(self) -> None:
        print(self.layer_report_text())

    def _save_report(self, output_dir: str | Path, suffix: str, content: str,
                      name: str = '') -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_" if name else ""
        path = output_dir / f"{stem}{suffix}.txt"
        path.write_text(content, encoding='utf-8')
        return path

    def save_layer_report(self, output_dir: str | Path, name: str = '') -> Path:
        return self._save_report(output_dir, 'layer_report',
                                 self.layer_report_text(), name)

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

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        sorted_nodes = []

        while queue:
            nid = queue.popleft()
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
        print(self.summary_text())

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
    # Parallelism analysis (B1-B3)
    # ------------------------------------------------------------------

    def compute_levels(self, weight_attr: str = "") -> dict[str, int]:
        """Assign a topological level to each node (B1).

        ``level[u] = max(level[v] + 1)`` for all predecessors ``v``.
        Nodes with the same level can be executed in parallel.

        Args:
            weight_attr: If ``"flops"``, weight = node.flops (for critical path).
                         If empty or ``"ops"``, each node counts as 1.

        Returns:
            dict mapping ``op_id`` → level.
        """
        order = self.topo_sort()
        if not order:
            return {}

        # op_id → in-degree for the DP
        indeg: dict[str, int] = {}
        for node in self._nodes.values():
            indeg[node.op_id] = indeg.get(node.op_id, 0)
            for cid in node.children:
                indeg[cid] = indeg.get(cid, 0) + 1

        level: dict[str, int] = {}
        init_queue = [n.op_id for n in order if indeg.get(n.op_id, 0) == 0]
        queue = deque(init_queue)

        for nid in init_queue:
            level[nid] = 0

        while queue:
            nid = queue.popleft()
            node = self._nodes[nid]
            w = 1
            if weight_attr == "flops":
                w = max(node.flops, 1)
            for cid in node.children:
                level[cid] = max(level.get(cid, 0), level[nid] + w)
                indeg[cid] -= 1
                if indeg[cid] == 0:
                    queue.append(cid)

        return level

    def critical_path(self, weight_attr: str = "") -> tuple[float, list[str]]:
        """Find the critical (longest) path in the DAG (B2).

        Args:
            weight_attr: ``""`` (unweighted, each op = 1), ``"flops"``, or
                         ``"memory_bytes"``.

        Returns:
            ``(length, [op_id_0, op_id_1, ..., op_id_k])``.
        """
        order = self.topo_sort()
        if not order:
            return 0.0, []

        # DP
        cp: dict[str, float] = {}
        prev: dict[str, str] = {}
        for nid, node in [(n.op_id, n) for n in order]:
            w = 1
            if weight_attr == "flops":
                w = max(node.flops, 1)
            elif weight_attr == "memory_bytes":
                w = max(node.memory_bytes, 1)

            best = 0.0
            best_pred = ""
            for pid in node.parents:
                if cp.get(pid, 0) > best:
                    best = cp.get(pid, 0)
                    best_pred = pid
            cp[nid] = best + w
            prev[nid] = best_pred

        # Trace back from the node with max cp
        last = max(cp, key=lambda x: cp[x])
        length = cp[last]
        path: list[str] = []
        u = last
        while u:
            path.append(u)
            u = prev.get(u, "")
        path.reverse()
        return length, path

    def _stats_for_report(self, label: str) -> list[str]:
        """Build parallelism stats lines for a single graph (internal helper)."""
        levels = self.compute_levels("")
        if not levels:
            return [f"  {label}: (empty)"]

        dist: dict[int, int] = defaultdict(int)
        for lv in levels.values():
            dist[lv] += 1

        total = len(levels)
        max_lv = max(dist.keys()) if dist else 0
        max_par = max(dist.values()) if dist else 0
        avg_par = total / (max_lv + 1) if max_lv >= 0 else 0
        cp_ops_len, _ = self.critical_path("")
        cp_flops_len, _ = self.critical_path("flops")

        lines = [
            f"  {label}",
            f"    Nodes:               {total}",
            f"    Max parallelism:     {max_par} ops/level",
            f"    Avg parallelism:     {avg_par:.2f} ops/level",
            f"    Critical path (ops): {cp_ops_len:.0f} steps",
            f"    Critical path (FLOPs): {cp_flops_len:,.0f} FLOPs",
        ]
        # Level distribution snippet (top 5 levels + tail)
        dist_lines = []
        for lv, cnt in sorted(dist.items()):
            pct = cnt / total * 100
            bar = "█" * int(cnt / max_par * 20) if max_par > 0 else ""
            dist_lines.append(f"      Lv {lv:3d}  {cnt:5d}  {bar}  {pct:.1f}%")
        if len(dist_lines) > 10:
            lines.extend(dist_lines[:5])
            lines.append(f"      ... ({len(dist_lines)-10} more levels)")
            lines.extend(dist_lines[-5:])
        else:
            lines.extend(dist_lines)
        return lines

    def parallelism_report(self) -> dict:
        """Generate a full parallelism analysis report (B1-B3), with
        prefill vs decode comparison where applicable.

        Returns a dict with:
            - ``level_distribution``: dict ``{level: count}``
            - ``max_parallelism``: max nodes at any level
            - ``avg_parallelism``: total_nodes / critical_path_length (ops)
            - ``critical_path_length``: unweighted critical path length
            - ``critical_path_flops``: FLOPs-weighted critical path length
            - ``critical_path_nodes``: op_ids on the unweighted critical path
            - ``text``: full human-readable report (incl. stage comparison)
        """
        levels = self.compute_levels("")
        if not levels:
            return {"text": "Graph empty, no parallelism data."}

        # Level distribution
        dist: dict[int, int] = defaultdict(int)
        for lv in levels.values():
            dist[lv] += 1

        max_lv = max(dist.keys()) if dist else 0
        max_par = max(dist.values()) if dist else 0
        total = len(levels)
        avg_par = total / (max_lv + 1) if max_lv >= 0 else 0

        # Critical paths
        cp_ops_len, cp_ops_path = self.critical_path("")
        cp_flops_len, _ = self.critical_path("flops")

        # Level distribution text
        dist_lines = []
        for lv in sorted(dist.keys()):
            bar_len = int(dist[lv] / max_par * 30) if max_par > 0 else 0
            bar = "█" * bar_len
            pct = dist[lv] / total * 100
            dist_lines.append(f"    Level {lv:3d}  {dist[lv]:5d} ops  {bar}  {pct:.1f}%")

        text_lines = [
            "=" * 62,
            "  Parallelism Analysis",
            "=" * 62,
            f"  Total nodes:           {total}",
            f"  Max parallelism:       {max_par} ops at a single level",
            f"  Avg parallelism:       {avg_par:.2f} ops/level",
            f"  Critical path (ops):   {cp_ops_len:.0f} steps",
            f"  Critical path (FLOPs): {cp_flops_len:,.0f} FLOPs",
            "",
            "  Level distribution:",
            *dist_lines,
            "",
            "  Stage comparison (Prefill vs Decode):",
        ]

        prefill_g = self.filter_by_stage("prefill")
        decode_g = self.filter_by_stage("decode")
        if prefill_g.num_nodes > 0 and decode_g.num_nodes > 0:
            pf = prefill_g._stats_for_report("Prefill")
            dc = decode_g._stats_for_report("Decode")
            text_lines.append("")
            text_lines.extend(pf)
            text_lines.append("")
            text_lines.extend(dc)
            # Ratio
            pf_avg = total / (max_lv + 1) if max_lv >= 0 else 0
            dc_levels = decode_g.compute_levels("")
            dc_max_lv = max(dc_levels.values()) if dc_levels else 1
            dc_avg = len(dc_levels) / (dc_max_lv + 1) if dc_max_lv >= 0 else 0
            if dc_avg > 0:
                text_lines.append(f"    Ratio (Prefill/Decode avg): {pf_avg/dc_avg:.2f}×")
        elif prefill_g.num_nodes > 0:
            text_lines.extend(prefill_g._stats_for_report("Prefill"))
        elif decode_g.num_nodes > 0:
            text_lines.extend(decode_g._stats_for_report("Decode"))

        text_lines.extend([
            "",
            "  Critical path nodes (unweighted):",
        ])

        # Show first 5 and last 5 nodes on the critical path
        cp_nodes = cp_ops_path
        if len(cp_nodes) <= 10:
            for nid in cp_nodes:
                n = self._nodes[nid]
                text_lines.append(f"    {nid}  [{n.op_type}]")
        else:
            for nid in cp_nodes[:5]:
                n = self._nodes[nid]
                text_lines.append(f"    {nid}  [{n.op_type}]")
            text_lines.append(f"    ... ({len(cp_nodes) - 10} more)")  # actually 5+5, adjust
            for nid in cp_nodes[-5:]:
                n = self._nodes[nid]
                text_lines.append(f"    {nid}  [{n.op_type}]")

        text = "\n".join(text_lines)

        return {
            "level_distribution": dict(dist),
            "max_parallelism": max_par,
            "avg_parallelism": avg_par,
            "critical_path_length": cp_ops_len,
            "critical_path_flops": cp_flops_len,
            "critical_path_nodes": cp_ops_path,
            "top_levels": "\n".join(dist_lines),
            "text": text,
        }

    def print_parallelism_report(self) -> None:
        """Print the parallelism analysis report to stdout."""
        print(self.parallelism_report()["text"])

    def save_parallelism_report(self, output_dir: str | Path, name: str = "") -> Path:
        return self._save_report(output_dir, 'parallel_report',
                                 self.parallelism_report()["text"], name)

    # ------------------------------------------------------------------
    # KV Cache cross-input dependency
    # ------------------------------------------------------------------

    def _find_attention_chain(self) -> list[OperatorNode]:
        """从图中找到所有 Attention 链上的算子: MUL→GEMM→ADD→SOFTMAX→ISNAN→WHERE→GEMM"""
        results: list[OperatorNode] = []
        for node in self._nodes.values():
            if node.op_type == "SOFTMAX":
                results.append(node)
                # 向上找前驱
                for pid in node.parents:
                    p = self._nodes.get(pid)
                    if p and p not in results:
                        results.append(p)
                        for ppid in p.parents:
                            pp = self._nodes.get(ppid)
                            if pp and pp not in results:
                                results.append(pp)
                # 向下找后继
                for cid in node.children:
                    c = self._nodes.get(cid)
                    if c and c not in results:
                        results.append(c)
                        for ccid in c.children:
                            cc = self._nodes.get(ccid)
                            if cc and cc not in results:
                                results.append(cc)
        return results

    def kv_cache_analysis(self, num_decode_tokens: int = 0) -> dict:
        """分析 KV cache 跨输入依赖。

        KV cache 使 Decode 阶段的 Attention 具有 **跨输入依赖**：
        生成第 N 个 token 时, Attention 的 Q@K^T 需要读取之前所有 token 的 K。
        这使得 Attention FLOPs 随生成长度 **二次增长** O(T²)。

        Args:
            num_decode_tokens: 生成的 token 数。默认取 ``prompt_tokens``。

        Returns:
            dict 包含 per_step_flops, total_flops, cross_edges, text。
        """
        attn_ops = self._find_attention_chain()
        if not attn_ops:
            return {"per_step_flops": 0, "total_flops": 0,
                    "cross_edges": 0,
                    "text": "  (No attention chain found in this graph)"}

        T = num_decode_tokens or self.prompt_tokens or 1

        # 收集所有 GEMM 在 Attention 链上的 FLOPs (per-step)
        per_step_attn_flops = sum(
            max(n.flops, 0) for n in attn_ops if n.op_type in ("GEMM", "BMM"))

        # 简化模型: per-step FLOPs 不变(单步导出), 但实际 F ~ T * per_step
        # 因为每次 Q@K^T 的 K 维度从 1 增长到 T
        # 所以总 Attention FLOPs ≈ per_step * T * (T+1) / 2
        num_layers = len([n for n in attn_ops if n.op_type == "SOFTMAX"])
        total_flops = per_step_attn_flops * T * (T + 1) // 2 if T > 1 else per_step_attn_flops
        cross_edges = T * (T - 1) // 2

        lines = [
            "=" * 62,
            "  KV Cache Cross-Input Dependency",
            "=" * 62,
            f"  Transformer layers:   {num_layers}",
            f"  Generated tokens (T): {T}",
            f"  Attn ops per layer:   {len(attn_ops)}",
            f"  Per-step attn FLOPs:  {per_step_attn_flops:,} "
            f"({per_step_attn_flops/1e6:.2f}M)",
            "",
            "  Dependency model:",
            f"    Token[i]'s Attention depends on K[0..i-1], V[0..i-1]",
            f"    → {cross_edges} cross-input dependency edges",
            f"    → Total attn FLOPs (with KV cache): {total_flops:,} "
            f"({total_flops/1e6:.2f}M)",
            f"    → vs single-step attn FLOPs: "
            f"{per_step_attn_flops:,} ({per_step_attn_flops/1e6:.2f}M)",
            f"    → Ratio: {total_flops / max(per_step_attn_flops, 1):.1f}x",
            "",
            "  Impact:",
            "    • Decode steps are strictly sequential (no inter-step parallel)",
            "    • Within a step: attention heads can be parallelized",
            "    • KV cache causes O(T²) attention cost growth",
            "    • Suggestion: reuse prefill KV cache, minimize decode steps",
        ]
        text = "\n".join(lines)
        return {
            "num_layers": num_layers,
            "per_step_flops": per_step_attn_flops,
            "total_flops": total_flops,
            "cross_edges": cross_edges,
            "text": text,
        }

    def save_kv_cache_report(self, output_dir: str | Path, name: str = "",
                              num_decode_tokens: int = 0) -> Path:
        return self._save_report(output_dir, 'kvcache_report',
                                 self.kv_cache_analysis(num_decode_tokens=num_decode_tokens)["text"], name)

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
        return self._save_report(output_dir, 'phase_report',
                                 self.stage_comparison_text(hardware_profile=hardware_profile), name)

    def save_to_json(self, output_dir: str | Path,
                     registry: OperatorRegistry | None = None,
                     name: str = "") -> Path:
        """Export the graph as a versioned JSON file."""
        if registry is None:
            registry = OperatorRegistry.get_default()
        stem = f"{name}_" if name else ""
        output_path = Path(output_dir) / f"{stem}graph.json"
        return graph_to_json(self, registry, output_path)

    def save_summary(self, output_dir: str | Path,
                     name: str = "") -> Path:
        return self._save_report(output_dir, 'summary', self.summary_text(), name)

    def to_dict(self) -> dict:
        """Export the graph as a serializable dictionary (legacy, prefer save_to_json)."""
        return {
            "model_name": self.model_name,
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }
