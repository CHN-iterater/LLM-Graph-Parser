"""层边界划分 — 通过 ADD 节点深度分析或高频算子检测识别 Transformer 层。"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional
from .operator_node import LayerNode

_RESIDUAL_DEPTH_THRESHOLD = 8
def _get_attn_ops():
    """从注册表动态获取 attention 算子名 + SOFTMAX。"""
    try:
        from .operator_registry import OperatorRegistry
        ops = [s.name for s in OperatorRegistry.get_default().get_by_tag("attention")]
        ops.append("SOFTMAX")
        return frozenset(ops)
    except Exception:
        return frozenset({"SOFTMAX", "ATTENTION", "GROUPQUERYATTENTION",
                          "FLASH_ATTENTION", "MULTIHEADATTENTION"})


class LayerPartitioner:
    def __init__(self, graph):
        self._graph = graph
        try:
            self._nodes = graph.topo_sort()
        except Exception:
            self._nodes = list(graph.nodes)
        self._topo_index = {}
        for i, n in enumerate(self._nodes):
            self._topo_index[n.op_id] = i

    def partition(self) -> LayerNode:
        if not self._nodes:
            return LayerNode(layer_id="root", layer_type="unknown")
        boundaries = self._find_block_boundaries()
        tree = LayerNode(layer_id=self._graph.model_name or "model", layer_type="model")
        prev = 0
        for b in boundaries:
            if b > prev:
                label = self._label_block(prev)
                layer = self._build_layer(self._nodes[prev:b], label)
                tree.add_child(layer)
            prev = b
        remaining = self._nodes[prev:]
        if remaining:
            label = "lm_head" if len(tree.children) > 0 else "embedding"
            layer = self._build_layer(remaining, label)
            tree.add_child(layer)
        if tree.children and tree.children[0].layer_type == "transformer_block":
            tree.children[0].layer_id = "embedding"
            tree.children[0].layer_type = "embedding"
        self._assign_layer_ids(tree)
        return tree

    def _find_block_boundaries(self) -> list[int]:
        """通过 ADD 深度或高频算子检测层边界。"""
        n = len(self._nodes)
        if n < 10:
            return []

        # 方法1: 残差 ADD 深度分析
        residual_adds = []
        for i, node in enumerate(self._nodes):
            if node.op_type != "ADD":
                continue
            depth = self._add_depth(i)
            if depth >= _RESIDUAL_DEPTH_THRESHOLD:
                residual_adds.append((i, node, depth))

        if len(residual_adds) >= 2:
            boundaries = []
            for idx, (i, node, depth) in enumerate(residual_adds):
                if idx % 2 == 1:  # 每2个一组，第2个是 mlp residual
                    boundaries.append(i + 1)
            boundaries = [b for b in boundaries if 3 < b < n - 3]
            if boundaries:
                return boundaries

        # 方法2: 回退到检测高频算子(Attention/Softmax等)
        return self._fallback_boundaries()

    def _add_depth(self, topo_idx: int) -> int:
        node = self._nodes[topo_idx]
        max_depth = 0
        for pid in node.parents:
            p_idx = self._topo_index.get(pid)
            if p_idx is not None:
                depth = topo_idx - p_idx
                if depth > max_depth:
                    max_depth = depth
        return max_depth

    def _fallback_boundaries(self) -> list[int]:
        """回退: 用高频算子估算层数。"""
        # 找 Attention 相关算子的位置作为层标记
        attn_positions = []
        for i, node in enumerate(self._nodes):
            if node.op_type in _get_attn_ops():
                attn_positions.append(i)

        if len(attn_positions) < 2:
            return []

        # 每层可能有多个 attention 相关算子，按位置聚簇
        # 簇之间的间隔就是层边界
        if len(attn_positions) <= 1:
            return []

        # 相邻 attention 算子的中间位置作为层边界
        boundaries = []
        for j in range(len(attn_positions) - 1):
            mid = (attn_positions[j] + attn_positions[j + 1]) // 2
            if boundaries and mid - boundaries[-1] < 5:
                continue
            boundaries.append(mid)

        return boundaries

    def _label_block(self, topo_start: int) -> str:
        return f"layer_{self._find_layer_index_at(topo_start)}"

    def _find_layer_index_at(self, topo_pos: int) -> int:
        boundaries = self._find_block_boundaries()
        for i, b in enumerate(boundaries):
            if topo_pos < b:
                return i
        return len(boundaries)

    def _build_layer(self, nodes, layer_id: str) -> LayerNode:
        layer = LayerNode(layer_id=layer_id)
        for node in nodes:
            layer.op_ids.append(node.op_id)

        # 用算子结构判断层类型，不依赖具体激活函数名
        has_attn = any(n.op_type in _get_attn_ops() for n in nodes)
        has_linear = any(n.op_type == "LINEAR" for n in nodes)
        has_embedding = any(n.op_type == "EMBEDDING" for n in nodes)

        if has_attn:
            # Attention 算子存在 = 这是 Transformer block
            layer.layer_type = "transformer_block"
            self._add_sublayers(layer, nodes, layer_id)
        elif layer_id == "embedding" or has_embedding:
            layer.layer_type = "embedding"
        elif layer_id == "lm_head":
            layer.layer_type = "lm_head"
        elif "output" in layer_id:
            layer.layer_type = "output"
        elif has_linear:
            layer.layer_type = "mlp"
        else:
            layer.layer_type = "other"
        return layer

    def _add_sublayers(self, parent_layer: LayerNode, nodes, parent_id: str) -> None:
        """在 transformer_block 内划分 attention 和 mlp 子层。"""
        residual_indices = []
        for i, node in enumerate(nodes):
            if node.op_type == "ADD":
                depth = self._add_depth_from_node(node, nodes, i)
                if depth >= _RESIDUAL_DEPTH_THRESHOLD:
                    residual_indices.append(i)
        if len(residual_indices) >= 2:
            attn_end = residual_indices[0]
            mlp_end = residual_indices[1]
            attn_node = LayerNode(
                layer_id=f"{parent_id}_attn", layer_type="attention",
                op_ids=[n.op_id for n in nodes[:attn_end]])
            parent_layer.add_child(attn_node)
            mlp_node = LayerNode(
                layer_id=f"{parent_id}_mlp", layer_type="mlp",
                op_ids=[n.op_id for n in nodes[attn_end:mlp_end + 1]])
            parent_layer.add_child(mlp_node)

    def _add_depth_from_node(self, node, node_list, idx: int) -> int:
        max_depth = 0
        for pid in node.parents:
            for j, n in enumerate(node_list):
                if n.op_id == pid:
                    depth = idx - j
                    if depth > max_depth:
                        max_depth = depth
                    break
        return max_depth

    def _assign_layer_ids(self, node: LayerNode) -> None:
        for oid in node.op_ids:
            op_node = self._graph.get_node(oid)
            if op_node:
                op_node.layer_id = node.layer_id
        for child in node.children:
            self._assign_layer_ids(child)
