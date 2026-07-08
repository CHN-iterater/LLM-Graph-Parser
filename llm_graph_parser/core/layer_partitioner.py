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
        """回退: 用 SkipLayerNorm 或 ATTENTION 算子位置检测层边界。

        策略:
          1) 找 SkipLayerNorm(残差连接+层归一化融合算子)位置,
             每 2 个 SkipLayerNorm = 1 个 block.
             第 2,4,6… 个 SkipLayerNorm 是 block 边界。
          2) 回退到 ATTENTION 位置检测。"""
        # 策略1: SkipLayerNorm 定位
        skip_norm_positions = []
        for i, node in enumerate(self._nodes):
            if node.op_type in ("LAYER_NORM", "RMS_NORM"):
                # 检查是否为 Skip 变体
                if "Skip" in node.raw_target or "skip" in node.raw_target:
                    skip_norm_positions.append(i)

        if len(skip_norm_positions) >= 4:  # 至少 2 个 block
            # 每 2 个一组,第 2 个是 block 边界
            boundaries = [skip_norm_positions[i] for i in range(1, len(skip_norm_positions), 2)]
            # 过滤太靠近开头/结尾的
            n = len(self._nodes)
            boundaries = [b for b in boundaries if 3 < b < n - 3]
            if len(boundaries) >= 1:
                # 检查边界间距是否合理(太近则回退)
                gaps = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
                if gaps and min(gaps) >= 5:
                    return boundaries

        # 策略2: ATTENTION 位置回退
        attn_positions = []
        for i, node in enumerate(self._nodes):
            if node.op_type in _get_attn_ops():
                attn_positions.append(i)

        if len(attn_positions) >= 2:
            n_total = len(self._nodes)
            # 第一个 attention 到最后一个 attention 之间的范围
            first = attn_positions[0]
            last = attn_positions[-1]
            span = last - first
            # 按 attention 数量等分
            num_blocks = len(attn_positions)
            step = max(span // num_blocks, 5)
            boundaries = [first + step * (i + 1) for i in range(num_blocks - 1)]
            boundaries = [b for b in boundaries if 3 < b < n_total - 3]
            return boundaries

        return []

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

        if layer_id == "lm_head":
            layer.layer_type = "lm_head"
        elif has_attn:
            layer.layer_type = "transformer_block"
            self._add_sublayers(layer, nodes, layer_id)
        elif layer_id == "embedding" or has_embedding:
            layer.layer_type = "embedding"
        elif "output" in layer_id:
            layer.layer_type = "output"
        elif has_linear:
            layer.layer_type = "mlp"
        else:
            layer.layer_type = "other"

        return layer

    def _add_sublayers(self, parent_layer: LayerNode, nodes, parent_id: str) -> None:
        """在 transformer_block 内划分 attention 和 mlp 子层。

        方法:
          1) ADD 深度分析(适用于分解式 Attention)
          2) 回退: 用 ATTENTION 算子定位(适用于融合式 Attention)
        """
        # 方法1: ADD 深度分析
        residual_indices = []
        for i, node in enumerate(nodes):
            if node.op_type == "ADD":
                depth = self._add_depth_from_node(node, nodes, i)
                if depth >= _RESIDUAL_DEPTH_THRESHOLD:
                    residual_indices.append(i)
        if len(residual_indices) >= 2:
            attn_end = residual_indices[0]
            mlp_end = residual_indices[1]
            attn_ids = [n.op_id for n in nodes[:attn_end]]
            mlp_ids = [n.op_id for n in nodes[attn_end:mlp_end + 1]]
            parent_layer.add_child(LayerNode(
                layer_id=f"{parent_id}_attn", layer_type="attention",
                op_ids=attn_ids))
            parent_layer.add_child(LayerNode(
                layer_id=f"{parent_id}_mlp", layer_type="mlp",
                op_ids=mlp_ids))
            return

        # 方法2: 用 ATTENTION 算子定位(融合式)
        attn_end = 0
        for i, node in enumerate(nodes):
            if node.op_type in _get_attn_ops():
                attn_end = i
        if attn_end > 0:
            # Attention 之后可能还有输出投影 GEMM
            attn_end += 1
            while attn_end < len(nodes) and nodes[attn_end].op_type in (
                    "GEMM", "LINEAR", "RESHAPE", "ADD"):
                attn_end += 1
            # 划分
            attn_ids = [n.op_id for n in nodes[:attn_end]]
            mlp_ids = [n.op_id for n in nodes[attn_end:]]
            parent_layer.add_child(LayerNode(
                layer_id=f"{parent_id}_attn", layer_type="attention",
                op_ids=attn_ids))
            parent_layer.add_child(LayerNode(
                layer_id=f"{parent_id}_mlp", layer_type="mlp",
                op_ids=mlp_ids))

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
