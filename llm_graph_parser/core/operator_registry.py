"""Extensible operator registry with pattern-based matching.

New category system (for energy modelling):
    compute_bound  - 计算密集型, high AI (GEMM, Attention, BMM)
    memory_bound   - 访存密集型, low AI (Softmax, LayerNorm, Reduction)
    activation     - 激活函数 (GELU, SiLU, ReLU)
    data_movement  - 数据搬移 (Reshape, Transpose, Copy)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

OPERATOR_CATEGORIES = {
    "compute_bound": "计算密集型（GEMM, Attention, BMM; AI≥5）",
    "memory_bound": "访存密集型（Softmax, LayerNorm, Reduction; AI≤1）",
    "activation": "激活函数（GELU, SiLU, ReLU; AI 1~3）",
    "data_movement": "数据搬移（Reshape, Transpose, Copy; AI≈0）",
    "embedding": "嵌入查找",
    "other": "其他",
}


@dataclass
class OperatorSpec:
    name: str
    category: str = "other"
    description: str = ""
    matching_patterns: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    flops_fn: Optional[Callable] = None
    memory_fn: Optional[Callable] = None


class OperatorRegistry:
    def __init__(self):
        self._specs: dict[str, OperatorSpec] = {}
        self._patterns: list[tuple[str, str]] = []

    def register(self, spec: OperatorSpec) -> None:
        self._specs[spec.name] = spec
        for pattern in spec.matching_patterns:
            self._patterns.append((pattern, spec.name))

    def get(self, name: str) -> Optional[OperatorSpec]:
        return self._specs.get(name)

    def lookup(self, target_str: str) -> OperatorSpec:
        target_lower = target_str.lower()
        for pattern, name in self._patterns:
            if (target_lower == pattern or target_lower.endswith(pattern)
                    or pattern in target_lower):
                return self._specs[name]
        name, category = self._parse_target(target_str)
        spec = OperatorSpec(name=name, category=category,
                            description=f"Dynamically recognized: {target_str}",
                            matching_patterns=[target_str])
        self.register(spec)
        return spec

    @staticmethod
    def _parse_target(target_str: str) -> tuple[str, str]:
        name = target_str
        if name.startswith("<") and ">" in name:
            inner = name.split(" ")[-1].rstrip(">")
            return inner.upper(), "data_movement"

        for prefix in ("torch.ops.aten.", "torch.ops.", "aten."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        name = re.split(r"\.\w+", name)[0].upper().strip("_")
        name = name.replace("__AND__", "AND").replace("__OR__", "OR")

        cat_map = [
            (["COPY", "VIEW", "RESHAPE", "TRANSPOSE", "PERMUTE", "SLICE",
              "SELECT", "UNSQUEEZE", "SQUEEZE", "EXPAND", "CAT",
              "CONTIGUOUS", "UNFLATTEN", "INDEX", "T", "CHUNK", "SPLIT",
              "AS_STRIDED", "SIZE", "STRIDE", "DETACH", "ALIAS",
              "NEW_EMPTY", "NEW_ZEROS", "NEW_ONES", "EMPTY", "ZEROS",
              "ONES", "FULL", "ARANGE", "LINSPACE", "RAND", "RANDN",
              "SCALAR_TENSOR", "FILL_", "ZERO_", "COPY_"], "data_movement"),
            (["MM", "MATMUL", "BMM", "LINEAR", "ADDMM", "CONV",
              "CONVOLUTION", "ATTENTION", "FLASH", "ATTN"], "compute_bound"),
            (["ADD", "MUL", "SUB", "DIV", "EQ", "NE", "GT", "LT", "GE", "LE",
              "WHERE", "MASKED_FILL", "CLAMP", "CLIP", "ABS", "NEG", "SIGN",
              "CEIL", "FLOOR", "ROUND", "SQRT", "RSQRT", "EXP", "LOG",
              "POW", "ERF", "RECIP"], "memory_bound"),
            (["SOFTMAX", "LOG_SOFTMAX", "HARDSWISH", "HARDTANH",
              "NORM", "LAYER_NORM", "RMS_NORM", "BATCH_NORM",
              "INSTANCE_NORM"], "memory_bound"),
            (["MEAN", "SUM", "MAX", "MIN", "PROD", "STD", "VAR",
              "ARGMAX", "ARGMIN", "TOPK", "SORT", "CUMSUM",
              "REDUCE"], "memory_bound"),
            (["RELU", "GELU", "SILU", "SIGMOID", "TANH"], "activation"),
            (["BERNOULLI", "DROPOUT", "DROPOUT_"], "other"),
        ]
        for keywords, cat in cat_map:
            if any(kw in name for kw in keywords):
                return name, cat
        return name, "data_movement"

    def get_by_tag(self, tag: str) -> list[OperatorSpec]:
        return [s for s in self._specs.values() if tag in s.tags]

    def list_specs(self) -> list[OperatorSpec]:
        return list(self._specs.values())

    @property
    def num_operators(self) -> int:
        return len(self._specs)

    @staticmethod
    def get_default() -> OperatorRegistry:
        registry = OperatorRegistry()

        registry.register(OperatorSpec(name="UNKNOWN", category="other",
            description="Unrecognized operator", matching_patterns=["unknown"]))

        # ---- compute_bound ----
        registry.register(OperatorSpec(name="LINEAR", category="compute_bound",
            description="y = x @ W^T + b", matching_patterns=["linear", "addmm"]))
        registry.register(OperatorSpec(name="GEMM", category="compute_bound",
            description="C = A @ B", matching_patterns=["mm", "matmul"]))
        registry.register(OperatorSpec(name="BMM", category="compute_bound",
            description="Batch matrix multiply", matching_patterns=["bmm"]))
        registry.register(OperatorSpec(name="ATTENTION", category="compute_bound",
            description="Scaled dot-product attention",
            matching_patterns=["scaled_dot_product_attention"], tags={"attention"}))
        registry.register(OperatorSpec(name="FLASH_ATTENTION", category="compute_bound",
            description="FlashAttention fused kernel",
            matching_patterns=["flash_attention", "flash_attn"], tags={"attention"}))

        # ---- memory_bound ----
        registry.register(OperatorSpec(name="LAYER_NORM", category="memory_bound",
            description="Layer Normalization",
            matching_patterns=["layer_norm", "native_layer_norm"]))
        registry.register(OperatorSpec(name="RMS_NORM", category="memory_bound",
            description="Root Mean Square Normalization",
            matching_patterns=["rms_norm"]))
        registry.register(OperatorSpec(name="SOFTMAX", category="memory_bound",
            description="Softmax normalization",
            matching_patterns=["softmax", "_softmax"]))
        registry.register(OperatorSpec(name="ADD", category="data_movement",
            description="Element-wise addition",
            matching_patterns=["add", "add_", "add.Tensor"]))
        registry.register(OperatorSpec(name="MUL", category="data_movement",
            description="Element-wise multiplication", matching_patterns=["mul"]))
        registry.register(OperatorSpec(name="SUB", category="data_movement",
            description="Element-wise subtraction", matching_patterns=["sub"]))
        registry.register(OperatorSpec(name="DIV", category="data_movement",
            description="Element-wise division", matching_patterns=["div"]))
        registry.register(OperatorSpec(name="MEAN", category="memory_bound",
            description="Mean reduction",
            matching_patterns=["mean.dim", "aten.mean"]))

        # ---- activation ----
        registry.register(OperatorSpec(name="GELU", category="activation",
            description="Gaussian Error Linear Unit", matching_patterns=["gelu"]))
        registry.register(OperatorSpec(name="SILU", category="activation",
            description="SiLU / Swish", matching_patterns=["silu"]))
        registry.register(OperatorSpec(name="RELU", category="activation",
            description="Rectified Linear Unit", matching_patterns=["relu"]))
        registry.register(OperatorSpec(name="SIGMOID", category="activation",
            description="Sigmoid", matching_patterns=["sigmoid"]))

        # ---- data_movement ----
        registry.register(OperatorSpec(name="VIEW", category="data_movement",
            description="Tensor view (no data copy)", matching_patterns=["view"]))
        registry.register(OperatorSpec(name="RESHAPE", category="data_movement",
            description="Tensor reshape", matching_patterns=["reshape"]))
        registry.register(OperatorSpec(name="TRANSPOSE", category="data_movement",
            description="Transpose dimensions", matching_patterns=["transpose"]))
        registry.register(OperatorSpec(name="PERMUTE", category="data_movement",
            description="Permute dimensions", matching_patterns=["permute"]))
        registry.register(OperatorSpec(name="CAT", category="data_movement",
            description="Concatenate", matching_patterns=["cat"]))
        registry.register(OperatorSpec(name="SLICE", category="data_movement",
            description="Slice tensor", matching_patterns=["slice"]))
        registry.register(OperatorSpec(name="EXPAND", category="data_movement",
            description="Expand / broadcast", matching_patterns=["expand"]))
        registry.register(OperatorSpec(name="EMBEDDING", category="embedding",
            description="Token embedding lookup", matching_patterns=["embedding"]))
        registry.register(OperatorSpec(name="DROPOUT", category="other",
            description="Dropout", matching_patterns=["dropout", "dropout_", "native_dropout"]))
        registry.register(OperatorSpec(name="CONTIGUOUS", category="data_movement",
            description="Make contiguous", matching_patterns=["contiguous"]))
        registry.register(OperatorSpec(name="SELECT", category="data_movement",
            description="Select along dim", matching_patterns=["select.int", "aten.select"]))
        registry.register(OperatorSpec(name="UNFLATTEN", category="data_movement",
            description="Unflatten dim", matching_patterns=["unflatten"]))
        registry.register(OperatorSpec(name="SQUEEZE", category="data_movement",
            description="Remove size-1 dims", matching_patterns=["squeeze"]))
        registry.register(OperatorSpec(name="UNSQUEEZE", category="data_movement",
            description="Add size-1 dim", matching_patterns=["unsqueeze"]))
        registry.register(OperatorSpec(name="CLONE", category="data_movement",
            description="Tensor clone", matching_patterns=["clone"]))
        registry.register(OperatorSpec(name="DETACH", category="data_movement",
            description="Detach from graph", matching_patterns=["detach"]))

        return registry
