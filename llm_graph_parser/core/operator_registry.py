"""Extensible operator registry with pattern-based matching.

Design:
- Operators register themselves with matching patterns and estimation functions
- New operator types can be added at runtime without modifying core code
- Each operator spec defines: name, category, matching patterns, FLOPs/memory estimators
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---- Categories ----

OPERATOR_CATEGORIES = {
    "compute": "密集型计算（GEMM, Attention 等）",
    "memory": "访存密集型（KV Cache, Data Movement 等）",
    "normalization": "归一化（LayerNorm, RMSNorm 等）",
    "activation": "激活函数（ReLU, GELU, SiLU 等）",
    "elementwise": "逐元素运算（Add, Mul 等）",
    "data_movement": "数据搬移（Reshape, Transpose, View 等）",
    "embedding": "嵌入查找",
    "other": "其他",
}


@dataclass
class OperatorSpec:
    """Specification for a single operator type.

    To add a new operator, create an OperatorSpec and register it::

        registry = OperatorRegistry.get_default()
        registry.register(OperatorSpec(
            name="MY_OP",
            category="compute",
            matching_patterns=["my_op", "torch.ops.aten.my_op"],
            flops_fn=lambda inputs, outputs: 0,
            memory_fn=lambda inputs, outputs: 0,
            description="My custom operator",
        ))
    """
    name: str
    category: str = "other"
    description: str = ""
    matching_patterns: list[str] = field(default_factory=list)
    flops_fn: Optional[Callable] = None
    memory_fn: Optional[Callable] = None


class OperatorRegistry:
    """Registry for operator types with pattern-based matching.

    Provides lookup from torch.export node targets to OperatorSpec,
    and serves as the single source of truth for all known operators.
    """

    def __init__(self):
        self._specs: dict[str, OperatorSpec] = {}
        self._patterns: list[tuple[str, str]] = []  # (pattern, name)

    def register(self, spec: OperatorSpec) -> None:
        """Register an operator spec.

        If a spec with the same name already exists, it is overwritten.
        """
        self._specs[spec.name] = spec
        for pattern in spec.matching_patterns:
            self._patterns.append((pattern, spec.name))

    def get(self, name: str) -> Optional[OperatorSpec]:
        return self._specs.get(name)

    def lookup(self, target_str: str) -> OperatorSpec:
        """Match a torch operator target string to an OperatorSpec.

        Matching logic:
        1. Try registered patterns (exact / suffix / substring).
        2. If no match, dynamically extract a name from the target string,
           infer a category, and register it on the fly.
           This guarantees every op gets a name — no UNKNOWN.
        """
        target_lower = target_str.lower()
        for pattern, name in self._patterns:
            if (
                target_lower == pattern
                or target_lower.endswith(pattern)
                or pattern in target_lower
            ):
                return self._specs[name]

        # ---- Dynamic fallback: derive name from target string ----
        name, category = self._parse_target(target_str)
        spec = OperatorSpec(
            name=name,
            category=category,
            description=f"Dynamically recognized: {target_str}",
            matching_patterns=[target_str],
        )
        self.register(spec)
        return spec

    @staticmethod
    def _parse_target(target_str: str) -> tuple[str, str]:
        """Extract an operator name and infer category from a target string.

        ``aten.copy_.default`` → ``"COPY_"``, ``"data_movement"``
        ``<built-in function getitem>`` → ``"GETITEM"``, ``"data_movement"``
        """
        name = target_str

        # Special: <built-in function xxx>
        if name.startswith("<") and ">" in name:
            inner = name.split(" ")[-1].rstrip(">")
            name = inner.upper()
            return name, "data_movement"

        # Remove common torch prefixes
        for prefix in ("torch.ops.aten.", "torch.ops.", "aten."):
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Remove trailing .default, .Tensor, .Scalar etc.
        name = re.split(r"\.\w+", name)[0]
        name = name.upper().strip("_")

        # Clean up common ugly patterns
        name = name.replace("__AND__", "AND").replace("__OR__", "OR")

        # --- Infer category from name ---
        cat_map = [
            (["COPY", "VIEW", "RESHAPE", "TRANSPOSE", "PERMUTE", "SLICE",
              "SELECT", "UNSQUEEZE", "SQUEEZE", "EXPAND", "CAT",
              "CONTIGUOUS", "UNFLATTEN", "INDEX", "T", "CHUNK", "SPLIT",
              "AS_STRIDED", "SIZE", "STRIDE", "DETACH", "ALIAS",
              "NEW_EMPTY", "NEW_ZEROS", "NEW_ONES", "EMPTY", "ZEROS",
              "ONES", "FULL", "ARANGE", "LINSPACE", "RAND", "RANDN",
              "SCALAR_TENSOR"], "data_movement"),
            (["ADD", "MUL", "SUB", "DIV", "EQ", "NE", "GT", "LT", "GE", "LE",
              "WHERE", "MASKED_FILL", "CLAMP", "CLIP", "ABS", "NEG",
              "SIGN", "CEIL", "FLOOR", "ROUND", "SQRT", "RSQRT",
              "EXP", "LOG", "POW", "ERF", "RECIP"], "elementwise"),
            (["RELU", "GELU", "SILU", "SIGMOID", "TANH", "SOFTMAX",
              "LOG_SOFTMAX", "HARDSWISH", "HARDTANH"], "activation"),
            (["NORM", "LAYER_NORM", "RMS_NORM", "BATCH_NORM",
              "INSTANCE_NORM"], "normalization"),
            (["MM", "MATMUL", "BMM", "LINEAR", "ADDMM", "CONV",
              "CONVOLUTION"], "compute"),
            (["MEAN", "SUM", "MAX", "MIN", "PROD", "STD", "VAR",
              "ARGMAX", "ARGMIN", "TOPK", "SORT", "CUMSUM"], "reduction"),
            (["BERNOULLI", "DROPOUT", "DROPOUT_"], "stochastic"),
            (["FILL_", "ZERO_", "COPY_"], "inplace"),
        ]
        for keywords, category in cat_map:
            if any(kw in name for kw in keywords):
                return name, category
        return name, "other"

    def list_specs(self) -> list[OperatorSpec]:
        return list(self._specs.values())

    @property
    def num_operators(self) -> int:
        return len(self._specs)

    @staticmethod
    def get_default() -> OperatorRegistry:
        """Create the default registry with all known PyTorch operators.

        This is the main extension point: call ``register()`` on the
        returned registry to add custom operators before parsing.
        """
        registry = OperatorRegistry()

        # ---- UNKNOWN (fallback) ----
        registry.register(OperatorSpec(
            name="UNKNOWN", category="other",
            description="Unrecognized operator",
            matching_patterns=["unknown"],
        ))

        # ---- Compute-intensive ----
        registry.register(OperatorSpec(
            name="LINEAR", category="compute",
            description="Fully connected layer: y = x @ W^T + b",
            matching_patterns=["linear", "addmm"],
        ))
        registry.register(OperatorSpec(
            name="GEMM", category="compute",
            description="General matrix multiply: C = A @ B",
            matching_patterns=["mm", "matmul"],
        ))
        registry.register(OperatorSpec(
            name="BMM", category="compute",
            description="Batch matrix multiply",
            matching_patterns=["bmm"],
        ))
        registry.register(OperatorSpec(
            name="ATTENTION", category="compute",
            description="Scaled dot-product attention (torch implementation)",
            matching_patterns=["scaled_dot_product_attention"],
        ))
        registry.register(OperatorSpec(
            name="FLASH_ATTENTION", category="compute",
            description="FlashAttention (fused attention kernel)",
            matching_patterns=["flash_attention", "flash_attn"],
        ))

        # ---- Normalization ----
        registry.register(OperatorSpec(
            name="LAYER_NORM", category="normalization",
            description="Layer Normalization",
            matching_patterns=["layer_norm", "native_layer_norm"],
        ))
        registry.register(OperatorSpec(
            name="RMS_NORM", category="normalization",
            description="Root Mean Square Normalization",
            matching_patterns=["rms_norm"],
        ))

        # ---- Activation functions ----
        registry.register(OperatorSpec(
            name="GELU", category="activation",
            description="Gaussian Error Linear Unit",
            matching_patterns=["gelu"],
        ))
        registry.register(OperatorSpec(
            name="SILU", category="activation",
            description="Sigmoid Linear Unit (SiLU / Swish)",
            matching_patterns=["silu"],
        ))
        registry.register(OperatorSpec(
            name="RELU", category="activation",
            description="Rectified Linear Unit",
            matching_patterns=["relu"],
        ))
        registry.register(OperatorSpec(
            name="SOFTMAX", category="activation",
            description="Softmax normalization",
            matching_patterns=["softmax", "_softmax"],
        ))
        registry.register(OperatorSpec(
            name="SIGMOID", category="activation",
            description="Sigmoid activation",
            matching_patterns=["sigmoid"],
        ))

        # ---- Element-wise ----
        registry.register(OperatorSpec(
            name="ADD", category="elementwise",
            description="Element-wise addition",
            matching_patterns=["add", "add_", "add.Tensor"],
        ))
        registry.register(OperatorSpec(
            name="MUL", category="elementwise",
            description="Element-wise multiplication",
            matching_patterns=["mul"],
        ))
        registry.register(OperatorSpec(
            name="SUB", category="elementwise",
            description="Element-wise subtraction",
            matching_patterns=["sub"],
        ))
        registry.register(OperatorSpec(
            name="DIV", category="elementwise",
            description="Element-wise division",
            matching_patterns=["div"],
        ))

        # ---- Data movement ----
        registry.register(OperatorSpec(
            name="VIEW", category="data_movement",
            description="Tensor view (no data copy)",
            matching_patterns=["view"],
        ))
        registry.register(OperatorSpec(
            name="RESHAPE", category="data_movement",
            description="Tensor reshape (may copy data)",
            matching_patterns=["reshape"],
        ))
        registry.register(OperatorSpec(
            name="TRANSPOSE", category="data_movement",
            description="Transpose two dimensions",
            matching_patterns=["transpose"],
        ))
        registry.register(OperatorSpec(
            name="PERMUTE", category="data_movement",
            description="Permute multiple dimensions",
            matching_patterns=["permute"],
        ))
        registry.register(OperatorSpec(
            name="CAT", category="data_movement",
            description="Concatenate tensors along a dimension",
            matching_patterns=["cat"],
        ))
        registry.register(OperatorSpec(
            name="SLICE", category="data_movement",
            description="Slice a tensor",
            matching_patterns=["slice"],
        ))
        registry.register(OperatorSpec(
            name="EXPAND", category="data_movement",
            description="Expand tensor dimensions (broadcast)",
            matching_patterns=["expand"],
        ))

        # ---- Embedding ----
        registry.register(OperatorSpec(
            name="EMBEDDING", category="embedding",
            description="Token embedding lookup",
            matching_patterns=["embedding"],
        ))

        # ---- Other ----
        registry.register(OperatorSpec(
            name="DROPOUT", category="other",
            description="Dropout regularization",
            matching_patterns=["dropout", "dropout_", "native_dropout"],
        ))

        # ---- Data movement (common aten ops from torch.export) ----
        registry.register(OperatorSpec(
            name="CONTIGUOUS", category="data_movement",
            description="Make tensor contiguous in memory",
            matching_patterns=["contiguous"],
        ))
        registry.register(OperatorSpec(
            name="SELECT", category="data_movement",
            description="Select a slice along a dimension",
            matching_patterns=["select.int", "aten.select"],
        ))
        registry.register(OperatorSpec(
            name="UNFLATTEN", category="data_movement",
            description="Unflatten a tensor dimension",
            matching_patterns=["unflatten"],
        ))
        registry.register(OperatorSpec(
            name="SQUEEZE", category="data_movement",
            description="Remove dimensions of size 1",
            matching_patterns=["squeeze"],
        ))
        registry.register(OperatorSpec(
            name="UNSQUEEZE", category="data_movement",
            description="Add a dimension of size 1",
            matching_patterns=["unsqueeze"],
        ))

        # ---- Reduction ----
        registry.register(OperatorSpec(
            name="MEAN", category="elementwise",
            description="Mean reduction",
            matching_patterns=["mean.dim", "aten.mean"],
        ))
        registry.register(OperatorSpec(
            name="CLONE", category="data_movement",
            description="Tensor clone (data copy)",
            matching_patterns=["clone"],
        ))
        registry.register(OperatorSpec(
            name="DETACH", category="data_movement",
            description="Tensor detach from computation graph",
            matching_patterns=["detach"],
        ))

        return registry
