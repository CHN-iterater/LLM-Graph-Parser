"""FLOPs estimation for common operators.

All functions accept string-based operator type names (matching
OperatorSpec.name from the registry) rather than an enum, making
them extensible for custom operator types.
"""

from __future__ import annotations

from llm_graph_parser.core.operator_node import TensorMeta


def estimate_flops(op_type: str,
                   inputs: list[TensorMeta],
                   outputs: list[TensorMeta]) -> int:
    """Estimate FLOPs for a given operator based on tensor shapes.

    Returns the number of floating-point operations (multiply-add = 2 FLOPs).
    """
    op = op_type.upper()

    if op in ("LINEAR", "GEMM"):
        return _linear_flops(inputs, outputs)
    elif op == "BMM":
        return _bmm_flops(inputs, outputs)
    elif op in ("ATTENTION", "FLASH_ATTENTION"):
        return _attention_flops(inputs, outputs)
    elif op in ("LAYER_NORM", "RMS_NORM"):
        return _norm_flops(inputs, outputs)
    elif op in ("SILU", "GELU", "RELU", "SIGMOID"):
        return _elementwise_flops(inputs, outputs)
    elif op == "SOFTMAX":
        return _softmax_flops(inputs, outputs)
    elif op == "ADD":
        return _add_flops(inputs, outputs)
    elif op in ("EMBEDDING",):
        return 0  # Lookup table, no compute
    elif op in ("RESHAPE", "VIEW", "TRANSPOSE", "PERMUTE", "SLICE", "CAT",
                "CLONE", "DETACH", "CONTIGUOUS", "SELECT", "UNFLATTEN",
                "SQUEEZE", "UNSQUEEZE"):
        return 0  # Data movement, no compute
    elif op == "MEAN":
        return _elementwise_flops(inputs, outputs)  # ~1 read per element
    else:
        return 0


def _safe_numel(shape: tuple[int, ...]) -> int:
    """Compute number of elements, treating -1 (dynamic) as 1."""
    count = 1
    for d in shape:
        if d > 0:
            count *= d
    return count


def _linear_flops(inputs: list[TensorMeta],
                  outputs: list[TensorMeta]) -> int:
    """FLOPs for Linear/GEMM: 2 * M * N * K (multiply-add counted as 2)."""
    if not inputs or not outputs:
        return 0
    in_shape = inputs[0].shape
    out_shape = outputs[0].shape

    if len(in_shape) < 2 or len(out_shape) < 1:
        return 0

    batch_dims = in_shape[:-1]
    K = in_shape[-1] if in_shape[-1] > 0 else 1
    N = out_shape[-1] if out_shape[-1] > 0 else 1

    batch_size = 1
    for d in batch_dims:
        if d > 0:
            batch_size *= d

    return 2 * batch_size * K * N


def _bmm_flops(inputs: list[TensorMeta],
               outputs: list[TensorMeta]) -> int:
    """FLOPs for batched matmul: 2 * B * M * N * K."""
    if len(inputs) < 2:
        return 0
    shape_a = inputs[0].shape
    shape_b = inputs[1].shape
    if len(shape_a) < 3 or len(shape_b) < 3:
        return _linear_flops(inputs, outputs)

    B = shape_a[0] if shape_a[0] > 0 else 1
    M = shape_a[-2] if shape_a[-2] > 0 else 1
    K = shape_a[-1] if shape_a[-1] > 0 else 1
    N = shape_b[-1] if shape_b[-1] > 0 else 1
    return 2 * B * M * K * N


def _attention_flops(inputs: list[TensorMeta],
                     outputs: list[TensorMeta]) -> int:
    """FLOPs for attention: 4 * B * H * T * T * d.

    Simplified estimate for scaled dot-product attention.
    """
    if not inputs:
        return 0
    q_shape = inputs[0].shape
    if len(q_shape) < 4:
        return 0
    dims = [d if d > 0 else 1 for d in q_shape[-4:]]
    B, H, T, d = dims
    return 4 * B * H * T * T * d


def _norm_flops(inputs: list[TensorMeta],
                outputs: list[TensorMeta]) -> int:
    """FLOPs for layer norm / rms norm: ~3 * num_elements."""
    if not inputs:
        return 0
    return 3 * _safe_numel(inputs[0].shape)


def _elementwise_flops(inputs: list[TensorMeta],
                       outputs: list[TensorMeta]) -> int:
    """FLOPs for element-wise activations: ~num_elements."""
    if not inputs:
        return 0
    return _safe_numel(inputs[0].shape)


def _softmax_flops(inputs: list[TensorMeta],
                   outputs: list[TensorMeta]) -> int:
    """FLOPs for softmax: ~5 * num_elements (exp + sum + div)."""
    if not inputs:
        return 0
    return 5 * _safe_numel(inputs[0].shape)


def _add_flops(inputs: list[TensorMeta],
               outputs: list[TensorMeta]) -> int:
    """FLOPs for addition: num_elements."""
    if not inputs:
        return 0
    return _safe_numel(inputs[0].shape)
