"""Memory access (bytes) estimation for common operators.

All functions accept string-based operator type names for extensibility.
"""

from __future__ import annotations

from llm_graph_parser.core.operator_node import TensorMeta

# Bytes per element for common dtypes
DTYPE_BYTES = {
    "float32": 4, "float": 4, "torch.float32": 4, "torch.float": 4,
    "float16": 2, "half": 2, "torch.float16": 2, "torch.half": 2,
    "bfloat16": 2, "torch.bfloat16": 2,
    "int64": 8, "torch.int64": 8,
    "int32": 4, "torch.int32": 4,
    "int8": 1, "torch.int8": 1,
    "uint8": 1, "torch.uint8": 1,
}


def _dtype_bytes(dtype_str: str) -> int:
    return DTYPE_BYTES.get(dtype_str, 4)


def _numel(shape: tuple[int, ...]) -> int:
    count = 1
    for d in shape:
        if d > 0:
            count *= d
    return count


def estimate_memory_bytes(op_type: str,
                          inputs: list[TensorMeta],
                          outputs: list[TensorMeta]) -> int:
    """Estimate memory access in bytes for a given operator.

    Accounts for reading inputs and writing outputs.
    """
    op = op_type.upper()

    if op == "LINEAR":
        return _linear_memory(inputs, outputs)
    elif op in ("GEMM", "BMM"):
        return _matmul_memory(inputs, outputs)
    elif op in ("ATTENTION", "FLASH_ATTENTION"):
        return _attention_memory(inputs, outputs)
    elif op in ("LAYER_NORM", "RMS_NORM"):
        return _norm_memory(inputs, outputs)
    elif op in ("SILU", "GELU", "RELU", "SIGMOID", "SOFTMAX"):
        return _elementwise_memory(inputs, outputs)
    elif op == "ADD":
        return _elementwise_memory(inputs, outputs)
    elif op == "EMBEDDING":
        return 0
    elif op in ("RESHAPE", "VIEW", "TRANSPOSE", "PERMUTE", "SLICE", "EXPAND",
                "CLONE", "DETACH"):
        return 0
    elif op == "CAT":
        total = 0
        for t in inputs:
            total += _numel(t.shape) * _dtype_bytes(t.dtype)
        if outputs:
            total += _numel(outputs[0].shape) * _dtype_bytes(outputs[0].dtype)
        return total
    else:
        total = 0
        for t in inputs:
            total += _numel(t.shape) * _dtype_bytes(t.dtype)
        for t in outputs:
            total += _numel(t.shape) * _dtype_bytes(t.dtype)
        return total


def _linear_memory(inputs: list[TensorMeta],
                   outputs: list[TensorMeta]) -> int:
    total = 0
    for t in inputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    for t in outputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    return total


def _matmul_memory(inputs: list[TensorMeta],
                   outputs: list[TensorMeta]) -> int:
    total = 0
    for t in inputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    for t in outputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    return total


def _attention_memory(inputs: list[TensorMeta],
                      outputs: list[TensorMeta]) -> int:
    total = 0
    for t in inputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    for t in outputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    return total


def _norm_memory(inputs: list[TensorMeta],
                 outputs: list[TensorMeta]) -> int:
    total = 0
    for t in inputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    for t in outputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    return total


def _elementwise_memory(inputs: list[TensorMeta],
                        outputs: list[TensorMeta]) -> int:
    total = 0
    for t in inputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    for t in outputs:
        total += _numel(t.shape) * _dtype_bytes(t.dtype)
    return total
