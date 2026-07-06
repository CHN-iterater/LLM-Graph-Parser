"""LLM Graph Parser - 大语言模型计算图的层级化解析工具。

一句话用法::

    from llm_graph_parser import parse_model, parse_onnx

    # 场景 A: 你有 PyTorch 模型（HuggingFace 等）
    graph = parse_model(model, dummy_input)

    # 场景 B: 你有 .onnx 文件
    graph = parse_onnx("model.onnx")

    # 查看结果
    graph.print_summary()
    graph.save_to_json("./output")

你需要准备:
    PyTorch 场景: 模型本身 + 一个形状正确的假输入 (torch.randint(...))
    ONNX  场景:   .onnx 文件路径
"""

from .core import (
    OperatorNode,
    TensorMeta,
    OperatorRegistry,
    OperatorSpec,
    ComputationGraph,
    PhaseSplitter,
    graph_to_dict,
    graph_to_json,
    SCHEMA_VERSION,
)
from .hardware import HardwareProfile, get_profile
from .config import ParserConfig


def parse_model(model, *example_args, model_name: str = "model",
                registry=None, onnx_path: str = "") -> ComputationGraph:
    """解析 PyTorch 模型的计算图。

    流程: PyTorch 模型 → 导出为 ONNX → OnnxParser → ComputationGraph
    ONNX 是统一的中间表示，后续所有分析基于 ONNX 进行。

    Args:
        model:        torch.nn.Module，已处于 eval 模式。
        example_args: 与模型 forward() 签名一致的假输入（tuple of tensors）。
        model_name:   给你的模型起个名字（用于输出文件名）。
        registry:     OperatorRegistry，默认使用内置的。
        onnx_path:    ONNX 文件保存路径。若文件已存在则跳过导出（复用）。
                      为空则用临时文件（不保留 .onnx）。

    Returns:
        ComputationGraph，包含算子 DAG。
    """
    import os, tempfile
    import torch
    from .parser.onnx_parser import OnnxParser
    from .utils.flops_calculator import estimate_flops
    from .utils.memory_calculator import estimate_memory_bytes

    if registry is None:
        registry = OperatorRegistry.get_default()

    model.eval()

    # HuggingFace 模型的 use_cache 与 ONNX 导出不兼容，临时禁掉
    old_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        old_cache = model.config.use_cache
        model.config.use_cache = False

    if not onnx_path:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = tmp.name
        tmp.close()
    else:
        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    try:
        torch.onnx.export(
            model,
            args=example_args,
            f=onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
    finally:
        if old_cache is not None:
            model.config.use_cache = old_cache

    # 用 OnnxParser 解析
    parser = OnnxParser(registry=registry)
    parser.load(onnx_path)
    graph = parser.parse(model_name=model_name)

    # 标注 FLOPs / Memory
    for node in graph.nodes:
        node.flops = estimate_flops(node.op_type, node.input_tensors, node.output_tensors)
        node.memory_bytes = estimate_memory_bytes(
            node.op_type, node.input_tensors, node.output_tensors
        )

    return graph


def parse_onnx(path: str, model_name: str = "", registry=None) -> ComputationGraph:
    """直接解析 .onnx 文件的计算图（无需 PyTorch 模型）。

    Args:
        path:        .onnx 文件路径。
        model_name:  给你的模型起个名字（可选）。
        registry:    OperatorRegistry，默认使用内置的。

    Returns:
        ComputationGraph，包含算子 DAG。

    示例::

        graph = parse_onnx("llama.onnx", model_name="Llama")
    """
    from .parser.onnx_parser import OnnxParser
    from .utils.flops_calculator import estimate_flops
    from .utils.memory_calculator import estimate_memory_bytes

    if registry is None:
        registry = OperatorRegistry.get_default()

    parser = OnnxParser(registry=registry)
    parser.load(path)
    graph = parser.parse(model_name=model_name)

    for node in graph.nodes:
        node.flops = estimate_flops(node.op_type, node.input_tensors, node.output_tensors)
        node.memory_bytes = estimate_memory_bytes(
            node.op_type, node.input_tensors, node.output_tensors
        )

    return graph


__all__ = [
    # 高人口 API
    "parse_model",
    "parse_onnx",
    # 核心类
    "OperatorNode",
    "TensorMeta",
    "OperatorRegistry",
    "OperatorSpec",
    "ComputationGraph",
    "PhaseSplitter",
    "HardwareProfile",
    "get_profile",
    "ParserConfig",
    "graph_to_dict",
    "graph_to_json",
    "SCHEMA_VERSION",
]
