"""LLM Graph Parser - 大语言模型计算图的层级化解析工具。"""

from .core import (
    OperatorNode, TensorMeta, OperatorRegistry, OperatorSpec,
    ComputationGraph, PhaseSplitter,
    graph_to_dict, graph_to_json, SCHEMA_VERSION,
)
from .hardware import HardwareProfile, get_profile
from .config import ParserConfig


def parse_model(model, *example_args, model_name: str = "model",
                registry=None, onnx_path: str = "") -> ComputationGraph:
    """解析 PyTorch 模型的计算图。"""
    import os, tempfile, warnings
    import torch
    from .parser.onnx_parser import OnnxParser
    from .utils.flops_calculator import estimate_flops
    from .utils.memory_calculator import estimate_memory_bytes

    warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
    warnings.filterwarnings("ignore", message=".*torchvision.*")

    if registry is None:
        registry = OperatorRegistry.get_default()

    model.eval()
    old_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        old_cache = model.config.use_cache
        model.config.use_cache = False

    use_tmp = not onnx_path
    if use_tmp:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = tmp.name
        tmp.close()
    else:
        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    # 导出 ONNX
    graph = None
    try:
        torch.onnx.export(
            model, example_args, onnx_path, opset_version=18,
            input_names=["input"], output_names=["output"],
            external_data=False)  # 权重嵌入 .onnx，不产生 .onnx.data
        parser = OnnxParser(registry=registry)
        parser.load(onnx_path)
        graph = parser.parse(model_name=model_name)
    except Exception as e:
        if "GuardOnDataDependentSymNode" in type(e).__name__:
            print("  [export] FAILED: model has data-dependent control flow")
            print("            (common in RoPE dynamic frequency update, e.g. HunYuan)")
            print("            Suggestion: use GPT-2/LLaMA/Qwen standard models")
            raise RuntimeError(
                "Model " + model_name + " cannot export to ONNX. "
                "Data-dependent control flow detected. "
                "PyTorch 2.12 torch.export does not support this pattern.")
        else:
            print(f"  [export] FAILED: {type(e).__name__}: {e}")
            raise
    finally:
        # 无论成功还是失败，都清理临时文件
        if old_cache is not None:
            model.config.use_cache = old_cache
        if use_tmp:
            for ext in ("", ".data"):
                f = onnx_path + ext
                if os.path.exists(f):
                    os.unlink(f)

    for node in graph.nodes:
        node.flops = estimate_flops(node.op_type, node.input_tensors, node.output_tensors)
        node.memory_bytes = estimate_memory_bytes(
            node.op_type, node.input_tensors, node.output_tensors)
    return graph


def parse_onnx(path: str, model_name: str = "", registry=None) -> ComputationGraph:
    """直接解析 .onnx 文件的计算图。"""
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
            node.op_type, node.input_tensors, node.output_tensors)
    return graph


__all__ = [
    "parse_model", "parse_onnx",
    "OperatorNode", "TensorMeta", "OperatorRegistry", "OperatorSpec",
    "ComputationGraph", "PhaseSplitter",
    "HardwareProfile", "get_profile", "ParserConfig",
    "graph_to_dict", "graph_to_json", "SCHEMA_VERSION",
]
