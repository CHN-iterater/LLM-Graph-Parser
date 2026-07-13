"""LLM Graph Parser — 高入 API: parse_model() / parse_onnx()."""

from .core import (
    OperatorNode, TensorMeta, OperatorRegistry, OperatorSpec,
    ComputationGraph, PhaseSplitter,
    graph_to_dict, graph_to_json, SCHEMA_VERSION,
)
from .hardware import HardwareProfile, get_profile
from .config import ParserConfig


def _annotate_flops(graph):
    """Annotate each node with estimated FLOPs and memory bytes."""
    from .utils.flops_calculator import estimate_flops
    from .utils.memory_calculator import estimate_memory_bytes
    for node in graph.nodes:
        node.flops = estimate_flops(node.op_type, node.input_tensors, node.output_tensors)
        node.memory_bytes = estimate_memory_bytes(
            node.op_type, node.input_tensors, node.output_tensors)


def parse_model(model, *example_args, model_name: str = "model",
                registry=None, onnx_path: str = "") -> ComputationGraph:
    """Export a PyTorch model to ONNX and parse its computation graph."""
    import os, tempfile, warnings
    import torch
    from .parser.onnx_parser import OnnxParser

    warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
    warnings.filterwarnings("ignore", message=".*torchvision.*")

    if registry is None:
        registry = OperatorRegistry.get_default()

    model.eval()
    old_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        old_cache = model.config.use_cache
        model.config.use_cache = False

    # 注册 DynamicCache 为 pytree 节点，兼容 torch.export（Gemma 等模型需要）
    try:
        from transformers.cache_utils import DynamicCache
        import torch.utils._pytree as pytree
        def _dc_flatten(c):
            d = c.__dict__
            k = d.get('_key_cache') or d.get('key_cache', [])
            v = d.get('_value_cache') or d.get('value_cache', [])
            return ([k, v], None)
        def _dc_unflatten(v, _):
            cache = DynamicCache()
            d = cache.__dict__
            if '_key_cache' in d:
                d['_key_cache'] = v[0]; d['_value_cache'] = v[1]
            else:
                d['key_cache'] = v[0]; d['value_cache'] = v[1]
            return cache
        pytree.register_pytree_node(DynamicCache, _dc_flatten, _dc_unflatten)
    except Exception:
        pass

    use_tmp = not onnx_path
    if use_tmp:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = tmp.name
        tmp.close()
    else:
        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    graph = None
    try:
        torch.onnx.export(
            model, example_args, onnx_path, opset_version=18,
            input_names=["input"], output_names=["output"],
            external_data=False)
        parser = OnnxParser(registry=registry)
        parser.load(onnx_path)
        graph = parser.parse(model_name=model_name)
    except Exception as e:
        if "GuardOnDataDependentSymNode" in type(e).__name__:
            raise RuntimeError(
                f"Model {model_name} cannot export to ONNX. "
                "The model has data-dependent control flow "
                "(common in RoPE dynamic frequency update). "
                "Use a standard model like GPT-2 / LLaMA / Qwen.")
        print(f"  [export] FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        if old_cache is not None:
            model.config.use_cache = old_cache
        if use_tmp:
            for ext in ("", ".data"):
                f = onnx_path + ext
                if os.path.exists(f):
                    os.unlink(f)

    _annotate_flops(graph)
    return graph


def parse_onnx(path: str, model_name: str = "", registry=None) -> ComputationGraph:
    """Load an ``.onnx`` file and parse its computation graph directly."""
    from .parser.onnx_parser import OnnxParser

    if registry is None:
        registry = OperatorRegistry.get_default()
    parser = OnnxParser(registry=registry)
    parser.load(path)
    graph = parser.parse(model_name=model_name)
    _annotate_flops(graph)
    return graph


__all__ = [
    "parse_model", "parse_onnx",
    "OperatorNode", "TensorMeta", "OperatorRegistry", "OperatorSpec",
    "ComputationGraph", "PhaseSplitter",
    "HardwareProfile", "get_profile", "ParserConfig",
    "graph_to_dict", "graph_to_json", "SCHEMA_VERSION",
]
