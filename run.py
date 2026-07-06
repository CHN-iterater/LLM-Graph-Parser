"""LLM Graph Parser — 用 Prompt 驱动计算图解析。

用法:
    1. 选择模式（pytorch / onnx）
    2. 修改下面的配置
    3. python run.py
    4. 结果输出到 output/模型名_时间戳/ 目录

流程: PyTorch 模型 → ONNX（临时文件，自动清理）→ 标准化 graph.json + summary.txt
output 目录只保留每个 Prompt 的最终结果，不保留中间 ONNX 文件。
"""

import os
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from llm_graph_parser import parse_model, parse_onnx


# ====================================================================
# 0. 配置区 — 你只需要改这里
# ====================================================================

MODE = "pytorch"                         # "pytorch" 或 "onnx"

# ---- PyTorch 模式 ----
MODEL_SOURCE = "../Models/gpt2_local"    # 本地模型路径 / HuggingFace 模型名

PROMPTS = [                               # 每条 Prompt 生成一份独立的计算图
    "Hello, how are you?",
    "What is the capital of France?",
    "Tell me something about the Artificial Intelligence.",
]

# ---- ONNX 模式 ----
ONNX_PATH = "path/to/your/model.onnx"


# ====================================================================
# 1. 工具函数
# ====================================================================
def make_output_dir(model_label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = f"{model_label}_{timestamp}"
    os.makedirs(f"output/{dir_name}", exist_ok=True)
    return f"output/{dir_name}"


# ====================================================================
# 2. PyTorch 模式
# ====================================================================
def run_pytorch_mode():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_label = os.path.basename(MODEL_SOURCE.replace("\\", "/"))
    output_dir = make_output_dir(model_label)

    print(f"\n加载模型: {MODEL_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE)
    model.eval()
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # 每条 Prompt 独立导出 ONNX（临时文件，自动清理），
    # output 只保留 graph.json + summary.txt
    results = []
    for i, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        print(f"\n{'=' * 60}")
        print(f"  Prompt [{i}]: \"{prompt}\"")
        print(f"  tokens: {seq_len}")
        print(f"{'=' * 60}")

        graph = parse_model(model, input_ids, model_name=model_label,
                            onnx_path="")  # 空 = 用临时文件，解析完自动删除

        graph.prompt_text = prompt
        graph.prompt_tokens = seq_len

        # 统计
        counts = graph.get_operator_counts()
        total_flops = sum(n.flops for n in graph.nodes)
        print(f"  算子调用: {graph.num_nodes}")
        print(f"  总算力:   {total_flops/1e6:.2f} MFLOPs")
        for op_name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    {op_name:25s}: {cnt}")

        # 保存（多条 Prompt 时用 prompt_0_ / prompt_1_ 区分）
        prefix = f"prompt_{i}" if len(PROMPTS) > 1 else ""
        graph.save_to_json(output_dir, name=prefix)
        graph.save_summary(output_dir, name=prefix)
        results.append((seq_len, total_flops))

    print(f"\n结果目录: {output_dir}/")

    if len(results) > 1:
        print(f"\n  Prompt 长度对比:")
        for seq_len, flops in results:
            print(f"    {seq_len:>4d} tokens  →  {flops/1e6:.2f} MFLOPs")


# ====================================================================
# 3. ONNX 模式
# ====================================================================
def run_onnx_mode():
    model_label = os.path.basename(ONNX_PATH.replace("\\", "/"))
    if model_label.lower().endswith(".onnx"):
        model_label = model_label[:-5]
    output_dir = make_output_dir(model_label)

    graph = parse_onnx(ONNX_PATH, model_name=model_label)

    print(f"\n{'=' * 60}")
    print(f"  ONNX 模型: {ONNX_PATH}")
    print(f"{'=' * 60}")
    print(f"  算子节点数: {graph.num_nodes}")

    counts = graph.get_operator_counts()
    total_flops = sum(n.flops for n in graph.nodes)
    print(f"  总算力: {total_flops/1e6:.2f} MFLOPs")
    for op_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {op_name:25s}: {cnt}")

    graph.save_to_json(output_dir)
    graph.save_summary(output_dir)
    print(f"\n结果目录: {output_dir}/")


# ====================================================================
# 4. 入口
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Graph Parser")
    print(f"  模式: {MODE}")
    print("=" * 60)

    if MODE == "pytorch":
        run_pytorch_mode()
    elif MODE == "onnx":
        run_onnx_mode()
    else:
        print(f"未知模式: {MODE}")
