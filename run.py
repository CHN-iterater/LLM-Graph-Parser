"""LLM Graph Parser — 完整推理流程阶段分析。

流程:
  1. Prefill: 输入完整 Prompt → 导出 ONNX → 解析 → 标记为 Prefill
  2. Decode:  取最后 1 个 token 模拟单步生成 → 导出 ONNX → 解析 → 标记为 Decode
  3. 生成:     model.generate() 获取实际回答长度
  4. 汇总:     Prefill + Decode per-token × 实际生成数 的完整管线统计

output 目录:
  ├── *graph.json              ← 全量图
  ├── *prefill_graph.json      ← Prefill 子图
  ├── *decode_graph.json       ← Decode 子图
  ├── *summary.txt             ← 管线统计摘要
  └── *phase_report.txt        ← 两阶段对比 + Roofline
"""

import os
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "D:/Hugging Face")          # 模型缓存目录

from llm_graph_parser import parse_model, parse_onnx
from llm_graph_parser.core.phase_splitter import PhaseSplitter


# ====================================================================
# 0. 配置区 — 你只需要改这里
# ====================================================================

MODE = "pytorch"                         # "pytorch" 或 "onnx"

# ---- PyTorch 模式 ----
MODEL_SOURCE = "../Models/gpt2"    # 本地路径 / Hugging Face 模型名

PROMPTS = [                               # 每条 Prompt 独立分析
    "Hello, how are you?",
    "What is the capital of France?",
    "A car travels from A to B at 40 km/h and returns at 60 km/h. What is the average speed for the whole trip?",
]

MAX_NEW_TOKENS = 20                       # Decode 阶段最大生成 token 数
SKIP_GENERATION = False                    # True = 跳过生成阶段，只解析计算图
TRUST_REMOTE_CODE = True

# ---- ONNX 模式 ----
ONNX_PATH = "path/to/your/model.onnx"

# ---- 硬件参数（用于 Roofline 分析） ----
HARDWARE = {"peak_flops": 1979e12, "memory_bw": 3350e9}  # H100


# ====================================================================
# 1. 工具函数
# ====================================================================
def make_output_dir(model_label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = f"{model_label}_{timestamp}"
    os.makedirs(f"output/{dir_name}", exist_ok=True)
    return f"output/{dir_name}"


# ====================================================================
# 2. 完整管线: Prefill + Decode + 生成
# ====================================================================
def run_pytorch_mode():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_label = os.path.basename(MODEL_SOURCE.replace("\\", "/"))
    output_dir = make_output_dir(model_label)

    print(f"\n加载模型: {MODEL_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE)
    model.eval()
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()):,}")

    for i, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        prompt_len = prompt_ids.shape[1]
        prefix = f"prompt_{i}" if len(PROMPTS) > 1 else ""

        print(f"\n{'=' * 60}")
        print(f"  Prompt [{i}]: \"{prompt}\"")
        print(f"{'=' * 60}")

        # ================================================================
        # Step 1: Prefill — 完整 Prompt 前向
        # ================================================================
        print(f"\n  [Phase 1/3] Prefill (prompt_len={prompt_len})")
        prefill_graph = parse_model(model, prompt_ids, model_name=model_label,
                                    onnx_path="")
        prefill_graph.prompt_text = prompt
        prefill_graph.prompt_tokens = prompt_len
        prefill_graph.tag_unassigned_as("prefill")

        pf_stats = prefill_graph.get_stage_stats("prefill")
        print(f"    ops={pf_stats['num_ops']}, "
              f"FLOPs={pf_stats['total_flops']/1e6:.2f}M, "
              f"Mem={pf_stats['total_memory_bytes']/1e6:.2f}MB, "
              f"AI={pf_stats['arith_intensity']:.2f}")

        # ================================================================
        # Step 2: Decode — 模拟单 token 生成
        # ================================================================
        print(f"  [Phase 2/3] Decode (1 token)")
        decode_token = prompt_ids[:, -1:]  # 取最后 1 个 token
        decode_graph = parse_model(model, decode_token, model_name=model_label,
                                   onnx_path="")
        decode_graph.prompt_text = f"[decode step] {prompt}"
        decode_graph.prompt_tokens = 1
        decode_graph.tag_unassigned_as("decode")

        dc_stats = decode_graph.get_stage_stats("decode")
        dc_per_token_flops = dc_stats["total_flops"]
        print(f"    per-step: ops={dc_stats['num_ops']}, "
              f"FLOPs={dc_stats['total_flops']/1e6:.2f}M, "
              f"AI={dc_stats['arith_intensity']:.2f}")

        # ================================================================
        # Step 3: 实际生成 — 确定回答了多长
        # ================================================================
        if SKIP_GENERATION:
            print(f"  [Phase 3/3] Skip generation (SKIP_GENERATION=True)")
            gen_len = 0
            answer = ""
        else:
            print(f"  [Phase 3/3] Generating (max {MAX_NEW_TOKENS} tokens)...")
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                for k, v in model.generation_config.to_dict().items():
                    if (v is not None and k not in gen_kwargs
                            and k not in ("_from_model_config", "transformers_version")):
                        gen_kwargs[k] = v
                gen_out = model.generate(prompt_ids, **gen_kwargs)
            gen_len = gen_out.shape[1] - prompt_len
            answer = tokenizer.decode(gen_out[0, prompt_len:],
                                      skip_special_tokens=True).strip()
            print(f"    generated: {gen_len} tokens")
            print(f"    answer: \"{answer[:120]}{'...' if len(answer) > 120 else ''}\"")

            if gen_len == 0:
                print("    ⚠ 模型未生成任何 token（跳过 Decode 统计）")
                print("      可能原因：Prompt 语言与模型不匹配，或模型需要特殊对话格式")

        # ================================================================
        # 汇总计算
        # ================================================================
        pf_flops = pf_stats["total_flops"]
        pf_mem = pf_stats["total_memory_bytes"]
        dc_flops_total = dc_per_token_flops * gen_len
        dc_mem_total = dc_stats["total_memory_bytes"] * gen_len
        total_flops = pf_flops + dc_flops_total
        total_mem = pf_mem + dc_mem_total

        print(f"\n  --- Pipeline Summary ---")
        print(f"  {'':20s} {'Ops':>8s} {'FLOPs':>15s} {'Mem(MB)':>10s}")
        print(f"  {'-' * 56}")
        print(f"  {'Prefill':20s} {pf_stats['num_ops']:>8d} "
              f"{pf_flops:>15,} {pf_mem/1e6:>10.2f}")
        if gen_len > 0:
            print(f"  {'Decode ×' + str(gen_len):20s} "
                  f"{dc_stats['num_ops']:>8d} "
                  f"{dc_flops_total:>15,} {dc_mem_total/1e6:>10.2f}")
        print(f"  {'-' * 56}")
        print(f"  {'Total':20s} {pf_stats['num_ops'] + dc_stats['num_ops'] * gen_len:>8d} "
              f"{total_flops:>15,} {total_mem/1e6:>10.2f}")

        # ================================================================
        # 保存
        # ================================================================
        # 合并两图到一个 graph（用于完整图输出）
        combined = prefill_graph
        for node in decode_graph.nodes:
            node.op_id = f"dc_{node.op_id}"  # 避免 op_id 冲突
            combined.add_node(node)

        combined.save_to_json(output_dir, name=prefix)
        prefill_graph.save_to_json(output_dir, name=f"{prefix}_prefill" if prefix else "prefill")
        if gen_len > 0:
            decode_graph.save_to_json(output_dir, name=f"{prefix}_decode" if prefix else "decode")

        # 带管线统计的摘要
        summary_lines = []
        summary_lines.append(f"Model: {model_label}")
        summary_lines.append(f"Prompt: \"{prompt}\"")
        summary_lines.append(f"Answer: \"{answer}\"")
        summary_lines.append(f"Prompt tokens: {prompt_len}  |  Generated tokens: {gen_len}")
        summary_lines.append("")
        summary_lines.append(f"{'Phase':20s} {'Ops':>8s} {'FLOPs':>15s} {'Mem(MB)':>10s} {'AI':>8s}")
        summary_lines.append("-" * 64)
        summary_lines.append(f"{'Prefill':20s} {pf_stats['num_ops']:>8d} "
                             f"{pf_flops:>15,} {pf_mem/1e6:>10.2f} "
                             f"{pf_stats['arith_intensity']:>8.2f}")
        if gen_len > 0:
            summary_lines.append(f"{'Decode ×' + str(gen_len):20s} "
                                 f"{dc_stats['num_ops']:>8d} "
                                 f"{dc_flops_total:>15,} {dc_mem_total/1e6:>10.2f} "
                                 f"{dc_stats['arith_intensity']:>8.2f}")
        summary_lines.append("-" * 64)
        summary_lines.append(f"{'Total':20s} "
                             f"{pf_stats['num_ops'] + dc_stats['num_ops'] * gen_len:>8d} "
                             f"{total_flops:>15,} {total_mem/1e6:>10.2f}")
        summary_lines.append("")
        summary_lines.append("Operator counts:")
        for op_name, cnt in sorted(combined.get_operator_counts().items(),
                                    key=lambda x: -x[1]):
            summary_lines.append(f"  {op_name:25s}: {cnt}")

        output_dir_p = Path(output_dir)
        stem = prefix or "graph"
        with open(output_dir_p / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print("\n".join(summary_lines))

        # 阶段对比报告
        combined.save_phase_report(output_dir, name=prefix,
                                   hardware_profile=HARDWARE)

    print(f"\n所有结果已保存到: {output_dir}/")


from pathlib import Path


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

    splitter = PhaseSplitter()
    prefill_g, decode_g = splitter.split_by_layer_pattern(graph)
    graph.tag_unassigned_as("prefill")

    pf_stats = graph.get_stage_stats("prefill")
    dc_stats = graph.get_stage_stats("decode")
    print(f"  Prefill: {pf_stats['num_ops']} ops, "
          f"{pf_stats['total_flops']/1e6:.2f} MFLOPs, "
          f"AI={pf_stats['arith_intensity']:.2f}")
    if dc_stats["num_ops"] > 0:
        print(f"  Decode:  {dc_stats['num_ops']} ops, "
              f"{dc_stats['total_flops']/1e6:.2f} MFLOPs, "
              f"AI={dc_stats['arith_intensity']:.2f}")

    graph.save_to_json(output_dir)
    graph.save_summary(output_dir)
    graph.save_phase_report(output_dir, hardware_profile=HARDWARE)
    print(f"\n结果目录: {output_dir}/")


# ====================================================================
# 4. 入口
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Graph Parser — 完整管线阶段分析")
    print(f"  模式: {MODE}")
    print("=" * 60)

    if MODE == "pytorch":
        run_pytorch_mode()
    elif MODE == "onnx":
        run_onnx_mode()
    else:
        print(f"未知模式: {MODE}")
