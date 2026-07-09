"""LLM Graph Parser — 完整推理流程阶段分析。"""
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "D:/Hugging Face")

from llm_graph_parser import parse_model, parse_onnx
from llm_graph_parser.core.phase_splitter import PhaseSplitter
from llm_graph_parser.hardware import HardwareProfiler

# ====================================================================
# 配置区
# ====================================================================
MODE = "pytorch"                         # "pytorch" 或 "onnx"

# ---- PyTorch ----
MODEL_SOURCE = "../Models/Forge-1-Mini"
PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
]
MAX_NEW_TOKENS = 20
SKIP_GENERATION = False
TRUST_REMOTE_CODE = True
HARDWARE_PROFILING = True               # True = 用 CUDA Event 测量推理延迟(需 GPU)

# ---- ONNX ----
ONNX_PATH = "../Models/ONNXs/Kokoro-82M.onnx"

# ---- 硬件 ----
HARDWARE = {"peak_flops": 1979e12, "memory_bw": 3350e9}  # H100


# ====================================================================
def make_output_dir(model_label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    d = f"output/{model_label}_{timestamp}"
    os.makedirs(d, exist_ok=True)
    return d


# ====================================================================
# PyTorch 模式
# ====================================================================
def run_pytorch_mode():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_label = os.path.basename(MODEL_SOURCE.replace("\\", "/"))
    output_dir = make_output_dir(model_label)

    print(f"\n加载模型: {MODEL_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE)
    model.eval()
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()):,}")

    profiler = HardwareProfiler()
    if HARDWARE_PROFILING and not profiler.available:
        print("  [hardware] HARDWARE_PROFILING=True 但未检测到 GPU,跳过 profiling")
    if profiler.available:
        print(f"  [hardware] GPU: {torch.cuda.get_device_name(0)}")
        if not HARDWARE_PROFILING:
            print("  [hardware] 设置 HARDWARE_PROFILING=True 启用延迟测量")

    for i, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        seq_len = prompt_ids.shape[1]
        prefix = f"prompt_{i}" if len(PROMPTS) > 1 else ""

        print(f"\n{'=' * 60}")
        print(f"  Prompt [{i}]: \"{prompt}\"")
        print(f"  tokens: {seq_len}")

        # Step 1: Prefill
        print(f"  [Phase 1/3] Prefill")
        if HARDWARE_PROFILING and profiler.available:
            _ = profiler.trace(model, prompt_ids, label="prefill")
        prefill_graph = parse_model(model, prompt_ids, model_name=model_label, onnx_path="")
        prefill_graph.prompt_text = prompt
        prefill_graph.prompt_tokens = seq_len
        prefill_graph.tag_unassigned_as("prefill")
        pf = prefill_graph.get_stage_stats("prefill")
        print(f"    ops={pf['num_ops']}, FLOPs={pf['total_flops']/1e6:.2f}M, "
              f"AI={pf['arith_intensity']:.2f}")
        if HARDWARE_PROFILING and profiler.available:
            print(f"    time={profiler._prefill_total_us/1000:.2f}ms")

        # Step 2: Decode
        print(f"  [Phase 2/3] Decode (1 token)")
        decode_token = prompt_ids[:, -1:]
        decode_graph = parse_model(model, decode_token, model_name=model_label, onnx_path="")
        decode_graph.prompt_tokens = 1
        decode_graph.tag_unassigned_as("decode")
        dc = decode_graph.get_stage_stats("decode")
        dc_flops_per = dc["total_flops"]
        print(f"    per-step: ops={dc['num_ops']}, FLOPs={dc_flops_per/1e6:.2f}M, "
              f"AI={dc['arith_intensity']:.2f}")

        # Step 3: Generation
        if SKIP_GENERATION:
            gen_len = 0
            answer = ""
            print(f"  [Phase 3/3] Skipped (SKIP_GENERATION=True)")
        else:
            print(f"  [Phase 3/3] Generating (max {MAX_NEW_TOKENS})...")
            with torch.no_grad():
                kw = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
                if attention_mask is not None:
                    kw["attention_mask"] = attention_mask
                for k, v in model.generation_config.to_dict().items():
                    if v is not None and k not in kw and k not in ("_from_model_config", "transformers_version"):
                        kw[k] = v
                if HARDWARE_PROFILING and profiler.available:
                    try:
                        gen_len, gen_time = profiler.trace_generate(model, prompt_ids, **kw)
                        out = None
                        print(f"    generate time={gen_time/1000:.2f}ms")
                    except Exception as pe:
                        print(f"    [profiler] generate trace failed: {pe}")
                else:
                    out = model.generate(prompt_ids, **kw)
            if out is not None:
                gen_len = out.shape[1] - seq_len
            answer = tokenizer.decode(out[0, seq_len:] if out is not None else prompt_ids[0],
                                      skip_special_tokens=True).strip()
            print(f"    generated: {gen_len} tokens")
            print(f"    answer: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"")
            if gen_len == 0:
                print("    (no output - model may need different prompts)")

        # ---- Layer partitioner ----
        from llm_graph_parser.core.layer_partitioner import LayerPartitioner
        for g in (prefill_graph, decode_graph):
            try:
                g.set_layer_tree(LayerPartitioner(g).partition())
            except Exception:
                pass

        # ---- Combine (保留 DAG 边) ----
        combined = prefill_graph
        for n in decode_graph.nodes:
            old_id = n.op_id
            n.op_id = f"dc_{n.op_id}"
            # 更新 parents/children 中的旧 ID
            n.parents = [f"dc_{p}" if not p.startswith("dc_") else p for p in n.parents]
            n.children = [f"dc_{c}" if not c.startswith("dc_") else c for c in n.children]
            combined.add_node(n)
            # 重新建立 DAG 边
            combined._layer_map[n.layer_id].append(n.op_id)
        combined._layer_tree = prefill_graph._layer_tree
        combined._layer_map = prefill_graph._layer_map

        # ---- Summary ----
        pf_flops = pf["total_flops"]
        pf_mem = pf["total_memory_bytes"]
        dc_flops = dc_flops_per * gen_len
        dc_mem = dc["total_memory_bytes"] * gen_len
        total_flops = pf_flops + dc_flops
        total_mem = pf_mem + dc_mem

        lines = [f"Model: {model_label}", f"Prompt: \"{prompt}\"", f"Answer: \"{answer}\"",
                 f"Prompt tokens: {seq_len}  |  Generated tokens: {gen_len}", "",
                 f"{'Phase':20s} {'Ops':>8s} {'FLOPs':>15s} {'Mem(MB)':>10s} {'AI':>8s}",
                 "-" * 64,
                 f"{'Prefill':20s} {pf['num_ops']:>8d} {pf_flops:>15,} {pf_mem/1e6:>10.2f} {pf['arith_intensity']:>8.2f}"]
        if gen_len > 0:
            lines.append(f"{'Decode x'+str(gen_len):20s} {dc['num_ops']:>8d} {dc_flops:>15,} {dc_mem/1e6:>10.2f} {dc['arith_intensity']:>8.2f}")
        lines.append("-" * 64)
        lines.append(f"{'Total':20s} {pf['num_ops'] + dc['num_ops'] * gen_len:>8d} {total_flops:>15,} {total_mem/1e6:>10.2f}")

        # Layer tree
        if prefill_graph.get_layer_tree():
            t = prefill_graph.get_layer_tree()
            nb = len([c for c in t.children if c.layer_type == "transformer_block"])
            lines.append(""); lines.append(f"Layer hierarchy ({nb} blocks):")
            lines.append(prefill_graph._layer_tree_to_text(t, "  "))

        # Parallelism + Roofline
        par = combined.parallelism_report()
        lines.append("")
        lines.append(f"Parallelism:  Max={par['max_parallelism']} ops/level, Avg={par['avg_parallelism']:.2f}, Critical={par['critical_path_length']:.0f} steps")
        if gen_len > 0:
            kvc = decode_graph.kv_cache_analysis(num_decode_tokens=gen_len)
            if kvc["cross_edges"] > 0:
                lines.append(f"KV cache:    {kvc['num_layers']} layers, {kvc['cross_edges']} cross-input edges")
        ai = pf["arith_intensity"]
        ridge = HARDWARE["peak_flops"] / HARDWARE["memory_bw"]
        bound = "COMPUTE BOUND" if ai >= ridge else "MEMORY BOUND"
        lines.append(f"Roofline:    Prefill AI={ai:.2f}, {bound} (vs H100)")

        # Operator counts
        lines.append(""); lines.append("Operator counts:")
        for op, cnt in sorted(combined.get_operator_counts().items(), key=lambda x: -x[1]):
            lines.append(f"  {op:25s}: {cnt}")

        text = "\n".join(lines)

        # ---- Save ----
        combined.save_to_json(output_dir, name=prefix)
        prefill_graph.save_to_json(output_dir, name=f"{prefix}_prefill" if prefix else "prefill")
        if gen_len > 0:
            decode_graph.save_to_json(output_dir, name=f"{prefix}_decode" if prefix else "decode")
        stem_s = f"{prefix}_" if prefix else ""
        Path(output_dir).joinpath(f"{stem_s}summary.txt").write_text(text, encoding="utf-8")
        combined.save_phase_report(output_dir, name=prefix, hardware_profile=HARDWARE)
        combined.save_parallelism_report(output_dir, name=prefix)
        prefill_graph.save_layer_report(output_dir, name=prefix)
        if gen_len > 0:
            decode_graph.save_kv_cache_report(output_dir, name=prefix, num_decode_tokens=gen_len)

        print("\n" + "=" * 60)
        print(text)
        print("=" * 60)

    print(f"\n所有结果已保存到: {output_dir}/")


# ====================================================================
# ONNX 模式
# ====================================================================
def run_onnx_mode():
    model_label = os.path.basename(ONNX_PATH.replace("\\", "/"))
    if model_label.lower().endswith(".onnx"):
        model_label = model_label[:-5]
    output_dir = make_output_dir(model_label)

    print(f"\n加载 ONNX: {ONNX_PATH}")
    graph = parse_onnx(ONNX_PATH, model_name=model_label)
    print(f"  算子节点数: {graph.num_nodes}")

    # Stage tagging
    graph.prompt_text = os.path.basename(ONNX_PATH)
    graph.prompt_tokens = 0
    graph.tag_unassigned_as("prefill")
    pf = graph.get_stage_stats("prefill")
    print(f"  Prefill: {pf['num_ops']} ops, {pf['total_flops']/1e6:.2f}M FLOPs, AI={pf['arith_intensity']:.2f}")

    # Layer partition
    try:
        from llm_graph_parser.core.layer_partitioner import LayerPartitioner
        graph.set_layer_tree(LayerPartitioner(graph).partition())
    except Exception:
        pass

    # Build summary
    pf_flops = pf["total_flops"]
    pf_mem = pf["total_memory_bytes"]
    lines = [f"Model: {model_label}", f"Source: {ONNX_PATH}", "",
             f"{'Phase':20s} {'Ops':>8s} {'FLOPs':>15s} {'Mem(MB)':>10s} {'AI':>8s}",
             "-" * 64,
             f"{'Prefill':20s} {pf['num_ops']:>8d} {pf_flops:>15,} {pf_mem/1e6:>10.2f} {pf['arith_intensity']:>8.2f}",
             ""]

    if graph.get_layer_tree():
        t = graph.get_layer_tree()
        nb = len([c for c in t.children if c.layer_type == "transformer_block"])
        lines.append(f"Layer hierarchy ({nb} blocks):")
        lines.append(graph._layer_tree_to_text(t, "  "))
        lines.append("")

    par = graph.parallelism_report()
    lines.append(f"Parallelism:  Max={par['max_parallelism']} ops/level, Avg={par['avg_parallelism']:.2f}, Critical={par['critical_path_length']:.0f} steps")
    ai = pf["arith_intensity"]
    ridge = HARDWARE["peak_flops"] / HARDWARE["memory_bw"]
    lines.append(f"Roofline:    Prefill AI={ai:.2f}, {'COMPUTE BOUND' if ai >= ridge else 'MEMORY BOUND'} (vs H100)")
    lines.append("")

    lines.append("Operator counts:")
    for op, cnt in sorted(graph.get_operator_counts().items(), key=lambda x: -x[1]):
        lines.append(f"  {op:25s}: {cnt}")

    text = "\n".join(lines)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

    # Save
    graph.save_to_json(output_dir)
    Path(output_dir).joinpath("summary.txt").write_text(text, encoding="utf-8")
    graph.save_phase_report(output_dir, hardware_profile=HARDWARE)
    graph.save_parallelism_report(output_dir)
    graph.save_layer_report(output_dir)
    print(f"\n结果目录: {output_dir}/")


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
