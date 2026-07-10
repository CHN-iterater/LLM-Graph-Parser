"""LLM Graph Parser — 完整推理流程阶段分析。"""
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "D:/Hugging Face")

from llm_graph_parser import parse_model, parse_onnx
from llm_graph_parser.hardware import HardwareProfiler


# ====================================================================
# 配置区
# ====================================================================
MODE = "pytorch"
MODEL_SOURCE = "../Models/Qwen3-0.6B"
PROMPT = "What's the capital of France?"
MAX_NEW_TOKENS = 20
SKIP_GENERATION = False
TRUST_REMOTE_CODE = True
HARDWARE_PROFILING = True
PROFILING_RUNS = 20
ONNX_PATH = "../Models/ONNXs/Kokoro-82M.onnx"
HARDWARE = {"peak_flops": 1979e12, "memory_bw": 3350e9}


# ====================================================================
# 通用工具
# ====================================================================
def make_output_dir(model_label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    d = f"output/{model_label}_{timestamp}"
    os.makedirs(d, exist_ok=True)
    return d


def ridge_point(hw: dict) -> float:
    return hw["peak_flops"] / hw["memory_bw"]



TYPE_15_NAME = {
    1: "GEMM", 2: "FlashAttention", 3: "BMM",
    4: "Softmax", 5: "LayerNorm", 6: "RMSNorm", 7: "Reduction",
    8: "GELU", 9: "SiLU/Swish", 10: "ReLU",
    11: "KV Cache Read", 12: "KV Cache Write",
    13: "AllReduce", 14: "AllGather", 15: "Memcpy D2D",
}
TYPE_15_CLASS = {
    1: "compute_bound", 2: "compute_bound", 3: "compute_bound",
    4: "memory_bound", 5: "memory_bound", 6: "memory_bound", 7: "memory_bound",
    8: "activation", 9: "activation", 10: "activation",
    11: "data_movement", 12: "data_movement",
    13: "communication", 14: "communication", 15: "data_movement",
}
_KCACHE_PATTERNS = (b"attention", b"attn", b"qkv")
_NCCL_PATTERNS = (b"nccl", b"allreduce", b"allgather", b"broadcast")
_MEMCPY_PATTERNS = (b"memcpy", b"d2d", b"dtoD")


def classify_15(op_type: str, category: str) -> tuple[int, str]:
    t = op_type.upper()
    if category == "compute_bound":
        if "ATTENTION" in t or "FLASH" in t:
            return 2, TYPE_15_NAME[2]
        if t == "BMM":
            return 3, TYPE_15_NAME[3]
        return 1, TYPE_15_NAME[1]
    if category == "memory_bound":
        if "SOFTMAX" in t:
            return 4, TYPE_15_NAME[4]
        if "RMS" in t:
            return 6, TYPE_15_NAME[6]
        if "NORM" in t:
            return 5, TYPE_15_NAME[5]
        return 7, TYPE_15_NAME[7]
    if category == "activation":
        if t in ("GELU",):
            return 8, TYPE_15_NAME[8]
        if t in ("SILU", "SIGMOID"):
            return 9, TYPE_15_NAME[9]
        if t in ("RELU",):
            return 10, TYPE_15_NAME[10]
        return 8, TYPE_15_NAME[8]
    return 0, "Auxiliary"


def classify_profiler_kernel(kernel_name: str) -> tuple[int, str]:
    kn = kernel_name.encode() if isinstance(kernel_name, str) else kernel_name
    if any(p in kn for p in _MEMCPY_PATTERNS):
        return 15, TYPE_15_NAME[15]
    if any(p in kn for p in _NCCL_PATTERNS):
        if b"allreduce" in kn:
            return 13, TYPE_15_NAME[13]
        return 14, TYPE_15_NAME[14]
    if any(p in kn for p in _KCACHE_PATTERNS):
        return 11, TYPE_15_NAME[11]
    return 0, "Auxiliary"



def _summary_header(model_label, prompt, answer, seq_len, gen_len, pf, dc, dc_total):
    """管线统计表头 + Prefill/Decode 数据行。"""
    pf_flops = pf["total_flops"]
    pf_mem = pf["total_memory_bytes"]
    lines = [
        f"Model: {model_label}", f"Prompt: \"{prompt}\"", f"Answer: \"{answer}\"",
        f"Prompt tokens: {seq_len}  |  Generated tokens: {gen_len}", "",
        f"{'Phase':20s} {'Ops':>8s} {'FLOPs':>15s} {'Mem(MB)':>10s} {'AI':>8s}",
        "-" * 64,
        f"{'Prefill':20s} {pf['num_ops']:>8d} {pf_flops:>15,} {pf_mem/1e6:>10.2f} {pf['arith_intensity']:>8.2f}",
    ]
    if gen_len > 0 and dc_total > 0:
        lines.append(
            f"{'Decode x'+str(gen_len):20s} {dc['num_ops']:>8d} {dc_total:>15,} "
            f"{dc['total_memory_bytes'] * gen_len / 1e6:>10.2f} {dc['arith_intensity']:>8.2f}")
    lines.append("-" * 64)
    total_ops = pf['num_ops'] + dc['num_ops'] * gen_len
    total_f = pf_flops + dc_total
    total_m = pf_mem + dc['total_memory_bytes'] * gen_len
    lines.append(f"{'Total':20s} {total_ops:>8d} {total_f:>15,} {total_m/1e6:>10.2f}")
    return lines


def _summary_extra(graph, combined, decode_graph, gen_len, pf_ai, profiler=None):
    """层结构 + 并行性 + Roofline + 算子分布。"""
    lines = []
    tree = graph.get_layer_tree()
    if tree:
        nb = len([c for c in tree.children if c.layer_type == "transformer_block"])
        lines.append("")
        lines.append(f"Layer hierarchy ({nb} blocks):")
        lines.append(graph._layer_tree_to_text(tree, "  "))

    par = combined.parallelism_report()
    lines.append("")
    lines.append(f"Parallelism:  Max={par['max_parallelism']} ops/level, "
                 f"Avg={par['avg_parallelism']:.2f}, Critical={par['critical_path_length']:.0f} steps")
    if gen_len > 0:
        kvc = decode_graph.kv_cache_analysis(num_decode_tokens=gen_len)
        if kvc["cross_edges"] > 0:
            lines.append(f"KV cache:    {kvc['num_layers']} layers, {kvc['cross_edges']} cross-input edges")

    ridge = ridge_point(HARDWARE)
    bound = "COMPUTE BOUND" if pf_ai >= ridge else "MEMORY BOUND"
    lines.append(f"Roofline:    Prefill AI={pf_ai:.2f}, {bound} (vs H100)")

    # Hardware profiling info
    if profiler and profiler.available:
        pf_t = profiler._prefill_total_us / 1000
        dc_t = profiler._decode_total_us / 1000
        tot_t = pf_t + dc_t
        mem = max(profiler._memory_peak - profiler._memory_start, 0) / 1e6
        if pf_t > 0 or dc_t > 0:
            lines.append(f"    GPU time: Prefill={pf_t:.2f}ms, Decode={dc_t:.2f}ms")
            lines.append(f"    GPU memory: {mem:.0f}MB")
        tot_b = sum(n.memory_bytes for n in combined.nodes)
        if tot_t > 0 and tot_b > 0:
            bw = tot_b / tot_t / 1e6
            peak_bw = HARDWARE["memory_bw"] / 1e9
            lines.append(f"    Achieved BW: {bw:.0f} GB/s ({bw/peak_bw*100:.0f}% of H100)")
        if gen_len > 0 and dc_t > 0:
            lines.append(f"    Throughput: {gen_len/(dc_t/1000):.1f} tokens/s")

    # 15-type classification
    counts = combined.get_operator_counts()
    types: dict[int, int] = {}
    aux: dict[str, int] = {}
    for op, cnt in counts.items():
        node = combined._nodes.get(list(combined._nodes.keys())[0]) if combined._nodes else None
        # Find any node of this type to get its category
        cat = "other"
        for n in combined._nodes.values():
            if n.op_type == op:
                cat = n.category; break
        tid, tname = classify_15(op, cat)
        if tid > 0:
            types.setdefault(tid, 0)
            types[tid] += cnt
        else:
            aux[op] = cnt

    lines.append("")
    lines.append(f"  {'ID':>4s}  {'Type':30s}  {'Count':>8s}  {'Energy Class':>18s}")
    lines.append("  " + "-" * 66)
    for tid in sorted(types):
        tname = {1:"GEMM",2:"FlashAttention",3:"BMM",4:"Softmax",5:"LayerNorm",
                 6:"RMSNorm",7:"Reduction",8:"GELU",9:"SiLU/Swish",10:"ReLU",
                 11:"KV Cache",12:"KV Cache Write",13:"AllReduce",14:"AllGather",15:"Memcpy D2D"}.get(tid, f"Type-{tid}")
        eclass = {1:"compute_bound",2:"compute_bound",3:"compute_bound",
                  4:"memory_bound",5:"memory_bound",6:"memory_bound",7:"memory_bound",
                  8:"activation",9:"activation",10:"activation",
                  11:"data_movement",12:"data_movement",13:"communication",14:"communication",15:"data_movement"}.get(tid, "")
        lines.append(f"  #{tid:2d}  {tname:30s}  {types[tid]:>8d}  {eclass:>18s}")

    if aux:
        lines.append("  " + "-" * 66)
        lines.append(f"  {'--':>4s}  {'Auxiliary operators':30s}  {sum(aux.values()):>8d}")
    total_main = sum(types.values())
    total_aux = sum(aux.values())
    lines.append(f"  {'--':>4s}  {'---':30s}  {'---':>8s}")
    lines.append(f"  {'Main':>4s}  {'(types 1-15)':30s}  {total_main:>8d}")
    lines.append(f"  {'Aux':>4s}  {'(assist ops)':30s}  {total_aux:>8d}")

    return lines


# ====================================================================
# PyTorch 模式
# ====================================================================
def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def run_pytorch_mode():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from llm_graph_parser.core.layer_partitioner import LayerPartitioner

    model_label = os.path.basename(MODEL_SOURCE.replace("\\", "/"))
    output_dir = make_output_dir(model_label)
    ts_path = os.path.join(output_dir, "timestamps.txt")

    print(f"\n加载模型: {MODEL_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE)
    model.eval()
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 硬件 profiling 初始化 ----
    profiler = HardwareProfiler()
    device = torch.device("cuda" if (HARDWARE_PROFILING and torch.cuda.is_available()) else "cpu")
    if HARDWARE_PROFILING and profiler.available:
        model = model.to(device)
        for p in model.parameters():
            if str(p.device) != str(device):
                p.data = p.data.to(device)
        print(f"  [hardware] 模型已移至 {device}")
    if HARDWARE_PROFILING and not profiler.available:
        print("  [hardware] HARDWARE_PROFILING=True 但未检测到 GPU,跳过 profiling")
    if profiler.available:
        print(f"  [hardware] GPU: {torch.cuda.get_device_name(0)}")
        if not HARDWARE_PROFILING:
            print("  [hardware] 设置 HARDWARE_PROFILING=True 启用延迟测量")

    prompt = PROMPT
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if HARDWARE_PROFILING and profiler.available:
        prompt_ids = prompt_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    seq_len = prompt_ids.shape[1]
    prefix = ""

    print(f"\n{'=' * 60}")
    with open(ts_path, "w") as tf:
        tf.write("start:" + _ts() + "\n")
    print(f"  start: {_ts()}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  tokens: {seq_len}")

    # Step 1: Prefill
    print(f"  [Phase 1/3] Prefill")
    if HARDWARE_PROFILING and profiler.available:
        _ = profiler.time_forward(model, prompt_ids, label="prefill", num_runs=PROFILING_RUNS)
    prefill_graph = parse_model(model, prompt_ids, model_name=model_label, onnx_path="")
    prefill_graph.prompt_text = prompt
    prefill_graph.prompt_tokens = seq_len
    prefill_graph.tag_unassigned_as("prefill")
    pf = prefill_graph.get_stage_stats("prefill")
    print(f"    ops={pf['num_ops']}, FLOPs={pf['total_flops']/1e6:.2f}M, AI={pf['arith_intensity']:.2f}")
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
    print(f"    per-step: ops={dc['num_ops']}, FLOPs={dc_flops_per/1e6:.2f}M, AI={dc['arith_intensity']:.2f}")

    # Step 3: Generation
    if SKIP_GENERATION:
        gen_len, answer = 0, ""
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
            out = model.generate(prompt_ids, **kw)
        gen_len = out.shape[1] - seq_len
        answer = tokenizer.decode(out[0, seq_len:], skip_special_tokens=True).strip()
        print(f"    generated: {gen_len} tokens")
        print(f"    answer: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"")
        if gen_len == 0:
            print("    (no output - model may need different prompts)")
        if HARDWARE_PROFILING and profiler.available:
            try:
                _ = profiler.time_generate(model, prompt_ids, num_runs=PROFILING_RUNS, **kw)
            except Exception as pe:
                print(f"    [profiler] trace failed: {pe}")

    # ---- Layer partitioner ----
    for g in (prefill_graph, decode_graph):
        try:
            g.set_layer_tree(LayerPartitioner(g).partition())
        except Exception:
            pass

    # ---- Combine ----
    combined = prefill_graph
    for n in decode_graph.nodes:
        n.op_id = f"dc_{n.op_id}"
        n.parents = [f"dc_{p}" if not p.startswith("dc_") else p for p in n.parents]
        n.children = [f"dc_{c}" if not c.startswith("dc_") else c for c in n.children]
        combined.add_node(n)
        combined._layer_map[n.layer_id].append(n.op_id)
    combined._layer_tree = prefill_graph._layer_tree
    combined._layer_map = prefill_graph._layer_map

    # ---- Build summary ----
    total_dc = dc_flops_per * gen_len
    lines = _summary_header(model_label, prompt, answer, seq_len, gen_len, pf, dc, total_dc)
    lines += _summary_extra(prefill_graph, combined, decode_graph, gen_len, pf["arith_intensity"], profiler)
    text = "\n".join(lines)

    # ---- Save ----
    combined.save_to_json(output_dir, name=prefix)
    prefill_graph.save_to_json(output_dir, name=f"{prefix}_prefill" if prefix else "prefill")
    if gen_len > 0:
        decode_graph.save_to_json(output_dir, name=f"{prefix}_decode" if prefix else "decode")
    stem = f"{prefix}_" if prefix else ""
    Path(output_dir).joinpath(f"{stem}summary.txt").write_text(text, encoding="utf-8")
    combined.save_phase_report(output_dir, name=prefix, hardware_profile=HARDWARE)
    combined.save_parallelism_report(output_dir, name=prefix)
    prefill_graph.save_layer_report(output_dir, name=prefix)
    if gen_len > 0:
        decode_graph.save_kv_cache_report(output_dir, name=prefix, num_decode_tokens=gen_len)
    if HARDWARE_PROFILING and profiler.available:
        try:
            profiler.save_report(output_dir, name=prefix)
            profiler.trace_to_json(output_dir, name=prefix)
        except Exception as pe:
            print(f"    [profiler] save failed: {pe}")

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    tf.write("end:" + _ts() + "\n")
    print(f"  end:   {_ts()}")
    print(f"\n所有结果已保存到: {output_dir}/")


# ====================================================================
# ONNX 模式
# ====================================================================
def run_onnx_mode():
    from llm_graph_parser.core.layer_partitioner import LayerPartitioner

    model_label = os.path.basename(ONNX_PATH.replace("\\", "/"))
    if model_label.lower().endswith(".onnx"):
        model_label = model_label[:-5]
    output_dir = make_output_dir(model_label)
    ts_path = os.path.join(output_dir, "timestamps.txt")

    print(f"\n加载 ONNX: {ONNX_PATH}")
    graph = parse_onnx(ONNX_PATH, model_name=model_label)
    print(f"  算子节点数: {graph.num_nodes}")

    graph.prompt_text = os.path.basename(ONNX_PATH)
    graph.prompt_tokens = 0
    graph.tag_unassigned_as("prefill")
    pf = graph.get_stage_stats("prefill")
    print(f"  Prefill: {pf['num_ops']} ops, {pf['total_flops']/1e6:.2f}M FLOPs, AI={pf['arith_intensity']:.2f}")

    try:
        graph.set_layer_tree(LayerPartitioner(graph).partition())
    except Exception:
        pass

    # Summary
    lines = _summary_header(model_label, ONNX_PATH, "", 0, 0, pf,
                            {"num_ops": 0, "total_flops": 0, "total_memory_bytes": 0, "arith_intensity": 0}, 0)
    lines += _summary_extra(graph, graph, None, 0, pf["arith_intensity"])
    text = "\n".join(lines)

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

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
