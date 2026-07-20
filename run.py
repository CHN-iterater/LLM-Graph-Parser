"""LLM Graph Parser — 完整推理流程阶段分析。"""
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from llm_graph_parser import parse_model, parse_onnx
from llm_graph_parser.hardware import HardwareProfiler


# ====================================================================
# 配置区
# ====================================================================
MODE = "pytorch"
MODEL_SOURCE = "../Models/MiniMind2-Small"
PROMPT = "What's the capital of France?"
MAX_NEW_TOKENS = 20
SKIP_GENERATION = False
TRUST_REMOTE_CODE = True
HARDWARE_PROFILING = True
PROFILING_RUNS = 20
GEN_REPEATS = 50        # 生成阶段每步重复次数（增大可降噪）
ONNX_PATH = "../Models/ONNXs/Kokoro-82M.onnx"
HARDWARE = {"peak_flops": 1979e12, "memory_bw": 3350e9}
ENERGY_COUNTER = True  # 用 nvml 硬件能量计数器替代功率采样积分


# ====================================================================
# 通用工具
# ====================================================================
def make_output_dir(model_label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = f"output/{model_label}_{timestamp}"
    os.makedirs(d, exist_ok=True)
    return d


def ridge_point(hw: dict) -> float:
    return hw["peak_flops"] / hw["memory_bw"]


def write_timestamp(label: str, path="timestamps.txt"):
    from datetime import datetime
    now = datetime.now()
    ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    with open(path, "a") as f:
        f.write(ts + " " + label + chr(10))


def read_energy_j():
    """返回 GPU 0 累计能耗（J），失败返回 None。"""
    if not ENERGY_COUNTER:
        return None
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
        except Exception:
            pass
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
    except Exception:
        return None


def write_energy(label: str, path: str):
    e = read_energy_j()
    if e is not None:
        with open(path, "a") as f:
            f.write(f"{label}_energy_j {e:.4f}\n")




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

    # 在 CUDA 初始化前测量 GPU 0 真实空闲功率
    write_timestamp("idle_before_start", ts_path)
    write_energy("idle_before_start", ts_path)
    import time; time.sleep(2)
    write_timestamp("idle_before_end", ts_path)
    write_energy("idle_before_end", ts_path)

    print(f"\n加载模型: {MODEL_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=False)
    model.eval()
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # 关闭 KV cache，保证每次 forward 都是完整前向（与方向 1 对齐）
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        print("  [config] use_cache=False")

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

    # 稳态空闲功率：等待模型加载瞬态消退后快速测量
    import time; time.sleep(30)
    write_timestamp("idle_cuda_start", ts_path)
    write_energy("idle_cuda_start", ts_path)
    time.sleep(2)
    write_timestamp("idle_cuda_end", ts_path)
    write_energy("idle_cuda_end", ts_path)

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
    write_timestamp("start", ts_path)
    write_energy("start", ts_path)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  tokens: {seq_len}")

    # Warmup：运行 2 秒 forward 让 GPU 升温至满负荷稳态
    if HARDWARE_PROFILING and profiler.available:
        print(f"  [cooling] 30s before warmup...", end=" ", flush=True)
        time.sleep(30)
        print(f"done")
        print(f"  [warmup] running 2s forward passes...", end=" ", flush=True)
        t0 = time.time()
        with torch.no_grad():
            while (time.time() - t0) < 2.0:
                _ = model(prompt_ids)
                torch.cuda.synchronize()
        time.sleep(1.0)
        print(f"done, starting measurement")

    # Step 1b: Prefill 能耗测量（稳态下的 forward）
    write_timestamp("prefill_start", ts_path)
    write_energy("prefill_start", ts_path)
    print(f"  [Phase 1/3] Prefill (profiling, {PROFILING_RUNS} runs)")
    if HARDWARE_PROFILING and profiler.available:
        pf_kwargs = {}
        if attention_mask is not None:
            pf_kwargs["attention_mask"] = attention_mask
        _ = profiler.time_forward(model, prompt_ids, label="prefill", num_runs=PROFILING_RUNS, **pf_kwargs)
    if HARDWARE_PROFILING and profiler.available:
        total_gpu_us = profiler._prefill_total_us * PROFILING_RUNS
        print(f"    time={profiler._prefill_total_us/1000:.2f}ms (per run), total GPU time={total_gpu_us/1000:.2f}ms")
        with open(ts_path, "a") as tf:
            tf.write(f"prefill_gpu_us {int(total_gpu_us)}\n")
    write_energy("prefill_end", ts_path)
    write_timestamp("prefill_end", ts_path)

    # 冷却 10s，让 GPU 温度在 decode 测量前回落到接近 idle
    if HARDWARE_PROFILING and profiler.available:
        print(f"  [cooling] 30s...", end=" ", flush=True)
        time.sleep(30)
        torch.cuda.synchronize()
        print(f"done")

    # Step 2: Decode — 单 token 前向能耗测量（多次 forward 取平均）
    decode_token = prompt_ids[:, -1:]
    decode_token = decode_token.to(device) if HARDWARE_PROFILING and profiler.available else decode_token
    if HARDWARE_PROFILING and profiler.available:
        # 准备 decode 时需要的额外参数（attention_mask 等）
        dc_kwargs = {}
        if attention_mask is not None:
            dc_kwargs["attention_mask"] = attention_mask[:, -1:]
        print(f"  [warmup decode] running 2s forward passes...", end=" ", flush=True)
        t0 = time.time()
        with torch.no_grad():
            while (time.time() - t0) < 2.0:
                _ = model(decode_token, **dc_kwargs)
                torch.cuda.synchronize()
        time.sleep(1.0)
        print(f"done, starting measurement")
    write_timestamp("decode_start", ts_path)
    write_energy("decode_start", ts_path)
    if HARDWARE_PROFILING and profiler.available:
        _ = profiler.time_forward(model, decode_token, label="decode", num_runs=PROFILING_RUNS, **dc_kwargs)
        total_dc_gpu_us = profiler._decode_total_us * PROFILING_RUNS
        with open(ts_path, "a") as tf:
            tf.write(f"decode_gpu_us {int(total_dc_gpu_us)}\n")
    write_energy("decode_end", ts_path)
    write_timestamp("decode_end", ts_path)
    print(f"  [Phase 2/3] Decode ({PROFILING_RUNS} token forwards)")

    # ONNX 导出
    prefill_graph = parse_model(model, prompt_ids, model_name=model_label, onnx_path="")
    prefill_graph.prompt_text = prompt
    prefill_graph.prompt_tokens = seq_len
    prefill_graph.tag_unassigned_as("prefill")
    pf = prefill_graph.get_stage_stats("prefill")
    print(f"    prefill ops={pf['num_ops']}, FLOPs={pf['total_flops']/1e6:.2f}M, AI={pf['arith_intensity']:.2f}")

    decode_graph = parse_model(model, decode_token, model_name=model_label, onnx_path="")
    decode_graph.prompt_tokens = 1
    decode_graph.tag_unassigned_as("decode")
    dc = decode_graph.get_stage_stats("decode")
    dc_flops_per = dc["total_flops"]
    print(f"    decode per-step: ops={dc['num_ops']}, FLOPs={dc_flops_per/1e6:.2f}M, AI={dc['arith_intensity']:.2f}")
    # ---- Subprocess profiling (isolated from main CUDA context) ----
    if HARDWARE_PROFILING:
        import subprocess as _sp, json as _js
        _prof_script = os.path.join(os.path.dirname(__file__), "profile_kernels.py")
        _all_cats = ["compute_bound", "memory_bound", "data_movement", "communication"]
        for _ph in ("prefill", "decode"):
            try:
                _res = _sp.run([sys.executable, _prof_script, MODEL_SOURCE, PROMPT, _ph],
                               capture_output=True, text=True, timeout=120)
                if _res.returncode != 0:
                    continue
                _dat = _js.loads(_res.stdout.strip().splitlines()[-1])
                with open(ts_path, "a") as _tf:
                    for _ck in _all_cats:
                        print(f"{_ph}_kernel_ratio_{_ck} {_dat[_ck]:.4f}", file=_tf)
                print(f"    kernel profile {_ph}: " +
                      f"compute={_dat['compute_bound']*100:.0f}% " +
                      f"memory={_dat['memory_bound']*100:.0f}% " +
                      f"move={_dat['data_movement']*100:.0f}%")
            except Exception:
                print(f"    kernel profile {_ph}: skipped")


    # Step 3: Generation
    if SKIP_GENERATION:
        gen_len, answer = 0, ""
        print(f"  [Phase 3/3] Skipped (SKIP_GENERATION=True)")
    else:
        write_timestamp("gen_start", ts_path)
        write_energy("gen_start", ts_path)
        print(f"  [Phase 3/3] Generating (max {MAX_NEW_TOKENS})...")

        # 从 idle_before 计算基准功率（与 power_analyze.py 一致）
        P_bl = 80.0
        with open(ts_path) as f:
            ib_start_e = ib_end_e = None
            for line in f:
                if "idle_before_start_energy_j" in line:
                    ib_start_e = float(line.strip().split()[1])
                elif "idle_before_end_energy_j" in line:
                    ib_end_e = float(line.strip().split()[1])
            if ib_start_e is not None and ib_end_e is not None:
                P_bl = (ib_end_e - ib_start_e) / 2.0

        from transformers.generation import LogitsProcessorList, StoppingCriteriaList
        import copy

        # 和 generate() 内部一样：深拷贝 + 预处理 config
        gen_config = copy.deepcopy(model.generation_config)
        # generate() 的 _prepare_generation_config 会设这些 tensor 属性
        if hasattr(gen_config, "eos_token_id") and gen_config.eos_token_id is not None:
            eos_ids = gen_config.eos_token_id if isinstance(gen_config.eos_token_id, list) else [gen_config.eos_token_id]
            gen_config._eos_token_tensor = torch.tensor(eos_ids, dtype=torch.long)
        if hasattr(gen_config, "pad_token_id") and gen_config.pad_token_id is not None:
            gen_config._pad_token_tensor = torch.tensor(gen_config.pad_token_id, dtype=torch.long)

        logits_processor = model._get_logits_processor(
            generation_config=gen_config,
            input_ids_seq_length=prompt_ids.shape[1],
            encoder_input_ids=prompt_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
        )
        stopping_criteria = model._get_stopping_criteria(
            generation_config=gen_config,
            stopping_criteria=StoppingCriteriaList(),
        )

        # 逐位置生成 + 测量：每个位置重复 GEN_REPEATS 次 forward（KV cache 不变态）
        input_ids = prompt_ids.clone()
        model_kwargs_gen = {"use_cache": False}
        if attention_mask is not None:
            model_kwargs_gen["attention_mask"] = attention_mask.clone()
        token_data = []
        generated_ids = []

        with torch.no_grad():
            for step in range(MAX_NEW_TOKENS):
                e_before = read_energy_j()
                t_before = time.time()
                for _ in range(GEN_REPEATS):
                    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs_gen)
                    outputs = model(**model_inputs, return_dict=True)
                    torch.cuda.synchronize()
                    del outputs
                e_after = read_energy_j()
                dt = time.time() - t_before

                raw_per_forward = (e_after - e_before) - P_bl * dt
                e_step = max(raw_per_forward / GEN_REPEATS, 0.0)

                # 真正一次 forward 更新状态 + 选 token
                model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs_gen)
                outputs = model(**model_inputs, return_dict=True)
                model_kwargs_gen = model._update_model_kwargs_for_generation(
                    outputs, model_kwargs_gen,
                    is_encoder_decoder=model.config.is_encoder_decoder,
                )
                logits = outputs.logits[:, -1, :].float()
                scores = logits_processor(input_ids, logits)
                next_token = torch.argmax(scores, dim=-1, keepdim=True)
                vocab_size = getattr(model.config, "vocab_size", scores.shape[-1])
                next_token = torch.clamp(next_token, 0, vocab_size - 1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                generated_ids.append(next_token[0, 0].item())
                token_data.append((step, e_step, dt / GEN_REPEATS))

                if stopping_criteria(input_ids, None):
                    break
                del outputs

        gen_len = len(generated_ids)

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"    generated: {gen_len} tokens")
        print(f"    answer: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"")
        if gen_len == 0:
            print("    (no output - model may need different prompts)")

        # 逐 token 能耗输出
        print(f"\n  Per-token energy (P_bl={P_bl:.1f}W, each averaged over {GEN_REPEATS} forwards):")
        print(f"  {'Token':>6s}  {'Energy(J)':>10s}  {'dt(s)':>8s}  {'Growth(%)':>10s}")
        print(f"  {'-' * 40}")
        prev = None
        for pos, e, dt in token_data:
            g = (e / prev - 1) * 100 if prev else None
            gs = f"{g:>+9.2f}" if g is not None else "      --"
            print(f"  {pos+1:>6d}  {e:>8.4f}  {dt:>6.4f}  {gs:>10s}")
            prev = e

        energy_path = os.path.join(output_dir, "per_token_energy.txt")
        with open(energy_path, "w") as f:
            f.write(f"# P_bl={P_bl:.1f}W  total_tokens={gen_len}  forwards_per_step={GEN_REPEATS}\n")
            f.write(f"# Token  Energy(J)  dt(s)  Growth(%)\n")
            prev = None
            for pos, e, dt in token_data:
                g = (e / prev - 1) * 100 if prev else None
                gs = f"{g:+.2f}" if g is not None else "--"
                f.write(f"{pos+1}  {e:.6f}  {dt:.4f}  {gs}\n")
                prev = e
        print(f"    -> saved to {energy_path}")

        write_energy("gen_end", ts_path)
        write_timestamp("gen_end", ts_path)

        # ---- Per-token operator mapping ----
        if gen_len > 0 and decode_graph is not None:
            op_path = os.path.join(output_dir, "per_token_operators.txt")
            with open(op_path, "w", encoding="utf-8") as f:
                f.write(f"# Per-token operator mapping  prompt_len={seq_len}  gen_tokens={gen_len}\n")
                for pos in range(gen_len):
                    total_e = token_data[pos][1] if pos < len(token_data) else 0.0
                    f.write(f"\n=== Token {pos+1} (total={total_e:.4f}J) ===\n")
                    f.write(f"{'Operator':25s}  {'Input Shapes':50s}  {'Output Shapes':50s}\n")
                    f.write("-" * 130 + "\n")
                    for node in decode_graph.nodes:
                        ins = node.input_tensors
                        outs = node.output_tensors
                        in_shapes = [list(t.shape) for t in ins]
                        out_shapes = [list(t.shape) for t in outs]
                        # Adjust attention sequence dim: seq_len → prompt_len + pos + 1
                        if node.op_type in ("SOFTMAX", "BMM"):
                            for shapes in (in_shapes, out_shapes):
                                for s in shapes:
                                    for i in [-1]:
                                        if len(s) > 1 and isinstance(s[i], int) and s[i] > 0:
                                            s[i] = seq_len + pos + 1
                        ins_str = "; ".join(str(s) for s in in_shapes)
                        outs_str = "; ".join(str(s) for s in out_shapes)
                        f.write(f"{node.op_type:25s}  {ins_str:50s}  {outs_str:50s}\n")
            print(f"    -> saved to {op_path}")

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
        except Exception as pe:
            print(f"    [profiler] save failed: {pe}")

    # ---- 事后空闲基线测量 ----
    write_timestamp("idle_after_start", ts_path)
    write_energy("idle_after_start", ts_path)
    import time
    time.sleep(2)
    write_timestamp("idle_after_end", ts_path)
    write_energy("idle_after_end", ts_path)

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    write_timestamp("end", ts_path)
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
    import argparse
    parser = argparse.ArgumentParser(description="LLM Graph Parser")
    parser.add_argument("--mode", choices=["pytorch", "onnx"], default=None,
                        help="运行模式")
    parser.add_argument("-m", "--model", default=None,
                        help="模型名或路径（自动映射 ../Models/NAME，完整路径直接使用）")
    parser.add_argument("--prompt", default=None,
                        help="提示词（默认: What's the capital of France?）")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="最大生成 token 数")
    parser.add_argument("--no-hardware", action="store_true",
                        help="禁用硬件 profiling")
    parser.add_argument("--runs", type=int, default=None,
                        help="profiling 重复次数")
    parser.add_argument("--gen-repeats", type=int, default=None,
                        help="生成阶段每步重复次数（默认 50）")
    _a = parser.parse_args()

    if _a.mode:
        MODE = _a.mode
    if _a.model:
        MODEL_SOURCE = _a.model
        if not MODEL_SOURCE.startswith("/") and not MODEL_SOURCE.startswith("..") and not MODEL_SOURCE.startswith("."):
            MODEL_SOURCE = f"../Models/{MODEL_SOURCE}"
    if _a.prompt is not None:
        PROMPT = _a.prompt
    if _a.max_new_tokens is not None:
        MAX_NEW_TOKENS = _a.max_new_tokens
    if _a.no_hardware:
        HARDWARE_PROFILING = False
    if _a.runs is not None:
        PROFILING_RUNS = _a.runs
    if _a.gen_repeats is not None:
        GEN_REPEATS = _a.gen_repeats

    print("=" * 60)
    print("  LLM Graph Parser")
    print(f"  模式: {MODE}  |  模型: {MODEL_SOURCE}")
    print("=" * 60)
    if MODE == "pytorch":
        run_pytorch_mode()
    elif MODE == "onnx":
        run_onnx_mode()
