"""
能耗重构 — 将单算子能耗数据映射回 graph.json 计算图，估算推理总能耗。
支持 Prefill / Decode 分阶段输出。

用法:
    python energy_consumption_refactor.py -c single_operator_summary.csv \\
                                         -g LLM_Graph_Parser/output/Qwen3-0.6B_20260710_*/graph.json \\
                                         -o energy_report.txt --gen-len 20
"""
import argparse, json, csv
from collections import defaultdict
from pathlib import Path


def load_operator_energy(csv_path):
    """single_operator_summary.csv → (db, aux_energy)

    aux_energy_fb = 无 OP_MAP 映射的辅助算子的 fallback 系数。
    CSV 有两种格式：
      - 旧版：energy_j 是多次 profiling 的总能量，直接使用
      - 新版（200ms 窗口）：有 net_power_w，归一化到单次迭代
    """
    db = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_net = "net_energy_j" in reader.fieldnames
        for row in reader:
            name = row["operator"].strip()
            try:
                m, n, k = row.get("M", ""), row.get("N", ""), row.get("K", "")
                flops = 2 * int(m) * int(n) * int(k) if m and n and k else 0
            except ValueError:
                flops = 0
            time_ms = float(row["time_per_iter_ms"])

            if has_net:
                net_power = float(row["net_power_w"])
                energy = net_power * time_ms / 1000
            else:
                energy = float(row["energy_j"])

            cat = row.get("category", "").strip()
            # 计算 benchmark 该算子的访存量（字节），用于 memory 缩放
            if cat == "compute_intensive" and not (m and n and k):
                try:
                    bs = int(row.get("batch_size", 0) or 0)
                    sl = int(row.get("seq_len", 0) or 0)
                    nh = int(row.get("num_heads", 0) or 0)
                    hd = int(row.get("head_dim", 0) or 0)
                    mem = 4 * bs * sl * nh * hd * 2 if (bs and sl and nh and hd) else 0
                except ValueError:
                    mem = 0
            elif m and n:
                im, inn = int(m), int(n)
                if cat == "compute_intensive" and k:
                    ik = int(k)
                    mem = (im * ik + ik * inn + im * inn) * 2  # A+B+C, FP16
                else:
                    mem = 4 * im * inn  # read + write, FP16
            else:
                mem = 0

            db[name] = {
                "energy_j": energy,
                "power_w": float(row["power_mean_w"]),
                "time_ms": time_ms,
                "flops": flops,
                "mem_bytes": mem,
                "category": cat,
            }
    aux_energy = min(e["energy_j"] for e in db.values())
    return db, aux_energy


def load_graph(graph_path):
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("summary", {})


# ONNX op_type → CSV operator name
OP_MAP = {
    "LINEAR": "GEMM", "GEMM": "GEMM", "BMM": "BMM",
    "ATTENTION": "FlashAttention", "FLASH_ATTENTION": "FlashAttention",
    "SOFTMAX": "Softmax",
    "LAYER_NORM": "LayerNorm", "RMS_NORM": "RMSNorm",
    "GELU": "GELU", "SILU": "SiLU", "RELU": "ReLU", "SIGMOID": "SiLU",
    "REDUCESUM": "Reduction", "REDUCEMEAN": "Reduction", "MEAN": "Reduction",
    "KV_CACHE_READ": "KVCacheRead", "KV_CACHE_WRITE": "KVCacheWrite",
    "CAT": "MemcpyD2D", "SLICE": "MemcpyD2D", "EXPAND": "MemcpyD2D",
    "ADD": "MemcpyD2D", "MUL": "MemcpyD2D",
}

# 辅助算子 — 图分解产物，无独立测试数据
AUXILIARY_OPS = {
    "ADD", "MUL", "RESHAPE", "CAST", "TRANSPOSE",
    "POW", "SQRT", "RECIPROCAL", "SLICE", "NEG",
    "CAT", "EXPAND", "ISNAN", "WHERE", "EMBEDDING",
    "SHAPE", "GATHER", "UNSQUEEZE", "SQUEEZE", "PAD",
    "DIV", "SUB", "EQUAL", "IDENTITY", "CONSTANT",
}


def estimate(node, db, aux_energy_fb):
    t = node.get("op_type", "UNKNOWN")
    flops = node.get("flops", 0) or 0
    mem = node.get("memory_bytes", 0) or 0

    # 辅助算子：无 OP_MAP 映射的按 memory_bytes 缩放（仅当 mem > 0）
    if t in AUXILIARY_OPS and not OP_MAP.get(t):
        if mem > 0:
            unit_mem = 16384  # 4 × 4096 (FP16 读写各一次，一个 hidden_size 向量)
            return aux_energy_fb * max(1.0, mem / unit_mem)
        return aux_energy_fb

    csv_name = OP_MAP.get(t)
    if csv_name is None or csv_name not in db:
        return mem / 1e6 * 0.01

    entry = db[csv_name]
    ref_f = entry["flops"]

    # 有 FLOPs → FLOPs 缩放
    if ref_f > 0 and flops > 0:
        return entry["energy_j"] * (flops / ref_f)

    # 无 FLOPs → memory_bytes 缩放
    ref_mem = entry.get("mem_bytes", 0)
    if ref_mem > 0 and mem > 0:
        return entry["energy_j"] * (mem / ref_mem)

    return entry["energy_j"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--csv", required=True)
    p.add_argument("-g", "--graph", required=True)
    p.add_argument("-o", "--output", default="", help="输出路径（默认与 graph.json 同目录）")
    p.add_argument("-v", "--verbose", action="store_true", help="打印每个算子的映射与分类详情")
    p.add_argument("--gen-len", type=int, default=1, help="生成 token 数，decode 能量乘以该值（默认 1）")
    args = p.parse_args()

    db, aux_energy = load_operator_energy(args.csv)
    nodes, summary = load_graph(args.graph)
    graph_dir = Path(args.graph).parent
    output_path = Path(args.output) if args.output else graph_dir / "energy_report.txt"
    if not nodes:
        print(f"[energy] {args.graph}: empty")
        return

    # ---- verbose: 打印 CSV 侧的类别信息 ----
    if args.verbose:
        print(f"\n[CSV 算子 → 类别]  (aux fallback = {aux_energy:.6f}J)")
        for name, info in sorted(db.items()):
            print(f"  {name:20s} → category={info['category']:20s}  E={info['energy_j']:.6f}J  "
                  f"flops={info['flops']:,}  mem={info['mem_bytes']:,}")
        print(f"\n[ONNX → CSV 映射表 ({len(OP_MAP)} 条)]")
        for onnx, csv in sorted(OP_MAP.items()):
            print(f"  {onnx:20s} → {csv}")
        print(f"\n[逐算子映射详情]")
        print(f"  {'stage':8s} {'ONNX op_type':20s} {'→ CSV':20s} {'role':>7} {'flops':>12} {'mem':>10} {'energy(J)':>10}")
        print(f"  {'-'*8} {'-'*20} {'-'*20} {'-'*7} {'-'*12} {'-'*10} {'-'*10}")

    # ---- 按 stage 分组 ----
    pf_nodes = [n for n in nodes if n.get("stage") == "prefill"]
    dc_nodes = [n for n in nodes if n.get("stage") == "decode"]

    # ---- 逐行打印（verbose 模式）----
    if args.verbose:
        for node in nodes:
            t = node["op_type"]
            e = estimate(node, db, aux_energy)
            stage = node.get("stage", "?")
            role = "aux" if t in AUXILIARY_OPS else "primary"
            csv_name = OP_MAP.get(t, "<UNMAPPED>")
            flops = node.get("flops", 0) or 0
            mem = node.get("memory_bytes", 0) or 0
            print(f"  {stage:8s} {t:20s} → {csv_name:20s} {role:>7} {flops:>12,} {mem:>10,} {e:>10.6f}")

    # ---- 分阶段计算能耗 ----
    def stage_report(stage_nodes):
        op_energy = defaultdict(float)
        op_count = defaultdict(int)
        primary_e = 0.0
        aux_e = 0.0
        total_e = 0.0
        total_f = 0
        for n in stage_nodes:
            t = n["op_type"]
            e = estimate(n, db, aux_energy)
            op_energy[t] += e
            op_count[t] += 1
            total_e += e
            total_f += n.get("flops", 0) or 0
            if t in AUXILIARY_OPS:
                aux_e += e
            else:
                primary_e += e
        return op_energy, op_count, total_e, primary_e, aux_e, total_f

    stages = []
    for label, sn in [("Prefill", pf_nodes), ("Decode", dc_nodes)]:
        if sn:
            r = stage_report(sn)
            stages.append((label, sn, *r))  # 存储单 token 值，不在 stages 中乘以 gen_len

    # ---- 组装报告 ----
    lines = [
        "=" * 70,
        "  Energy Consumption Reconstruction",
        "=" * 70,
        f"  Graph:  {args.graph}",
        f"  Nodes:  {len(nodes)}  (prefill: {len(pf_nodes)} + decode: {len(dc_nodes)})",
        f"  Generation length: {args.gen_len} tokens",
        f"  Auxiliary fallback: {aux_energy:.6f} J/op (benchmark min; mapped ops use their target energy)",
    ]

    grand_total = 0.0
    grand_primary = 0.0
    grand_aux = 0.0
    grand_flops = 0

    for label, sn, _, _, total, pe, ae, tf in stages:
        mul = args.gen_len if label == "Decode" else 1
        n_count = len(sn)
        label_display = f"{label} (per token)" if mul > 1 else label
        lines += [
            "",
            f"  --- {label_display} ---",
            f"  Nodes: {n_count:>6d}  FLOPs: {tf/1e9:.2f} G",
            f"  Energy:     {total:.4f} J ({total*1000:.2f} mJ)",
            f"    ├─ Primary: {pe:.4f} J ({pe/total*100:.1f}%)",
            f"    └─ Auxiliary: {ae:.4f} J ({ae/total*100:.1f}%)",
        ]
        grand_total += total * mul
        grand_primary += pe * mul
        grand_aux += ae * mul
        grand_flops += tf * mul

    lines += [
        "",
        "-" * 70,
        f"  Aggregated (prefill + decode x{args.gen_len}):",
        f"  Total energy: {grand_total:.4f} J ({grand_total*1000:.2f} mJ)  "
        f"FLOPs: {grand_flops/1e9:.2f} G",
        f"    ├─ Primary: {grand_primary:.4f} J ({grand_primary/grand_total*100:.1f}%)",
        f"    └─ Auxiliary: {grand_aux:.4f} J ({grand_aux/grand_total*100:.1f}%)",
    ]

    # ---- 每个阶段的算符明细 ----
    for label, sn, op_energy, op_count, total, pe, ae, tf in stages:
        label_disp = f"{label} (per token)" if (label == "Decode" and args.gen_len > 1) else label
        lines += [
            "",
            f"  {label_disp} operators:",
            f"  {'Operator':25s} {'Cnt':>5s} {'Energy(J)':>10s} {'%':>6s}",
            "  " + "-" * 48,
        ]
        for op, e in sorted(op_energy.items(), key=lambda x: -x[1]):
            c = op_count[op]
            lines.append(f"  {op:25s} {c:>5d} {e:>8.6f}  {e/total*100:>5.1f}%")
        # 小计 primary/auxiliary
        pe_sub = sum(e for op, e in op_energy.items() if op not in AUXILIARY_OPS)
        ae_sub = sum(e for op, e in op_energy.items() if op in AUXILIARY_OPS)
        lines.append(f"  {'-'*48}")
        lines.append(f"  {'Primary subtotal':25s} {'':>5s} {pe_sub:>8.6f}  {pe_sub/total*100:>5.1f}%")
        lines.append(f"  {'Auxiliary subtotal':25s} {'':>5s} {ae_sub:>8.6f}  {ae_sub/total*100:>5.1f}%")

    lines.append("")
    lines.append("  Note: Prefill = 1 forward pass. Decode = 1 token (multiply by gen_len for total).")
    lines.append("  Primary ops scaled by FLOPs ratio; auxiliary mapped ops use target benchmark energy.")

    text = "\n".join(lines)
    print(text)
    output_path.write_text(text, encoding="utf-8")
    print(f"  -> saved to {output_path}")

    # ---- Token-level energy report ----
    decode_data = None
    for label, sn, op_energy, op_count, total, pe, ae, tf in stages:
        if label == "Decode":
            decode_data = (op_energy, op_count, total, ae)
            break

    if decode_data:
        op_energy, op_count, total_per, aux_per = decode_data
        pri_per = total_per - aux_per
        token_energy_txt = [
            "=" * 70,
            "  Token-level Energy Breakdown",
            "=" * 70,
            f"  Decode steps: {args.gen_len}",
            f"  Per-token energy: {total_per:.6f}J",
            f"  Each step executes the same operator graph ({sum(op_count.values())} nodes):",
            "",
        ]
        for op, e in sorted(op_energy.items(), key=lambda x: -x[1]):
            c = op_count[op]
            token_energy_txt.append(f"    {op:25s} x{c:>4d}  {e:>8.6f}J")
        token_energy_txt.append(f"    {'-'*48}")
        token_energy_txt.append(f"    {'Total per token':25s}  {total_per:>8.6f}J")
        token_energy_txt.append(f"    {'Auxiliary':25s}  {aux_per:>8.6f}J")
        token_energy_txt.append(f"    {'Primary':25s}  {pri_per:>8.6f}J")
        token_energy_txt.append("")
        token_energy_txt.append(f"  Per-token energy: {total_per:.6f}J (all {args.gen_len} tokens identical)")
        token_energy_txt.append(f"  Total decode:     {total_per * args.gen_len:.6f}J")
        token_energy_txt.append("")
        token_energy_txt.append("  Note: All decode tokens use the same ONNX graph traced at seq_len=1.")
        token_energy_txt.append("  The 1st token additionally initializes KV cache from prefill (included in prefill phase).")

        token_text = "\n".join(token_energy_txt)
        token_path = output_path.with_name(output_path.stem + "_token_energy.txt")
        token_path.write_text(token_text, encoding="utf-8")
        print(f"  -> saved to {token_path}")


if __name__ == "__main__":
    main()
