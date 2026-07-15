"""
能耗重构 — 将单算子能耗数据映射回 graph.json 计算图，估算推理总能耗。
支持 Prefill / Decode 分阶段输出。

用法:
    python energy_consumption_refactor.py -c operator_energy_comparison.csv \\
                                         -g output/Qwen3-0.6B_20260710_*/graph.json \\
                                         -o energy_report.txt --gen-len 20
"""
import argparse, json, csv
from collections import defaultdict
from pathlib import Path


def prod(shape):
    """返回 shape 各维乘积（忽略 -1 表示动态维度）。"""
    p = 1
    for d in shape:
        if d > 0:
            p *= d
    return p


def extract_mnk_ins(ins, op_type=""):
    """从输入 tensor 列表提取 (N, M, K)。

    GEMM/BMM: A[M×K] @ B[K×N] → 返回 (N, M, K) = (B[-1], prod(A[:-1]), A[-1])
    其余: 取元素数最多的输入 tensor 的 shape（广播等价于最大者）
    """
    if not ins:
        return 0, 0, 0
    shapes = [t["shape"] for t in ins]

    if op_type in ("GEMM", "LINEAR", "BMM") and len(shapes) >= 2:
        A, B = shapes[0], shapes[1]
        if len(A) >= 2 and len(B) >= 2:
            return B[-1], prod(A[:-1]), A[-1]

    # 取 broadcast 后的有效 shape（元素数最多的输入）
    best = max(shapes, key=prod)
    if len(best) >= 2:
        return best[-1], prod(best[:-1]), 0
    return best[0], 1, 0


def extract_mnk_outs(outs):
    """从输出 tensor 列表提取 (N, M, K)。"""
    if not outs:
        return 0, 0, 0
    s = outs[0]["shape"]
    if not s:
        return 0, 0, 0
    if len(s) >= 2:
        return s[-1], prod(s[:-1]), 0
    return s[0], 1, 0


def load_operator_energy(csv_path):
    """operator_energy_comparison.csv → (lookup_table, aux_energy)

    lookup_table: {(op_type, iN, iM, iK, oN, oM, oK): energy_mJ}
    """
    table = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        col = "csv_operator_summary_repeat5_方式1(mJ)"
        for row in reader:
            key = (
                row["operator"].strip(),
                int(row["input_N"]), int(row["input_M"]), int(row["input_K"]),
                int(row["output_N"]), int(row["output_M"]), int(row["output_K"]),
            )
            table[key] = float(row[col])  # mJ

    aux = min(v for v in table.values()) if table else 0.0
    return table, aux


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


def _csv_lookup(t, ins, outs, table):
    """从 CSV 查找匹配的 per-iter 能量（J）。返回 None 表示未命中。"""
    iN, iM, iK = extract_mnk_ins(ins, t)
    oN, oM, oK = extract_mnk_outs(outs)

    # 用 ONNX op_type 直接查
    key = (t, iN, iM, iK, oN, oM, oK)
    if key in table:
        return table[key] / 1000

    # 用 OP_MAP 映射名查
    csv_name = OP_MAP.get(t)
    if csv_name:
        key = (csv_name, iN, iM, iK, oN, oM, oK)
        if key in table:
            return table[key] / 1000
        key0 = (csv_name, iN, iM, 0, oN, oM, 0)
        if key0 in table:
            return table[key0] / 1000

    # 通配：同名首条
    for k, v in table.items():
        if k[0] == (csv_name or t):
            return v / 1000
    return None


def estimate(node, table, aux_fb):
    t = node.get("op_type", "UNKNOWN")
    mem = node.get("memory_bytes", 0) or 0
    ins = node.get("input_tensors", [])
    outs = node.get("output_tensors", [])

    # 1) CSV 精确查找优先
    e = _csv_lookup(t, ins, outs, table)
    if e is not None:
        return e

    # 2) 辅助算子按 mem 缩放
    if t in AUXILIARY_OPS:
        if mem > 0:
            unit_mem = 16384
            return aux_fb * max(1.0, mem / unit_mem)
        return aux_fb

    # 3) fallback
    return mem / 1e6 * 0.01


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--csv", required=True)
    p.add_argument("-g", "--graph", required=True)
    p.add_argument("-o", "--output", default="", help="输出路径（默认与 graph.json 同目录）")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--gen-len", type=int, default=1)
    args = p.parse_args()

    table, aux = load_operator_energy(args.csv)
    nodes, summary = load_graph(args.graph)
    graph_dir = Path(args.graph).parent
    output_path = Path(args.output) if args.output else graph_dir / "energy_report.txt"
    if not nodes:
        print(f"[energy] {args.graph}: empty")
        return

    if args.verbose:
        print(f"\n[CSV entries]  {len(table)}  (aux_fb = {aux:.4f}mJ)")
        for op in sorted(set(k[0] for k in table)):
            print(f"  {op}: {sum(1 for k in table if k[0] == op)} dims")

    # 分组
    pf_nodes = [n for n in nodes if n.get("stage") == "prefill"]
    dc_nodes = [n for n in nodes if n.get("stage") == "decode"]

    if args.verbose:
        print(f"\n[逐算子映射]")
        print(f"  {'stage':8s} {'op_type':20s} {'→CSV':12s} {'iN,iM,iK':>14s} {'oN,oM,oK':>14s} {'E(mJ)':>8s}")
        print(f"  {'-'*8} {'-'*20} {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
        for node in nodes:
            t = node["op_type"]
            csv_name = OP_MAP.get(t, t)
            iN, iM, iK = extract_mnk_ins(node.get("input_tensors", []), t)
            oN, oM, oK = extract_mnk_outs(node.get("output_tensors", []))
            e = estimate(node, table, aux)
            print(f"  {node.get('stage','?'):8s} {t:20s} {csv_name:12s} "
                  f"({iN},{iM},{iK})  ({oN},{oM},{oK})  {e*1000:>8.4f}")

    # 分阶段计算
    def stage_report(sn):
        op_energy = defaultdict(float)
        op_count = defaultdict(int)
        pe = ae = te = 0.0
        for n in sn:
            t = n["op_type"]
            e = estimate(n, table, aux)
            op_energy[t] += e
            op_count[t] += 1
            te += e
            if t in AUXILIARY_OPS:
                ae += e
            else:
                pe += e
        return op_energy, op_count, te, pe, ae

    stages = []
    for label, sn in [("Prefill", pf_nodes), ("Decode", dc_nodes)]:
        if sn:
            stages.append((label, *stage_report(sn)))

    # 输出
    lines = [
        "=" * 70,
        "  Energy Consumption Reconstruction",
        "=" * 70,
        f"  Graph:  {args.graph}",
        f"  Nodes:  {len(nodes)}  (prefill: {len(pf_nodes)} + decode: {len(dc_nodes)})",
        f"  Generation length: {args.gen_len} tokens",
        f"  CSV entries: {len(table)}, aux_fb = {aux:.4f}mJ",
    ]

    gt = gp = ga = 0.0
    for label, op_energy, op_count, total, pe, ae in stages:
        mul = args.gen_len if label == "Decode" else 1
        n_count = sum(op_count.values())
        disp = f"{label} (per token)" if mul > 1 else label
        lines += ["", f"  --- {disp} ---",
                  f"  Nodes: {n_count:>6d}",
                  f"  Energy:  {total:.4f}J ({total*1000:.2f}mJ)",
                  f"    ├─ Primary: {pe:.4f}J ({pe/total*100:.1f}%)",
                  f"    └─ Aux: {ae:.4f}J ({ae/total*100:.1f}%)"]
        gt += total * mul
        gp += pe * mul
        ga += ae * mul

    lines += ["", "-" * 70,
              f"  Aggregated (prefill + decode x{args.gen_len}):",
              f"  Total: {gt:.4f}J ({gt*1000:.2f}mJ)",
              f"    ├─ Primary: {gp:.4f}J ({gp/gt*100:.1f}%)",
              f"    └─ Aux: {ga:.4f}J ({ga/gt*100:.1f}%)"]

    for label, op_energy, op_count, total, pe, ae in stages:
        mul = args.gen_len if label == "Decode" else 1
        disp = f"{label} (per token)" if mul > 1 else label
        lines += ["", f"  {disp} operators:",
                  f"  {'Operator':25s} {'Cnt':>5s} {'Energy(mJ)':>12s} {'%':>6s}",
                  "  " + "-" * 48]
        for op, e in sorted(op_energy.items(), key=lambda x: -x[1]):
            c = op_count[op]
            lines.append(f"  {op:25s} {c:>5d} {e*1000:>10.4f}  {e/total*100:>5.1f}%")
        pe_sub = sum(e for op, e in op_energy.items() if op not in AUXILIARY_OPS)
        ae_sub = sum(e for op, e in op_energy.items() if op in AUXILIARY_OPS)
        lines += [f"  {'-'*48}",
                  f"  {'Primary subtotal':25s} {'':>5s} {pe_sub*1000:>10.4f}  {pe_sub/total*100:>5.1f}%",
                  f"  {'Aux subtotal':25s} {'':>5s} {ae_sub*1000:>10.4f}  {ae_sub/total*100:>5.1f}%"]

    lines += ["", "  Note: Per-operator energy looked up from benchmark CSV by exact dimensions."]
    text = "\n".join(lines)
    print(text)
    output_path.write_text(text, encoding="utf-8")
    print(f"  -> saved to {output_path}")


if __name__ == "__main__":
    main()
