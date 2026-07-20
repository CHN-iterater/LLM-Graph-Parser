"""
能耗重构 — 基于 benchmark 拟合公式估算推理总能耗。
支持 Prefill / Decode 分阶段输出。

用法:
    python energy_consumption_refactor.py -g output/Qwen3-0.6B_*/graph.json \\
                                         -o energy_report.txt --gen-len 20
"""
import argparse, json
import math
from collections import defaultdict
from pathlib import Path


def prod(shape):
    p = 1
    for d in shape:
        if d > 0:
            p *= d
    return p


def extract_mnk_ins(ins, op_type=""):
    if not ins:
        return 0, 0, 0
    shapes = [t["shape"] for t in ins]
    if op_type in ("GEMM", "LINEAR", "BMM") and len(shapes) >= 2:
        A, B = shapes[0], shapes[1]
        if len(A) >= 2 and len(B) >= 2:
            return B[-1], prod(A[:-1]), A[-1]
    best = max(shapes, key=prod)
    if len(best) >= 2:
        return best[-1], prod(best[:-1]), 0
    return best[0], 1, 0


def extract_mnk_outs(outs):
    if not outs:
        return 0, 0, 0
    s = outs[0]["shape"]
    if not s:
        return 0, 0, 0
    if len(s) >= 2:
        return s[-1], prod(s[:-1]), 0
    return s[0], 1, 0


# -------------------------------------------------------------------
# 拟合公式：E(N,M,K) = t(N,M,K) × P(N,M,K)
# -------------------------------------------------------------------
def _logistic(x, c, d, p1, p2):
    return c + d / (1 + math.exp(-p1 * x + p2))


def _logistic2(n, m, c, d, p1, p2, p3):
    return _logistic(p1 * math.log2(n) + p2 * math.log2(m), c, d, 1, p3)


def _linear_t(a, b, *args):
    return a * prod(args) + b


# -------------------------------------------------------------------
# 算子公式注册表
# key = (t_func_args, p_func, energy_func)
#   t_func_args: ("N*M", a, b) 或 ("N*M*K", a, b) 或 ("N,M", aN, bM, cMN, d)
#   p_func: ("logistic1", c, d, p1, p2) 或 ("logistic2", c, d, p1, p2, p3) 或 ("const", v)
_FORMULAS = {
    # ---- 逐元素类（N from output）----
    "ADD":        ("N*M", 1.91941e-09, 0.0046922,      "logistic1", 56.7262, 410.776, 1.31213, 28.4027),
    "MUL":        ("N*M", 1.91877e-09, 0.00485007,     "logistic1", 57.7616, 402.096, 1.33557, 28.8748),
    "NEG":        ("N*M", 1.29882e-09, 0.00485478,     "logistic1", 58.2715, 390.255, 1.23136, 27.3271),
    "POW":        ("N*M", 1.30003e-09, 0.00557941,     "logistic1", 62.7531, 415.721, 1.29888, 29.0723),
    "SQRT":       ("N*M", 3.91666e-09, 0.0150007,      "logistic1", 62.309,  430.164, 1.09706, 24.2661),
    "RECIPROCAL": ("N*M", 3.91462e-09, 0.0157049,      "logistic1", 63.4888, 430.83,  1.18387, 26.1887),
    "SIGMOID":    ("N*M", 1.37637e-09, 0.00446598,     "logistic1", 63.3293, 544.362, 1.10708, 23.9914),
    "SiLU":       ("N*M", 1.37637e-09, 0.00446598,     "logistic1", 63.3293, 544.362, 1.10708, 23.9914),  # SiLU≈Sigmoid
    "RELU":       ("N*M", 1.37637e-09, 0.00446598,     "logistic1", 63.3293, 544.362, 1.10708, 23.9914),  # ReLU≈Sigmoid
    "GELU":       ("N*M", 1.37637e-09, 0.00446598,     "logistic1", 63.3293, 544.362, 1.10708, 23.9914),  # GELU≈Sigmoid
    "CAST":       ("N*M", 4.89748e-09, 0.0117041,      "logistic1", 55.116,  364.295, 1.09685, 23.5387),
    "DIV":        ("N*M", 1.91301e-09, 0.00517741,     "logistic1", 59.6176, 475.238, 1.2691,  27.334),
    "ISNAN":      ("N*M", 9.57676e-10, 0.00483714,     "logistic1", 56.9498, 427.538, 1.12089, 25.3951),
    "WHERE":      ("N*M", 3.178e-09,   0.0108666,      "logistic1", 61.6106, 393.398, 1.53998, 33.416),
    "TANH":       ("N*M", 1.34214e-09, 0.00499552,     "logistic1", 56.3172, 553.536, 1.00828, 22.1186),
    "ERF":        ("N*M", 1.34214e-09, 0.00499552,     "logistic1", 63.2964, 550.586, 1.2725,  27.6518),
    # ---- 计算密集型 ---- use input K
    "GEMM":       ("N*M*K", 3.23329e-12, 0.00923393,   "logistic1", 94.2941, 516.929, 1.7072,  50.5924),
    "LINEAR":     ("N*M*K", 3.53285e-12, 0.0103349,    "logistic1", 93.7326, 519.756, 1.58347, 47.3039),
    "BMM":        ("N*M*K", 3.23329e-12, 0.00923393,   "logistic1", 94.2941, 516.929, 1.7072,  50.5924),
    # ---- 不对称（N,M 分别作用） ----
    "SOFTMAX":    ("N,M", 1.72302e-06, 2.91939e-06, 2.58744e-11, 0.00384187,  "logistic2", 48.1421, 570.69,  0.755547, 0.442078, 13.0659),
    "REDUCEMEAN": ("N,M", 7.34328e-07, 6.47576e-07, 2.49611e-11, 0.0116988,   "logistic2", 61.9219, 467.778, 0.672083, 0.972062, 20.038),
    "LAYER_NORM": ("N,M", 2.41765e-06, 2.85593e-06, 3.56326e-12, 0.00538381,  "logistic2", 63.5845, 513.813, 0.947821, 0.49298, 15.8803),
    "RMSNorm":    ("N,M", 2.41765e-06, 2.85593e-06, 3.56326e-12, 0.00538381,  "logistic2", 63.5845, 513.813, 0.947821, 0.49298, 15.8803),
    "EMBEDDING":  ("N,M", 6.76365e-09, 8.36773e-06, 0, 0.00121189,            "logistic2", 47.167,  454.384, 0.0348264, 3.05941, 33.9716),
    # ---- 数据搬运类（几乎常数） ----
    "RESHAPE":    ("N*M", 0, 0.00201895,                "const", 52.3686),
    "TRANSPOSE":  ("N*M", 0, 0.00278664,                "const", 50.6483),
    "SLICE":      ("N*M", 0, 0.00297642,                "const", 49.4025),
    "EXPAND":     ("N*M", 1.45904e-09, 0.0139601,       "logistic1", 55.4878, 341.649, 1.85321, 42.2219),
    "CAT":        ("N*M", 2.72884e-09, 0.00494385,      "logistic1", 58.5417, 351.322, 1.52265, 32.389),
    "KVCacheRead":  ("N*M", 4.89748e-09, 0.0117041,     "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "KVCacheWrite": ("N*M", 4.89748e-09, 0.0117041,     "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "AllReduce":  ("N*M", 4.89748e-09, 0.0117041,       "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "AllGather":  ("N*M", 4.89748e-09, 0.0117041,       "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "MemcpyD2D":  ("N*M", 4.89748e-09, 0.0117041,       "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "Reduction":  ("N*M", 4.89748e-09, 0.0117041,       "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "DROPOUT":    ("N*M", 0, 0.005,                     "const", 50.0),
    "UNKNOWN":    ("N*M", 0, 0.005,                     "const", 50.0),
}

# ONNX op_type → formula name
FORMULA_NAME = {
    "LINEAR": "GEMM", "GEMM": "GEMM", "BMM": "BMM",
    "ATTENTION": "SOFTMAX", "FLASH_ATTENTION": "SOFTMAX",
    "SOFTMAX": "SOFTMAX",
    "LAYER_NORM": "LAYER_NORM", "RMS_NORM": "RMSNorm",
    "GELU": "GELU", "SILU": "SiLU", "RELU": "RELU", "SIGMOID": "SIGMOID",
    "REDUCESUM": "REDUCEMEAN", "REDUCEMEAN": "REDUCEMEAN", "MEAN": "REDUCEMEAN",
    "KV_CACHE_READ": "KVCacheRead", "KV_CACHE_WRITE": "KVCacheWrite",
    "ADD": "ADD", "MUL": "MUL", "CAT": "CAT", "SLICE": "SLICE", "EXPAND": "EXPAND",
    "RESHAPE": "RESHAPE", "TRANSPOSE": "TRANSPOSE",
    "CAST": "CAST", "NEG": "NEG", "POW": "POW",
    "SQRT": "SQRT", "RECIPROCAL": "RECIPROCAL",
    "ISNAN": "ISNAN", "WHERE": "WHERE", "EMBEDDING": "EMBEDDING",
    "DIV": "DIV", "SUB": "ADD",
}

# Operator categories for compute/memory/data_movement/communication
OP_CAT: dict[str, str] = {}
for _k, _f in _FORMULAS.items():
    if _k in ("AllReduce", "AllGather"):
        OP_CAT[_k] = "communication"
    elif _f[0] == "N*M*K":
        OP_CAT[_k] = "compute_bound"
    elif _f[0] in ("N*M", "N*M*K") and len(_f) >= 5 and _f[3] == "const":
        OP_CAT[_k] = "data_movement"
    elif _f[0] == "N,M" and len(_f) >= 7 and _f[5] == "const":
        OP_CAT[_k] = "data_movement"
    else:
        OP_CAT[_k] = "memory_bound"
ONNX_CAT: dict[str, str] = {}
for _onnx_op, _fk in FORMULA_NAME.items():
    ONNX_CAT[_onnx_op] = OP_CAT.get(_fk, "memory_bound")


def estimate_by_category(nodes, stage=None):
    from collections import defaultdict
    result = defaultdict(float)
    for n in nodes:
        if stage and n.get("stage") != stage:
            continue
        op = n.get("op_type", "UNKNOWN")
        e = estimate(n)
        result[ONNX_CAT.get(op, "memory_bound")] += e
    return dict(result)


def energy_j(N, M, K, formula_key):
    f = _FORMULAS[formula_key]
    t_type = f[0]

    # --- time per iter (ms) ---
    if t_type == "N*M*K":
        _, a, b, p_type, *prest = f
        t = a * N * M * K + b
        prest = (p_type, *prest)
        size = N * M * K  # power 公式也用 N*M*K
    elif t_type == "N*M":
        _, a, b, p_type, *prest = f
        t = a * N * M + b
        prest = (p_type, *prest)
        size = N * M
    elif t_type == "N,M":
        _, a_n, b_m, c_mn, d, p_type, *prest = f
        t = a_n * N + b_m * M + c_mn * M * N + d
        prest = (p_type, *prest)
        size = N * M
    else:
        return 0.0

    t_ms = max(t, 0.0)

    # --- power (W) ---
    p_type = prest[0]
    if p_type == "logistic1":
        _, c, d, p1, p2 = prest
        p = c + d / (1 + math.exp(-p1 * math.log2(max(size, 1)) + p2))
    elif p_type == "logistic2":
        _, c, d, p1_n, p2_m, p3 = prest
        x = p1_n * math.log2(N) + p2_m * math.log2(M)
        p = c + d / (1 + math.exp(-x + p3))
    elif p_type == "const":
        _, v = prest
        p = v
    else:
        return 0.0

    return t_ms * p / 1000  # mJ → J


def estimate(node):
    t = node.get("op_type", "UNKNOWN")
    ins = node.get("input_tensors", [])
    outs = node.get("output_tensors", [])

    key = FORMULA_NAME.get(t)
    if key is None or key not in _FORMULAS:
        key = "UNKNOWN"

    iN, iM, iK = extract_mnk_ins(ins, t)

    if t in ("GEMM", "LINEAR", "BMM"):
        N, M, K = iN, iM, iK
    else:
        N, M, K = extract_mnk_outs(outs)

    if N <= 0 or M <= 0:
        N, M = max(N, 1), max(M, 1)

    return energy_j(N, M, K, key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-g", "--graph", required=True)
    p.add_argument("-o", "--output", default="", help="输出路径（默认与 graph.json 同目录）")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--gen-len", type=int, default=1)
    args = p.parse_args()

    nodes, summary = load_graph(args.graph)
    graph_dir = Path(args.graph).parent
    output_path = Path(args.output) if args.output else graph_dir / "energy_report.txt"
    if not nodes:
        print(f"[energy] {args.graph}: empty")
        return

    pf_nodes = [n for n in nodes if n.get("stage") == "prefill"]
    dc_nodes = [n for n in nodes if n.get("stage") == "decode"]

    def stage_report(sn):
        op_energy = defaultdict(float)
        op_count = defaultdict(int)
        te = 0.0
        for n in sn:
            op = n["op_type"]
            e = estimate(n)
            op_energy[op] += e
            op_count[op] += 1
            te += e
        return op_energy, op_count, te

    stages = []
    for label, sn in [("Prefill", pf_nodes), ("Decode", dc_nodes)]:
        if sn:
            stages.append((label, *stage_report(sn)))

    lines = [
        "=" * 70, "  Energy Consumption Reconstruction", "=" * 70,
        f"  Graph:  {args.graph}",
        f"  Nodes:  {len(nodes)}  (prefill: {len(pf_nodes)} + decode: {len(dc_nodes)})",
        f"  Generation length: {args.gen_len} tokens",
    ]

    gt = 0.0
    for label, op_energy, op_count, total in stages:
        mul = args.gen_len if label == "Decode" else 1
        disp = f"{label} (per token)" if mul > 1 else label
        lines += ["", f"  --- {disp} ---",
                  f"  Nodes: {sum(op_count.values()):>6d}",
                  f"  Energy:  {total:.4f}J ({total*1000:.2f}mJ)"]
        gt += total * mul

    lines += ["", "-" * 70,
              f"  Aggregated (prefill + decode x{args.gen_len}):",
              f"  Total: {gt:.4f}J ({gt*1000:.2f}mJ)"]

    for label, op_energy, op_count, total in stages:
        mul = args.gen_len if label == "Decode" else 1
        disp = f"{label} (per token)" if mul > 1 else label
        lines += ["", f"  {disp} operators:",
                  f"  {'Operator':25s} {'Cnt':>5s} {'Energy(mJ)':>12s} {'%':>6s}",
                  "  " + "-" * 48]
        for op, e in sorted(op_energy.items(), key=lambda x: -x[1]):
            c = op_count[op]
            lines.append(f"  {op:25s} {c:>5d} {e*1000:>10.4f}  {e/total*100:>5.1f}%")

    lines += ["", "  Note: Energy estimated from benchmark fitting formulas."]
    text = "\n".join(lines)
    print(text)
    output_path.write_text(text, encoding="utf-8")
    print(f"  -> saved to {output_path}")


def load_graph(graph_path):
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("summary", {})


if __name__ == "__main__":
    main()
