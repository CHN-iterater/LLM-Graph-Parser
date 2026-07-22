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
# Formula types (t = time per iter in ms, P = net GPU power in W):
#   t types:
#     N*M_pw:   b_const  (when N*M<th) | a*N*M+b  (when N*M>=th)
#     N*M*K_pw: b_const  (when N*M*K<th) | a*N*M*K+b  (when N*M*K>=th)
#     N+M_pw:   t(N)+t(M), each piecewise: const (N<|M<th) | a*N|M+b (N|M>=th)
#     N*M:      t = a*N*M + b
#     N*M*K:    t = a*N*M*K + b
#     N,M:      t = a_n*N + b_m*M + c_mn*M*N + d
#   P types:
#     logistic1: c + d/(1+exp(-p1*log2(vol) + p2))
#     logistic2: c + d/(1+exp(-p1*log2(N) - p2*log2(M) + p3))
#     const:     P = v
_FORMULAS = {
    # ---- 逐元素类（N*M_pw: t = b_const if vol < th else a*vol + b）----
    "ADD":        ("N*M_pw", 1.92454e-09, 0.00384307, 0.00606783, 4.79786e+06,  "logistic1", 56.7262, 410.776, 1.31213, 28.4027),
    "MUL":        ("N*M_pw", 1.92413e-09, 0.00393729, 0.00626854, 6.12206e+06,  "logistic1", 57.7616, 402.096, 1.33557, 28.8748),
    "NEG":        ("N*M_pw", 1.30722e-09, 0.00336362, 0.00603114, 6.64555e+06,  "logistic1", 58.2715, 390.255, 1.23136, 27.3271),
    "POW":        ("N*M_pw", 1.30552e-09, 0.00435395, 0.00688748, 5.84812e+06,  "logistic1", 62.7531, 415.721, 1.29888, 29.0723),
    "SQRT":       ("N*M_pw", 3.95174e-09, 0.00871169, 0.0190033,  7.0177e+06,   "logistic1", 62.309,  430.164, 1.09706, 24.2661),
    "RECIPROCAL": ("N*M_pw", 3.95271e-09, 0.00888308, 0.0198299,  5.19744e+06,  "logistic1", 63.4888, 430.83,  1.18387, 26.1887),
    "SIGMOID":    ("N*M_pw", 1.38351e-09, 0.00320094, 0.00563601, 5.2119e+06,   "logistic1", 63.3293, 544.362, 1.10708, 23.9914),
    "SiLU":       ("N*M_pw", 1.38351e-09, 0.00320094, 0.00563601, 5.2119e+06,   "logistic1", 63.3293, 544.362, 1.10708, 23.9914),  # SiLU≈Sigmoid
    "RELU":       ("N*M_pw", 1.30558e-09, 0.00334062, 0.00618347, 4.85261e+06,  "logistic1", 64.3431, 421.677, 1.20796, 26.8245),
    "GELU":       ("N*M_pw", 1.38351e-09, 0.00320094, 0.00563601, 5.2119e+06,   "logistic1", 63.3293, 544.362, 1.10708, 23.9914),  # GELU≈Sigmoid
    "CAST":       ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197,  3.88858e+06,  "logistic1", 55.116,  364.295, 1.09685, 23.5387),
    "DIV":        ("N*M_pw", 1.91982e-09, 0.0040125,  0.006656,   7.42963e+06,  "logistic1", 59.6176, 475.238, 1.2691,  27.334),
    "ISNAN":      ("N*M_pw", 9.64183e-10, 0.00374666, 0.00614436, 1.39596e+07,  "logistic1", 56.9498, 427.538, 1.12089, 25.3951),
    "WHERE":      ("N*M_pw", 3.20141e-09, 0.00669723, 0.0138635,  4.78315e+06,  "logistic1", 61.6106, 393.398, 1.53998, 33.416),
    "TANH":       ("N*M_pw", 1.30595e-09, 0.00380376, 0.00569155, 4.5093e+06,   "logistic1", 56.3172, 553.536, 1.00828, 22.1186),
    "ERF":        ("N*M_pw", 1.35364e-09, 0.00293892, 0.00633921, 7.92668e+06,  "logistic1", 63.2964, 550.586, 1.2725,  27.6518),
    # ---- 计算密集型（N*M*K_pw: t = b_const if vol < th else a*vol + b）----
    "GEMM":       ("N*M*K_pw", 3.24956e-12, 0.00624627, 0.0113172, 3.56456e+09,  "logistic1", 94.2933, 516.93,  1.70718, 50.5915),
    "LINEAR":     ("N*M*K_pw", 3.54739e-12, 0.00771382, 0.0132276, 6.54031e+09,  "logistic1", 93.7326, 519.756, 1.58347, 47.3039),
    "BMM":        ("N*M*K_pw", 3.24956e-12, 0.00624627, 0.0113172, 3.56456e+09,  "logistic1", 94.2933, 516.93,  1.70718, 50.5915),
    # ---- 不对称（N+M_pw: t(N,M) = t(N) + t(M)，各分段）----
    "SOFTMAX":    ("N+M_pw", 2392.3, 0.00284274, 1.74917e-06, 0.00240572,
                             1134.02, 0.00366614, 2.94565e-06, 0.0041931,
                             "logistic2", 48.1421, 570.69, 0.755547, 0.442078, 13.0659),
    "REDUCEMEAN": ("N+M_pw", 506.557, 0.000601891, 7.59524e-07, 0.00669025,
                             57.7648, 0.000138298, 6.72566e-07, 0.00622868,
                             "logistic2", 61.9221, 467.787, 0.672083, 0.972062, 20.0381),
    "LAYER_NORM": ("N+M_pw", 3108.03, 0.00526191, 2.43963e-06, 2.60804e-05,
                             7188.6, 0.00511926, 2.86917e-06, 1.62517e-05,
                             "logistic2", 52.196, 529.425, 0.895789, 0.464159, 14.9533),
    "RMSNorm":    ("N+M_pw", 3108.03, 0.00526191, 2.43963e-06, 2.60804e-05,
                             7188.6, 0.00511926, 2.86917e-06, 1.62517e-05,
                             "logistic2", 52.196, 529.425, 0.895789, 0.464159, 14.9533),
    "EMBEDDING":  ("N+M_pw", 2450.06, 0, 0, 0.00151323,
                             1661.56, 0.00952048, 8.37554e-06, 5.63047e-05,
                             "logistic2", 47.167, 454.384, 0.0348264, 3.05941, 33.9716),
    # ---- 数据搬运类（常数）----
    "RESHAPE":    ("N*M", 0, 0.00178114,                "const", 48.6435),
    "TRANSPOSE":  ("N*M_pw", 7.29336e-09, 0.0074075, 0.0101497, 1.78835e+06, "logistic1", 51.4168, 406.984, 0.758696, 15.2999),
    "TRANSPOSE2RESHAPE": ("N*M", 0, 0.00268636,               "const", 53.5681),
    "SLICE":      ("N*M", 0, 0.00271925,                "const", 54.7875),
    "EXPAND":     ("N*M", 0, 0.00184319,                "const", 53.633),
    "CAT":        ("N*M_pw", 2.74955e-09, 0.00127788, 0.00705622, 3.84259e+06, "logistic1", 58.5417, 351.322, 1.52265, 32.389),
    "KVCacheRead":  ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197, 3.88858e+06,  "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "KVCacheWrite": ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197, 3.88858e+06,  "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "AllReduce":  ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197, 3.88858e+06,  "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "AllGather":  ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197, 3.88858e+06,  "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "MemcpyD2D":  ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197, 3.88858e+06,  "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "Reduction":  ("N*M_pw", 4.92851e-09, 0.00613371, 0.0152197, 3.88858e+06,  "logistic1", 55.116, 364.295, 1.09685, 23.5387),
    "DROPOUT":    ("N*M", 0, 0.005,                     "const", 50.0),
    "UNKNOWN":    ("N*M", 0, 0.005,                     "const", 50.0),
}

_TABLE: dict = {}

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


_VIEW_OPS = {"RESHAPE", "VIEW", "SLICE", "EXPAND", "TRANSPOSE"}
_FUSABLE_ACT = {"ADD", "GELU", "RELU"}

def _build_idx(nodes):
    idx = {}
    for n in nodes:
        idx[n.get("op_id", "")] = n
    return idx

def _match_chain(node, expected, idx, skip):
    chain = [node]
    cur = node
    for exp in expected:
        found = None
        for cid in cur.get("children", []):
            c = idx.get(cid)
            if c and c.get("op_type") == exp and cid not in skip:
                found = c
                break
        if found is None:
            return None
        chain.append(found)
        cur = found
    return chain

def estimate_with_fusion(nodes, stage=None, summary=None):
    from energy_consumption_refactor import extract_mnk_ins as _emi
    from collections import defaultdict
    idx = _build_idx(nodes)
    skip = {}
    rule_counts = {"view": 0, "gemm_act": 0, "ln": 0, "rms": 0, "attn": 0, "swiglu": 0, "gemm_stack": 0}
    
    # Model architecture for framework overhead correction
    _hidden_dim = 0
    _num_layers = 0
    if summary:
        _num_layers = summary.get("num_layers", 0)
        # hidden_dim: most common K among GEMMs where K == N
        from collections import Counter
        _ks = Counter()
        for n in nodes:
            if n.get("op_type") in ("GEMM", "LINEAR", "BMM"):
                ins = n.get("input_tensors", [])
                if len(ins) >= 2:
                    s = ins[1].get("shape", [])
                    if len(s) >= 2:
                        _ks[s[0]] += 1
        _hidden_dim = max(_ks, key=_ks.get) if _ks else 0
    # Rule 1: view ops
    for n in nodes:
        if n.get("op_type") in {"RESHAPE","VIEW","SLICE","EXPAND"}:
            skip[n.get("op_id", "")] = 0.0
            rule_counts["view"] += 1

    # Rule 1b: weight transpose fusion (2D weight transpose feeding into GEMM is fused by cuBLAS)
    for n in nodes:
        if n.get("op_type") != "TRANSPOSE":
            continue
        _ins = n.get("input_tensors", [])
        if not _ins:
            continue
        _s = _ins[0].get("shape", [])
        if len(_s) != 2:  # only 2D weight tensors, skip 3D/4D activation transposes
            continue
        from functools import reduce as _rd
        import operator as _op
        _vol = _rd(_op.mul, [d for d in _s if d > 0], 1)
        if _vol <= 200000:
            continue
        for _cid in n.get("children", []):
            _c = idx.get(_cid)
            if _c and _c.get("op_type") in {"GEMM", "LINEAR", "BMM"}:
                skip[n.get("op_id", "")] = 0.0
                rule_counts.setdefault("wtrans", 0)
                rule_counts["wtrans"] += 1
                break

    LN_CHAIN = ["SUB", "POW", "SQRT", "DIV", "MUL", "ADD"]
    RMS_CHAIN = ["REDUCEMEAN", "ADD", "SQRT", "RECIPROCAL", "MUL", "CAST", "MUL"]
    ATTN_MID = ["SOFTMAX", "ISNAN", "WHERE"]

    # Rule 2: GEMM -> elementwise fusion
    for n in nodes:
        if n.get("op_type") not in {"GEMM", "LINEAR", "BMM"}:
            continue
        for cid in n.get("children", []):
            c = idx.get(cid)
            if not c or c.get("op_type") not in _FUSABLE_ACT:
                continue
            parents = [p for p in c.get("parents", []) if p in idx]
            if len(parents) <= 1:
                skip[cid] = 0.1

    # Rule 3: LayerNorm chain
    for n in nodes:
        if n.get("op_type") == "REDUCEMEAN" and n.get("op_id", "") not in skip:
            chain = _match_chain(n, LN_CHAIN, idx, skip)
            if chain and len(chain) == 7:
                for cn in chain[1:]:
                    skip[cn.get("op_id", "")] = 0.3
                    rule_counts["ln"] += 1

    # Rule 4: RMSNorm chain
    for n in nodes:
        if n.get("op_type") == "POW" and n.get("op_id", "") not in skip:
            chain = _match_chain(n, RMS_CHAIN, idx, skip)
            if chain and len(chain) == 8:
                for cn in chain[1:]:
                    skip[cn.get("op_id", "")] = 0.3
                    rule_counts["rms"] += 1

    # Rule 5: Attention chain
    for n in nodes:
        if n.get("op_type") not in {"BMM", "GEMM"} or n.get("op_id", "") in skip:
            continue
        if not any(c.get("op_type") == "SOFTMAX" for c in [idx.get(cid) for cid in n.get("children", [])] if c):
            continue
        chain = _match_chain(n, ATTN_MID, idx, skip)
        if chain and len(chain) == 4:
            last = chain[-1]
            for cid in last.get("children", []):
                c = idx.get(cid)
                if c and c.get("op_type") in {"BMM", "GEMM"}:
                    for cn in chain[1:]:
                        skip[cn.get("op_id", "")] = 0.2
                    break


        # Rule 7: GEMM stacking
    gemm_groups = defaultdict(list)
    import energy_consumption_refactor as _ecr
    for n in nodes:
        if n.get("op_type") == "GEMM":
            pset = tuple(sorted(n.get("parents", [])))
            ins = n.get("input_tensors", [])
            m = _emi(ins, 'GEMM')[1]
            k = _emi(ins, 'GEMM')[2]
            gemm_groups[(pset, m, k)].append(n)

    for (pset, m, k), gg in gemm_groups.items():
        if len(gg) <= 1:
            continue
        total_n = sum(_emi(g.get("input_tensors",[]), 'GEMM')[0] for g in gg)
        for g in gg[:-1]:
            skip[g.get("op_id", "")] = 0.0
            rule_counts["gemm_stack"] += 1
        gg[-1]["_fused_gemm_n"] = total_n

    result = defaultdict(float)
    total = 0.0
    for n in nodes:
        if stage and n.get("stage") != stage:
            continue
        nid = n.get("op_id", "")
        fused_n = n.get("_fused_gemm_n", 0)
        fuse_discount = skip.get(nid, 1.0)
        if fused_n:
            ins = n.get("input_tensors", [])
            t = n.get("op_type", "UNKNOWN")
            if t == "GEMM" and len(ins) >= 2:
                # Modify B's last dim for CSV lookup: [K, fused_n] instead of [K, N]
                b_shape = list(ins[1]["shape"])
                if len(b_shape) >= 2:
                    b_shape[-1] = fused_n
                syn_key = (t, ins[0]["shape"], b_shape, [t["shape"] for t in n.get("output_tensors",[])])
                # Directly look up CSV: use extract_mnk style
                from energy_consumption_refactor import extract_mnk_ins, extract_mnk_outs
                iN2, iM2, iK2 = fused_n, 0, 0
                # Actually just patch the input tensors and call estimate
                import copy
                syn = copy.deepcopy(n)
                syn["input_tensors"][1]["shape"] = b_shape
                e = estimate(syn)
            else:
                e = estimate(n)
        else:
            e = estimate(n)
        e *= fuse_discount
        result[ONNX_CAT.get(n.get("op_type", "UNKNOWN"), "memory_bound")] += e
        total += e
    
    # Framework overhead correction (EMNLP 2023: The Framework Tax)
    if _hidden_dim > 0 and _num_layers > 0:
        _stage_nodes = [n for n in nodes if (stage and n.get("stage") == stage) or not stage]
        _num_ops = len(_stage_nodes)
        _fw_energy = 3.130e-07 * _hidden_dim * _num_ops
        total += _fw_energy
        result["compute_bound"] = result.get("compute_bound", 0) + _fw_energy * 0.5
        result["memory_bound"] = result.get("memory_bound", 0) + _fw_energy * 0.3
        result["data_movement"] = result.get("data_movement", 0) + _fw_energy * 0.2
    
    return dict(result), total, len(skip)


def energy_j(N, M, K, formula_key):
    f = _FORMULAS[formula_key]
    t_type = f[0]

    # --- time per iter (ms) ---
    if t_type == "N*M*K_pw":
        _, a, b, b_const, threshold, p_type, *prest = f
        vol = N * M * K
        t = b_const if vol < threshold else a * vol + b
        prest = (p_type, *prest)
        size = vol
    elif t_type == "N*M*K":
        _, a, b, p_type, *prest = f
        t = a * N * M * K + b
        prest = (p_type, *prest)
        size = N * M * K
    elif t_type == "N*M_pw":
        _, a, b, b_const, threshold, p_type, *prest = f
        vol = N * M
        t = b_const if vol < threshold else a * vol + b
        prest = (p_type, *prest)
        size = vol
    elif t_type == "N*M":
        _, a, b, p_type, *prest = f
        t = a * N * M + b
        prest = (p_type, *prest)
        size = N * M
    elif t_type == "N+M_pw":
        _, n_th, n_const, n_a, n_b, m_th, m_const, m_a, m_b, p_type, *prest = f
        tn = n_const if N < n_th else n_a * N + n_b
        tm = m_const if M < m_th else m_a * M + m_b
        t = tn + tm
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


def _is_transpose_view(node):
    """判断 TRANSPOSE 是 view 还是 copy：通过数据量和 CSV 实测数据的对应关系"""
    ins = node.get("input_tensors", [])
    if not ins:
        return True
    shape = ins[0].get("shape", [])
    vol = 1
    for d in shape:
        if d > 0:
            vol *= d
    # CSV 中 128×16=2048 以下都是 view（t≈0.0026ms，iter>3M）
    # 1024×151936=155M 才是 copy（t≈1.16ms）
    return vol < 500000  # 小于 50 万元素认为是 view


def estimate(node):
    t = node.get("op_type", "UNKNOWN")
    ins = node.get("input_tensors", [])
    outs = node.get("output_tensors", [])

    # TRANSPOSE 区分 view 和 copy
    if t == "TRANSPOSE" and _is_transpose_view(node):
        f = _FORMULAS.get("TRANSPOSE2RESHAPE")
        if f and len(f) >= 3 and f[0] == "N*M" and len(f) >= 5 and f[3] == "const":
            return f[2] * f[4] / 1000  # t * P / 1000 = constant energy
        return 0.0

    # CSV 查表优先
    if _TABLE:
        iN, iM, iK = extract_mnk_ins(ins, t)
        oN, oM, oK = extract_mnk_outs(outs)
        key = (t, iN, iM, iK, oN, oM, oK)
        if key in _TABLE:
            return _TABLE[key]
        key0 = (t, iN, iM, 0, oN, oM, 0)
        if key0 in _TABLE:
            return _TABLE[key0]

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
    p.add_argument("--fusion", action="store_true", help="启用融合感知，识别 GPU kernel 融合后折扣能耗")
    p.add_argument("-c", "--csv", default="", help="Qwen_result.csv 路径，启用 CSV 查表替代公式")
    args = p.parse_args()

    nodes, summary = load_graph(args.graph)
    graph_dir = Path(args.graph).parent
    output_path = Path(args.output) if args.output else graph_dir / "energy_report.txt"
    if not nodes:
        print(f"[energy] {args.graph}: empty")
        return

    # Load CSV lookup table if provided
    import csv as _csv

    if args.csv:
        with open(args.csv, encoding="utf-8-sig") as _f:
            for _row in _csv.DictReader(_f):
                _vals = list(_row.values())
                _key = (_row['operator'].strip(),
                       int(_row['input_N']), int(_row['input_M']), int(_row['input_K']),
                       int(_row['output_N']), int(_row['output_M']), int(_row['output_K']))
                _TABLE[_key] = float(_vals[-1]) / 1000  # mJ -> J

    pf_nodes = [n for n in nodes if n.get("stage") == "prefill"]
    dc_nodes = [n for n in nodes if n.get("stage") == "decode"]

    # 设置 CSV 查表（影响 estimate() 和 estimate_with_fusion() 的全局行为）
    import energy_consumption_refactor as _ecr_mod
    _ecr_mod._TABLE = _TABLE

    def stage_report(sn, label):
        op_energy = defaultdict(float)
        op_count = defaultdict(int)
        te = 0.0
        # Always compute per-operator energy
        for n in sn:
            op = n["op_type"]
            e = estimate(n)
            op_energy[op] += e
            op_count[op] += 1
            te += e
        # If fusion enabled, also compute fusion-aware estimate
        fusion_info = None
        if args.fusion:
            cats, fte, n_skip = estimate_with_fusion(nodes, stage=label.lower(), summary=summary)
            fusion_info = (cats, fte, n_skip)
            if args.verbose:
                print(f"    [fusion] {label}: skipped {n_skip} fused ops, raw={te:.4f}J fused={fte:.4f}J")
        return op_energy, op_count, te, fusion_info
    stages = []
    for label, sn in [("Prefill", pf_nodes), ("Decode", dc_nodes)]:
        if sn:
            stages.append((label, *stage_report(sn, label)))

    lines = [
        "=" * 70, "  Energy Consumption Reconstruction", "=" * 70,
        f"  Graph:  {args.graph}",
        f"  Nodes:  {len(nodes)}  (prefill: {len(pf_nodes)} + decode: {len(dc_nodes)})",
        f"  Generation length: {args.gen_len} tokens",
    ]

    gt = 0.0
    for label, op_energy, op_count, total, fusion_info in stages:
        fused_total = fusion_info[1] if fusion_info else total
        mul = args.gen_len if label == "Decode" else 1
        disp = f"{label} (per token)" if mul > 1 else label
        lines += ["", f"  --- {disp} ---",
                  f"  Nodes: {sum(op_count.values()):>6d}",
                  f"  Energy:  {fused_total:.4f}J ({fused_total*1000:.2f}mJ)"]
        gt += fused_total * mul
    lines += ["", "-" * 70,
              f"  Aggregated (prefill + decode x{args.gen_len}):",
              f"  Total: {gt:.4f}J ({gt*1000:.2f}mJ)"]

    for label, op_energy, op_count, total, fusion_info in stages:
        mul = args.gen_len if label == "Decode" else 1
        disp = f"{label} (per token)" if mul > 1 else label
        lines += ["", f"  {disp} operators:",
                  f"  {'Operator':25s} {'Cnt':>5s} {'Energy(mJ)':>12s} {'%':>6s}",
                  "  " + "-" * 48]
        for op, e in sorted(op_energy.items(), key=lambda x: -x[1]):
            c = op_count[op]
            lines.append(f"  {op:25s} {c:>5d} {e*1000:>10.4f}  {e/total*100:>5.1f}%")
        # Fusion-aware summary (if enabled)
        if fusion_info is not None:
            cats, fte, n_skip = fusion_info
            lines += ["", f"  {disp} fusion-aware estimate:",
                      f"  {'Category':20s} {'Energy(J)':>12s} {'%':>8s}",
                      "  " + "-" * 42]
            for cat in ("compute_bound", "memory_bound", "data_movement", "communication"):
                e2 = cats.get(cat, 0)
                lines.append(f"  {cat:20s} {e2:>10.4f}  {e2/fte*100:>7.1f}%")
            lines.append(f"  {'Skipped fused':20s} {n_skip:>10d}")
            lines.append(f"  {'Total (fusion)':20s} {fte:>10.4f}")
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
