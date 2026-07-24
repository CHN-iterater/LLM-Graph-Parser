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
    # ---- element-wise (N*M_pw: t = b_const if vol < th else a*vol + b) ----
    "ADD":        ("N*M_pw", 1.92823e-09, 0.00289996, 0.00635157, 5.33521e+06,  "logistic1", 54.2356, 420.558, 1.16065, 25.1919),
    "MUL":        ("N*M_pw", 1.92895e-09, 0.00276984, 0.00623039, 6.37304e+06,  "logistic1", 58.7196, 435.736, 1.12697, 24.502),
    "NEG":        ("N*M_pw", 1.30702e-09, 0.00302957, 0.00598533, 7.18424e+06,  "logistic1", 60.651,  419.173, 1.07331, 23.9388),
    "POW":        ("N*M_pw", 1.31208e-09, 0.00266534, 0.00643007, 6.65575e+06,  "logistic1", 58.6,    413.63,  1.11445, 24.9389),
    "SQRT":       ("N*M_pw", 3.93839e-09, 0.00925112, 0.0188554,  7.18194e+06,   "logistic1", 59.7309, 432.193, 1.08655, 24.052),
    "RECIPROCAL": ("N*M_pw", 3.93948e-09, 0.00932756, 0.0188021,  5.02579e+06,  "logistic1", 60.3872, 426.47,  1.11488, 24.6024),
    "SIGMOID":    ("N*M_pw", 1.37863e-09, 0.00276357, 0.00564882, 5.12872e+06,  "logistic1", 58.45,   552.483, 1.03471, 22.5211),
    "SiLU":       ("N*M_pw", 1.37863e-09, 0.00276357, 0.00564882, 5.12872e+06,  "logistic1", 58.45,   552.483, 1.03471, 22.5211),
    "RELU":       ("N*M_pw", 1.30646e-09, 0.0028212,  0.00621647, 5.66914e+06,  "logistic1", 57.1621, 429.915, 1.06574, 23.7396),
    "GELU":       ("N*M_pw", 1.37863e-09, 0.00276357, 0.00564882, 5.12872e+06,  "logistic1", 58.45,   552.483, 1.03471, 22.5211),
    "CAST":       ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873,  3.96325e+06,  "logistic1", 54.0433, 373.33,  1.03891, 22.3404),
    "DIV":        ("N*M_pw", 1.92376e-09, 0.00332011, 0.00630557, 7.54482e+06,  "logistic1", 54.8931, 480.58,  1.12241, 24.203),
    "ISNAN":      ("N*M_pw", 9.66532e-10, 0.00264008, 0.00614313, 7.38949e+06,  "logistic1", 59.4337, 473.053, 0.996814, 22.7117),
    "WHERE":      ("N*M_pw", 3.19944e-09, 0.00582988, 0.0136162,  4.7571e+06,   "logistic1", 59.4053, 405.578, 1.35346, 29.4494),
    "TANH":       ("N*M_pw", 1.31472e-09, 0.00216877, 0.00554724, 6.76324e+06,  "logistic1", 61.4124, 555.822, 1.11587, 24.2938),
    "ERF":        ("N*M_pw", 1.37901e-09, 0.000147727, 0.00596899, 5.97458e+06, "logistic1", 64.6663, 548.331, 1.26942, 27.5136),
    # ---- compute-intensive (N+M+K_pw: t(N)+t(M)+t(K)) ----
    "GEMM":       ("N+M+K_pw", 124490, 0.00243222, 4.55771e-08, -0.00376641,
                            374111, 0.00219715, 3.82224e-08, -0.00937703,
                            424357, 0.00503167, 1.17229e-08, 0.022804,
                            "logistic3", 56.84, 358.092, 0.511654, 0.400197, 0.390121, 9.99382),
    "LINEAR":     ("N+M+K_pw", 434576, 0.00441065, 6.65187e-08, -0.0260789,
                            221389, 0.00141431, 4.60605e-08, -0.00733825,
                            235920, 0.00530255, 1.50132e-08, 0.00820528,
                            "logistic3", 60.0623, 386.098, 0.559769, 0.534287, 0.381541, 11.1402),
    "BMM":        ("N+M+K_pw", 124490, 0.00243222, 4.55771e-08, -0.00376641,
                            374111, 0.00219715, 3.82224e-08, -0.00937703,
                            424357, 0.00503167, 1.17229e-08, 0.022804,
                            "logistic3", 56.84, 358.092, 0.511654, 0.400197, 0.390121, 9.99382),
    # ---- asymmetric (N+M_pw: t(N,M) = t(N) + t(M)) ----
    "SOFTMAX":    ("N+M_pw", 737935, 0.000235484, 1.00787e-08, -0.00558565,
                            50598.8, 0.00616682, 2.48292e-07, -0.0073039,
                            "logistic2", 62.0281, 554.637, 0.865006, 0.427815, 17.3772),
    "REDUCEMEAN": ("N+M_pw", 262144, 5.08618e-05, 1.18356e-08, 0.00515896,
                            2.08625e+06, 0.00731883, 1.3516e-08, -0.0280151,
                            "logistic2", 53.5189, 479.24, 0.700456, 0.806632, 18.4096),
    "LAYER_NORM": ("N+M_pw", 11520, 0.000235511, 1.30606e-06, 0.00890085,
                            32085.7, 0.0118945, 2.46812e-06, -0.0790089,
                            "logistic2", 56.9395, 190.816, 0.939837, 0.217613, 12.0479),
    "RMSNorm":    ("N+M_pw", 11520, 0.000235511, 1.30606e-06, 0.00890085,
                            32085.7, 0.0118945, 2.46812e-06, -0.0790089,
                            "logistic2", 56.9395, 190.816, 0.939837, 0.217613, 12.0479),
    "EMBEDDING":  ("N+M_pw", 3.315e+06, 0.00293876, 8.22255e-13, 0.00485828,
                            1.98546e+06, 0.00624084, 1.52946e-09, 0.00394454,
                            "logistic2", 59.3521, 1e-10, 1e-12, 1e-12, 100),
    "TRANSPOSE":  ("N+M_pw", 32153.6, 0.00448473, 8.47557e-06, -0.268388,
                            6699.22, 0.00716886, 1.71288e-06, -0.0022271,
                            "logistic2", 60.5984, 365.67, 1.00404, 1.02429, 20.2118),
    # ---- data movement (mostly constant) ----
    "RESHAPE":    ("N*M", 0, 0.00185561,               "const", 51.932),
    "TRANSPOSE2RESHAPE": ("N*M", 0, 0.00268636,              "const", 53.5681),
    "SLICE":      ("N*M", 0, 0.00276413,               "const", 51.9363),
    "EXPAND":     ("N*M", 0, 0.001582,                 "const", 57.1697),
    "CAT":        ("N*M_pw", 3.00565e-09, 0, 0.0072653, 3.54246e+06, "logistic1", 56.3667, 364.122, 1.39447, 29.7414),
    "KVCacheRead":  ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873, 3.96325e+06, "logistic1", 54.0433, 373.33, 1.03891, 22.3404),
    "KVCacheWrite": ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873, 3.96325e+06, "logistic1", 54.0433, 373.33, 1.03891, 22.3404),
    "AllReduce":  ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873, 3.96325e+06, "logistic1", 54.0433, 373.33, 1.03891, 22.3404),
    "AllGather":  ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873, 3.96325e+06, "logistic1", 54.0433, 373.33, 1.03891, 22.3404),
    "MemcpyD2D":  ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873, 3.96325e+06, "logistic1", 54.0433, 373.33, 1.03891, 22.3404),
    "Reduction":  ("N*M_pw", 4.92387e-09, 0.00574644, 0.0149873, 3.96325e+06, "logistic1", 54.0433, 373.33, 1.03891, 22.3404),
    "DROPOUT":    ("N*M", 0, 0.005,                    "const", 50.0),
    "UNKNOWN":    ("N*M", 0, 0.005,                    "const", 50.0),
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
        _fw_energy = 2.967e-07 * _hidden_dim * _num_ops
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
    elif t_type == "N+M+K_pw":
        _, n_th, n_const, n_a, n_b, m_th, m_const, m_a, m_b, k_th, k_const, k_a, k_b, p_type, *prest = f
        tn = n_const if N < n_th else n_a * N + n_b
        tm = m_const if M < m_th else m_a * M + m_b
        tk = k_const if K < k_th else k_a * K + k_b
        t = tn + tm + tk
        prest = (p_type, *prest)
        size = N * M * K
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
    elif p_type == "logistic3":
        _, c, d, p1_n, p2_m, p3_k, p4 = prest
        x = p1_n * math.log2(N) + p2_m * math.log2(M) + p3_k * math.log2(K)
        p = c + d / (1 + math.exp(-x + p4))
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
