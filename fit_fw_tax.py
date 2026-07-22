"""
Fit Framework Tax coefficient C, with Leave-One-Model-Out cross validation.

Usage:
    python fit_fw_tax.py

Fill in Dir2 data below before running.
"""
import json, glob, numpy as np
from collections import Counter
from pathlib import Path
from energy_consumption_refactor import estimate, estimate_with_fusion


def compute(model_dir: str):
    """Extract raw formula sum, hidden_dim, num_ops for a model."""
    paths = sorted(glob.glob(f"output/{model_dir}/graph.json"))
    if not paths:
        paths = sorted(glob.glob(f"output/{model_dir}_*/graph.json"))
    if not paths:
        print(f"  [skip] {model_dir}: graph.json not found")
        return None

    with open(paths[0]) as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    summary = data.get("summary", {})

    # hidden_dim: mode of GEMM K dimensions
    ks = Counter()
    for n in nodes:
        if n.get("op_type") in ("GEMM", "LINEAR", "BMM"):
            ins = n.get("input_tensors", [])
            if len(ins) >= 2:
                s = ins[1].get("shape", [])
                if len(s) >= 2:
                    ks[s[0]] += 1
    h = max(ks, key=ks.get) if ks else 0
    n_layers = summary.get("num_layers", 0)

    result = {"model": model_dir, "hidden_dim": h, "layers": n_layers}

    for stage in ("prefill", "decode"):
        sn = [n for n in nodes if n.get("stage") == stage]
        num_ops = len(sn)

        # Raw formula sum (no fusion, no correction)
        raw = sum(estimate(n) for n in sn)

        # Fusion-aware sum minus framework correction
        cats, fused_total, _ = estimate_with_fusion(nodes, stage=stage, summary=summary)
        fw = 3.4996e-07 * h * num_ops
        fused_no_fw = fused_total - fw

        result[f"raw_{stage}"] = raw
        result[f"fusion_{stage}"] = fused_no_fw
        result[f"ops_{stage}"] = num_ops
        result[f"hxops_{stage}"] = h * num_ops

    return result


def main():
    # Scan all output dirs with graph.json, dedup by model name
    dirs = [d.name for d in Path("output").glob("*") if d.is_dir() and (d / "graph.json").exists()]
    seen = {}
    for d in dirs:
        base = d.rsplit("_", 2)[0]
        if base not in seen or d > seen[base]:
            seen[base] = d

    print("=" * 100)
    print(f"  {'Model':30s} {'h':>5s} {'Layers':>6s} "
          f"{'ops_pf':>6s} {'hxops_pf':>12s} {'raw_pf':>8s} {'fus_pf':>8s} "
          f"{'ops_dc':>6s} {'hxops_dc':>12s} {'raw_dc':>8s} {'fus_dc':>8s}")
    print("-" * 100)

    rows = []
    for model in sorted(seen.keys()):
        r = compute(seen[model])
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['model']:30s} {r['hidden_dim']:>5d} {r['layers']:>6d} "
              f"{r['ops_prefill']:>6d} {r['hxops_prefill']:>12d} {r['raw_prefill']:>8.4f} {r['fusion_prefill']:>8.4f} "
              f"{r['ops_decode']:>6d} {r['hxops_decode']:>12d} {r['raw_decode']:>8.4f} {r['fusion_decode']:>8.4f}")

    # ======================
    # Dir2 data (J per inference)
    # ======================
    Dir2 = {
        "Llama-3.1-8B-Instruct": (4.22, 4.10),
        "Mistral-7B-Instruct-v0.3": (4.12, 3.85),
        "Qwen3-0.6B": (2.18, 2.15),
        "deepseek-coder-7b-instruct-v1.5": (3.71, 3.56),
        "gpt2": (0.57, 0.40),
        "opt-125m": (0.38, 0.35),
        "gemma-3-4b-it": (5.49, 4.31),
    }

    if not Dir2:
        print("\nFill in Dir2 data first, then re-run.")
        return

    # Build dataset: (model_name, stage, gap, hxops, base, dir2)
    data = []
    for r in rows:
        model_base = r["model"].rsplit("_", 2)[0]
        if model_base not in Dir2:
            continue
        pf2, dc2 = Dir2[model_base]
        for stage in ("prefill", "decode"):
            d2 = pf2 if stage == "prefill" else dc2
            base_val = r[f"fusion_{stage}"]
            hxops = r[f"hxops_{stage}"]
            gap = d2 - base_val
            data.append((model_base, stage, gap, hxops, base_val, d2))

    gaps = np.array([d[2] for d in data])
    hxops_arr = np.array([d[3] for d in data])
    C_all = np.sum(gaps * hxops_arr) / np.sum(hxops_arr ** 2)

    # ---- Full fit ----
    print("\n" + "=" * 100)
    print("  Full fit: C on all 6 models (12 data points)")
    print("=" * 100)
    print(f"\n  C = {C_all:.3e}")
    print(f"\n  {'Model':28s} {'Stage':8s} {'base':>8s} {'pred':>9s} {'Dir2':>7s} {'Err%':>7s}")
    print("  " + "-" * 72)
    errs = []
    for m, st, gap, ho, b, d2 in data:
        pred = b + C_all * ho
        ep = (pred / d2 - 1) * 100
        errs.append(abs(ep))
        print(f"  {m:28s} {st:8s} {b:>7.4f}J {pred:>8.4f}J {d2:>5.2f}J {ep:>+6.1f}%")
    print(f"\n  Mean |Err%|: {np.mean(errs):.1f}%")

    # ---- LOOCV ----
    print("\n" + "=" * 100)
    print("  Leave-One-Model-Out Cross Validation")
    print("  (train on 5 models, predict on held-out model)")
    print("=" * 100)

    models = sorted(set(d[0] for d in data))
    cv_errs = []
    print(f"\n  {'Held-out':28s} {'C_train':>10s} {'pf_err':>8s} {'dc_err':>8s}  {'pf_pred':>8s} {'dc_pred':>8s}")
    print("  " + "-" * 78)

    for held_out in models:
        train = [(d[2], d[3]) for d in data if d[0] != held_out]
        test = [(d[1], d[4], d[3], d[5]) for d in data if d[0] == held_out]

        g_train = np.array([t[0] for t in train])
        h_train = np.array([t[1] for t in train])
        C_cv = np.sum(g_train * h_train) / np.sum(h_train ** 2)

        pf_err = dc_err = pf_pred = dc_pred = None
        for st, b, ho, d2 in test:
            pred = b + C_cv * ho
            ep = (pred / d2 - 1) * 100
            cv_errs.append(abs(ep))
            if st == "prefill":
                pf_err = ep
                pf_pred = pred
            else:
                dc_err = ep
                dc_pred = pred
        print(f"  {held_out:28s} {C_cv:>10.3e} {pf_err:>+7.1f}% {dc_err:>+7.1f}%  {pf_pred:>7.3f}J {dc_pred:>7.3f}J")

    print(f"\n  LOOCV Mean |Err%|: {np.mean(cv_errs):.1f}%")
    print(f"  LOOCV Max  |Err%|: {np.max(cv_errs):.1f}%")


if __name__ == "__main__":
    main()
