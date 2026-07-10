"""
GPU 功耗分析 — python power_analyze.py -t timestamps.txt -p power.txt [-n 20] [--gen-len 20]

自动扣除基准功耗：GPU 0 推理，其余 GPU 空载时取其平均功率作为基准。
decode 能耗除以 --gen-len 得到单 token 结果。
"""
import argparse
import numpy as np

DEFAULT_RUNS = 20  # 默认 profiling 重复次数（对应 run.py 中的 PROFILING_RUNS）


def parse_ts_value(line):
    raw = line.strip()
    parts = raw.split(None, 1)  # 最多拆成 2 段
    if len(parts) == 2:
        a, b = parts
        # 判断哪一段像是时间戳 (HH:MM:SS.mmm → 第 2、5 字符是 ':')
        if len(a) >= 8 and a[2] == ":" and a[5] == ":":
            ts_str = a  # 新格式: "09:15:23.456 start"
        elif len(b) >= 8 and b[2] == ":" and b[5] == ":":
            ts_str = b  # 旧格式: "start:09:15:23.456"
        else:
            ts_str = a
    else:
        # 无空格 → 旧格式 "start:09:15:23.456"
        _, ts_str = raw.split(":", 1)
    h, m = int(ts_str[0:2]), int(ts_str[3:5])
    s, ms = ts_str[6:8], ts_str[9:12]
    return h * 3600 + m * 60 + int(s) + int(ms) / 1000


def load_power(path):
    times, powers = [], []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ts = parts[0]
            h, m = int(ts[0:2]), int(ts[3:5])
            s, ms = ts[6:8], ts[9:12]
            times.append(h * 3600 + m * 60 + int(s) + int(ms) / 1000)
            powers.append([float(p) for p in parts[1:]])
    return np.array(times), np.array(powers)


def load_timestamps(path):
    ts = {}
    with open(path) as f:
        for line in f:
            # 普通时间戳
            for tag in ("prefill_start", "prefill_end", "decode_start", "decode_end", "gen_start", "gen_end", "end"):
                if tag in line:
                    ts[tag] = parse_ts_value(line)
            # GPU 执行时间（μs），格式: "prefill_gpu_us 12345"
            for tag in ("prefill_gpu_us", "decode_gpu_us"):
                if line.startswith(tag):
                    ts[tag] = int(line.strip().split()[1])
    return ts


def integrate(times, power_1d, t_start, t_end):
    """对一维功率序列在 [t_start, t_end] 上积分求能量。"""
    mask = (times >= t_start) & (times <= t_end)
    if not mask.any():
        return 0.0, 0.0
    energy = np.trapezoid(power_1d[mask], times[mask])
    return energy, float(power_1d[mask].mean())


def main():
    parser = argparse.ArgumentParser(description="GPU 功耗分析")
    parser.add_argument("-t", "--timestamps", default="timestamps.txt")
    parser.add_argument("-p", "--power", default="power.txt")
    parser.add_argument("-n", "--runs", type=int, default=DEFAULT_RUNS,
                        help=f"profiling 重复次数，能耗除以该值得到单次结果（默认 {DEFAULT_RUNS}）")
    parser.add_argument("--gen-len", type=int, default=1,
                        help="生成 token 数，decode 能耗除以该值得到单 token 结果（默认 1）")
    args = parser.parse_args()

    times, powers = load_power(args.power)
    ts = load_timestamps(args.timestamps)
    n_gpu = powers.shape[1] if powers.ndim > 1 else 0
    runs = max(args.runs, 1)
    gen_len = max(args.gen_len, 1)

    # ---- 基准功耗扣除（GPU 0 推理，GPU 1~N-1 空载）----
    if n_gpu >= 2:
        baseline = powers[:, 1:].mean(axis=1)  # 每时刻 idle GPU 平均功率
        inference_w = powers[:, 0] - baseline   # GPU 0 净推理功率
        avg_baseline = float(baseline.mean())
    else:
        baseline = np.zeros(len(times))
        inference_w = powers[:, 0] if powers.ndim > 1 else powers
        avg_baseline = 0.0

    phases = [
        ("Prefill", "prefill_start", "prefill_end", "prefill_gpu_us"),
        ("Decode", "gen_start", "gen_end", "decode_gpu_us"),
    ]
    results = []
    for name, s, e, gpu_tag in phases:
        if s in ts and e in ts:
            e_j_total, w = integrate(times, inference_w, ts[s], ts[e])
            wall_s = ts[e] - ts[s]
            if gpu_tag in ts:
                gpu_s = ts[gpu_tag] / 1e6
                ratio = min(gpu_s / wall_s, 1.0) if wall_s > 0 else 1.0
                e_j = e_j_total * ratio  # 按 GPU 活跃时间比例折算算子能耗
            else:
                e_j = e_j_total
            e_j /= runs
            if name == "Decode":
                e_j /= gen_len  # 除以生成 token 数，得到单 token 结果
            results.append((name, wall_s, e_j, w))

    if not results:
        print(f"[analyze] {args.timestamps}: timestamps not found")
        return

    # Per-GPU average power + 基准功耗
    t0 = ts.get("prefill_start")
    t1 = ts.get("prefill_end", ts.get("decode_end"))
    if t0 is not None and t1 is not None:
        avg = powers[(times >= t0) & (times <= t1)].mean(axis=0)
        print(f"  {'GPU':>5s}  {'Avg Power (W)':>14s}")
        print(f"  {'-' * 22}")
        for i in range(min(len(avg), 8)):
            print(f"  {i:>5d}  {avg[i]:>10.2f}")
        if n_gpu >= 2:
            print(f"\n  Baseline (idle GPU 1~{n_gpu-1} avg): {avg_baseline:.2f} W")
            print(f"  Inference-only (GPU 0 - baseline): {avg[0] - avg_baseline:.2f} W")

    # Phase summary (per-run)
    print(f"\n  (除以 {runs} 次 profiling runs，以下为单次推理结果)")
    print(f"  {'Phase':15s}  {'Duration':>10s}  {'Energy':>10s}  {'Avg Power':>10s}")
    print(f"  {'-' * 50}")
    for name, d, e, w in results:
        label = f"{name} (per token)" if (name == "Decode" and gen_len > 1) else name
        print(f"  {label:15s}  {d:>8.3f}s  {e:>8.2f}J  {w:>8.2f}W")
    # 算子 vs 框架开销分解
    has_gpu_data = any(t in ts for t in ("prefill_gpu_us", "decode_gpu_us"))
    if has_gpu_data and n_gpu >= 2:
        print(f"\n  [算子 vs 框架开销分解] (按 GPU busy/wall 比例折算)")
        for name, s, e, gpu_tag in phases:
            if s in ts and e in ts and gpu_tag in ts:
                total_ej, _ = integrate(times, inference_w, ts[s], ts[e])
                gpu_s = ts[gpu_tag] / 1e6
                wall_s = ts[e] - ts[s]
                ratio = min(gpu_s / wall_s, 1.0) if wall_s > 0 else 1.0
                div = gen_len if name == "Decode" else 1
                op = total_ej * ratio / runs / div
                fw = total_ej * (1 - ratio) / runs / div
                per = " (per token)" if div > 1 else ""
                print(f"  {name:15s}{per}  operator={op:.2f}J  framework={fw:.2f}J  total={total_ej/runs/div:.2f}J")
                print(f"  {'':15s}  GPU busy={gpu_s*1000:.0f}/{wall_s*1000:.0f}ms ({ratio*100:.1f}%)")

    if len(results) >= 2:
        td = sum(r[1] for r in results)
        te = sum(r[2] for r in results)
        print(f"  {'-' * 50}")
        print(f"  {'Total':15s}  {td:>8.3f}s  {te:>8.2f}J  {te/td:>8.2f}W")

    print(f"\n  [{len(times)} power samples, {n_gpu} GPUs]")


if __name__ == "__main__":
    main()
