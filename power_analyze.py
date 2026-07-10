"""
GPU 功耗分析 — python power_analyze.py -t timestamps.txt -p power.txt
"""
import argparse
import numpy as np


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
            for tag in ("prefill_start", "prefill_end", "decode_start", "decode_end", "gen_start", "end"):
                if tag in line:
                    ts[tag] = parse_ts_value(line)
    return ts


def integrate(times, powers, t_start, t_end):
    mask = (times >= t_start) & (times <= t_end)
    if not mask.any():
        return 0.0, 0.0
    total_w = powers[mask].sum(axis=1)
    energy = np.trapezoid(total_w, times[mask])
    return energy, float(total_w.mean())


def main():
    parser = argparse.ArgumentParser(description="GPU 功耗分析")
    parser.add_argument("-t", "--timestamps", default="timestamps.txt")
    parser.add_argument("-p", "--power", default="power.txt")
    args = parser.parse_args()

    times, powers = load_power(args.power)
    ts = load_timestamps(args.timestamps)
    n_gpu = powers.shape[1] if powers.ndim > 1 else 0

    phases = [
        ("Prefill", "prefill_start", "prefill_end"),
        ("Decode", "decode_start", "decode_end"),
        ("Generation", "gen_start", "end"),
    ]
    results = []
    for name, s, e in phases:
        if s in ts and e in ts:
            e_j, w = integrate(times, powers, ts[s], ts[e])
            results.append((name, ts[e] - ts[s], e_j, w))

    if not results:
        print(f"[analyze] {args.timestamps}: timestamps not found")
        return

    # Per-GPU average power (Prefill)
    t0 = ts.get("prefill_start")
    t1 = ts.get("prefill_end", ts.get("decode_end"))
    if t0 is not None and t1 is not None:
        avg = powers[(times >= t0) & (times <= t1)].mean(axis=0)
        print(f"  {'GPU':>5s}  {'Avg Power (W)':>14s}")
        print(f"  {'-' * 22}")
        for i in range(min(len(avg), 8)):
            print(f"  {i:>5d}  {avg[i]:>10.2f}")

    # Phase summary
    print(f"\n  {'Phase':15s}  {'Duration':>10s}  {'Energy':>10s}  {'Avg Power':>10s}")
    print(f"  {'-' * 50}")
    for name, d, e, w in results:
        print(f"  {name:15s}  {d:>8.3f}s  {e:>8.2f}J  {w:>8.2f}W")

    if len(results) >= 2:
        td = sum(r[1] for r in results)
        te = sum(r[2] for r in results)
        print(f"  {'-' * 50}")
        print(f"  {'Total':15s}  {td:>8.3f}s  {te:>8.2f}J  {te/td:>8.2f}W")

    print(f"\n  [{len(times)} samples, {n_gpu} GPUs]")


if __name__ == "__main__":
    main()
