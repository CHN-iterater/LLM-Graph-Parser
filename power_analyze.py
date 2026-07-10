"""
功耗分析脚本 — 从 power.txt + timestamp.txt 计算单次推理能耗。

用法:
    python power_analyze.py power.txt

输入:
    power.txt:      HH:MM:SS.mmm gpu0_W gpu1_W ... gpu7_W
    timestamp.txt:  run.py 同目录下自动生成
                    HH:MM:SS.mmm inference_start
                    HH:MM:SS.mmm inference_end

输出:
    prefill / decode / total 各阶段能耗 (J) 和平均功率 (W)
"""

import sys
import numpy as np
from datetime import datetime


def parse_hhmmss(line):
    """Parse 'HH:MM:SS.mmm' at start of line → seconds since epoch."""
    ts = line.split()[0]
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s, ms = parts[2].split(".")
    return h * 3600 + m * 60 + int(s) + int(ms) / 1000


def load_power(path):
    """Load power.txt → (seconds[], power_matrix[sample, gpu])."""
    times, powers = [], []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            times.append(parse_hhmmss(line))
            powers.append([float(p) for p in parts[1:]])
    return np.array(times), np.array(powers)


def load_timestamps(path="timestamp.txt"):
    """Load timestamp.txt → (start_seconds, end_seconds)."""
    start = end = None
    with open(path) as f:
        for line in f:
            if "inference_start" in line:
                start = parse_hhmmss(line)
            elif "inference_end" in line:
                end = parse_hhmmss(line)
    return start, end


def integrate(times, powers, t_start, t_end):
    """Trapezoidal integration of total GPU power over [t_start, t_end] → Joules."""
    mask = (times >= t_start) & (times <= t_end)
    if not mask.any():
        return 0.0, 0.0
    total_w = powers[mask].sum(axis=1)
    dt = times[mask]
    energy = np.trapz(total_w, dt)
    avg_w = float(total_w.mean())
    return energy, avg_w


def main():
    if len(sys.argv) < 2:
        power_file = "power.txt"
    else:
        power_file = sys.argv[1]

    times, powers = load_power(power_file)
    t_start, t_end = load_timestamps()

    if t_start is None or t_end is None:
        print("[analyze] timestamp.txt 中缺少 inference_start/inference_end")
        sys.exit(1)

    energy, avg_w = integrate(times, powers, t_start, t_end)
    duration = t_end - t_start

    print(f"  {'Phase':15s}  {'Duration':>10s}  {'Energy':>10s}  {'Avg Power':>10s}")
    print(f"  {'-' * 50}")
    print(f"  {'Inference':15s}  {duration:>8.3f}s  {energy:>8.2f}J  {avg_w:>8.2f}W")
    print(f"\n  [{len(times)} power samples, {powers.shape[1]} GPUs]")


if __name__ == "__main__":
    main()
