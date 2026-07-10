"""
GPU 功耗分析 — python power_analyze.py -t timestamp.txt -p power.txt
"""
import argparse
import numpy as np


def parse_hhmmss(line):
    ts = line.split()[0]
    h, m = int(ts[0:2]), int(ts[3:5])
    s, ms = ts[6:8], ts[9:12]
    return h * 3600 + m * 60 + int(s) + int(ms) / 1000


def load_power(path):
    times, powers = [], []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            times.append(parse_hhmmss(line))
            powers.append([float(p) for p in parts[1:]])
    return np.array(times), np.array(powers)


def load_timestamps(path):
    start = end = None
    with open(path) as f:
        for line in f:
            if "inference_start" in line:
                start = parse_hhmmss(line)
            elif "inference_end" in line:
                end = parse_hhmmss(line)
    return start, end


def integrate(times, powers, t_start, t_end):
    mask = (times >= t_start) & (times <= t_end)
    if not mask.any():
        return 0.0, 0.0
    total_w = powers[mask].sum(axis=1)
    energy = np.trapz(total_w, times[mask])
    return energy, float(total_w.mean())


def main():
    parser = argparse.ArgumentParser(description="GPU 功耗分析")
    parser.add_argument("-t", "--timestamps", default="timestamps.txt",
                        help="令 timestamps.txt（run.py 在 output/ 目录下输出）")
    parser.add_argument("-p", "--power", default="power.txt",
                        help="令 power.txt（power_monitor.py 输出）")
    args = parser.parse_args()

    times, powers = load_power(args.power)
    t_start, t_end = load_timestamps(args.timestamps)

    if t_start is None or t_end is None:
        print(f"[analyze] {args.timestamps}: inference_start/end 未找到")
        try:
            with open(args.timestamps) as f:
                print(f"  文件内容 ({len(f.readlines())} 行):")
                f.seek(0)
                for line in f:
                    print(f"    {line.rstrip()}")
        except Exception as e:
            print(f"  文件读取失败: {e}")
        return

    energy, avg_w = integrate(times, powers, t_start, t_end)
    duration = t_end - t_start
    n_gpu = powers.shape[1] if powers.ndim > 1 else 0

    print(f"  {'Phase':15s}  {'Duration':>10s}  {'Energy':>10s}  {'Avg Power':>10s}")
    print(f"  {'-' * 50}")
    print(f"  {'Inference':15s}  {duration:>8.3f}s  {energy:>8.2f}J  {avg_w:>8.2f}W")
    print(f"\n  [{len(times)} samples, {n_gpu} GPUs]")


if __name__ == "__main__":
    main()
