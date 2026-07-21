"""
GPU 功耗分析 — python power_analyze.py -t timestamps.txt -p power.txt [-n 20] [--gen-len 20]

自动扣除基准功耗：GPU 0 推理，其余 GPU 空载时取其平均功率作为基准。
decode 能耗除以 --gen-len 得到单 token 结果。
"""
import argparse
import numpy as np

DEFAULT_RUNS = 20


def parse_ts_value(line):
    raw = line.strip()
    parts = raw.split(None, 1)
    if len(parts) == 2:
        a, b = parts
        if len(a) >= 8 and a[2] == ":" and a[5] == ":":
            ts_str = a
        elif len(b) >= 8 and b[2] == ":" and b[5] == ":":
            ts_str = b
        else:
            ts_str = a
    else:
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
            for tag in ("prefill_start", "prefill_end", "decode_start", "decode_end",
                        "gen_start", "gen_end", "end",
                        "idle_before_start", "idle_before_end",
                        "idle_after_start", "idle_after_end",
                        "idle_cuda_start", "idle_cuda_end"):
                if f" {tag}" in line or line.startswith(tag + ":"):
                    ts[tag] = parse_ts_value(line)
            for tag in ("prefill_gpu_us", "decode_gpu_us"):
                if line.startswith(tag):
                    ts[tag] = int(line.strip().split()[1])
            for tag in ("start_energy_j", "prefill_start_energy_j", "prefill_end_energy_j",
                        "decode_start_energy_j", "decode_end_energy_j",
                        "gen_start_energy_j", "gen_end_energy_j",
                        "idle_before_start_energy_j", "idle_before_end_energy_j",
                        "idle_after_start_energy_j", "idle_after_end_energy_j",
                        "idle_cuda_start_energy_j", "idle_cuda_end_energy_j"):
                if line.startswith(tag):
                    ts[tag] = float(line.strip().split()[1])
            for tag in ("prefill_kernel_ratio", "decode_kernel_ratio"):
                if line.startswith(tag):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        ts[parts[0]] = float(parts[1])
    return ts


def integrate(times, power_1d, t_start, t_end):
    mask = (times >= t_start) & (times <= t_end)
    if not mask.any():
        return 0.0, 0.0
    energy = np.trapezoid(power_1d[mask], times[mask])
    return energy, float(power_1d[mask].mean())


def main():
    parser = argparse.ArgumentParser(description="GPU 功耗分析")
    parser.add_argument("-t", "--timestamps", default="timestamps.txt")
    parser.add_argument("-p", "--power", default="power.txt")
    parser.add_argument("-n", "--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--gen-len", type=int, default=1)
    args = parser.parse_args()

    times, powers = load_power(args.power)
    ts = load_timestamps(args.timestamps)
    n_gpu = powers.shape[1] if powers.ndim > 1 else 0
    runs = max(args.runs, 1)
    gen_len = max(args.gen_len, 1)

    idle_cuda = idle_after = 0.0
    if "idle_cuda_end_energy_j" in ts:
        e = ts["idle_cuda_end_energy_j"] - ts["idle_cuda_start_energy_j"]
        t = ts["idle_cuda_end"] - ts["idle_cuda_start"]
        if t > 0.1:
            idle_cuda = e / t
    if "idle_after_end_energy_j" in ts:
        e = ts["idle_after_end_energy_j"] - ts["idle_after_start_energy_j"]
        t = ts["idle_after_end"] - ts["idle_after_start"]
        if t > 0.1:
            idle_after = e / t

    if idle_cuda > 0 and idle_after > 0:
        P_bl = (idle_cuda + idle_after) / 2
    elif idle_cuda > 0:
        P_bl = idle_cuda
    else:
        P_bl = 0.0

    phases = [("Prefill", "prefill_start", "prefill_end"), ("Decode", "decode_start", "decode_end")]
    results = []
    for name, s, e in phases:
        if s not in ts or e not in ts:
            continue

        wall_s = ts[e] - ts[s]

        energy_tag_s = f"{s}_energy_j"
        energy_tag_e = f"{e}_energy_j"
        if energy_tag_s in ts and energy_tag_e in ts:
            e_j_all = ts[energy_tag_e] - ts[energy_tag_s]
            e_j_dynamic = e_j_all - P_bl * wall_s
        else:
            e_j_all = e_j_dynamic = 0.0

        # GPU_busy 比例
        gpu_us_tag = {"Prefill": "prefill_gpu_us", "Decode": "decode_gpu_us"}[name]
        if gpu_us_tag in ts:
            gpu_s = ts[gpu_us_tag] / 1e6
            ratio = min(gpu_s / wall_s, 1.0) if wall_s > 0 else 1.0
        else:
            ratio = 1.0

        e_j_op = e_j_dynamic * ratio

        avg_power = e_j_dynamic / wall_s if wall_s > 0 else 0
        e_j = e_j_op / runs

        _prefix = {"Prefill": "prefill", "Decode": "decode"}[name]
        _r_c = ts.get(f"{_prefix}_kernel_ratio_compute_bound", 0.0)
        _r_m = ts.get(f"{_prefix}_kernel_ratio_memory_bound", 0.0)
        _r_d = ts.get(f"{_prefix}_kernel_ratio_data_movement", 0.0)
        _r_co = ts.get(f"{_prefix}_kernel_ratio_communication", 0.0)
        _r_sum = _r_c + _r_m + _r_d + _r_co
        cat_results = {}
        if _r_sum > 0:
            cat_results = {"compute_bound": e_j * _r_c / _r_sum, "memory_bound": e_j * _r_m / _r_sum,
                           "data_movement": e_j * _r_d / _r_sum, "communication": e_j * _r_co / _r_sum}

        if name == "Decode":
            pass

        results.append((name, wall_s, e_j, avg_power, cat_results))

        # 调试信息
        e_total_display = e_j_all if energy_tag_s in ts and energy_tag_e in ts else e_j_dynamic
        gpu_ms = ts.get({"Prefill":"prefill_gpu_us","Decode":"decode_gpu_us"}[name], 0) / 1000
        print(f"  [{name}] wall={ts[e]-ts[s]:.4f}s E_total={e_total_display:.4f}J "
              f"P_bl={P_bl:.1f}W gpu={gpu_ms:.2f}ms ratio={ratio:.3f} "
              f"E_op={e_j_op:.4f}J per_run={e_j:.6f}J")

    use_ec = any(f"{p[1]}_energy_j" in ts for p in phases)
    print(f"  Energy source: {'hardware energy counter' if use_ec else 'power sampling + integration'}")

    if not results:
        print(f"[analyze] {args.timestamps}: timestamps not found")
        return

    t0 = ts.get("prefill_start")
    t1 = ts.get("prefill_end", ts.get("decode_end"))
    if t0 is not None and t1 is not None:
        mask = (times >= t0) & (times <= t1)
        if mask.any():
            avg = powers[mask].mean(axis=0)
            print(f"  {'GPU':>5s}  {'Avg Power (W)':>14s}")
            print(f"  {'-' * 22}")
            for i in range(min(len(avg), 8)):
                print(f"  {i:>5d}  {avg[i]:>10.2f}")

    print(f"\n  (除以 {runs} 次 profiling runs，以下为单次推理结果)")
    print(f"  {'Phase':15s}  {'Duration':>10s}  {'Energy':>10s}  {'Avg Power':>10s}")
    print(f"  {'-' * 50}")
    for r in results:
        name, d, e, w = r[0], r[1], r[2], r[3]
        print(f"  {name:15s}  {d:>8.3f}s  {e:>8.2f}J  {w:>8.2f}W")
        if len(r) >= 5 and r[4]:
            _cv = r[4]
            print(f"  {'':15s}  compute={_cv.get('compute_bound',0):.4f}J  "
                  f"memory={_cv.get('memory_bound',0):.4f}J  "
                  f"move={_cv.get('data_movement',0):.4f}J")

    if idle_before > 0 or idle_after > 0:
        print(f"\n  [参考] idle_before={idle_before:.1f}W  idle_after={idle_after:.1f}W")

    if len(results) >= 2:
        td = sum(r[1] for r in results)
        te = sum(r[2] for r in results)
        print(f"  {'-' * 50}")
        print(f"  {'Total':15s}  {td:>8.3f}s  {te:>8.2f}J  {te/td:>8.2f}W")

    print(f"\n  [{len(times)} power samples, {n_gpu} GPUs]")


if __name__ == "__main__":
    main()
