"""
GPU 功耗采集 — pynvml 或 nvidia-smi 采样 8 张 H100 实时功率。

用法:
    python power_monitor.py
    python power_monitor.py -i 50 --use-smi
"""
import argparse, subprocess, datetime
from threading import Thread, Event


def get_power(use_smi=False):
    """返回 [gpu0_W, ..., gpu7_W] 或 None。"""
    # 方法1: nvidia-smi
    if use_smi:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return [v.strip().replace(" W", "") for v in r.stdout.strip().split("\n")]
        except Exception:
            pass

    # 方法2: pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        vals = []
        for i in range(min(pynvml.nvmlDeviceGetCount(), 8)):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            vals.append(f"{pynvml.nvmlDeviceGetPowerUsage(h) / 1000:.1f}")
        return vals
    except Exception:
        pass
    return None


def sample(interval_ms, max_samples, output, stop, use_smi):
    with open(output, "w") as f:
        f.write("HH:MM:SS.mmm " + " ".join([f"gpu{i}_W" for i in range(8)]) + "\n")
        n = 0
        while not stop.is_set() and (max_samples <= 0 or n < max_samples):
            now = datetime.datetime.now()
            ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
            vals = get_power(use_smi) or ["N/A"] * 8
            while len(vals) < 8:
                vals.append("N/A")
            f.write(ts + " " + " ".join(vals[:8]) + "\n")
            f.flush()
            n += 1
            stop.wait(interval_ms / 1000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--interval", type=int, default=100)
    p.add_argument("-n", "--max-samples", type=int, default=0)
    p.add_argument("-o", "--output", default="power.txt")
    p.add_argument("--use-smi", action="store_true")
    a = p.parse_args()

    stop = Event()
    Thread(target=sample, args=(a.interval, a.max_samples, a.output, stop, a.use_smi),
           daemon=True).start()
    print(f"[power] sampling ({a.interval}ms) -> {a.output}")
    try:
        while True:
            Thread._sleep(1)
    except KeyboardInterrupt:
        stop.set()
    print(f"\n[power] saved to {a.output}")


if __name__ == "__main__":
    main()
