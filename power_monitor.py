"""
GPU 功耗采集脚本 — 用 pynvml 同时采样 8 张 H100 的实时功率。

用法:
    # 默认间隔 100ms，输出到 power.txt
    python power_monitor.py

    # 指定间隔 50ms，输出到 my_power.txt
    python power_monitor.py -i 50 -o my_power.txt

    # 配合 LLM Graph Parser 使用:
    # 终端 1: python power_monitor.py -i 50
    # 终端 2: HARDWARE_PROFILING=True python run.py

输出格式 (power.txt):
    HH:MM:SS.mmm gpu0_W gpu1_W gpu2_W gpu3_W gpu4_W gpu5_W gpu6_W gpu7_W
    14:30:01.023 86 87 85 87 88 83 84 81
    14:30:01.125 86 87 85 87 88 83 84 81
"""

import argparse
import time
from threading import Thread, Event


def get_power_handles():
    """初始化 pynvml 并返回 8 张 GPU 的功率句柄。"""
    try:
        import pynvml
    except ImportError:
        print("[power] 请先安装 pynvml: pip install nvidia-ml-py")
        raise
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    handles = []
    for i in range(min(device_count, 8)):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        handles.append(handle)
    return handles


def sample_power(handles, interval_ms, max_samples, output, stop):
    """Sample GPU power at fixed intervals."""
    header = "HH:MM:SS.mmm " + " ".join([f"gpu{i}_W" for i in range(len(handles))])
    with open(output, "w") as fp:
        fp.write(header + chr(10))
        sampled = 0
        while not stop.is_set() and (max_samples <= 0 or sampled < max_samples):
            import datetime as dt
            now = dt.datetime.now()
            ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
            powers = []
            for handle in handles:
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    powers.append(f"{mw / 1000:.1f}")
                except Exception:
                    powers.append("N/A")
            fp.write(ts + " " + " ".join(powers) + chr(10))
            fp.flush()
            sampled += 1
            stop.wait(interval_ms / 1000)
def main():
    parser = argparse.ArgumentParser(description="GPU 功耗采集（pynvml，8 张 H100）")
    parser.add_argument("-i", "--interval", type=int, default=100,
                        help="采样间隔 (ms)，默认 100")
    parser.add_argument("-n", "--max-samples", type=int, default=0,
                        help="最大采样次数，0=持续到手动停止")
    parser.add_argument("-o", "--output", default="power.txt",
                        help="输出文件，默认 power.txt")
    args = parser.parse_args()

    handles = get_power_handles()
    print(f"[power] 检测到 {len(handles)} 张 GPU")
    print(f"[power] 开始采集 (间隔={args.interval}ms, 输出={args.output})")
    if args.max_samples > 0:
        print(f"[power] 将采集 {args.max_samples} 次后自动停止")
    else:
        print("[power] 按 Ctrl+C 停止")

    stop = Event()
    t = Thread(target=sample_power,
               args=(handles, args.interval, args.max_samples, args.output, stop),
               daemon=True)
    t.start()

    try:
        while t.is_alive():
            t.join(1)
    except KeyboardInterrupt:
        print("\n[power] 停止采集")
        stop.set()

    print(f"[power] 结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
