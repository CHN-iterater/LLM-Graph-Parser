"""
批量测试脚本 — 对多个模型依次运行完整方向 1 + 方向 2 测试。
支持多次重复取平均。

用法:
    python batch_test.py
    python batch_test.py --models Qwen3-0.6B gpt2 --repeat 3
"""
import argparse, subprocess, sys, time, statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_MODELS = sorted(
    p.name for p in Path(__file__).resolve().parent.parent.glob("Models/*")
    if p.is_dir() and not p.name.startswith(".")
)

MAX_NEW_TOKENS = 20
PROMPT = "What's the capital of France?"
PROFILING_RUNS = 20
GEN_LEN = 20


def run_cmd(cmd: list, step_label: str, capture: bool = False) -> tuple[bool, str]:
    print(f"\n{'=' * 60}")
    print(f"  [{step_label}]")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 60}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BASE_DIR, capture_output=capture, text=True)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    status = "OK" if ok else f"FAILED (code {r.returncode})"
    print(f"  [{step_label}] {status}  ({elapsed:.1f}s)")
    if capture and r.stdout:
        sys.stdout.write(r.stdout[-3000:])
    return ok, (r.stdout if capture else "")


def parse_power_analyze(output: str) -> tuple[float | None, float | None]:
    """解析 power_analyze 输出, 返回 (prefill_J, decode_J)。"""
    import re
    pf = dc = None
    for line in output.split("\n"):
        m = re.search(r"Prefill.*?(\d+\.\d+)J", line)
        if m:
            pf = float(m.group(1))
        m = re.search(r"Decode.*?(\d+\.\d+)J", line)
        if m:
            dc = float(m.group(1))
    return pf, dc


def fmt_mean_std(vals: list[float]) -> str:
    """返回 'mean±std J' 格式。"""
    if not vals:
        return "N/A"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f}±{s:.2f}J"


def fmt_val(v: float | None) -> str:
    return f"{v:.2f}J" if v is not None else "N/A"


def main():
    p = argparse.ArgumentParser(description="批量测试 LLM Graph Parser")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--runs", type=int, default=PROFILING_RUNS)
    p.add_argument("--gen-len", type=int, default=GEN_LEN)
    p.add_argument("--repeat", type=int, default=1, help="每个模型重复测量次数（默认 1）")
    p.add_argument("--no-hardware", action="store_true")
    args = p.parse_args()

    models = args.models or DEFAULT_MODELS
    # 存储: {模型: {pf2: [v1,v2,...], dc2: [v1,v2,...]}}
    energy: dict[str, dict[str, list[float]]] = {}
    results: list[tuple[str, str]] = []

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    print(f"\n{'#' * 70}")
    print(f"  Batch Test — {len(models)} models × {args.repeat} reps")
    print(f"  Prompt: {args.prompt}")
    print(f"  Profiling runs: {args.runs}")
    print(f"{'#' * 70}")

    for model_name in models:
        print(f"\n{'#' * 70}")
        print(f"  Model: {model_name}  ({args.repeat} measurements)")
        print(f"{'#' * 70}")

        pf2_vals: list[float] = []
        dc2_vals: list[float] = []

        for rep in range(1, args.repeat + 1):
            label = f"{model_name}: run.py (rep {rep}/{args.repeat})"
            run_args = [sys.executable, "run.py", "--model", model_name,
                        "--prompt", args.prompt,
                        "--max-new-tokens", str(args.max_new_tokens),
                        "--runs", str(args.runs)]
            if args.no_hardware:
                run_args.append("--no-hardware")
            ok, _ = run_cmd(run_args, label)
            if not ok:
                break

            out_dir = find_latest_output(model_name)
            if out_dir is None:
                print(f"  [ERROR] output dir not found for {model_name}")
                break

            ts_path = out_dir / "timestamps.txt"
            ok, pa_out = run_cmd(
                [sys.executable, "power_analyze.py",
                 "-t", str(ts_path), "-n", str(args.runs),
                 "--gen-len", str(args.gen_len)],
                f"{model_name}: power_analyze (rep {rep}/{args.repeat})",
                capture=True)
            if not ok:
                break

            pf, dc = parse_power_analyze(pa_out)
            if pf is not None:
                pf2_vals.append(pf)
            if dc is not None:
                dc2_vals.append(dc)

            # 重复间冷却
            if rep < args.repeat:
                sec = 30
                print(f"\n  冷却 {sec}s 后进行第 {rep+1} 次测量...")
                for r in range(sec, 0, -1):
                    print(f"\r  冷却剩余: {r}s", end="")
                    time.sleep(1)
                print()

        if pf2_vals:
            energy.setdefault(model_name, {"pf2": [], "dc2": []})
            energy[model_name]["pf2"] = pf2_vals
            energy[model_name]["dc2"] = dc2_vals
            results.append((model_name, f"OK ({len(pf2_vals)} reps)"))
        else:
            results.append((model_name, "FAILED"))

        # 模型间冷却
        if model_name != models[-1]:
            sec = 60
            print(f"\n  冷却 {sec}s 后进入下一个模型...")
            for r in range(sec, 0, -1):
                print(f"\r  冷却剩余: {r}s", end="")
                time.sleep(1)
            print()

    # 汇总
    print(f"\n{'#' * 70}")
    print(f"  Batch Test Summary")
    print(f"{'#' * 70}")
    if energy:
        print(f"  {'Model':28s} {'Prefill(Dir2)':>18s} {'Decode(Dir2)':>18s}")
        print(f"  {'-' * 28} {'-' * 18} {'-' * 18}")
        for m in sorted(energy):
            d = energy[m]
            pf_s = fmt_mean_std(d["pf2"])
            dc_s = fmt_mean_std(d["dc2"])
            print(f"  {m:28s} {pf_s:>18s} {dc_s:>18s}")
        print()

    print(f"  {'Model':30s}  {'Status':>20s}")
    print(f"  {'-' * 52}")
    for name, status in results:
        print(f"  {name:30s}  {status:>20s}")
    print(f"  {'-' * 52}")
    passed = sum(1 for _, s in results if s.startswith("OK"))
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Time:   {time.strftime('%H:%M:%S')}")


def find_latest_output(model_label: str) -> Path | None:
    cands = sorted(OUTPUT_DIR.glob(f"{model_label}_*"), key=lambda p: p.name, reverse=True)
    return cands[0] if cands else None


if __name__ == "__main__":
    main()
