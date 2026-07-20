"""
批量测试脚本 — 对多个模型依次运行完整方向 1 + 方向 2 测试。

用法:
    python batch_test.py
    python batch_test.py --models Qwen3-0.6B gpt2
    python batch_test.py --prompt "Hello" --max-new-tokens 10
"""
import argparse, subprocess, sys, time, csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# 默认测试模型列表（从 ../Models 目录自动扫描）
DEFAULT_MODELS = sorted(
    p.name for p in Path(__file__).resolve().parent.parent.glob("Models/*")
    if p.is_dir() and not p.name.startswith(".")
)

MAX_NEW_TOKENS = 20
PROMPT = "What's the capital of France?"
PROFILING_RUNS = 20
GEN_LEN = 20


def run_cmd(cmd: list, step_label: str, capture: bool = False) -> tuple[bool, str]:
    """运行命令并实时输出，返回 (成功标志, 捕获的标准输出)。"""
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
        sys.stdout.write(r.stdout[-3000:])  # 打印最后 3000 字符
    return ok, (r.stdout if capture else "")


def _parse_energy(lines: list[str], section_start: int) -> float | None:
    """在 lines 中从 section_start 往后找 Energy: x.xxxx J。"""
    import re
    for i in range(section_start + 1, min(section_start + 5, len(lines))):
        if "Energy:" in lines[i]:
            m = re.search(r"Energy:\s+(\d+\.\d+)J", lines[i])
            if m:
                return float(m.group(1))
    return None


def find_latest_output(model_label: str) -> Path | None:
    """找 output/模型名_最新时间戳/ 目录。"""
    cands = sorted(OUTPUT_DIR.glob(f"{model_label}_*"), key=lambda p: p.name, reverse=True)
    return cands[0] if cands else None


def main():
    p = argparse.ArgumentParser(description="批量测试 LLM Graph Parser")
    p.add_argument("--models", nargs="*", default=None, help="测试的模型列表（默认使用 DEFAULT_MODELS）")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--runs", type=int, default=PROFILING_RUNS)
    p.add_argument("--gen-len", type=int, default=GEN_LEN)
    p.add_argument("--gen-repeats", type=int, default=None, help="生成阶段每步重复次数")
    p.add_argument("--no-hardware", action="store_true", help="禁用 hardware profiling")
    args = p.parse_args()

    models = args.models or DEFAULT_MODELS
    results = []

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    print(f"\n{'#' * 70}")
    print(f"  Batch Test — {len(models)} models")
    print(f"  Prompt: {args.prompt}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Profiling runs: {args.runs}")
    print(f"{'#' * 70}")

    # 汇总结果：模型 → { "pf1": ..., "pf2": ..., "dc1": ..., "dc2": ... }
    energy_summary = {}

    for model_name in models:
        print(f"\n{'#' * 70}")
        print(f"  Model: {model_name}")
        print(f"{'#' * 70}")

        # Step 1: run.py
        run_args = [sys.executable, "run.py", "--model", model_name, "--prompt", args.prompt,
                     "--max-new-tokens", str(args.max_new_tokens),
                     "--runs", str(args.runs)]
        if args.gen_repeats is not None:
            run_args += ["--gen-repeats", str(args.gen_repeats)]
        if args.no_hardware:
            run_args.append("--no-hardware")
        ok, _ = run_cmd(run_args, f"{model_name}: run.py")
        if not ok:
            results.append((model_name, "FAILED at run.py"))
            continue

        # 找输出目录
        out_dir = find_latest_output(model_name)
        if out_dir is None:
            results.append((model_name, "FAILED: output dir not found"))
            continue

        timestamps_path = out_dir / "timestamps.txt"
        graph_path = out_dir / "graph.json"

        # Step 2: power_analyze.py（捕获输出以解析方向 2 能耗）
        ok, pa_out = run_cmd(
            [sys.executable, "power_analyze.py",
             "-t", str(timestamps_path),
             "-n", str(args.runs),
             "--gen-len", str(args.gen_len)],
            f"{model_name}: power_analyze", capture=True)
        pf2 = dc2 = None
        pf2cats = dc2cats = None
        if ok:
            import re
            for line in pa_out.split("\n"):
                m = re.search(r"Prefill.*?(\d+\.\d+)J", line)
                if m:
                    pf2 = float(m.group(1))
                m = re.search(r"Decode.*?(\d+\.\d+)J", line)
                if m:
                    dc2 = float(m.group(1))
            _cur = None
            for line in pa_out.split("\n"):
                ls = line.strip()
                if ls.startswith("Prefill"):
                    _cur = "pf"
                elif ls.startswith("Decode"):
                    _cur = "dc"
                elif "compute=" in ls and _cur:
                    _d = {}
                    for _kv in ls.split():
                        if "=" in _kv:
                            k, v = _kv.split("=", 1)
                            _d[k] = float(v.replace("J", ""))
                    if _cur == "pf":
                        pf2cats = _d
                    else:
                        dc2cats = _d
                    _cur = None
        else:
            results.append((model_name, "FAILED at power_analyze"))
            continue

        # Step 3: energy_consumption_refactor.py（捕获输出以解析方向 1 能耗）
        ok, ec_out = run_cmd(
            [sys.executable, "energy_consumption_refactor.py",
             "-g", str(graph_path),
             "--gen-len", str(args.gen_len)],
            f"{model_name}: energy_consumption_refactor", capture=True)
        pf1 = dc1 = None
        if ok:
            ec_lines = ec_out.split("\n")
            for i, line in enumerate(ec_lines):
                ls = line.strip()
                if "--- Prefill" in ls:
                    pf1 = _parse_energy(ec_lines, i)
                if "--- Decode" in ls:
                    dc1 = _parse_energy(ec_lines, i)

        # Step 4: Direction 1 category breakdown
        pf1cats = dc1cats = None
        if graph_path.exists():
            try:
                from energy_consumption_refactor import estimate_by_category
                import json
                with open(graph_path) as _fg:
                    _gdata = json.load(_fg)
                _gnodes = _gdata.get("nodes", [])
                pf1cats = estimate_by_category(_gnodes, stage="prefill")
                dc1cats = estimate_by_category(_gnodes, stage="decode")
            except Exception:
                pass

        # Step 5: graph_operator_extractor.py（可选，失败不阻塞）
        run_cmd(
            [sys.executable, "graph_operator_extractor.py", "-g", str(graph_path)],
            f"{model_name}: graph_operator_extractor")

        status = "OK"
        if pf1 is None:
            status = "D2 OK (D1 N/A)"
        energy_summary[model_name] = (pf1, pf2, dc1, dc2, pf1cats, dc1cats, pf2cats, dc2cats)
        results.append((model_name, status))

        # 冷却等待（让 GPU 降温后再测下一个模型）
        if model_name != models[-1]:
            sec = 60
            print(f"\n  冷却 {sec}s 后进入下一个模型...")
            for remaining in range(sec, 0, -1):
                print(f"\r  冷却剩余: {remaining}s", end="")
                time.sleep(1)
            print()

    # 汇总
    print(f"\n{'#' * 70}")
    print(f"  Batch Test Summary")
    print(f"{'#' * 70}")

    # 能耗汇总表（12列：2阶段 × 3类别 × 2方向）
    if energy_summary:
        _cats = ["compute_bound", "memory_bound", "data_movement"]
        hdr = f"  {'Model':22s}"
        for _ph in ("PF", "DC"):
            for _d in ("D1", "D2"):
                hdr += f" {_ph}_{_d}_Tot"
                for _c in ("Cmp", "Mem", "Mov"):
                    hdr += f" {_ph}_{_d}_{_c:>3s}"
        print(hdr)
        print(f"  {'-' * 22} {'-' * (len(hdr)-26)}")
        for name, (pf1, pf2, dc1, dc2, pf1c, dc1c, pf2c, dc2c) in sorted(energy_summary.items()):
            def _fv(v):
                return f"{v*1000:.1f}" if v is not None else "N/A"
            def _fc(d):
                return tuple(f"{d.get(c,0)*1000:.1f}" if d else "N/A" for c in _cats)
            row = f"  {name:22s}"
            for _te, _tc in [(pf1, pf1c), (pf2, pf2c)]:
                row += f" {_fv(_te):>7s}"
                for _v in _fc(_tc):
                    row += f" {_v:>6s}"
            for _te, _tc in [(dc1, dc1c), (dc2, dc2c)]:
                row += f" {_fv(_te):>7s}"
                for _v in _fc(_tc):
                    row += f" {_v:>6s}"
            print(row)
        print()

    print(f"  {'Model':30s}  {'Status':>20s}")
    print(f"  {'-' * 52}")
    for name, status in results:
        print(f"  {name:30s}  {status:>20s}")
    print(f"  {'-' * 52}")
    passed = sum(1 for _, s in results if s == "OK")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Time:   {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
