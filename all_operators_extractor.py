"""
扫描所有 output/ 中 graph.json，提取各模型算子的并集与重合度。

输出 CSV：每行一个模型，每列一个算子，值为出现次数。

用法:
    python all_operators_extractor.py
    python all_operators_extractor.py -o operator_matrix.csv
"""
import argparse, json, csv
from collections import defaultdict
from pathlib import Path


def model_label(dirname: str) -> str:
    """提取真实模型名（第一个下划线之前的部分）。"""
    return dirname.split("_")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default="", help="输出 CSV 路径")
    args = p.parse_args()

    base = Path(__file__).resolve().parent / "output"
    dirs = sorted(base.iterdir())
    if not dirs:
        print(f"[ERROR] No output directories in {base}")
        return

    # 收集数据: {真实模型名: {算子名: 出现次数}}
    model_op_counts: dict[str, dict[str, int]] = {}

    for d in dirs:
        gp = d / "graph.json"
        if not gp.exists():
            continue
        name = model_label(d.name)
        with open(gp, encoding="utf-8") as f:
            data = json.load(f)
        counts: dict[str, int] = defaultdict(int)
        for n in data.get("nodes", []):
            counts[n["op_type"]] += 1
        if counts:
            # 同一模型可能有多个时间戳，合并
            if name in model_op_counts:
                for op, c in counts.items():
                    model_op_counts[name][op] += c
            else:
                model_op_counts[name] = dict(counts)

    if not model_op_counts:
        print("[ERROR] No operator data found")
        return

    all_ops = sorted({op for c in model_op_counts.values() for op in c})
    models = sorted(model_op_counts)

    # ---- 控制台输出：重合度 ----
    print(f"算子总计: {len(all_ops)} 种")
    print(f"模型总数: {len(models)}")
    print()
    print(f"{'模型':30s}  {'算子数':>8s}  {'占并集%':>10s}")
    print("-" * 52)
    for m in models:
        cnt = len(model_op_counts[m])
        pct = cnt / len(all_ops) * 100
        print(f"{m:30s}  {cnt:>8d}  {pct:>9.1f}%")
    print(f"\n并集算子列表: {', '.join(all_ops)}")

    # ---- CSV 输出 ----
    csv_path = args.output or str(base / "operator_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型"] + all_ops)
        for m in models:
            row = [m] + [model_op_counts[m].get(op, 0) for op in all_ops]
            w.writerow(row)
    print(f"\n已保存: {csv_path}")

    # 缺失算子的模型数统计
    print(f"\n[缺失算子统计]  (出现次数=0 的模型数)")
    for op in all_ops:
        missing = sum(1 for m in models if model_op_counts[m].get(op, 0) == 0)
        if missing > 0:
            print(f"  {op:30s}:  {missing}/{len(models)} 个模型缺失")


if __name__ == "__main__":
    main()
