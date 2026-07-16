"""
扫描所有 output/ 中 graph.json，提取各模型算子的并集与重合度。

用法:
    python all_operators_extractor.py
    python all_operators_extractor.py -o operator_union.txt
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="提取所有模型的算子并集与重合度")
    p.add_argument("-o", "--output", default="", help="输出路径（默认打印到控制台）")
    args = p.parse_args()

    base = Path(__file__).resolve().parent / "output"
    graphs = sorted(base.glob("*/graph.json"))
    if not graphs:
        print(f"[ERROR] No graph.json found in {base}")
        return

    # 每个模型的算子集合
    model_ops: dict[str, set[str]] = {}
    for gp in graphs:
        name = gp.parent.name
        with open(gp, encoding="utf-8") as f:
            data = json.load(f)
        ops = {n["op_type"] for n in data.get("nodes", [])}
        if ops:
            model_ops[name] = ops

    if not model_ops:
        print("[ERROR] No operator data found")
        return

    # 全面过并集
    all_ops = sorted(set().union(*model_ops.values()))

    # 每模型的详细算子（按并集顺序列出）
    lines = []

    # 表头：每个算子在哪个模型中出现
    lines.append(f"算子总计: {len(all_ops)} 种")
    lines.append(f"模型总数: {len(model_ops)}")
    lines.append("")
    lines.append(f"{'算子名':30s}" + "".join(f"{n:>18s}" for n in model_ops))
    lines.append("-" * (30 + 18 * len(model_ops)))

    for op in all_ops:
        row = f"{op:30s}"
        for n in model_ops:
            row += f"{'✓' if op in model_ops[n] else ' ':>18s}"
        lines.append(row)

    # 重合度
    lines.append("")
    lines.append(f"{'模型':30s}  {'算子数':>8s}  {'占并集%':>10s}")
    lines.append("-" * 50)
    for n in sorted(model_ops):
        cnt = len(model_ops[n])
        pct = cnt / len(all_ops) * 100
        lines.append(f"{n:30s}  {cnt:>8d}  {pct:>9.1f}%")

    # 遗漏算子（缺失最多的模型）
    min_cnt = min(len(v) for v in model_ops.values())
    lines.append(f"\n最少模型: {min_cnt}/{len(all_ops)} = {min_cnt/len(all_ops)*100:.1f}%")

    text = "\n".join(lines)
    print(text)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\n已保存: {args.output}")


if __name__ == "__main__":
    main()
