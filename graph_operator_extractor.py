"""
从 graph.json 提取算子的输入/输出维度及出现次数，供 benchmark 测试用。

用法:
    python graph_operator_extractor.py -g output/模型名/graph.json -o operator_table.csv
"""
import argparse, json, csv
from collections import defaultdict
from pathlib import Path


def prod(shape):
    """返回 shape 各维乘积（忽略 -1 表示动态维度）"""
    p = 1
    for d in shape:
        if d > 0:
            p *= d
    return p


def extract_mnk(op_type, input_shapes, output_shapes):
    """从 tensor shape 推测 N, M, K（尽力而为，部分纯启发式）。"""
    L = len(input_shapes)
    O = output_shapes[0] if output_shapes else []

    # GEMM / LINEAR: input[0]=activation, input[1]=weight
    if op_type in ("GEMM", "LINEAR") and L >= 2:
        A, B = input_shapes[0], input_shapes[1]
        M = prod(A[:-1])
        K = A[-1] if A else 1
        N = B[-1] if B else 1
        return N, M, K

    # BMM: [B,M,K] @ [B,K,N]
    if op_type == "BMM" and L >= 2:
        A, B = input_shapes[0], input_shapes[1]
        M = A[-2] if len(A) >= 2 else 1
        K = A[-1] if A else 1
        N = B[-1] if B else 1
        return N, M, K

    # ATTENTION / FLASH_ATTENTION
    if "ATTENTION" in op_type and L >= 2:
        N = O[-1] if O else 1
        M = prod(O[:-1]) if O else 1
        K = 0
        return N, M, K

    # 其余算子：以输出 shape 为准
    N = O[-1] if O else 1
    M = prod(O[:-1]) if O else 1
    K = 0

    # KVCache: 从 cache_size / num_heads / head_dim 提取 K
    if "KV_CACHE" in op_type and L >= 1:
        K = input_shapes[0][-1] if input_shapes and input_shapes[0] else 0

    return N, M, K


def shape_to_str(shapes):
    """tensor shape 列表 → 浓缩字符串，如 '1x7x1024|1024x2048'"""
    return "|".join("x".join(str(d) for d in s) for s in shapes)


def main():
    p = argparse.ArgumentParser(description="从 graph.json 提取算子输入/输出维度")
    p.add_argument("-g", "--graph", required=True, help="graph.json 路径")
    p.add_argument("-o", "--output", default="operator_table.csv", help="输出 CSV 路径")
    args = p.parse_args()

    with open(args.graph, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    if not nodes:
        print(f"[extract] {args.graph}: empty")
        return

    # 聚合: (op_type, input_shapes_str, output_shapes_str, stage)
    groups = defaultdict(lambda: {"cnt": 0, "input_shapes": None, "output_shapes": None})

    for node in nodes:
        t = node["op_type"]
        ins = [tuple(t["shape"]) for t in node.get("input_tensors", [])]
        outs = [tuple(t["shape"]) for t in node.get("output_tensors", [])]
        stage = node.get("stage", "unknown")

        key = (t, shape_to_str(ins), shape_to_str(outs), stage)
        if groups[key]["cnt"] == 0:
            groups[key]["input_shapes"] = ins
            groups[key]["output_shapes"] = outs
        groups[key]["cnt"] += 1

    # 写出 CSV
    rows = []
    for (op_type, in_str, out_str, stage), info in sorted(groups.items(), key=lambda x: -x[1]["cnt"]):
        N, M, K = extract_mnk(op_type, info["input_shapes"], info["output_shapes"])
        rows.append([op_type, in_str, out_str, stage, N, M, K, info["cnt"]])

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["算子名", "输入维度 (shape)", "输出维度 (shape)", "阶段",
                     "N", "M", "K", "出现次数"])
        w.writerows(rows)

    # 控制台打印摘要
    print(f"{'算子名':20s} {'N':>8s} {'M':>8s} {'K':>8s} {'Cnt':>6s} {'阶段':8s}  输入")
    print("-" * 90)
    for r in rows[:25]:
        op, ins, outs, stage, N, M, K, cnt = r
        in_short = ins[:40] + "..." if len(ins) > 40 else ins
        print(f"{op:20s} {str(N):>8s} {str(M):>8s} {str(K):>8s} {cnt:>6d} {stage:8s}  {in_short}")
    if len(rows) > 25:
        print(f"  ... 共 {len(rows)} 种组合")
    print(f"\n已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
