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


def extract_mnk(shapes, is_input=True, op_type=""):
    """从 tensor shape 列表推测 N, M, K。

    is_input=True: shapes 是输入张量列表，GEMM/BMM 的 N,M,K 来自多输入
    is_input=False: shapes 是输出张量列表，取第一个输出
    """
    if not shapes:
        return 0, 0, 0

    L = len(shapes)
    S0 = shapes[0]
    L0 = len(S0)

    if is_input and op_type in ("GEMM", "LINEAR") and L >= 2:
        # A[M×K], B[K×N]
        A, B = shapes[0], shapes[1]
        M = prod(A[:-1]) if A else 1
        K = A[-1] if A else 1
        N = B[-1] if B else 1
        return N, M, K

    if is_input and op_type == "BMM" and L >= 2:
        A, B = shapes[0], shapes[1]
        M = A[-2] if len(A) >= 2 else 1
        K = A[-1] if A else 1
        N = B[-1] if B else 1
        return N, M, K

    # 默认：取第一个 tensor 的 shape
    if L0 >= 2:
        N = S0[-1]
        M = prod(S0[:-1])
    elif L0 == 1:
        N = S0[0]
        M = 1
    else:
        N, M = 1, 1

    K = 0
    if is_input and "KV_CACHE" in op_type:
        K = S0[-1] if S0 else 0

    return N, M, K


def shape_to_str(shapes):
    """tensor shape 列表 → 浓缩字符串，如 '1x7x1024|1024x2048'"""
    return "|".join("x".join(str(d) for d in s) for s in shapes)


def main():
    p = argparse.ArgumentParser(description="从 graph.json 提取算子输入/输出维度")
    p.add_argument("-g", "--graph", required=True, help="graph.json 路径")
    p.add_argument("-o", "--output", default="", help="输出 CSV 路径（默认与 graph.json 同目录）")
    args = p.parse_args()

    graph_dir = Path(args.graph).parent
    out_path = Path(args.output) if args.output else graph_dir / "operator_table.csv"

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
        iN, iM, iK = extract_mnk(info["input_shapes"], is_input=True, op_type=op_type)
        oN, oM, oK = extract_mnk(info["output_shapes"], is_input=False, op_type=op_type)
        rows.append([op_type, iN, iM, iK, oN, oM, oK, stage, info["cnt"]])

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["算子名", "输入N", "输入M", "输入K",
                     "输出N", "输出M", "输出K", "阶段", "出现次数"])
        w.writerows(rows)

    # 控制台打印摘要
    print(f"{'算子名':20s} {'入N':>6s} {'入M':>6s} {'入K':>6s} {'出N':>6s} {'出M':>6s} {'出K':>6s} {'Cnt':>5s} {'阶段':8s}")
    print("-" * 85)
    for r in rows[:25]:
        op, iN, iM, iK, oN, oM, oK, stage, cnt = r
        print(f"{op:20s} {str(iN):>6s} {str(iM):>6s} {str(iK):>6s} {str(oN):>6s} {str(oM):>6s} {str(oK):>6s} {cnt:>5d} {stage:8s}")
    if len(rows) > 25:
        print(f"  ... 共 {len(rows)} 种组合")
    print(f"\n已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
