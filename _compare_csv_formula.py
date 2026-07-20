"""CSV 查表 vs 公式法对比"""
import json, csv
from collections import defaultdict
from pathlib import Path
from energy_consumption_refactor import (
    _FORMULAS, FORMULA_NAME, energy_j,
    extract_mnk_ins, extract_mnk_outs
)
OP_MAP = {
    "LINEAR":"GEMM","GEMM":"GEMM","BMM":"BMM","SOFTMAX":"Softmax",
    "LAYER_NORM":"LayerNorm","RMS_NORM":"RMSNorm","GELU":"GELU",
    "SILU":"SiLU","RELU":"ReLU","SIGMOID":"SiLU",
    "REDUCESUM":"Reduction","REDUCEMEAN":"Reduction","MEAN":"Reduction",
    "KV_CACHE_READ":"KVCacheRead","KV_CACHE_WRITE":"KVCacheWrite",
    "ADD":"ADD","MUL":"MUL","CAT":"CAT","SLICE":"SLICE","EXPAND":"EXPAND",
    "RESHAPE":"RESHAPE","TRANSPOSE":"TRANSPOSE","CAST":"CAST",
    "NEG":"NEG","POW":"POW","SQRT":"SQRT","RECIPROCAL":"RECIPROCAL",
    "ISNAN":"ISNAN","WHERE":"WHERE","EMBEDDING":"EMBEDDING","DIV":"DIV",
}

# Load graph
with open('output/Qwen3-0.6B_20260715_201409/graph.json') as f:
    data = json.load(f)

# Load CSV
table = {}
with open('../operator_energy_comparison.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        key = (row['operator'].strip(),
               int(row['input_N']), int(row['input_M']), int(row['input_K']),
               int(row['output_N']), int(row['output_M']), int(row['output_K']))
        table[key] = float(row['csv_operator_summary_repeat5_方式1(mJ)'])

def csv_lookup(t, ins, outs):
    iN,iM,iK = extract_mnk_ins(ins, t)
    oN,oM,oK = extract_mnk_outs(outs)
    key = (t, iN,iM,iK, oN,oM,oK)
    if key in table:
        return table[key] / 1000
    csv_name = OP_MAP.get(t)
    if csv_name:
        key = (csv_name, iN,iM,iK, oN,oM,oK)
        if key in table:
            return table[key] / 1000
        key0 = (csv_name, iN,iM,0, oN,oM,0)
        if key0 in table:
            return table[key0] / 1000
    return None

stats = defaultdict(lambda: {'cnt':0, 'csv':0.0, 'fml':0.0})

for n in data['nodes']:
    if n.get('stage') != 'prefill':
        continue
    op = n['op_type']
    ins = n.get('input_tensors',[])
    outs = n.get('output_tensors',[])
    stats[op]['cnt'] += 1

    fname = FORMULA_NAME.get(op, 'UNKNOWN')
    if fname in _FORMULAS:
        iN,iM,iK = extract_mnk_ins(ins, op)
        if op in ('GEMM','LINEAR','BMM'):
            N,M,K = iN,iM,iK
        else:
            N,M,K = extract_mnk_outs(outs)
        N = max(N,1); M = max(M,1)
        stats[op]['fml'] += energy_j(N, M, K, fname)

    e = csv_lookup(op, ins, outs)
    if e is not None:
        stats[op]['csv'] += e

print(f"{'Operator':25s} {'Cnt':>5s} {'CSV(mJ)':>10s} {'Fml(mJ)':>10s} {'Fml/CSV':>8s}")
print('-' * 60)
tc, tf = 0.0, 0.0
for op in sorted(stats):
    s = stats[op]
    c, f = s['csv']*1000, s['fml']*1000
    r = f/c if c > 0 else 0
    tc += s['csv']; tf += s['fml']
    tag = ' Y' if 0.5 <= r <= 2.0 else ' X'
    print(f"{op:25s} {s['cnt']:>5d} {c:>10.4f} {f:>10.4f} {r:>7.2f}x{tag:>3s}")

R = tf/tc if tc else 0
print('-' * 60)
print(f"{'TOTAL':25s} {'':>5s} {tc*1000:>10.4f} {tf*1000:>10.4f} {R:>7.2f}x")
