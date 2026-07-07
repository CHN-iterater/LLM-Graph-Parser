# LLM Graph Parser

给定一个模型，自动输出 **标准化算子级计算图（DAG）**：
每个算子叫什么、调了多少次、输入输出多大、FLOPs 多少、数据依赖关系如何。
基于 ONNX，框架无关。

---

## 两种模式

| | PyTorch 模式 | ONNX 模式 |
|--|-------------|-----------|
| 输入 | 模型路径 + Prompt | `.onnx` 文件路径 |
| Prefill/Decode 拆分 | ✅ | ❌ 静态图 |
| 实际文本生成 | ✅ | ❌ |
| 层边界划分 | ✅ | ✅ |
| 并行性分析 | ✅ | ✅ |
| KV cache 依赖 | ✅ | ❌ |
| 需要模型代码 | ✅ | ❌ |

在 `run.py` 顶部切换:

```python
MODE = "pytorch"     # PyTorch 模式
# MODE = "onnx"     # ONNX 模式
```

---

## PyTorch 模式

```
Prompt
  │
  ├── Step 1: Prefill ─── 完整 Prompt 前向 ─── ONNX → 解析 → Prefill 子图
  │
  ├── Step 2: Decode ──── 单 token 模拟生成 ── ONNX → 解析 → Decode 子图
  │
  ├── Step 3: Generation ─ model.generate() ── 获取真实回答长度
  │
  └── Step 4: 汇总 ────── Prefill + Decode × gen_len
                            ├── 管线统计 (Ops, FLOPs, Mem, AI)
                            ├── 两阶段对比 + Roofline vs H100
                            └── 标准化 graph.json (schema v1.0)
```

### 配置

```python
MODEL_SOURCE = "../Models/gpt2"              # 本地路径或 HuggingFace 模型名
PROMPTS = ["Hello, how are you?", "法国的首都是什么？"]
MAX_NEW_TOKENS = 20                           # Decode 最大生成长度
SKIP_GENERATION = False                       # True = 仅解析，不生成
```

### 运行

```bash
pip install -r requirements.txt
python run.py
```

---

## ONNX 模式

直接加载 `.onnx` 文件（不需要模型代码、不需要下载权重）。`OnnxParser` 自动跳过权重加载 (`load_external_data=False`)，只需 `.onnx` 文件本身。

### 配置

```python
MODE = "onnx"
ONNX_PATH = "../Models/ONNXs/Qwen2.5-0.5B.onnx"
```

### 运行

```bash
python run.py
```

---

## 输出文件

```
output/模型名_时间戳/
├── graph.json              ← 标准化算子 DAG (schema v1.0)
├── summary.txt             ← 管线统计摘要
├── phase_report.txt        ← Prefill/Decode 对比 + Roofline
├── parallel_report.txt     ← 并行性分析 (B1-B3)
├── layer_report.txt        ← 层级别统计 + 层次树
├── kvcache_report.txt      ← KV cache 依赖 (仅 PyTorch)
├── prefill_graph.json      ← Prefill 子图 (仅 PyTorch)
└── decode_graph.json       ← Decode 子图 (仅 PyTorch)
```

### 终端输出示例

```
Phase                     Ops           FLOPs    Mem(MB)       AI
----------------------------------------------------------------
Prefill                   524   1,484,645,760     519.61     2.86
Decode x20                524   4,945,046,400    9974.95     0.50
----------------------------------------------------------------
Total                   11004   6,429,692,160   10494.55

Layer hierarchy (11 transformer blocks):
  gpt2 [model] (1048 ops)
    embedding [embedding] (556 ops)
    layer_0 [transformer_block] (43 ops)
    ...

Parallelism:  Max=527 ops/level, Avg=2.33 ops/level, Critical=450 steps
KV cache:    12 layers, 190 cross-input edges (O(T²) growth for T=20)
Roofline:    Prefill AI=2.86, MEMORY BOUND (vs H100 peak)
```

---

## 功能清单

| 功能 | 说明 |
|------|------|
| **算子级 DAG** | 基于 ONNX，框架无关的标准化计算图 |
| **算子注册表** | 插件式注册 + 动态回退，永不 UNKNOWN |
| **层边界划分** | ADD 深度分析 + Attention 算子位置检测两策略 |
| **动态 Attention 识别** | 从注册表查询 tags={"attention"} 算子，自动适配融合算子 |
| **并行性分析** | B1 level 分配 / B2 关键路径 / B3 并行度统计 |
| **Roofline 分析** | vs H100/A100/V100 理论峰值，判断计算/访存瓶颈 |
| **Prefill/Decode 拆分** | 两阶段独立子图 + 对比 + KV cache 依赖分析 |
| **硬件 Profile** | A100 / H100 / V100 参数抽象 |

---

## 扩展

### 注册自定义算子

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec
registry = OperatorRegistry.get_default()
registry.register(OperatorSpec(
    name="MY_FUSED_ATTENTION", category="compute",
    tags={"attention"},        # 层划分自动识别
    matching_patterns=["my_fused_attn"],
))
```

### 添加硬件 Profile

```python
from llm_graph_parser.hardware import HardwareProfile
h100 = HardwareProfile(name="H100-SXM", peak_flops_fp16=1979e12,
                       memory_bandwidth=3350e9, memory_size=80e9, tdp=700)
```

---

## 项目结构

```
LLM_Graph_Parser/
├── run.py                       ← 用户入口
├── requirements.txt
├── llm_graph_parser/
│   ├── __init__.py              ← parse_model() / parse_onnx()
│   ├── core/
│   │   ├── operator_node.py     ← OperatorNode, LayerNode
│   │   ├── operator_registry.py ← 算子注册表 (tags 支持)
│   │   ├── computation_graph.py ← DAG + 并行度 + 层统计
│   │   ├── layer_partitioner.py ← 层边界检测 (双层策略)
│   │   ├── phase_splitter.py    ← Prefill/Decode 拆分
│   │   └── serialization.py     ← JSON schema v1.0
│   ├── parser/
│   │   ├── onnx_parser.py       ← ONNX 解析器
│   │   ├── operator_parser.py   ← torch.export 解析器
│   │   └── module_parser.py     ← Module 级遍历
│   └── utils/
│       ├── flops_calculator.py
│       └── memory_calculator.py
└── output/
```
