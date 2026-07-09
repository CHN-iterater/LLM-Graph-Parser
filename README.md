# LLM Graph Parser

自动将大语言模型推理过程解析为 **标准化算子级有向无环图（DAG）**。

**输出**：每个算子叫什么、调了多少次、输入输出 shape 多大、FLOPs 多少、数据依赖关系如何、属于哪个 Transformer 层。

基于 ONNX，框架无关。

---

## 目录

- [为什么需要这个工具](#为什么需要这个工具)
- [如何工作](#如何工作)
- [快速开始](#快速开始)
- [两种模式](#两种模式)
- [配置文件说明](#配置文件说明)
- [输出文件详解](#输出文件详解)
- [核心算法](#核心算法)
- [扩展指南](#扩展指南)
- [运行测试](#运行测试)
- [研究路线中的位置](#研究路线中的位置)
- [项目结构](#项目结构)

---

## 为什么需要这个工具

在做算子级能耗建模之前，必须先回答三个基础问题：

> 1. **模型推理时实际执行了哪些算子？**
> 2. **每种算子执行了多少次？**
> 3. **算子之间的数据依赖和执行顺序是什么？**

LLM Graph Parser 解决的就是这三个问题。它输出标准化的计算图 JSON，作为后续能耗建模、性能瓶颈分析的数据入口。

---

## 如何工作

```
PyTorch 模型 / HuggingFace ──→ ONNX 导出 ──→ OnnxParser 解析 ──→ ComputationGraph
           .onnx 文件 ────────────────────────→ OnnxParser 解析 ──→ ComputationGraph
                                                                        │
                                      ┌── graph.json (标准化算子 DAG)
                                      ├── summary.txt (管线统计摘要)
                                      ├── phase_report.txt (阶段对比 + Roofline)
                                      ├── parallel_report.txt (并行性分析)
                                      ├── layer_report.txt (层级别统计)
                                      └── kvcache_report.txt (KV cache 依赖)
```

核心设计决策：**以 ONNX 作为统一中间格式**。任何框架（PyTorch/TensorFlow/JAX）的模型都可以导出为 ONNX，然后用同一套解析逻辑处理。

---

## 快速开始

### 安装

```bash
cd LLM_Graph_Parser
pip install -r requirements.txt
```

### PyTorch 模式（需要模型 + Prompt）

```python
# run.py 顶部配置
MODE = "pytorch"
MODEL_SOURCE = "../Models/gpt2"                    # 本地路径或 HuggingFace 模型名
PROMPTS = ["Hello, how are you?", "What is the capital of France?"]
MAX_NEW_TOKENS = 20
SKIP_GENERATION = False
```

```bash
python run.py
```

### ONNX 模式（只需 .onnx 文件）

```python
# run.py 顶部配置
MODE = "onnx"
ONNX_PATH = "../Models/ONNXs/Qwen2.5-0.5B.onnx"
```

```bash
python run.py
```

---

## 两种模式

| 对比项 | PyTorch 模式 | ONNX 模式 |
|--------|-------------|-----------|
| 输入 | 模型代码/路径 + 自然语言 Prompt | `.onnx` 文件 |
| 数据流 | Prompt → Tokenize → ONNX 导出 → 解析 | `.onnx` → 加载 → 解析 |
| Prefill 图 | ✅ 完整 Prompt 前向 | ✅ 从 ONNX 读取 |
| Decode 图 | ✅ 单 token 前向导出 | ❌ 静态图只有一张 |
| 实际生成 | ✅ `model.generate()` 获取 gen_len | ❌ 需手动传入 |
| 层边界划分 | ✅ | ✅ |
| 并行性分析 | ✅ | ✅ |
| KV cache 依赖分析 | ✅ 基于 Decode 图 | ❌ 无多步信息 |
| Roofline 分析 | ✅ | ✅ |
| 需要模型代码/权重 | ✅ | ❌ 只需 .onnx |

在 `run.py` 顶部切换:

```python
MODE = "pytorch"     # 或 "onnx"
```

---

## 配置文件说明

```python
MODE = "pytorch"                          # "pytorch" 或 "onnx"

# ---- PyTorch 模式 ----
MODEL_SOURCE = "../Models/gpt2"           # 本地路径或 HuggingFace 模型名
PROMPTS = ["Hello, how are you?"]          # 支持多条 Prompt 对比
MAX_NEW_TOKENS = 20                       # 最大生成 token 数
SKIP_GENERATION = False                   # True = 跳过生成阶段
TRUST_REMOTE_CODE = True                  # 加载自定义模型时需要

# ---- ONNX 模式 ----
ONNX_PATH = "path/to/model.onnx"

# ---- 硬件参数（Roofline 分析用） ----
HARDWARE = {"peak_flops": 1979e12, "memory_bw": 3350e9}  # H100
HARDWARE_PROFILING = False                # True = 用 torch.profiler 做 Kernel 级分析（需 GPU）
```

---

## 输出文件详解

每次运行在 `output/模型名_时间戳/` 下生成以下文件：

### graph.json

标准化算子计算图（schema v1.0），是所有后续分析的入口。

```json
{
  "schema_version": "1.0",
  "model_name": "gpt2",
  "prompt": {"text": "Hello, how are you?", "tokens": 6},
  "summary": {
    "num_nodes": 524,
    "num_layers": 12,
    "operator_counts": {"LINEAR": 49, "SOFTMAX": 24, ...},
    "total_flops": 1484651520,
    "total_memory_bytes": 519605976
  },
  "layer_tree": {
    "layer_id": "gpt2", "layer_type": "model",
    "children": [
      {"layer_id": "layer_0", "layer_type": "transformer_block", ...},
      ...
    ]
  },
  "nodes": [
    {
      "op_id": "op_0000",
      "op_type": "LINEAR",
      "category": "compute",
      "stage": "prefill",
      "layer_id": "layer_0",
      "flops": 3538944,
      "memory_bytes": 1048576,
      "arith_intensity": 3.375,
      "parents": [],
      "children": ["op_0001"],
      "input_tensors": [{"shape": [6, 768], "dtype": "float32"}],
      "output_tensors": [{"shape": [6, 2304], "dtype": "float32"}]
    }
  ]
}
```

### summary.txt

终端输出的主要内容，包含：

- 模型名、Prompt、回答文本
- 管线统计表（Prefill / Decode / Total 的 Ops、FLOPs、访存量、AI）
- 层次化层结构树（embedding → layer_0 → ... → lm_head）
- 并行性关键指标（最大/平均并行度、关键路径长度）
- KV cache 跨输入依赖（层数、跨步边数）
- Roofline 结论（COMPUTE BOUND 或 MEMORY BOUND）
- 算子类型分布统计

### phase_report.txt

两阶段对比分析 + Roofline 分析：

- Prefill vs Decode 的算子数、FLOPs、AI 对比表
- 各类别 FLOPs 分解（compute / activation / elementwise / normalization）
- 硬件理论峰值对比，计算密度 vs 访存带宽 Roofline 曲线

### parallel_report.txt

并行性分析（B1-B3）：

- B1: level 分配 — 每个拓扑 level 的算子数
- B2: 关键路径 — 无权关键路径长度 + FLOPs 加权关键路径
- B3: 并行度统计 — 峰值并行度、平均并行度
- Prefill vs Decode 两阶段并行度对比

### layer_report.txt

层级别统计：

- 层次化层结构树（完整展开）
- 每层的算子数、FLOPs、访存量、算子类型分布

### kvcache_report.txt (仅 PyTorch 模式)

KV cache 跨输入依赖分析：

- Transformer 层数
- attention 算子识别
- 跨输入边数（T×T/2 增长）
- 总 attention FLOPs（含 KV cache 增长）

### prefill_graph.json / decode_graph.json (仅 PyTorch 模式)

独立的两阶段子图，格式同 `graph.json`，但只包含对应阶段的算子节点。

---

## 核心算法

### 算子注册表

```python
# 查找: 先匹配模式, 未命中则动态提取名称
spec = registry.lookup("torch.ops.aten.linear.default")
# → LINEAR, category=compute

spec = registry.lookup("unknown_aten_op_123")
# → 自动提取 "UNKNOWN_ATEN_OP_123", category=other
# → 注册到表中, 下次直接命中
```

永不出现 UNKNOWN。每个新算子第一次遇到时自动从 target 字符串提取名称并缓存。

### 层边界划分

三种策略按优先级尝试:

1. **ADD 深度分析** — 残差连接 ADD 的拓扑深度远大于普通 ADD，每 2 个残差 ADD 标记一个 block 边界
2. **SkipLayerNorm 定位** — 对融合算子模型，检测 SkipSimplifiedLayerNormalization，每 2 个为一组
3. **Attention 位置回退** — 检测 SOFTMAX / GROUPQUERYATTENTION 等算子的位置等分

层类型判断基于算子结构而非激活函数名:

| 条件 | 判定结果 |
|------|---------|
| 含 `SOFTMAX` / `ATTENTION` / `GROUPQUERYATTENTION` 等 | `transformer_block` |
| 含 `EMBEDDING` 或名称为 `embedding` | `embedding` |
| 名称为 `lm_head` | `lm_head` |
| 含 `LINEAR` | `mlp` |

### 并行性分析

```
B1: level[u] = max(level[v] + 1)   对前驱 v
B2: cp[u] = max(cp[v]) + weight[u]  对前驱 v (weight = 1 或 FLOPs)
B3: 平均并行度 = total_nodes / critical_path_length
    峰值并行度 = max(每个 level 的节点数)
```

### FLOPs 估算

| 算子 | 公式 |
|------|------|
| LINEAR / GEMM | `2 × B × K × N` |
| ATTENTION (分解式) | `4 × B × H × T × T × d` |
| ATTENTION (融合式 GQA) | `4 × B × T × T × hidden` |
| LAYER_NORM | `3 × numel` |
| SOFTMAX | `5 × numel` |
| GELU / SiLU / ReLU | `numel` |
| RESHAPE / VIEW / TRANSPOSE | `0`（纯数据搬运） |

### Roofline 分析

```
ridge_point = peak_flops / memory_bw    (H100: 590.7 FLOPs/byte)
AI = total_flops / total_memory_bytes

if AI >= ridge_point → COMPUTE BOUND (算力瓶颈)
if AI < ridge_point  → MEMORY BOUND (访存瓶颈)
```

---

## 扩展指南

### 注册新的算子类型

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec
registry = OperatorRegistry.get_default()
registry.register(OperatorSpec(
    name="MY_FUSED_KERNEL",
    category="compute",
    tags={"attention"},                 # 层划分自动识别
    matching_patterns=["my_fused_kernel"],
))
```

### 添加新的硬件 Profile

```python
from llm_graph_parser.hardware import HardwareProfile

h100 = HardwareProfile(
    name="H100-SXM",
    peak_flops_fp16=1979e12,
    peak_flops_fp32=989e12,
    memory_bandwidth=3350e9,
    memory_size=80e9,
    tdp=700,
)
```

### 获取 Attention 类算子的方法

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry
reg = OperatorRegistry.get_default()
attn_ops = reg.get_by_tag("attention")   # 所有带 tags={"attention"} 的算子
```

---

## 运行测试

```bash
cd LLM_Graph_Parser
pip install pytest
python -m pytest tests/ -v
```

55 个测试，覆盖:

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/test_operator_registry.py` | 算子查找、动态回退、标签、自定义算子 |
| `tests/test_computation_graph.py` | DAG 构建、拓扑排序、并行度、阶段统计、边类型 |
| `tests/test_layer_partitioner.py` | 边界检测、层类型、子层划分 |
| `tests/test_flops_calculator.py` | 各类算子 FLOPs、负维度、融合 Attention |
| `tests/test_memory_calculator.py` | 访存量估算、数据类型、未知算子 |

---

## 研究路线中的位置

本项目服务于 **基于算子解构的算力-电力耦合机理与能耗映射模型** 研究课题的模块 3（算子级能耗测试与任务能耗重构）：

```
当前 (软件层完成)
  ① 模型解析      ② 算子解析      ③ 计算图构建
  ④ 算子属性标注  ⑤ Prefill/Decode 拆分  ⑥ 标准化表示

下一步 (硬件层)
  ⑦ 算子能耗测试  ⑧ GPU 功耗采集  ⑨ 能耗特征提取
  ⑩ 算子→任务功耗重构  ⑪ 映射模型验证
```

每个算子节点的 `hardware_metrics` 字段已预留，后续能耗测试数据可直接填入。

---

## 项目结构

```
LLM_Graph_Parser/
├── run.py                           # 用户入口（配置 + 运行）
├── requirements.txt                 # 核心依赖
├── setup.py                         # Python 包配置
│
├── llm_graph_parser/
│   ├── __init__.py                  # 高入 API: parse_model() / parse_onnx()
│   │
│   ├── core/                        # 核心数据结构和算法
│   │   ├── operator_node.py         # OperatorNode, TensorMeta, LayerNode
│   │   ├── operator_registry.py     # 算子注册表 + 动态回退 (tags 支持)
│   │   ├── computation_graph.py     # DAG 构建 + 并行度分析 + 层统计 + Roofline
│   │   ├── layer_partitioner.py     # 层边界检测（三策略）
│   │   ├── phase_splitter.py        # Prefill/Decode 阶段拆分
│   │   └── serialization.py         # 版本化 JSON (schema v1.0)
│   │
│   ├── parser/                      # 解析引擎
│   │   ├── onnx_parser.py           # ONNX → OperatorNode（主要路径）
│   │   ├── operator_parser.py       # torch.export 解析器（备用）
│   │   ├── module_parser.py         # nn.Module 树遍历
│   │   └── tensor_recorder.py       # Tensor 元数据提取
│   │
│   ├── hooks/                       # PyTorch hook（torch.export 备用路径）
│   │   ├── operator_hook.py
│   │   └── module_hook.py
│   │
│   ├── utils/                       # 工具函数
│   │   ├── flops_calculator.py      # 逐算子 FLOPs 估算
│   │   └── memory_calculator.py     # 逐算子访存量估算
│   │
│   └── hardware/                    # GPU 硬件参数抽象
│       └── abstraction.py           # A100 / H100 / V100 算力带宽参数
│
├── tests/                           # 55 个单元测试 (pytest)
│   ├── test_operator_registry.py
│   ├── test_computation_graph.py
│   ├── test_layer_partitioner.py
│   ├── test_flops_calculator.py
│   └── test_memory_calculator.py
│
└── output/                          # 运行结果（自动按 模型名_时间戳 归档）
```
