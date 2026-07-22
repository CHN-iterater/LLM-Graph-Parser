# LLM Graph Parser

自动将大语言模型推理过程解析为 **标准化算子级有向无环图（DAG）**。

每个算子叫什么、调了多少次、输入输出 Shape 多大、FLOPs 多少、数据依赖关系如何、属于哪个 Transformer 层。可选 **GPU 硬件 Profiling** 捕获真实 CUDA Kernel 执行轨迹。基于 ONNX，框架无关。

---

## 目录

- [为什么需要这个工具](#为什么需要这个工具)
- [安装](#安装)
- [快速开始](#快速开始)
- [两种模式](#两种模式)
- [配置文件说明](#配置文件说明)
- [核心算法](#核心算法)
- [硬件 Profiling（可选）](#硬件-profiling可选)
- [输出文件详解](#输出文件详解)
- [扩展指南](#扩展指南)
- [能耗重构（可选）](#能耗重构可选)
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

## 安装

```bash
cd LLM_Graph_Parser
pip install -r requirements.txt
```

---

## 快速开始

### PyTorch 模式

```bash
# run.py 顶部配置
MODE = "pytorch"
MODEL_SOURCE = "../Models/gpt2"
PROMPT = "What's the capital of France?"

python run.py
```

### ONNX 模式

```bash
# run.py 顶部配置
MODE = "onnx"
ONNX_PATH = "../Models/ONNXs/Qwen2.5-0.5B.onnx"

python run.py
```

---

## 两种模式

| 对比项 | PyTorch 模式 | ONNX 模式 |
|--------|-------------|-----------|
| 输入 | 模型代码/路径 + Prompt | `.onnx` 文件 |
| Prefill/Decode 拆分 | ✅ 两阶段独立子图 | ❌ 静态图 |
| 实际文本生成 | ✅ `model.generate()` | ❌ |
| 层边界划分 | ✅ 三策略 | ✅ |
| 并行性分析 | ✅ B1‑B3 | ✅ |
| KV cache 依赖 | ✅ | ❌ |
| GPU Kernel 级 Profiling | ✅ 可选 | ❌ |
| 需要模型代码/权重 | ✅ | ❌ 只需 .onnx |

---

## 配置文件说明

```python
MODE = "pytorch"                          # "pytorch" 或 "onnx"

# ---- PyTorch 模式 ----
MODEL_SOURCE = "../Models/gpt2"           # 本地路径或 HuggingFace 模型名
PROMPT = "What's the capital of France?"   # 自然语言 Prompt
MAX_NEW_TOKENS = 20                       # 最大生成 token 数
SKIP_GENERATION = False                   # True = 跳过生成，只解析图
TRUST_REMOTE_CODE = True                  # 加载自定义模型时需要

# ---- ONNX 模式 ----
ONNX_PATH = "path/to/model.onnx"

# ---- 硬件参数 ----
HARDWARE = {"peak_flops": 1979e12, "memory_bw": 3350e9}  # H100
HARDWARE_PROFILING = False                # True = GPU profiling via torch.profiler
PROFILING_RUNS = 20                       # Prefill / Decode repeat count
GEN_REPEATS = 50                          # Generation per-step forward repeats (higher = less noise)
```

---

## 核心算法

### 算子注册表

插件式注册 + 动态回退，永不出现 `UNKNOWN`。

```python
spec = registry.lookup("torch.ops.aten.linear.default")  # → LINEAR, category=compute
spec = registry.lookup("unknown_aten_op")                # → 自动提取名称并缓存
```

### 层边界划分

三种策略按优先级尝试：

| 策略 | 原理 | 适用模型 |
|------|------|---------|
| ADD 深度分析 | 残差 ADD 拓扑深度远大于普通 ADD | GPT-2（分解式 Attention） |
| SkipLayerNorm 定位 | 检测 SkipSimplifiedLayerNormalization | LLaMA（融合算子） |
| Attention 位置回退 | SOFTMAX / GQA 等位置等分 | 前两种都失败的兜底 |

层类型基于算子结构判断，不依赖具体激活函数名（GELU / SiLU / Sigmoid 均可自动识别）。

### 并行性分析

```
B1: level[u] = max(level[v] + 1)     → 可并行算子识别
B2: cp[u] = max(cp[v]) + weight[u]   → 关键路径（FLOPs 加权）
B3: avg = total / critical_path_len  → 并行度统计
```

### FLOPs 估算

| 算子 | 公式 |
|------|------|
| LINEAR / GEMM | `2 × B × K × N` |
| ATTENTION | `4 × B × H × T × T × d` |
| LAYER_NORM | `3 × numel` |
| SOFTMAX | `5 × numel` |
| GELU / SiLU / ReLU | `numel` |
| RESHAPE / VIEW / TRANSPOSE | `0` |

### Roofline 分析

```
ridge_point = peak_flops / memory_bw    (H100: 590.7)
AI >= ridge_point → COMPUTE BOUND      瓶颈在算力
AI <  ridge_point → MEMORY BOUND       瓶颈在访存
```

---

## 硬件 Profiling（可选）

当 `HARDWARE_PROFILING = True` 且 CUDA 可用时，用 `torch.profiler` 捕获 GPU 上的真实 CUDA Kernel 执行轨迹。

### 开启方式

```python
HARDWARE_PROFILING = True   # run.py 配置
```

无 GPU 时自动跳过，不影响软件侧分析。

### 采集的硬件指标

| 指标 | 来源 | 能耗建模用途 |
|------|------|-------------|
| **GPU time (Prefill/Decode ms)** | CUDA Event 计时 | 功耗时间基座，总能耗 = ∫P(t)dt |
| **GPU kernels (compute/copy)** | torch.profiler | 调度能耗 + 不同 Kernel 类型功耗差异 |
| **Memory peak (MB)** | `cuda.max_memory_allocated` | 显存占用量决定静态功耗 |
| **Avg kernel time (μs)** | 总时间 / Kernel 数 | 短 Kernel 跑不满 GPU，影响功耗分布 |
| **Compute time ratio (%)** | 计算 Kernel 时间占比 | GPU 高功耗状态时间占比 |
| **Achieved BW (GB/s)** | `total_bytes / total_time` | 访存单元功耗占 GPU 30‑40% |
| **FLOPs utilization (%)** | `achieved_flops / peak` | 计算单元利用率与功耗正相关 |
| **Throughput (tokens/s)** | `gen_len / decode_time` | 能效比 J/token 的基础 |

### 输出文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `*_hardware_report.txt` | 文本 | Kernel 分解报告（分类、时段统计） |
| `*_kernel_trace.json` | JSON | 原始 Kernel 事件列表（名称、时长），供外部分析 |

### summary.txt 中的硬件信息

```
GPU time: Prefill=4.66ms, Decode=90.79ms
GPU kernels: 1421 compute + 180 copy, peak mem=1237MB
Avg kernel: 8.5us, compute time ratio=87.3%
Achieved BW: 102 GB/s (3% of H100 peak)
FLOPs util: 15.5% (1.48 TFLOPs / 0.095s)
Throughput: 220.3 tokens/s
```

---

## 输出文件详解

```
output/模型名_时间戳/
├── graph.json              ← 标准化算子 DAG + layer_tree (schema v1.0)
├── summary.txt             ← 管线统计 + 层结构 + 并行性 + Roofline + 硬件指标
├── phase_report.txt        ← Prefill/Decode 对比 + Roofline 分析
├── parallel_report.txt     ← 并行性分析 (B1‑B3)
├── layer_report.txt        ← 层级别统计 + 层次树
├── kvcache_report.txt      ← KV cache 跨输入依赖 (仅 PyTorch)
├── hardware_report.txt     ← GPU Kernel 级分解报告 (需 HARDWARE_PROFILING=True)
├── kernel_trace.json       ← 原始 CUDA Kernel 事件列表
├── per_token_energy.txt    ← 生成阶段逐 token 能耗 + 增长率
├── prefill_graph.json      ← Prefill 子图 (仅 PyTorch)
└── decode_graph.json       ← Decode 子图 (仅 PyTorch)
```

---

## 扩展指南

### 注册自定义算子

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec
registry = OperatorRegistry.get_default()
registry.register(OperatorSpec(
    name="MY_FUSED_ATTENTION", category="compute",
    tags={"attention"},
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

## 能耗重构（可选）

将算子级能耗映射回计算图 DAG，重构推理总能耗并交叉验证。支持两种方法（方向 1: 公式拟合、方向 2: 硬件计数器），内置算子融合折扣与框架开销修正。

详见 [LLM_Graph_Parser.md](LLM_Graph_Parser.md) 第十章。

### 方向 1: 公式法重构

基于 benchmark 拟合公式 `E(N,M,K) = t(N,M,K) × P(N,M,K)` 估算每算子能耗。支持算子融合感知（`--fusion`），自动应用 cuBLASLt epilogue 融合折扣：

| 融合类型 | 折扣 |
|---------|------|
| GEMM + bias (ADD) | 10% |
| GEMM + GELU | 10% |
| GEMM + SiLU + Mul (SwiGLU) | 20% |
| VIEW / RESHAPE / TRANSPOSE | 0%（零开销） |
| LayerNorm / RMSNorm 前后 | 30% |

并内置框架开销（Framework Tax）修正：`E_corrected = E_raw + 349.96 nJ × hidden_size × num_ops`。

```bash
python energy_consumption_refactor.py \
    -g output/ModelName_timestamp/graph.json \
    --gen-len 20 --fusion
```

### 方向 2: 硬件计数器测量

nvml 硬件能量计数器直接读取 GPU 总能耗。Prefill/Decode 阶段各重复 PROFILING_RUNS 次取平均，生成阶段逐 token 测量（每步 GEN_REPEATS 次 forward 降噪）。

```bash
python run.py --runs 20 --gen-repeats 50
python power_analyze.py -t output/ModelName_timestamp/timestamps.txt -n 20
```

### 逐 token 能耗

生成阶段自动输出 `per_token_energy.txt`，记录每个 token 的能耗(J)、耗时(s)、较上一步增长率。

### 相关脚本

| 脚本 | 用途 | 位置 |
|------|------|------|
| `energy_consumption_refactor.py` | 方向 1: 公式法算子能耗重构（支持 --fusion） | `LLM_Graph_Parser/` |
| `power_analyze.py` | 方向 2: 硬件计数器分析 + 框架开销校准 | `LLM_Graph_Parser/` |
| `batch_test.py` | 批量跑多模型对比（同时输出方向 1 + 方向 2） | `LLM_Graph_Parser/` |
| `profile_kernels.py` | 子进程 GPU kernel profiling（避免 CUDA context 污染） | `LLM_Graph_Parser/` |

---

## 运行测试

```bash
cd LLM_Graph_Parser
pip install pytest
python -m pytest tests/ -v
```

62 个测试，全部在 CPU 上运行（硬件 Profiling 测试用 mock 模拟 GPU）：

| 测试文件 | 测试数 | 覆盖内容 |
|---------|--------|---------|
| `test_operator_registry.py` | 10 | 查找、动态回退、tag、自定义算子 |
| `test_computation_graph.py` | 16 | DAG、拓扑排序、并行度、阶段统计、边类型 |
| `test_layer_partitioner.py` | 7 | 边界检测、层类型、子层划分 |
| `test_flops_calculator.py` | 11 | 各类算子 FLOPs、负维度、融合 Attention |
| `test_memory_calculator.py` | 7 | 访存量估算、数据类型、未知算子 |
| `test_hardware_profiler.py` | 7 | Kernel 捕获、报告生成、图回填（Mock CUDA） |

---

## 研究路线中的位置

本项目服务于 **基于算子解构的算力‑电力耦合机理与能耗映射模型** 研究课题：

```
当前 (软件层完成)
  ① 模型解析  ② 算子解析  ③ 计算图构建
  ④ 算子属性标注  ⑤ Prefill/Decode 拆分
  ⑥ 标准化表示  ⑦ GPU Kernel 级 Profiling

下一步
  ⑧ 算子能耗测试  ⑨ GPU 功耗采集  ⑩ 能耗特征提取
  ⑪ 算子→任务功耗重构  ⑫ 映射模型验证
```

每个算子节点的 `hardware_metrics` 字段已预留，后续硬件侧数据可直接填入。

---

## 项目结构

```
LLM_Graph_Parser/
├── run.py                       ← 用户入口（配置 + 运行）
├── requirements.txt
│
├── llm_graph_parser/
│   ├── __init__.py              ← parse_model() / parse_onnx()
│   │
│   ├── core/                    ← 数据结构和算法
│   │   ├── operator_node.py     ← OperatorNode, TensorMeta, LayerNode
│   │   ├── operator_registry.py ← 插件式注册表 + 动态回退 (tags 支持)
│   │   ├── computation_graph.py ← DAG + 并行度 + 层统计 + Roofline
│   │   ├── layer_partitioner.py ← 层边界检测（三策略）
│   │   ├── phase_splitter.py    ← Prefill/Decode 阶段拆分
│   │   └── serialization.py     ← JSON schema v1.0
│   │
│   ├── parser/                  ← 解析引擎
│   │   ├── onnx_parser.py       ← ONNX → OperatorNode（主要路径）
│   │   ├── operator_parser.py   ← torch.export 解析器（备用）
│   │   ├── module_parser.py     ← nn.Module 树遍历
│   │   └── tensor_recorder.py   ← Tensor 元数据提取
│   │
│   ├── utils/                   ← 工具函数
│   │   ├── flops_calculator.py
│   │   └── memory_calculator.py
│   │
│   └── hardware/                ← GPU 硬件参数 + Profiling
│       ├── abstraction.py       ← A100 / H100 / V100 参数
│       └── profiler.py          ← torch.profiler 封装（可选）
│
├── tests/                       ← 62 个单元测试 (pytest)
│   ├── test_operator_registry.py
│   ├── test_computation_graph.py
│   ├── test_layer_partitioner.py
│   ├── test_flops_calculator.py
│   ├── test_memory_calculator.py
│   └── test_hardware_profiler.py  ← Mock GPU
│
└── output/                      ← 运行结果（按 模型名_时间戳 归档）
```
