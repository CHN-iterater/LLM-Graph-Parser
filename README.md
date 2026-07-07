# LLM Graph Parser

给定一个模型和一段 Prompt，自动输出 **标准化算子级计算图（DAG）**：
每个算子叫什么、调了多少次、输入输出多大、FLOPs 多少、数据依赖关系如何。
基于 ONNX，框架无关。

---

## 完整推理管线

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

---

## 用法

### 1. 配置

打开 `run.py`，修改顶部配置区：

```python
MODEL_SOURCE = "../Models/gpt2_local"          # 本地路径或 HuggingFace 模型名
PROMPTS = ["Hello, how are you?", "法国的首都是什么？"]

MAX_NEW_TOKENS = 20                             # Decode 最大生成长度
SKIP_GENERATION = False                         # True = 仅解析，不生成回答
TRUST_REMOTE_CODE = True                        # 新模型需要此选项
```

### 2. 运行

```bash
pip install -r requirements.txt
python run.py
```

### 3. 输出

```
output/
└── gpt2_local_20260706_1658/                  ← 模型名_时间戳（自动创建，不会覆盖）
    ├── prompt_0_graph.json                    ← 全量算子 DAG (schema v1.0)
    ├── prompt_0_prefill_graph.json            ← Prefill 阶段子图
    ├── prompt_0_decode_graph.json             ← Decode 阶段子图
    ├── prompt_0_summary.txt                   ← 管线统计摘要
    └── prompt_0_phase_report.txt              ← 两阶段对比 + Roofline
```

终端输出示例：

```
  [Phase 1/3] Prefill (prompt_len=6)
    ops=524, FLOPs=1484.65M, Mem=519.61MB, AI=2.86
  [Phase 2/3] Decode (1 token)
    per-step: ops=524, FLOPs=247.25M, AI=0.50
  [Phase 3/3] Generating (max 20 tokens)...
    generated: 20 tokens
    answer: "The capital of France is Paris..."

  --- Pipeline Summary ---
                            Ops           FLOPs    Mem(MB)
  --------------------------------------------------------
  Prefill                   524   1,484,645,760     519.61
  Decode ×20                524   4,945,046,400    9974.95
  --------------------------------------------------------
  Total                   11004   6,429,692,160   10494.55
```

---

## 输出格式

### graph.json（schema v1.0）

```json
{
  "schema_version": "1.0",
  "model_name": "gpt2_local",
  "prompt": {"text": "Hello, how are you?", "tokens": 6},
  "nodes": [
    {
      "op_id": "op_0000",
      "op_type": "LINEAR",
      "category": "compute",
      "stage": "prefill",
      "flops": 3538944,
      "memory_bytes": 1048576,
      "arith_intensity": 3.375,
      "parents": [],
      "children": ["op_0001"],
      "input_tensors": [{"shape": [6, 768], "dtype": "float32"}],
      "output_tensors": [{"shape": [6, 2304], "dtype": "float32"}]
    }
  ],
  "summary": {
    "num_nodes": 524,
    "operator_counts": {"LINEAR": 48, "MUL": 72, ...},
    "total_flops": 1484651520
  }
}
```

### summary.txt

```
Model: gpt2_local
Prompt: "Hello, how are you?"
Answer: "The capital of France is Paris..."
Prompt tokens: 6  |  Generated tokens: 20

Phase                     Ops           FLOPs    Mem(MB)       AI
----------------------------------------------------------------
Prefill                   524   1,484,645,760     519.61     2.86
Decode ×20                524   4,945,046,400    9974.95     0.50
----------------------------------------------------------------
Total                   11004   6,429,692,160   10494.55
```

### phase_report.txt（Prefill/Decode 对比 + Roofline）

```
  Phase Comparison: Prefill vs Decode
  Operator nodes           524              0
  Total FLOPs       1,484,645,760          0
  Arith intensity           2.86          0.00

  Roofline Analysis
  Hardware: Peak FP = 1979.0 TFLOPS, Memory BW = 3350 GB/s
  Ridge point:          590.7 FLOPs/byte

  Prefill      AI=2.86      MEMORY BOUND  Util=0.5%
  Decode       AI=0.50      MEMORY BOUND
```

---

## 功能清单

| 研究方案要求 | 实现 | 状态 |
|------------|------|------|
| **(1) Module 级解析** | `ModuleParser` — 模块树遍历与层识别 | ✅ 完成 |
| **(2) Operator 级解析** | `OnnxParser` + `OperatorRegistry` — 动态算子识别（0 UNKNOWN） | ✅ 完成 |
| **(3) Tensor 信息提取** | `TensorRecorder` — shape/dtype/device | ✅ 完成 |
| **(4) 计算图构建** | `ComputationGraph` — DAG + 拓扑排序 + 层级分组 | ✅ 完成 |
| **(5) Prefill/Decode 拆分** | `PhaseSplitter` — 序列长度 / 算子模式两种拆分 | ✅ 完成 |
| **完整推理管线** | Prefill + Decode 模拟 + 实际生成 + 汇总 | ✅ 完成 |
| **两阶段对比 + Roofline** | 独立 `phase_report.txt`，含 AI 和硬件理论峰值对比 | ✅ 完成 |
| **跨模型兼容** | 支持 GPT-2 / LLaMA / Qwen / MiniMind；继承模型 generation_config | ✅ 完成 |
| **动态算子识别** | 所有 ONNX op 自动提取名称，永不出现 UNKNOWN | ✅ 完成 |
| **标准化 JSON 输出** | schema v1.0，版本化自描述 | ✅ 完成 |
| **FLOPs / 访存量 / AI** | 基于 tensor 形状逐算子估算 | ✅ 完成 |
| **硬件 Profile** | A100 / H100 / V100 算力带宽参数 | ✅ 完成 |
| **模型缓存可配置** | `HF_HOME` 环境变量指定缓存目录 | ✅ 完成 |

---

## 项目结构

```
LLM_Graph_Parser/
├── run.py                       ← 唯一入口（配置区在顶部）
├── requirements.txt             ← 核心依赖
│
├── llm_graph_parser/
│   ├── __init__.py              ← 高入 API: parse_model() / parse_onnx()
│   │
│   ├── core/                    ← 数据结构
│   │   ├── operator_node.py     ← OperatorNode, TensorMeta
│   │   ├── operator_registry.py ← 插件式算子注册表（动态回退）
│   │   ├── computation_graph.py ← DAG 构建 + 阶段统计 + Roofline
│   │   ├── phase_splitter.py    ← Prefill/Decode 阶段拆分
│   │   └── serialization.py     ← 版本化 JSON（schema v1.0）
│   │
│   ├── parser/                  ← 解析引擎
│   │   ├── onnx_parser.py       ← ONNX 解析器（主要路径）
│   │   ├── operator_parser.py   ← torch.export 解析器（备用）
│   │   ├── module_parser.py     ← Module 层级遍历
│   │   └── tensor_recorder.py   ← Tensor 元数据提取
│   │
│   ├── hooks/                   ← PyTorch hook 工具
│   ├── utils/                   ← FLOPs / 访存量计算
│   └── hardware/                ← GPU 硬件参数（A100/H100/V100）
│
└── output/                      ← 每次运行按 模型名_时间戳 归档
```

---

## 后续研究路线

```
当前已完成                              下一步
┌──────────────────────┐           ┌──────────────────────┐
│  ① 模型解析            │           │  ⑤ 算子能耗测试        │
│  ② 算子解析            │  ───→    │  ⑥ GPU 功耗采集        │
│  ③ 计算图构建          │           │  ⑦ 能耗特征提取        │
│  ④ 算子属性标注        │           │  ⑧ 算子→任务功耗重构    │
│ ⑤ Prefill/Decode 拆分 │           │  ⑨ 映射模型验证        │
│  ⑥ 标准化表示          │           │                      │
└──────────────────────┘           └──────────────────────┘
```

`graph.json` 中每个算子节点已预留 `hardware_metrics` 字段，后续能耗数据可直接填入。

---

## 扩展

### 注册自定义算子

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec
registry = OperatorRegistry.get_default()
registry.register(OperatorSpec(
    name="MY_CUSTOM_KERNEL", category="compute",
    matching_patterns=["my_custom_kernel"],
))
```

### 添加硬件 Profile

```python
from llm_graph_parser.hardware import HardwareProfile
h100 = HardwareProfile(name="H100-SXM", peak_flops_fp16=1979e12,
                       memory_bandwidth=3350e9, memory_size=80e9, tdp=700)
```
