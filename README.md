# LLM Graph Parser

给定一个模型和一段 Prompt，自动输出 **标准化算子级计算图（DAG）**：
每个算子叫什么、调了多少次、输入输出多大、FLOPs 多少、数据依赖关系如何。
基于 ONNX，框架无关。

---

## 核心流程

```
PyTorch 模型 ──→ ONNX（统一中间格式）──→ OnnxParser ──→ ComputationGraph (DAG)
                                                             ↓
                                                    graph.json (schema v1.0)
                                                    summary.txt (文本摘要)
```

两种入口：
- **PyTorch 模式**：加载模型（HuggingFace / 本地）→ 导出 ONNX（临时文件，自动清理）→ 解析
- **ONNX 模式**：直接加载已有的 `.onnx` 文件

---

## 用法

### 1. 配置

打开 `run.py`，修改顶部配置区：

```python
MODEL_SOURCE = "../Models/gpt2_local"    # 本地路径或 HuggingFace 模型名
PROMPTS = ["Hello, how are you?", "What is the capital of France?"]
```

### 2. 运行

```bash
pip install -r requirements.txt
python run.py
```

### 3. 输出

```
output/
└── gpt2_local_20260706_1658/          ← 模型名_时间戳（自动创建，不会覆盖）
    ├── prompt_0_graph.json            ← 标准化算子 DAG
    ├── prompt_0_summary.txt
    ├── prompt_1_graph.json
    └── prompt_1_summary.txt
```

多条 Prompt 时每个独立文件，单条时不加前缀。

---

## 输出格式

### graph.json（schema v1.0）

```json
{
  "schema_version": "1.0",
  "model_name": "gpt2_local",
  "prompt": {
    "text": "Hello, how are you?",
    "tokens": 6
  },
  "nodes": [
    {
      "op_id": "op_0000",
      "op_type": "LINEAR",
      "category": "compute",
      "flops": 3538944,
      "memory_bytes": 1048576,
      "arith_intensity": 3.375,
      "stage": "prefill",
      "layer_id": "root",
      "parents": [],
      "children": ["op_0001"],
      "input_tensors": [{"shape": [6, 768], "dtype": "float32", "device": "cpu"}],
      "output_tensors": [{"shape": [6, 2304], "dtype": "float32", "device": "cpu"}]
    }
  ],
  "summary": {
    "num_nodes": 524,
    "num_layers": 1,
    "operator_counts": {"LINEAR": 48, "MUL": 72, ...},
    "total_flops": 1484651520,
    "total_memory_bytes": 524288000
  }
}
```

### summary.txt

```
Model: gpt2_local
Prompt: "Hello, how are you?"
Prompt tokens: 6
Total operator nodes: 524

Operator counts:
  RESHAPE    : 158
  MUL        : 72
  ADD        : 61
  TRANSPOSE  : 61
  LINEAR     : 48
  ...
```

---

## 功能清单（对应研究方案）

| 研究方案要求 | 实现 | 状态 |
|------------|------|------|
| **(1) Module 级解析** | `ModuleParser` — 遍历模块树，识别 Transformer 层 | ✅ 完成 |
| **(2) Operator 级解析** | `OnnxParser` + `OperatorRegistry` — 算子识别与调用统计 | ✅ 完成 |
| **(3) Tensor 信息提取** | `TensorRecorder` — shape/dtype/device 记录 | ✅ 完成 |
| **(4) 计算图构建** | `ComputationGraph` — DAG 节点+边、拓扑排序、层级分组 | ✅ 完成 |
| **(5) Prefill/Decode 拆分** | `PhaseSplitter` — 按序列长度和算子模式两种方式拆分 | ✅ 完成 |
| **标准化 JSON 输出** | schema v1.0，版本化自描述 | ✅ 完成 |
| **Prompt 驱动分析** | 多条 Prompt 自动对比长度 vs FLOPs | ✅ 完成 |
| **算子注册表** | 插件式注册 + 动态回退（永不出现 UNKNOWN） | ✅ 完成 |
| **FLOPs/访存量估算** | 基于 tensor 形状逐算子估算 | ✅ 完成 |
| **ONNX 作为统一中间格式** | 框架无关，支持 PyTorch/TF/JAX 导出的模型 | ✅ 完成 |
| **硬件 Profile** | A100 / H100 / V100 算力带宽参数 | ✅ 完成 |

---

## 后续研究路线

当前项目完成的是整个课题的 **算子级可解构基础**，后续工作在此之上展开：

```
当前已完成 ──→ ⑤ 算子能耗测试       ──→ ⑥ GPU 功耗采集       ──→ ⑦ 能耗特征提取
① 模型解析         为每类算子设计           用 NVML/DCGM          建立算子属性与功耗
② 算子解析         微基准测试程序           实时采集 GPU 功耗      之间的映射关系
③ 计算图构建       获取单次调用             与算子执行时间对齐
④ 算子属性标注     的功耗数据
                  ↓                        ↓                        ↓
                 ⑧ 算子→任务功耗重构       ──→ ⑨ 映射模型验证
                 利用 DAG 按执行顺序        跨模型（GPT-2/LLaMA/Qwen）
                 聚合各算子功耗             跨 GPU 验证预测精度
                 考虑非线性叠加效应
```

`graph.json` 中的每个算子节点已预留 `hardware_metrics` 字段，后续能耗数据可直接填入。

---

## 项目结构

```
LLM_Graph_Parser/
├── run.py                       ← 唯一人口
├── requirements.txt             ← 依赖
│
├── llm_graph_parser/
│   ├── __init__.py              ← 高入 API: parse_model() / parse_onnx()
│   │
│   ├── core/                    ← 核心数据结构
│   │   ├── operator_node.py     ← OperatorNode, TensorMeta
│   │   ├── operator_registry.py ← 插件式算子注册表（动态回退）
│   │   ├── computation_graph.py ← DAG 构建与分析
│   │   ├── phase_splitter.py    ← Prefill/Decode 阶段拆分
│   │   └── serialization.py     ← 版本化 JSON 输出（schema v1.0）
│   │
│   ├── parser/                  ← 解析引擎
│   │   ├── onnx_parser.py       ← ONNX 解析器（主要路径）
│   │   ├── operator_parser.py   ← torch.export 解析器（备用路径）
│   │   ├── module_parser.py     ← Module 层级遍历
│   │   └── tensor_recorder.py   ← Tensor 元数据提取
│   │
│   ├── hooks/                   ← PyTorch hook 工具
│   ├── utils/                   ← FLOPs / 访存量计算
│   └── hardware/                ← GPU 硬件参数（A100/H100/V100）
│
└── output/                      ← 每次运行结果自动按模型_时间戳归档
```

---

## 扩展

### 注册自定义算子

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec

registry = OperatorRegistry.get_default()
registry.register(OperatorSpec(
    name="MY_CUSTOM_KERNEL",
    category="compute",
    description="自定义算子",
    matching_patterns=["my_custom_kernel"],
))
```

### 添加硬件 Profile

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
