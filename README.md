# LLM Graph Parser

给定一个模型和一段 Prompt，自动输出 **算子级计算图（DAG）**：
每个算子叫什么、调了多少次、输入输出多大、FLOPs 多少。

**核心流程：** PyTorch 模型 → ONNX（统一中间格式）→ 标准化 graph.json + summary.txt

---

## 用法

### 1. 配置

打开 `run.py`，修改顶部配置区：

```python
MODEL_SOURCE = "../Models/gpt2_local"   # 本地模型路径，或 HuggingFace 模型名

PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
]
```

单条 Prompt 也可以，多条会自动对比。

### 2. 运行

```bash
pip install -r requirements.txt
python run.py
```

### 3. 结果

```
output/
└── gpt2_local_20260706_1658/          ← 模型名_时间戳，每次运行独立
    ├── prompt_0_graph.json            ← 标准化算子 DAG（schema v1.0）
    ├── prompt_0_summary.txt
    ├── prompt_1_graph.json
    └── prompt_1_summary.txt
```

终端输出对比：

```
  Prompt [0]: "Hello, how are you?"         tokens: 6
  Prompt [1]: "What is the capital of France?"  tokens: 7

  Prompt 长度对比:
       6 tokens  →  1484.65 MFLOPs
       7 tokens  →  1732.35 MFLOPs
```

---

## 两种模式

在 `run.py` 顶部切换：

```python
MODE = "pytorch"        # 加载本地/HuggingFace 模型，自动导出 ONNX 后解析
# MODE = "onnx"         # 直接加载已有的 .onnx 文件
```

| 模式 | 输入 | 适用场景 |
|------|------|---------|
| `pytorch` | 模型路径/名称 + Prompt | 你有 PyTorch 模型（HuggingFace、本地训练等） |
| `onnx` | `.onnx` 文件路径 | 你已经导出了 ONNX 文件，或从其他框架转换而来 |

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
      "input_tensors": [{"shape": [6, 768], "dtype": "float32"}],
      "output_tensors": [{"shape": [6, 2304], "dtype": "float32"}],
      "parents": [],
      "children": ["op_0001"]
    }
  ]
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

## 你只需要准备

| 东西 | 说明 |
|------|------|
| **模型** | 本地路径（如 `"../Models/gpt2_local"`）或 HuggingFace ID（如 `"gpt2"`） |
| **Prompt** | 你想分析的自然语言文本 |
| **运行环境** | `pip install -r requirements.txt` |

### 关于 ONNX

解析流程内部通过 ONNX 作为统一中间格式。这样做的好处：

- **框架无关** — 同一套解析逻辑处理 PyTorch、TensorFlow、JAX 等导出的模型
- **标准化** — ONNX 是业界标准，图结构清晰（节点=算子，边=Tensor）
- **透明性** — 如需保留 ONNX，在 `parse_model()` 中传入 `onnx_path="path/to/model.onnx"` 即可

注意：ONNX 路径的算子粒度比 PyTorch 直接导出更细（如 Attention 会被拆为 MatMul + Softmax + Reshape），这是标准化过程的正常现象。

---

## 项目文件

```
LLM_Graph_Parser/
├── run.py                       # ← 唯一入口，改 Prompt 和模型
├── requirements.txt             # 依赖：torch、onnx、numpy
├── llm_graph_parser/
│   ├── __init__.py              # 人口：parse_model() / parse_onnx()
│   ├── core/                    # 数据结构：OperatorNode、DAG、PhaseSplitter
│   ├── parser/                  # 解析引擎：OnnxParser、OperatorParser
│   ├── utils/                   # FLOPs / 访存量计算
│   └── hardware/                # GPU 硬件 Profile（A100/H100 等）
└── output/                      # 运行结果（自动按模型_时间戳分类）
```
