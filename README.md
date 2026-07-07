# LLM Graph Parser

Parse any LLM into a **standardized operator-level computation graph (DAG)**.
Given a model and prompt, outputs every operator, its count, tensor shapes, FLOPs,
and data dependencies. ONNX-based, framework-agnostic.

---

## Core Pipeline

```
PyTorch model ──→ ONNX (standard intermediate format) ──→ OnnxParser ──→ ComputationGraph (DAG)
                                                                              ↓
                                                                     graph.json (schema v1.0)
                                                                     summary.txt (text summary)
```

Two entry modes:
- **PyTorch mode**: load a model (HuggingFace / local) → export to ONNX (temp, auto-cleaned) → parse
- **ONNX mode**: load a pre-existing `.onnx` file directly

---

## Usage

### 1. Configure

Open `run.py`, modify the top section:

```python
MODEL_SOURCE = "../Models/gpt2_local"    # local path or HuggingFace model ID
PROMPTS = ["Hello, how are you?", "What is the capital of France?"]
```

### 2. Run

```bash
pip install -r requirements.txt
python run.py
```

### 3. Output

```
output/
└── gpt2_local_20260706_1658/          ← model_timestamp (auto, no overwrite)
    ├── prompt_0_graph.json            ← standardized operator DAG
    ├── prompt_0_summary.txt
    ├── prompt_1_graph.json
    └── prompt_1_summary.txt
```

---

## Output Format

### graph.json (schema v1.0)

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

## Capabilities

| Feature | Status | Implementation |
|---------|--------|---------------|
| **(1) Module-level parsing** | ✅ Done | `ModuleParser` — walks `nn.Module` tree, identifies transformer layers |
| **(2) Operator-level parsing** | ✅ Done | `OnnxParser` / `OperatorParser` — counts every operator call |
| **(3) Tensor information** | ✅ Done | `TensorRecorder` — shape, dtype, device per operator |
| **(4) Computation graph (DAG)** | ✅ Done | `ComputationGraph` — nodes + edges, topological sort, layer grouping |
| **(5) Prefill/Decode splitting** | ✅ Done | `PhaseSplitter` — sequence-based and pattern-based splitting |
| **Standardized JSON output** | ✅ Done | `graph.json` — schema v1.0, versioned |
| **Prompt-driven analysis** | ✅ Done | Multi-prompt input, length vs FLOPs comparison |
| **Operator registry** | ✅ Done | `OperatorRegistry` — plugin-style, dynamic fallback (no UNKNOWN) |
| **FLOPs / memory estimation** | ✅ Done | Per-operator estimation based on tensor shapes |
| **ONNX intermediate format** | ✅ Done | ONNX as universal IR for framework-agnostic parsing |
| **Hardware profiles** | ✅ Done | `HardwareProfile` — A100, H100, V100 specs for roofline analysis |
| Multi-prompt comparison with per-sample output | ✅ Done | Timestamped directory, `prompt_0_`/`prompt_1_` naming |

---

## Project Structure

```
LLM_Graph_Parser/
├── run.py                       ← Single entry point
├── requirements.txt
├── llm_graph_parser/
│   ├── __init__.py              ← High-level API: parse_model() / parse_onnx()
│   │
│   ├── core/                    ← Core data structures
│   │   ├── operator_node.py     ← OperatorNode, TensorMeta
│   │   ├── operator_registry.py ← Plugin-style operator registry with dynamic fallback
│   │   ├── computation_graph.py ← DAG construction & analysis
│   │   ├── phase_splitter.py    ← Prefill/Decode phase splitting
│   │   └── serialization.py     ← Versioned JSON (schema v1.0)
│   │
│   ├── parser/                  ← Parsing engines
│   │   ├── onnx_parser.py       ← ONNX-based parser (primary)
│   │   ├── operator_parser.py   ← torch.export-based parser (secondary)
│   │   ├── module_parser.py     ← Module hierarchy walker
│   │   └── tensor_recorder.py   ← Tensor metadata extraction
│   │
│   ├── hooks/                   ← PyTorch hook utilities (torch.export path)
│   │   ├── operator_hook.py
│   │   └── module_hook.py
│   │
│   ├── utils/                   ← FLOPs / memory calculators
│   │   ├── flops_calculator.py
│   │   └── memory_calculator.py
│   │
│   └── hardware/                ← GPU hardware profiles
│       └── abstraction.py       ← A100 / H100 / V100 specs
│
└── output/                      ← Per-run results (auto-organized)
```

---

## Research Roadmap

This tool addresses **Stage 1** of a broader research project on operator-level energy modeling.

```
Current (✅ Complete)
  ├── ① Model parsing          → ModuleParser
  ├── ② Operator parsing       → OnnxParser + dynamic registry (no UNKNOWN)
  ├── ③ Computation graph      → ComputationGraph (DAG, topo sort, layer grouping)
  ├── ④ Operator annotation    → FLOPs, memory bytes, arith_intensity per node
  └── ⑤ Prefill/Decode split   → PhaseSplitter

Next (🔜 Upcoming)
  ├── ⑥ Operator energy testing           → benchmark programs per operator type
  ├── ⑦ GPU power measurement             → NVML / DCGM-based real-time sampling
  ├── ⑧ Operator→task power aggregation   → non-linear power composition model
  └── ⑨ Mapping model validation          → cross-model, cross-GPU verification
```

---

## Extending

### Add a custom operator type

```python
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec

registry = OperatorRegistry.get_default()
registry.register(OperatorSpec(
    name="FLASH_ATTENTION_V2",
    category="compute",
    description="Flash Attention v2 kernel",
    matching_patterns=["flash_attention_v2", "flash_attn_v2"],
))
```

### Add a hardware profile

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
