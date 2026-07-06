"""Versioned graph serialization with forward-compatible schema.

Schema version history:
    "1.0" - Initial standardized format
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .operator_registry import OperatorRegistry

if TYPE_CHECKING:
    from .computation_graph import ComputationGraph

SCHEMA_VERSION = "1.0"


def graph_to_dict(graph: ComputationGraph, registry: OperatorRegistry,
                  include_metadata: bool = True) -> dict:
    """Serialize a ComputationGraph to a standardized dictionary.

    The output schema is versioned and self-describing.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "model_name": graph.model_name,
        "prompt": {
            "text": graph.prompt_text,
            "tokens": graph.prompt_tokens,
        },
        "registry_info": {
            "num_operator_types": registry.num_operators,
            "operator_types": [s.name for s in registry.list_specs()],
        },
        "summary": {
            "num_nodes": graph.num_nodes,
            "num_layers": len(graph.get_layers()),
            "layers": graph.get_layers(),
            "operator_counts": graph.get_operator_counts(),
            "total_flops": sum(n.flops for n in graph.nodes),
            "total_memory_bytes": sum(n.memory_bytes for n in graph.nodes),
        },
        "nodes": [n.to_dict(include_metadata=include_metadata) for n in graph.nodes],
    }


def graph_to_json(graph: ComputationGraph, registry: OperatorRegistry,
                  output_path: str | Path, include_metadata: bool = True,
                  indent: int = 2) -> Path:
    """Serialize a ComputationGraph to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = graph_to_dict(graph, registry, include_metadata=include_metadata)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return output_path
