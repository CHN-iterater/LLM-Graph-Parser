"""Default configuration for the parsing pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParserConfig:
    """Configuration for the LLM Graph Parser pipeline.

    All fields have sensible defaults. Users can override selectively.
    """
    # ---- General ----
    model_name: str = "model"
    output_dir: str = "output"
    hardware_profile: Optional[str] = None  # None = A100-80G default
    schema_version: str = "1.0"

    # ---- Operator matching ----
    exclude_data_movement: bool = False
    custom_operator_patterns: dict[str, list[str]] = field(default_factory=dict)

    # ---- Layer detection ----
    layer_keywords: tuple[str, ...] = ("layer", "block", "transformer_block")
    auto_detect_layers: bool = True

    # ---- FLOPs / Memory ----
    compute_flops: bool = True
    compute_memory: bool = True

    # ---- Output ----
    save_json: bool = True
    save_summary: bool = True
    include_metadata: bool = True
    json_indent: int = 2

    # ---- Phase split ----
    auto_split_phases: bool = True
    prefill_seq_len: int = 0  # 0 = infer from tensors
