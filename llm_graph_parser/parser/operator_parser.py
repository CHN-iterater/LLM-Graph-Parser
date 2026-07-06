"""Operator-level parser.

Statistically analyze the operators executed during model inference,
using torch.export or torch.fx captured graphs.

Uses ``OperatorRegistry`` for extensible operator matching.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch

from llm_graph_parser.core.operator_registry import OperatorRegistry


class OperatorParser:
    """Parse and analyze operators from a torch.export ExportedProgram.

    Args:
        model: Optional PyTorch model to export.
        registry: OperatorRegistry to use. Defaults to ``get_default()``.
    """

    def __init__(self, model: Optional[torch.nn.Module] = None,
                 registry: Optional[OperatorRegistry] = None):
        self.model = model
        self._registry = registry or OperatorRegistry.get_default()
        self._exported_program = None
        self._graph = None

    def export(self, *example_args) -> None:
        """Export the model using torch.export to capture the computation graph."""
        if self.model is None:
            raise ValueError("Model must be provided for export.")

        self.model.eval()
        self._exported_program = torch.export.export(self.model, args=example_args)
        self._graph = self._exported_program.graph

    def load_from_exported(self, exported_program) -> None:
        """Load from an already-exported program."""
        self._exported_program = exported_program
        self._graph = exported_program.graph

    @property
    def graph(self):
        return self._graph

    def count_operators(self, exclude_data_movement: bool = False) -> Counter:
        """Count operator call frequencies in the captured graph.

        Args:
            exclude_data_movement: If True, skip data_movement ops.

        Returns:
            Counter mapping operator type name -> call count.
        """
        counter: Counter = Counter()
        if self._graph is None:
            return counter

        for node in self._graph.nodes:
            if node.op != "call_function":
                continue
            op_spec = self._resolve_op(node)
            if exclude_data_movement and op_spec.category in (
                "data_movement", "inplace"
            ):
                continue
            counter[op_spec.name] += 1

        return counter

    def classify_operators(self) -> dict[str, list]:
        """Group operators by category (compute, normalization, activation, etc.).

        Returns:
            dict of ``{category_name: [(node_name, op_type_name), ...]}``
        """
        categories: dict[str, list] = defaultdict(list)
        if self._graph is None:
            return categories

        for node in self._graph.nodes:
            if node.op != "call_function":
                continue
            op_spec = self._resolve_op(node)
            categories[op_spec.category].append((node.name, op_spec.name))

        return dict(categories)

    def _resolve_op(self, node):
        """Map a torch graph node's target to an OperatorSpec via registry."""
        target = node.target
        target_str = str(target)
        return self._registry.lookup(target_str)

    @staticmethod
    def print_report(counter: Counter, title: str = "Operator Count Report") -> None:
        """Print a formatted operator count report."""
        total = sum(counter.values())
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
        print(f"  Total operator calls: {total}\n")
        print(f"  {'Operator':30s} {'Count':>8s}")
        print(f"  {'-' * 40}")
        for op_name, count in counter.most_common():
            print(f"  {op_name:30s} {count:8d}")
        print(f"{'=' * 50}")

    @staticmethod
    def save_report(counter: Counter, output_dir: str | Path,
                    filename: str = "operator_counts.txt") -> Path:
        """Save operator count report to a text file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        total = sum(counter.values())
        lines = [
            "Operator Count Report",
            "=" * 50,
            f"Total operator calls: {total}\n",
            f"{'Operator':30s} {'Count':>8s}",
            "-" * 40,
        ]
        for op_name, count in counter.most_common():
            lines.append(f"{op_name:30s} {count:8d}")
        lines.append("=" * 50)
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path
