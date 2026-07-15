"""Optional GPU profiling for operator-level latency measurement.

When enabled and CUDA is available, measures total inference time and
GPU memory using ``torch.cuda.Event``. Granularity is operator-level,
not CUDA kernel-level.
"""

from __future__ import annotations
from pathlib import Path


class HardwareProfiler:
    """GPU profiler for operator-level latency measurement.

    Gracefully handles missing GPU — all methods are no-ops when
    ``torch.cuda.is_available()`` is ``False``.
    """

    def __init__(self):
        import torch
        self._available = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._available else "cpu")
        self._prefill_total_us: float = 0.0
        self._decode_total_us: float = 0.0
        self._memory_peak: int = 0
        self._memory_start: int = 0

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Timing (CUDA Event)
    # ------------------------------------------------------------------

    def time_forward(self, model, input_ids, label: str = "forward",
                     num_runs: int = 1) -> float:
        """Run forward pass(es), return avg elapsed time in us."""
        if not self._available:
            return 0.0
        import torch

        torch.cuda.reset_peak_memory_stats(self._device)
        self._memory_start = torch.cuda.memory_allocated(self._device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            base_len = input_ids.shape[1]
            for i in range(num_runs):
                pad = torch.zeros(1, i, dtype=torch.long, device=input_ids.device)
                x = torch.cat([input_ids, pad], dim=1)
                _ = model(x)
        end.record()
        torch.cuda.synchronize()

        self._memory_peak = torch.cuda.max_memory_allocated(self._device)
        us = start.elapsed_time(end) * 1000 // num_runs

        if label == "prefill":
            self._prefill_total_us = us
        elif label == "decode":
            self._decode_total_us = us
        return us

    def time_generate(self, model, input_ids, max_new_tokens=20,
                      num_runs: int = 1,
                      **gen_kwargs) -> tuple[int, float]:
        """Run generate (num_runs times), return (tokens, avg time us)."""
        if not self._available:
            return 0, 0.0
        import torch

        torch.cuda.reset_peak_memory_stats(self._device)
        self._memory_start = torch.cuda.memory_allocated(self._device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(num_runs):
                out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                    **gen_kwargs)
        end.record()
        torch.cuda.synchronize()

        self._memory_peak = torch.cuda.max_memory_allocated(self._device)
        us = start.elapsed_time(end) * 1000 // num_runs
        self._decode_total_us = us
        return out.shape[1] - input_ids.shape[1], us

    # ------------------------------------------------------------------
    # Graph annotation
    # ------------------------------------------------------------------

    def attach_to_graph(self, graph) -> None:
        """Store measured latency into ``hardware_metrics``."""
        if not self._available or not graph:
            return
        total_f = sum(n.flops for n in graph.nodes) or 1
        total_us = self._prefill_total_us + self._decode_total_us
        mem = max(self._memory_peak - self._memory_start, 0) / 1e6
        for node in graph.nodes:
            node.hardware_metrics.update({
                "gpu_time_us": round((node.flops / total_f) * total_us, 2),
                "gpu_memory_peak_mb": round(mem, 2),
            })

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report_text(self, gen_len: int = 0, total_flops: float = 0,
                    total_bytes: float = 0) -> str:
        """Format report with measured latency, bandwidth, throughput."""
        if not self._available:
            return "  GPU profiling: not available (no CUDA device detected)"
        import torch

        pf = self._prefill_total_us / 1000
        dc = self._decode_total_us / 1000
        total = pf + dc
        mem = max(self._memory_peak - self._memory_start, 0) / 1e6

        lines = [
            "=" * 60,
            "  Hardware Profiling",
            "=" * 60,
            f"  Device: {torch.cuda.get_device_name(0)}",
            f"  Memory peak: {mem:.0f} MB",
            "",
            f"  {'Phase':20s} {'Time (ms)':>12s}",
            "-" * 34,
        ]
        if pf > 0:
            lines.append(f"{'Prefill':20s} {pf:>12.2f}")
        if dc > 0:
            lines.append(f"{'Decode':20s} {dc:>12.2f}")
        if total > 0:
            lines.append(f"{'Total':20s} {total:>12.2f}")
        if total > 0 and total_bytes > 0:
            bw = total_bytes / total / 1e6
            try:
                from .abstraction import get_profile
                p = get_profile("H100-SXM")
                peak = p.memory_bandwidth / 1e9
                lines.append(f"\n  Achieved BW: {bw:.0f} GB/s ({bw/peak*100:.0f}% of H100)")
            except Exception:
                lines.append(f"\n  Achieved BW: {bw:.0f} GB/s")
        if gen_len > 0 and dc > 0:
            lines.append(f"  Throughput: {gen_len/(dc/1000):.1f} tokens/s")
        return "\n".join(lines)

    def save_report(self, output_dir: str | Path, name: str = "",
                    gen_len: int = 0, total_flops: float = 0,
                    total_bytes: float = 0) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_" if name else ""
        path = output_dir / f"{stem}hardware_report.txt"
        path.write_text(self.report_text(gen_len=gen_len, total_flops=total_flops,
                                        total_bytes=total_bytes), encoding="utf-8")
        return path
