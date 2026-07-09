"""Tests for HardwareProfiler: GPU profiling with mocked CUDA events.

All tests run on CPU using ``unittest.mock`` — no GPU required.
Verifies that kernel capture, report generation, and graph annotation
work correctly even when no real hardware is present.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock


@pytest.fixture
def mock_cuda():
    """Mock CUDA availability so profiler.available = True."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Mock GPU (H100)"), \
         patch("torch.version.cuda", "12.6"), \
         patch("torch.cuda.memory_allocated", return_value=512 * 1024 * 1024), \
         patch("torch.cuda.max_memory_allocated", return_value=2048 * 1024 * 1024), \
         patch("torch.cuda.reset_peak_memory_stats"):
        yield


@pytest.fixture
def mock_events():
    """Mock torch.profiler.profile context manager returning fake kernel events."""

    class FakeEvent:
        def __init__(self, name, dur):
            self.name = name
            self.duration_us = dur
            self.duration = dur
            self.device_time = dur

    events = [
        FakeEvent("gemm_bf16_256x128", 125),
        FakeEvent("elementwise_kernel", 45),
        FakeEvent("softmax_kernel", 30),
        FakeEvent("gemm_bf16_128x64", 98),
        FakeEvent("reduce_kernel", 22),
    ]

    # Build a mock profiler context manager
    mock_prof = MagicMock()
    mock_prof.events.return_value = events
    mock_prof.__enter__.return_value = mock_prof
    with patch("torch.profiler.profile", return_value=mock_prof):
        yield events


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_no_gpu_fallback():
    """Without GPU, all profiler methods should return safe defaults."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    assert profiler.available is False
    assert profiler.report_text() == "  GPU profiling: not available (no CUDA device detected)"
    assert profiler.trace(None, None) == 0.0
    assert profiler.trace_generate(None, None) == (0, 0.0)


def test_mock_gpu_detection(mock_cuda):
    """With mocked CUDA, profiler should detect GPU and report specs."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    assert profiler.available is True
    report = profiler.report_text()
    assert "Mock GPU (H100)" in report
    assert "Total CUDA kernels: 0" in report  # no events captured yet


def test_trace_captures_events(mock_cuda, mock_events):
    """trace() should capture and process kernel events from profiler."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()

    # Mock a forward pass
    mock_model = MagicMock()
    mock_input = MagicMock()

    elapsed = profiler.trace(mock_model, mock_input, label="prefill")
    assert elapsed > 0
    assert profiler._prefill_total_us > 0

    # Check that events were captured
    report = profiler.report_text()
    assert "Total CUDA kernels: 5" in report
    assert "gemm" in report
    assert "elementwise" in report
    assert "softmax" in report


def test_trace_generate_captures_events(mock_cuda, mock_events):
    """trace_generate() should also capture kernel events."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()

    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.shape = [1, 10]
    mock_model.generate.return_value = mock_out

    class FakeInput:
        shape = [1, 5]
    gen_len, total_us = profiler.trace_generate(mock_model, FakeInput(), max_new_tokens=5)
    assert isinstance(gen_len, (int, float))  # should be numeric
    assert total_us > 0

    report = profiler.report_text()
    assert "Total CUDA kernels: 5" in report


def test_attach_to_graph_fills_metrics(mock_cuda, mock_events):
    """attach_to_graph() should populate hardware_metrics on each node."""
    from llm_graph_parser.hardware import HardwareProfiler
    from llm_graph_parser.core.computation_graph import ComputationGraph
    from llm_graph_parser.core.operator_node import OperatorNode

    profiler = HardwareProfiler()
    profiler._prefill_total_us = 320
    profiler._trace_data = [
        {"name": "gemm", "duration_us": 125},
        {"name": "elementwise", "duration_us": 45},
        {"name": "softmax", "duration_us": 30},
        {"name": "gemm_2", "duration_us": 98},
        {"name": "reduce", "duration_us": 22},
    ]

    g = ComputationGraph("test")
    for i in range(3):
        n = OperatorNode(op_id=f"op_{i}", op_type="LINEAR", op_name=f"linear_{i}",
                         flops=1000 * (i + 1))
        g.add_node(n)

    profiler.attach_to_graph(g)

    for node in g.nodes:
        assert "kernels" in node.hardware_metrics
        assert "total_kernel_time_us" in node.hardware_metrics
        assert "num_kernels" in node.hardware_metrics
        assert node.hardware_metrics["total_kernel_time_us"] > 0


def test_save_report_writes_file(mock_cuda, mock_events, tmp_path):
    """save_report() should write a non-empty hardware report."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    mock_model = MagicMock()
    mock_input = MagicMock()
    profiler.trace(mock_model, mock_input)

    path = profiler.save_report(tmp_path, name="test")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "CUDA kernels" in content
    assert "Mock GPU" in content


def test_trace_to_json_export(mock_cuda, mock_events, tmp_path):
    """trace_to_json() should export kernel trace as JSON."""
    from llm_graph_parser.hardware import HardwareProfiler
    import json

    profiler = HardwareProfiler()
    mock_model = MagicMock()
    profiler.trace(mock_model, MagicMock())

    path = profiler.trace_to_json(tmp_path, name="test")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 5
    assert data[0]["name"] == "gemm_bf16_256x128"
