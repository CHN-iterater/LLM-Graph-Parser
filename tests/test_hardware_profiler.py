"""Tests for HardwareProfiler: operator-level latency measurement.

All tests run on CPU using ``unittest.mock`` — no GPU required.
Verifies that CUDA Event timing, report generation, and graph annotation
work correctly even when no real hardware is present.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_cuda():
    """Mock CUDA availability so profiler.available = True."""
    with patch("torch.cuda.is_available", return_value=True),          patch("torch.cuda.get_device_name", return_value="Mock H100"),          patch("torch.cuda.synchronize"),          patch("torch.cuda.reset_peak_memory_stats"),          patch("torch.cuda.memory_allocated", return_value=512 * 1024 * 1024),          patch("torch.cuda.max_memory_allocated", return_value=2048 * 1024 * 1024),          patch("torch.cuda.Event") as mock_event:
        mock_event.return_value.elapsed_time.return_value = 10.0
        yield


@pytest.fixture
def mock_cuda_legacy():
    """Mock CUDA availability so profiler.available = True."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Mock H100"), \
         patch("torch.cuda.Event") as mock_event, \
         patch("torch.cuda.memory_allocated", return_value=512 * 1024 * 1024), \
         patch("torch.cuda.max_memory_allocated", return_value=2048 * 1024 * 1024), \
         patch("torch.cuda.reset_peak_memory_stats"):
        # Mock CUDA Event to return a fixed elapsed time
        mock_event.return_value.elapsed_time.return_value = 10.0  # 10ms
        yield


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_no_gpu_fallback():
    """Without GPU, all profiler methods should return safe defaults."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    assert profiler.available is False
    assert "not available" in profiler.report_text()
    assert profiler.time_forward(None, None) == 0.0
    assert profiler.time_generate(None, None) == (0, 0.0)


def test_mock_gpu_detection(mock_cuda):
    """With mocked CUDA, profiler should detect GPU."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    assert profiler.available is True
    report = profiler.report_text()
    assert "Mock H100" in report


def test_time_forward(mock_cuda):
    """time_forward() should measure and store prefill time."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()

    mock_model = MagicMock()
    mock_input = MagicMock()

    elapsed = profiler.time_forward(mock_model, mock_input, label="prefill")
    assert elapsed == 10000.0  # 10ms → 10000us
    assert profiler._prefill_total_us == 10000.0


def test_time_generate(mock_cuda):
    """time_generate() should measure decode time and return gen_len."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()

    class FakeOutput:
        shape = [1, 15]

    mock_model = MagicMock()
    mock_model.generate.return_value = FakeOutput()

    class FakeInput:
        shape = [1, 5]

    gen_len, total_us = profiler.time_generate(mock_model, FakeInput(), max_new_tokens=10)
    assert gen_len == 10  # 15 - 5
    assert total_us == 10000.0
    assert profiler._decode_total_us == 10000.0


def test_attach_to_graph(mock_cuda):
    """attach_to_graph() should populate hardware_metrics."""
    from llm_graph_parser.hardware import HardwareProfiler
    from llm_graph_parser.core.computation_graph import ComputationGraph
    from llm_graph_parser.core.operator_node import OperatorNode

    profiler = HardwareProfiler()
    profiler._prefill_total_us = 5000
    profiler._decode_total_us = 5000
    profiler._memory_peak = 1024 * 1024 * 1024
    profiler._memory_start = 512 * 1024 * 1024

    g = ComputationGraph("test")
    for i in range(3):
        n = OperatorNode(op_id=f"op_{i}", op_type="LINEAR", op_name=f"linear_{i}",
                         flops=1000 * (i + 1))
        g.add_node(n)

    profiler.attach_to_graph(g)
    for node in g.nodes:
        assert "gpu_time_us" in node.hardware_metrics
        assert "gpu_memory_peak_mb" in node.hardware_metrics
        assert node.hardware_metrics["gpu_time_us"] > 0


def test_save_report(mock_cuda, tmp_path):
    """save_report() should write a non-empty hardware report."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    profiler._prefill_total_us = 5000
    profiler._decode_total_us = 10000

    path = profiler.save_report(tmp_path, name="test")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Hardware Profiling" in content
    assert "Mock H100" in content


def test_report_with_metrics(mock_cuda):
    """report_text() with gen_len/bytes should include throughput and BW."""
    from llm_graph_parser.hardware import HardwareProfiler
    profiler = HardwareProfiler()
    profiler._prefill_total_us = 5000
    profiler._decode_total_us = 10000

    report = profiler.report_text(gen_len=20, total_flops=1e9, total_bytes=1e8)
    assert "Throughput" in report
