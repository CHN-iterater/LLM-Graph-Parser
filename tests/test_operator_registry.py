"""Tests for OperatorRegistry: lookup, dynamic fallback, tags."""

import pytest
from llm_graph_parser.core.operator_registry import OperatorRegistry, OperatorSpec


@pytest.fixture
def registry():
    return OperatorRegistry.get_default()


class TestOperatorRegistry:
    def test_lookup_known_linear(self, registry):
        spec = registry.lookup("linear")
        assert spec.name == "LINEAR"
        assert spec.category == "compute_bound"

    def test_lookup_known_softmax(self, registry):
        spec = registry.lookup("softmax")
        assert spec.name == "SOFTMAX"
        assert spec.category == "memory_bound"

    def test_lookup_known_aten_format(self, registry):
        spec = registry.lookup("torch.ops.aten.addmm.default")
        assert spec.name == "LINEAR"

    def test_dynamic_fallback_no_unknown(self, registry):
        """Any target string should return a named spec, never UNKNOWN."""
        for target in ["custom_op_123", "aten.foobar", "my.fused.kernel"]:
            spec = registry.lookup(target)
            assert spec.name != "UNKNOWN", f"{target} returned UNKNOWN"

    def test_dynamic_fallback_caches(self, registry):
        """Dynamic fallback should register and cache the new spec."""
        first = registry.lookup("some.rare.op")
        second = registry.lookup("some.rare.op")
        assert first.name == second.name

    def test_dynamic_name_extraction(self, registry):
        spec = registry.lookup("torch.ops.aten.copy_.default")
        assert spec.name == "COPY"
        assert spec.category == "data_movement"

    def test_dynamic_name_builtin(self, registry):
        spec = registry.lookup("<built-in function getitem>")
        assert spec.name == "GETITEM"

    def test_register_custom_operator(self, registry):
        registry.register(OperatorSpec(
            name="MY_CUSTOM_OP", category="compute",
            matching_patterns=["my_custom_pattern"],
        ))
        spec = registry.lookup("my_custom_pattern")
        assert spec.name == "MY_CUSTOM_OP"

    def test_register_overwrite(self, registry):
        registry.register(OperatorSpec(
            name="LINEAR", category="elementwise",
            matching_patterns=["linear"],
        ))
        spec = registry.lookup("linear")
        assert spec.category == "elementwise"  # overwritten

    def test_tags_fresh(self):
        reg = OperatorRegistry()
        reg.register(OperatorSpec(
            name="TEST_ATTN", category="compute",
            matching_patterns=["test_attn"],
            tags={"attention"},
        ))
        assert len(reg.get_by_tag("attention")) == 1
        assert reg.get_by_tag("attention")[0].name == "TEST_ATTN"

    def test_get_unknown(self, registry):
        spec = registry.get("UNKNOWN")
        assert spec is not None
        assert spec.name == "UNKNOWN"
