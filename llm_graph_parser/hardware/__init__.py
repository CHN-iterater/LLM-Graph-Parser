"""Hardware abstraction layer for device-agnostic computation graph analysis.

Provides predefined hardware profiles (A100, H100, etc.) and utilities
for mapping operator properties to specific GPU models.
"""
from .abstraction import HardwareProfile, get_profile, list_profiles
from .profiler import HardwareProfiler

__all__ = ["HardwareProfile", "get_profile", "list_profiles", "HardwareProfiler"]
