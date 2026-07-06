"""Hardware abstraction layer.

Provides hardware profiles that map operator properties to specific GPU models.
This allows the computation graph to be analyzed in a device-agnostic way,
then mapped to concrete hardware for energy/performance estimation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    """Profile of a GPU hardware platform.

    These values serve as reference points for roofline analysis
    and energy estimation. Measured values should be calibrated
    against actual hardware.
    """
    name: str
    peak_flops_fp16: float = 0.0       # TFLOPS
    peak_flops_fp32: float = 0.0       # TFLOPS
    peak_flops_bf16: float = 0.0       # TFLOPS
    memory_bandwidth: float = 0.0      # GB/s
    hbm_capacity: int = 0              # GB
    l2_cache: int = 0                  # MB
    tdp: int = 0                       # Watts (thermal design power)
    architecture: str = ""
    description: str = ""

    def roofline_ceiling(self, flops: int, bytes_: int) -> float:
        """Compute arithmetic intensity: FLOPs/Byte."""
        return flops / max(bytes_, 1)


# ---- Predefined profiles ----
# These are approximate reference values; calibrate for precise modeling.

PROFILES: dict[str, HardwareProfile] = {
    "A100-80G": HardwareProfile(
        name="NVIDIA A100-80G",
        peak_flops_fp16=312.0,
        peak_flops_bf16=312.0,
        peak_flops_fp32=156.0,
        memory_bandwidth=2039.0,
        hbm_capacity=80,
        l2_cache=40,
        tdp=400,
        architecture="Ampere",
        description="NVIDIA A100 SXM with 80GB HBM2e",
    ),
    "H100-80G": HardwareProfile(
        name="NVIDIA H100-80G",
        peak_flops_fp16=1979.0,
        peak_flops_bf16=1979.0,
        peak_flops_fp32=989.0,
        memory_bandwidth=3350.0,
        hbm_capacity=80,
        l2_cache=50,
        tdp=700,
        architecture="Hopper",
        description="NVIDIA H100 SXM with 80GB HBM3",
    ),
    "V100-32G": HardwareProfile(
        name="NVIDIA V100-32G",
        peak_flops_fp16=125.0,
        peak_flops_fp32=62.5,
        memory_bandwidth=900.0,
        hbm_capacity=32,
        l2_cache=6,
        tdp=300,
        architecture="Volta",
        description="NVIDIA V100 SXM with 32GB HBM2",
    ),
    "A10-24G": HardwareProfile(
        name="NVIDIA A10-24G",
        peak_flops_fp16=125.0,
        peak_flops_fp32=62.5,
        memory_bandwidth=600.0,
        hbm_capacity=24,
        l2_cache=6,
        tdp=150,
        architecture="Ampere",
        description="NVIDIA A10 with 24GB GDDR6",
    ),
}

# Default profile used when no specific hardware is selected
DEFAULT_PROFILE = PROFILES["A100-80G"]


def get_profile(name: Optional[str] = None) -> HardwareProfile:
    """Get a hardware profile by name.

    Args:
        name: Profile name (e.g., "A100-80G", "H100-80G").
              If None, returns the default profile (A100-80G).

    Returns:
        The matching HardwareProfile.
    """
    if name is None:
        return DEFAULT_PROFILE
    name = name.upper().replace("-", "").replace(" ", "")
    for key, profile in PROFILES.items():
        if key.upper().replace("-", "") == name:
            return profile
    raise KeyError(f"Unknown hardware profile: {name}. Available: {list(PROFILES.keys())}")


def list_profiles() -> list[str]:
    """List all available hardware profile names."""
    return list(PROFILES.keys())
