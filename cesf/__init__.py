"""
Causal Event Space Framework (CESF) — operational reduction of possibility spaces.

Pipeline: Ω_H → Γ_H(C) → G_{ε,H} → Γ_H(C)/~ → R̃_Q → R_Q → E_H(Q)
"""

from cesf.types import OperationalQuery, AdmissibilityConstraints, RelevanceConfig
from cesf.dynamics import GBMSystem, JumpDiffusionSystem
from cesf.pipeline import CESFPipeline, PipelineResult
from cesf.relevance import RelevanceFunctional
from cesf.metrics import operational_complexity_bits, compression_ratio

__version__ = "0.1.0"

__all__ = [
    "OperationalQuery",
    "AdmissibilityConstraints",
    "RelevanceConfig",
    "GBMSystem",
    "JumpDiffusionSystem",
    "CESFPipeline",
    "PipelineResult",
    "RelevanceFunctional",
    "operational_complexity_bits",
    "compression_ratio",
]
