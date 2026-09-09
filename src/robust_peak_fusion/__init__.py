"""Robust peak timing and sensor fusion."""

from .detectors import DurationAwareDetector, GMMPeakDetector, PeakEstimate
from .fusion import FusedEstimate, fuse_duration_evidence, fuse_peak_estimates
from .synthetic import SyntheticCase, generate_case, generate_dataset

__all__ = [
    "DurationAwareDetector",
    "GMMPeakDetector",
    "PeakEstimate",
    "FusedEstimate",
    "fuse_duration_evidence",
    "fuse_peak_estimates",
    "SyntheticCase",
    "generate_case",
    "generate_dataset",
]

