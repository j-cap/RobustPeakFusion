"""Missing-sensor-safe fusion strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import PeakEstimate


@dataclass
class FusedEstimate:
    time: float
    confidence: float
    agreement: bool
    contributing_sensors: tuple[str, ...]
    evidence: np.ndarray | None = None


def fuse_peak_estimates(
    estimates: dict[str, PeakEstimate],
    *,
    delays: dict[str, float] | None = None,
    agreement_tolerance: float = 0.08,
) -> FusedEstimate:
    """Confidence-weighted late fusion, valid for one or more sensors."""
    if not estimates:
        raise ValueError("at least one sensor estimate is required")
    delays = delays or {}
    names = tuple(estimates)
    corrected = np.array([estimates[n].time - delays.get(n, 0.0) for n in names])
    weights = np.array([max(estimates[n].confidence, 0.05) for n in names])
    if len(names) == 1:
        return FusedEstimate(
            time=float(corrected[0]),
            confidence=float(estimates[names[0]].confidence),
            agreement=True,
            contributing_sensors=names,
        )

    spread = float(corrected.max() - corrected.min())
    agreement = spread <= agreement_tolerance
    if agreement:
        time = float(np.average(corrected, weights=weights))
        agreement_factor = np.exp(-spread / max(agreement_tolerance, 1e-9))
        confidence = float(np.clip(np.average(weights) * (0.75 + 0.25 * agreement_factor), 0, 1))
    else:
        # Averaging incompatible events can create a time where no peak exists.
        # In disagreement, select the stronger sensor and report low confidence.
        best = int(np.argmax(weights))
        time = float(corrected[best])
        confidence = float(0.4 * weights[best])
    return FusedEstimate(time, confidence, agreement, names)


def fuse_duration_evidence(
    estimates: dict[str, PeakEstimate],
    *,
    delays: dict[str, float] | None = None,
) -> FusedEstimate:
    """Fuse duration-aware evidence curves before extracting the peak center."""
    if not estimates:
        raise ValueError("at least one sensor estimate is required")
    delays = delays or {}
    names = tuple(estimates)
    reference = estimates[names[0]].time_grid
    if len(names) == 1:
        name = names[0]
        estimate = estimates[name]
        return FusedEstimate(
            time=float(estimate.time - delays.get(name, 0.0)),
            confidence=float(estimate.confidence),
            agreement=True,
            contributing_sensors=names,
            evidence=estimate.evidence.copy(),
        )
    accumulated = np.zeros_like(reference)
    total_weight = 0.0
    corrected_times = []
    for name in names:
        est = estimates[name]
        delay = delays.get(name, 0.0)
        shifted = np.interp(reference + delay, est.time_grid, est.evidence, left=0.0, right=0.0)
        weight = max(est.confidence, 0.05)
        accumulated += weight * shifted
        total_weight += weight
        corrected_times.append(est.time - delay)
    accumulated /= total_weight

    level = 0.80 * float(accumulated.max())
    high = np.clip(accumulated - level, 0.0, None)
    time = float(np.sum(reference * high) / np.sum(high)) if high.sum() else float(reference[np.argmax(accumulated)])
    spread = float(np.ptp(corrected_times)) if len(corrected_times) > 1 else 0.0
    agreement = spread <= 0.08
    confidence = float(np.clip(np.mean([estimates[n].confidence for n in names]) * (1.0 if agreement else 0.5), 0, 1))
    return FusedEstimate(time, confidence, agreement, names, evidence=accumulated)
