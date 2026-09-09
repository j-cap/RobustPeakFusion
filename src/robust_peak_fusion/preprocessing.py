"""Shared signal preprocessing utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def prepare_series(
    time: np.ndarray,
    values: np.ndarray,
    *,
    median_fraction: float = 0.012,
    baseline_quantile: float = 0.10,
    upper_quantile: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate, de-spike, baseline-correct, and robustly normalize a series.

    Returns ``(time, cleaned_values, positive_normalized_evidence)``.
    The time vector must be strictly increasing after finite samples are selected.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("time and values must be one-dimensional arrays of equal length")

    finite_t = np.isfinite(t)
    if finite_t.sum() < 5:
        raise ValueError("at least five finite time samples are required")
    t = t[finite_t]
    y = y[finite_t]
    order = np.argsort(t)
    t, y = t[order], y[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("time samples must be unique")

    finite_y = np.isfinite(y)
    if finite_y.sum() < 5:
        raise ValueError("at least five finite sensor samples are required")
    y_interp = np.interp(t, t[finite_y], y[finite_y])

    kernel = max(3, int(round(median_fraction * len(y_interp))))
    kernel += 1 - kernel % 2
    cleaned = median_filter(y_interp, size=kernel, mode="nearest")

    baseline = float(np.quantile(cleaned, baseline_quantile))
    upper = float(np.quantile(cleaned, upper_quantile))
    scale = max(upper - baseline, np.finfo(float).eps)
    evidence = np.clip((cleaned - baseline) / scale, 0.0, None)
    return t, cleaned, evidence


def normalized_entropy(weights: np.ndarray) -> float:
    """Return entropy in [0, 1], ignoring zero entries."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if w.sum() <= 0:
        return 1.0
    p = w / w.sum()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))

