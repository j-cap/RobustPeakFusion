"""Synthetic broad-event signals with narrow distractor peaks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class SyntheticCase:
    time: np.ndarray
    sensors: dict[str, np.ndarray]
    true_peak_time: float
    metadata: dict[str, float]


def _broad_event(t: np.ndarray, center: float, width: float, shape: str) -> np.ndarray:
    if shape == "gaussian":
        return np.exp(-0.5 * ((t - center) / width) ** 2)
    if shape == "plateau":
        half = 0.7 * width
        edge = 0.18 * width
        return expit((t - (center - half)) / edge) * expit(((center + half) - t) / edge)
    if shape == "asymmetric":
        left_width, right_width = 0.65 * width, 1.35 * width
        return np.where(
            t <= center,
            np.exp(-0.5 * ((t - center) / left_width) ** 2),
            np.exp(-0.5 * ((t - center) / right_width) ** 2),
        )
    raise ValueError(f"unknown shape: {shape}")


def generate_case(
    seed: int,
    *,
    n_samples: int = 800,
    missing_probability: float = 0.25,
) -> SyntheticCase:
    """Generate one case with a known broad event and taller narrow spikes."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_samples)
    center = float(rng.uniform(0.38, 0.62))
    width = float(rng.uniform(0.07, 0.14))
    shape = str(rng.choice(["gaussian", "plateau", "asymmetric"]))
    sensors: dict[str, np.ndarray] = {}

    for sensor_index, name in enumerate(("blue", "orange")):
        delay = float(rng.normal(0.0, 0.009))
        amplitude = float(rng.uniform(0.65, 1.25))
        y = amplitude * _broad_event(t, center + delay, width * rng.uniform(0.85, 1.2), shape)
        y += rng.uniform(-0.18, 0.18) + rng.uniform(-0.15, 0.15) * t
        y += rng.normal(0.0, rng.uniform(0.02, 0.065), size=n_samples)

        for _ in range(int(rng.integers(1, 5))):
            spike_center = center + rng.choice([-1.0, 1.0]) * rng.uniform(0.13, 0.34)
            spike_width = rng.uniform(0.002, 0.012)
            spike_height = amplitude * rng.uniform(1.05, 2.2)
            y += spike_height * np.exp(-0.5 * ((t - spike_center) / spike_width) ** 2)

        # Make missingness possible but never remove both sensors.
        if sensor_index == 0 or rng.random() >= missing_probability:
            sensors[name] = y

    if len(sensors) == 2 and rng.random() < missing_probability:
        sensors.pop(str(rng.choice(list(sensors))))
    return SyntheticCase(
        time=t,
        sensors=sensors,
        true_peak_time=center,
        metadata={"width": width, "shape": shape},
    )


def generate_dataset(n_cases: int = 200, *, seed: int = 17) -> list[SyntheticCase]:
    rng = np.random.default_rng(seed)
    return [generate_case(int(s)) for s in rng.integers(0, 2**31 - 1, size=n_cases)]

