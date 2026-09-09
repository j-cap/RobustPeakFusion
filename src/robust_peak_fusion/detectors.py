"""Peak-time estimators used in the comparison notebook."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn.mixture import GaussianMixture

from .preprocessing import normalized_entropy, prepare_series


@dataclass
class PeakEstimate:
    """A peak-time estimate plus diagnostic information."""

    time: float
    confidence: float
    width: float
    method: str
    score: float
    time_grid: np.ndarray = field(repr=False)
    evidence: np.ndarray = field(repr=False)
    candidates: list[dict[str, float]] = field(default_factory=list)


class DurationAwareDetector:
    """Detect the broad, high-area event while suppressing narrow impulses."""

    def __init__(
        self,
        *,
        median_fraction: float = 0.012,
        broad_scale_fraction: float = 0.035,
        min_width_fraction: float = 0.035,
        min_prominence: float = 0.035,
        centroid_level: float = 0.80,
    ) -> None:
        self.median_fraction = median_fraction
        self.broad_scale_fraction = broad_scale_fraction
        self.min_width_fraction = min_width_fraction
        self.min_prominence = min_prominence
        self.centroid_level = centroid_level

    def estimate(self, time: np.ndarray, values: np.ndarray) -> PeakEstimate:
        t, _, raw_evidence = prepare_series(
            time, values, median_fraction=self.median_fraction
        )
        n = len(t)
        dt = float(np.median(np.diff(t)))
        sigma_samples = max(1.0, self.broad_scale_fraction * n)
        broad = gaussian_filter1d(raw_evidence, sigma=sigma_samples, mode="nearest")
        if broad.max() <= 0:
            raise ValueError("signal contains no positive event evidence")
        broad /= broad.max()

        min_width_samples = max(3.0, self.min_width_fraction * n)
        peaks, props = find_peaks(
            broad,
            prominence=self.min_prominence,
            width=min_width_samples,
            rel_height=0.65,
        )
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(broad))])
            prominences = peak_prominences(broad, peaks)[0]
            widths, _, left_ips, right_ips = peak_widths(broad, peaks, rel_height=0.65)
        else:
            prominences = props["prominences"]
            widths = props["widths"]
            left_ips = props["left_ips"]
            right_ips = props["right_ips"]

        candidates: list[dict[str, float]] = []
        for peak, prominence, width, left, right in zip(
            peaks, prominences, widths, left_ips, right_ips
        ):
            lo = max(0, int(np.floor(left)))
            hi = min(n, int(np.ceil(right)) + 1)
            area = float(np.trapezoid(raw_evidence[lo:hi], t[lo:hi]))
            width_time = float(width * dt)
            # Area is primary; width rewards broad physical events. Prominence is
            # deliberately capped so an impulse cannot win on height alone.
            score = area * np.sqrt(max(width_time, dt)) * (0.5 + min(float(prominence), 1.0))
            candidates.append(
                {
                    "index": float(peak),
                    "time": float(t[peak]),
                    "left": float(t[lo]),
                    "right": float(t[hi - 1]),
                    "width": width_time,
                    "prominence": float(prominence),
                    "area": area,
                    "score": score,
                }
            )

        selected = max(candidates, key=lambda item: item["score"])
        lo = int(np.searchsorted(t, selected["left"], side="left"))
        hi = int(np.searchsorted(t, selected["right"], side="right"))
        local = broad[lo:hi]
        local_t = t[lo:hi]
        threshold = self.centroid_level * float(local.max())
        weights = np.clip(local - threshold, 0.0, None)
        if weights.sum() <= 0:
            peak_time = selected["time"]
        else:
            peak_time = float(np.sum(local_t * weights) / np.sum(weights))

        peak_snr = selected["prominence"] / max(
            float(np.median(np.abs(raw_evidence - broad))), 1e-3
        )
        confidence = float(
            np.clip((1.0 - np.exp(-peak_snr / 4.0)) * (1.0 - 0.45 * normalized_entropy(weights)), 0.0, 1.0)
        )
        return PeakEstimate(
            time=peak_time,
            confidence=confidence,
            width=selected["width"],
            method="duration-aware",
            score=selected["score"],
            time_grid=t,
            evidence=broad,
            candidates=candidates,
        )


class GMMPeakDetector:
    """Fit a GMM to time samples weighted by positive signal evidence.

    The selected component is the one with the greatest probability mass. This
    is the most direct implementation of the proposed 'dominant GMM component'
    idea, while BIC chooses the component count.
    """

    def __init__(
        self,
        *,
        max_components: int = 5,
        pseudo_samples: int = 5000,
        median_fraction: float = 0.012,
        random_state: int = 7,
    ) -> None:
        self.max_components = max_components
        self.pseudo_samples = pseudo_samples
        self.median_fraction = median_fraction
        self.random_state = random_state

    def estimate(self, time: np.ndarray, values: np.ndarray) -> PeakEstimate:
        t, _, evidence = prepare_series(
            time, values, median_fraction=self.median_fraction
        )
        # Remove low-level background so the mixture models events, not the full
        # measurement interval. Squaring gives clear peaks more influence while
        # retaining the area advantage of a broad, lower peak.
        cutoff = float(np.quantile(evidence, 0.35))
        weights = np.clip(evidence - cutoff, 0.0, None) ** 2
        if weights.sum() <= 0:
            raise ValueError("signal contains no positive event evidence")
        probabilities = weights / weights.sum()
        rng = np.random.default_rng(self.random_state)
        sampled = rng.choice(t, size=self.pseudo_samples, replace=True, p=probabilities)
        x = sampled.reshape(-1, 1)

        models: list[GaussianMixture] = []
        max_k = min(self.max_components, max(1, len(np.unique(sampled)) // 5))
        for k in range(1, max_k + 1):
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-6,
                n_init=5,
                random_state=self.random_state,
            ).fit(x)
            models.append(model)
        model = min(models, key=lambda item: item.bic(x))

        means = model.means_.ravel()
        stds = np.sqrt(model.covariances_.reshape(-1))
        component_weights = model.weights_.ravel()
        chosen = int(np.argmax(component_weights))

        density = np.exp(model.score_samples(t.reshape(-1, 1)))
        density /= max(float(density.max()), np.finfo(float).eps)
        candidates = [
            {
                "time": float(means[k]),
                "width": float(2.355 * stds[k]),
                "weight": float(component_weights[k]),
                "score": float(component_weights[k]),
            }
            for k in range(model.n_components)
        ]
        sorted_weights = np.sort(component_weights)[::-1]
        margin = float(sorted_weights[0] - sorted_weights[1]) if len(sorted_weights) > 1 else 1.0
        confidence = float(np.clip(0.5 * component_weights[chosen] + 0.5 * margin, 0.0, 1.0))
        return PeakEstimate(
            time=float(means[chosen]),
            confidence=confidence,
            width=float(2.355 * stds[chosen]),
            method=f"GMM (K={model.n_components})",
            score=float(component_weights[chosen]),
            time_grid=t,
            evidence=density,
            candidates=candidates,
        )
