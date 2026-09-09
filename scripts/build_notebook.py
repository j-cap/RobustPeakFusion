"""Build the executable comparison notebook committed with this project."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_compare_peak_detectors.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Robust peak timing: GMM versus duration-aware fusion

This notebook compares two ways of locating the desired **broad physical event** when shorter but taller peaks may appear before or after it:

1. a Gaussian mixture model (GMM) fitted to amplitude-weighted timestamps;
2. a duration-aware detector that explicitly rewards broad, high-area events and suppresses narrow impulses.

Both are evaluated with both sensors, blue only, and orange only. The two supplied screenshots provide realistic shapes; synthetic data provide known ground truth for quantitative testing.

> **Operational definition:** “peak time” means the center of the dominant broad event—not necessarily the largest observed sample. If the application instead needs the instantaneous maximum, the target definition and labels must change."""))

cells.append(nbf.v4.new_code_cell(r"""from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from robust_peak_fusion import (
    DurationAwareDetector,
    GMMPeakDetector,
    fuse_duration_evidence,
    fuse_peak_estimates,
    generate_dataset,
)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"blue": "tab:blue", "orange": "tab:orange"}
duration_detector = DurationAwareDetector()
gmm_detector = GMMPeakDetector(max_components=5, pseudo_samples=3000, random_state=7)
data = {
    name: pd.read_csv(ROOT / "data" / f"{name}.csv")
    for name in ("AP_01", "AP_02")
}
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
data["AP_01"].head()"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 1. Digitized measurements

The traces were recovered by HSV color segmentation followed by the median colored-pixel position in each image column. Because the figures contain no numerical axis labels, both axes are normalized. These data preserve shape and relative timing, but not physical units. The reproducible extraction script is included; the original screenshots are intentionally not published."""))

cells.append(nbf.v4.new_code_cell(r"""fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ax, (name, frame) in zip(axes, data.items()):
    for sensor, color in COLORS.items():
        ax.plot(frame.time_normalized, frame[sensor], color=color, lw=1, label=sensor)
    ax.set(title=f"{name}: digitized traces", ylabel="Image-derived amplitude")
    ax.legend(loc="best")
axes[-1].set_xlabel("Normalized time")
fig.tight_layout()
fig.savefig(FIGURES / "01_digitized_measurements.png", dpi=160, bbox_inches="tight")
plt.close(fig)"""))

cells.append(nbf.v4.new_markdown_cell(r"""![Digitized blue and orange sensor measurements](../figures/01_digitized_measurements.png)"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 2. The two estimators

### GMM baseline

After median filtering and baseline subtraction, normalized positive amplitude is treated as probability mass over time. Weighted pseudo-samples are drawn, BIC chooses $K\in\{1,\ldots,5\}$, and the mean of the component with greatest mixture weight is selected. This is a direct implementation of the proposed “detect components and select the dominant one” idea.

### Duration-aware candidate

The alternative uses:

1. median filtering to suppress short impulses;
2. robust baseline/scale normalization;
3. Gaussian integration at the expected broad-event scale;
4. candidate scoring by area, width, and capped prominence;
5. a high-evidence centroid for the event time.

For two sensors, the duration evidence is fused before extracting the center. With one sensor, the same code simply uses that sensor. Sensor-specific propagation delays can be supplied when calibration data become available."""))

cells.append(nbf.v4.new_code_cell(r"""def estimate_measurement(frame, detector, sensors=("blue", "orange")):
    t = frame.time_normalized.to_numpy()
    estimates = {
        sensor: detector.estimate(t, frame[sensor].to_numpy())
        for sensor in sensors
    }
    if isinstance(detector, DurationAwareDetector):
        fused = fuse_duration_evidence(estimates)
    else:
        fused = fuse_peak_estimates(estimates)
    return estimates, fused


def plot_comparison(name, frame, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    methods = [("GMM", gmm_detector), ("Duration-aware", duration_detector)]
    t = frame.time_normalized.to_numpy()
    for col, (label, detector) in enumerate(methods):
        estimates, fused = estimate_measurement(frame, detector)
        top, bottom = axes[0, col], axes[1, col]
        for sensor, color in COLORS.items():
            top.plot(t, frame[sensor], color=color, alpha=.75, lw=1, label=sensor)
            top.axvline(estimates[sensor].time, color=color, ls="--", lw=1.5,
                        label=f"{sensor} estimate")
            bottom.plot(estimates[sensor].time_grid, estimates[sensor].evidence,
                        color=color, lw=1.8, label=f"{sensor} evidence")
        top.axvline(fused.time, color="black", lw=2.2, label=f"fused = {fused.time:.3f}")
        if fused.evidence is not None:
            bottom.plot(estimates["blue"].time_grid, fused.evidence,
                        color="black", lw=2.2, label="fused evidence")
        bottom.axvline(fused.time, color="black", lw=2)
        top.set_title(f"{name} — {label}")
        top.set_ylabel("Amplitude")
        bottom.set_ylabel("Normalized evidence")
        bottom.set_xlabel("Normalized time")
        top.legend(fontsize=8, ncol=2)
        bottom.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


plot_comparison("AP_01", data["AP_01"], "02_ap_01_detector_comparison.png")
plot_comparison("AP_02", data["AP_02"], "03_ap_02_detector_comparison.png")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Detector comparison on AP_01

![GMM and duration-aware detector comparison on AP_01](../figures/02_ap_01_detector_comparison.png)

### Detector comparison on AP_02

![GMM and duration-aware detector comparison on AP_02](../figures/03_ap_02_detector_comparison.png)"""))

cells.append(nbf.v4.new_code_cell(r"""rows = []
for name, frame in data.items():
    for method_name, detector in (("GMM", gmm_detector), ("Duration-aware", duration_detector)):
        estimates, fused = estimate_measurement(frame, detector)
        rows.append({
            "measurement": name,
            "method": method_name,
            "blue time": estimates["blue"].time,
            "orange time": estimates["orange"].time,
            "sensor disagreement": abs(estimates["blue"].time - estimates["orange"].time),
            "fused time": fused.time,
            "fused confidence": fused.confidence,
            "GMM components (blue/orange)": (
                f"{len(estimates['blue'].candidates)}/{len(estimates['orange'].candidates)}"
                if method_name == "GMM" else "—"
            ),
        })
real_summary = pd.DataFrame(rows)
real_summary.round(3)"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Interpretation of the two measurements

There is no ground-truth timestamp in the screenshots, so these plots cannot establish absolute accuracy. They can reveal structural behavior:

- BIC frequently uses several Gaussian components for one broad plateau. The selected mean is therefore a subregion of the event and can jump when one sensor is removed.
- The duration-aware response remains one broad region and places its estimate near the center of that region.
- The blue/orange difference may include a real location-dependent propagation delay. It should be calibrated from paired measurements before treating disagreement as error.

The next section explicitly checks whether removing one sensor causes a large change."""))

cells.append(nbf.v4.new_code_cell(r"""rows = []
for name, frame in data.items():
    for method_name, detector in (("GMM", gmm_detector), ("Duration-aware", duration_detector)):
        _, both = estimate_measurement(frame, detector, ("blue", "orange"))
        for available in (("blue",), ("orange",), ("blue", "orange")):
            estimates, fused = estimate_measurement(frame, detector, available)
            rows.append({
                "measurement": name,
                "method": method_name,
                "available": "+".join(available),
                "estimate": fused.time,
                "change from two-sensor estimate": abs(fused.time - both.time),
                "confidence": fused.confidence,
            })
missing_sensor_results = pd.DataFrame(rows)
missing_sensor_results.round(3)"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 3. Synthetic data with known ground truth

The generator is based on features visible in the figures and the stated failure mode:

- Gaussian, plateau, or asymmetric broad events;
- sensor-specific amplitude, offset, slope, width, delay, and noise;
- one to four narrow peaks before or after the desired event, often taller than it;
- random loss of either sensor, while ensuring at least one remains.

The exact parameter ranges are intentionally centralized in `synthetic.py`, so they can later be fitted to a larger real dataset rather than hand-selected."""))

cells.append(nbf.v4.new_code_cell(r"""example_cases = generate_dataset(4, seed=41)
fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
for ax, case in zip(axes.ravel(), example_cases):
    for sensor, values in case.sensors.items():
        ax.plot(case.time, values, color=COLORS[sensor], lw=1, label=sensor)
    ax.axvline(case.true_peak_time, color="black", lw=2, label="true broad-event time")
    ax.set_title(f"shape={case.metadata['shape']}, sensors={len(case.sensors)}")
    ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Normalized time")
fig.tight_layout()
fig.savefig(FIGURES / "04_synthetic_examples.png", dpi=160, bbox_inches="tight")
plt.close(fig)"""))

cells.append(nbf.v4.new_markdown_cell(r"""![Representative synthetic signals with broad target events and narrow distractor peaks](../figures/04_synthetic_examples.png)"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 4. Monte Carlo comparison

Mean absolute error (MAE), median error, 90th-percentile error, and the rate of errors above 0.08 normalized-time units are reported. This benchmark evaluates the algorithmic principle, not final production parameters."""))

cells.append(nbf.v4.new_code_cell(r"""benchmark_cases = generate_dataset(100, seed=17)
benchmark_gmm = GMMPeakDetector(max_components=4, pseudo_samples=1200, random_state=7)
records = []

for case_index, case in enumerate(benchmark_cases):
    for method_name, detector in (("GMM", benchmark_gmm), ("Duration-aware", duration_detector)):
        estimates = {
            sensor: detector.estimate(case.time, values)
            for sensor, values in case.sensors.items()
        }
        fused = (
            fuse_duration_evidence(estimates)
            if method_name == "Duration-aware"
            else fuse_peak_estimates(estimates)
        )
        records.append({
            "case": case_index,
            "method": method_name,
            "available sensors": len(case.sensors),
            "shape": case.metadata["shape"],
            "absolute error": abs(fused.time - case.true_peak_time),
            "confidence": fused.confidence,
        })

benchmark = pd.DataFrame(records)
summary = (
    benchmark.groupby("method")["absolute error"]
    .agg(MAE="mean", median="median", p90=lambda x: x.quantile(.9), maximum="max")
)
summary["error > 0.08"] = benchmark.groupby("method")["absolute error"].apply(lambda x: (x > .08).mean())
summary.round(4)"""))

cells.append(nbf.v4.new_code_cell(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

groups = []
labels = []
for sensor_count in (1, 2):
    for method in ("GMM", "Duration-aware"):
        groups.append(benchmark.loc[(benchmark["available sensors"] == sensor_count) &
                                    (benchmark.method == method), "absolute error"])
        labels.append(f"{method}\n{sensor_count} sensor{'s' if sensor_count > 1 else ''}")
axes[0].boxplot(groups, tick_labels=labels, showfliers=False)
axes[0].set_ylabel("Absolute timing error")
axes[0].set_title("Missing-sensor robustness")

for method, color in (("GMM", "tab:purple"), ("Duration-aware", "tab:green")):
    errors = np.sort(benchmark.loc[benchmark.method == method, "absolute error"])
    cdf = np.arange(1, len(errors) + 1) / len(errors)
    axes[1].plot(errors, cdf, lw=2, label=method, color=color)
axes[1].set(xlabel="Absolute timing error", ylabel="Fraction of cases", title="Empirical error distribution")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "05_benchmark_results.png", dpi=160, bbox_inches="tight")
plt.close(fig)

benchmark.groupby(["method", "available sensors"])["absolute error"].agg(["mean", "median", "count"]).round(4)"""))

cells.append(nbf.v4.new_markdown_cell(r"""![Missing-sensor error distributions and empirical timing-error CDF](../figures/05_benchmark_results.png)"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 5. Conclusion

For this problem formulation, the **duration-aware detector is the recommended starting point**:

- its assumptions match the domain statement directly: the desired event is broad, while distractors are short and potentially taller;
- it works unchanged with one sensor;
- it combines comparable evidence curves when two sensors are present;
- it does not require a discrete and sometimes unstable choice of Gaussian component count.

In this deterministic 100-case proof-of-concept benchmark, it achieves an MAE
of **0.0174**, compared with **0.0355** for the GMM. Its 90th-percentile error is
**0.0375** rather than **0.0784**, and errors above 0.08 occur in **4%** rather
than **10%** of cases. The largest difference is in failure severity: the
maximum errors are 0.1327 and 0.3229, respectively. The exact values are
conditional on the current synthetic distribution, but the GMM's heavier tail
matches its observed component-fragmentation failure mode.

The GMM remains useful as a baseline and as a way to describe genuinely separate temporal modes. Its central weakness here is semantic: one physical plateau can require multiple Gaussian components, so “dominant component” does not necessarily mean “dominant event.”

### What is still needed before deployment

1. Define the target precisely: broad-event center, plateau midpoint, or another physically meaningful timestamp.
2. Label representative real examples, including difficult short spikes.
3. Estimate systematic blue/orange time delays and sensor reliabilities.
4. Choose the median and broad-scale windows in physical time units—not sample counts.
5. Add an abstention threshold for ambiguous or low-quality measurements.

The synthetic conclusion is conditional on the current generator. As real measurements become available, its distributions should be estimated from data and the benchmark rerun."""))

nb["cells"] = cells
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT}")
