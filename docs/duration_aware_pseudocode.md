# Duration-aware peak detection and sensor fusion

This document gives language-independent pseudocode for locating the center of
the dominant **broad event** when short, higher-amplitude peaks may occur before
or after it. It also covers the cases in which either one sensor or several
sensors are available.

The method deliberately does not define the desired peak as the largest sample.
It favors events with substantial area and duration, suppresses narrow impulses,
and combines normalized evidence rather than raw sensor amplitudes.

## Inputs and outputs

Inputs:

- timestamps `t[1:N]`;
- measurements `y_j[1:N]` for every available sensor `j`;
- optional calibrated delay `delay_j` for each sensor;
- preprocessing and event-scale parameters.

Outputs:

- estimated event time `t_peak`;
- confidence score in `[0, 1]`;
- agreement flag when multiple sensors are available;
- individual sensor estimates and evidence curves for diagnostics.

The default parameters in the current proof of concept are expressed relative
to the record length because the digitized data have normalized time. In a real
application, window widths should be specified in physical time units.

## Algorithm 1: duration-aware estimate for one sensor

```text
FUNCTION DURATION_AWARE_ESTIMATE(t, y, parameters):
    # A. Validate and prepare the series
    keep samples having finite timestamps
    sort samples by increasing timestamp
    require unique timestamps and at least five finite measurements
    interpolate missing measurements on the valid time grid

    median_width <- odd integer near median_fraction * number_of_samples
    y_clean <- MEDIAN_FILTER(y, width = median_width)

    baseline <- QUANTILE(y_clean, baseline_quantile)
    upper    <- QUANTILE(y_clean, upper_quantile)
    scale    <- MAX(upper - baseline, machine_epsilon)

    # Positive amplitude above the robust baseline
    raw_evidence <- MAX((y_clean - baseline) / scale, 0)

    # B. Integrate evidence at the expected broad-event scale
    sigma_samples <- MAX(1, broad_scale_fraction * number_of_samples)
    broad_evidence <- GAUSSIAN_FILTER(raw_evidence, sigma_samples)
    broad_evidence <- broad_evidence / MAX(broad_evidence)

    # C. Generate broad-event candidates
    candidates <- FIND_PEAKS(
        broad_evidence,
        minimum_prominence = min_prominence,
        minimum_width      = min_width_fraction * number_of_samples,
        relative_height    = 0.65
    )

    IF candidates is empty:
        candidates <- {ARGMAX(broad_evidence)}

    # D. Score each candidate by duration and integrated evidence
    FOR each candidate k:
        [left_k, right_k] <- candidate bounds at relative height 0.65
        area_k <- INTEGRAL(raw_evidence between left_k and right_k)
        width_k <- time duration between candidate bounds
        prominence_k <- prominence of broad_evidence at candidate k

        capped_prominence <- MIN(prominence_k, 1)
        score_k <- area_k
                   * SQRT(MAX(width_k, median_sample_period))
                   * (0.5 + capped_prominence)
    END FOR

    selected <- candidate having maximum score

    # E. Estimate the center of the selected broad event
    local_evidence <- broad_evidence within selected candidate bounds
    level <- centroid_level * MAX(local_evidence)
    center_weights <- MAX(local_evidence - level, 0)

    IF SUM(center_weights) > 0:
        t_peak <- WEIGHTED_MEAN(local timestamps, center_weights)
    ELSE:
        t_peak <- timestamp at selected candidate maximum

    # F. Produce a diagnostic confidence value
    residual_noise <- MEDIAN(ABS(raw_evidence - broad_evidence))
    peak_snr <- selected prominence / MAX(residual_noise, 0.001)
    concentration <- 1 - 0.45 * NORMALIZED_ENTROPY(center_weights)
    confidence <- CLIP((1 - EXP(-peak_snr / 4)) * concentration, 0, 1)

    RETURN {
        time: t_peak,
        confidence: confidence,
        width: selected width,
        score: selected score,
        evidence: broad_evidence,
        candidates: candidates
    }
END FUNCTION
```

The score is dominated by area and duration. Prominence is capped so that a
very high but short impulse cannot win solely because of its amplitude.

## Algorithm 2: fusion with intermittently available sensors

```text
FUNCTION FUSED_DURATION_AWARE_ESTIMATE(sensor_series, delays):
    require at least one available sensor

    FOR each available sensor j:
        estimate_j <- DURATION_AWARE_ESTIMATE(t_j, y_j, parameters)
    END FOR

    # Missing-sensor case: no special fallback detector is required
    IF exactly one sensor j is available:
        RETURN {
            time: estimate_j.time - delay_j,
            confidence: estimate_j.confidence,
            agreement: TRUE,
            evidence: estimate_j.evidence,
            contributing_sensors: {j}
        }

    choose one sensor time grid as the common reference grid
    fused_evidence <- zeros on the reference grid
    total_weight <- 0

    FOR each available sensor j:
        # Shift into the common physical-event time frame
        shifted_evidence_j <- INTERPOLATE(
            estimate_j.evidence,
            from t_j onto reference_time + delay_j,
            using zero outside the recorded range
        )

        reliability_j <- MAX(estimate_j.confidence, 0.05)
        fused_evidence <- fused_evidence
                          + reliability_j * shifted_evidence_j
        total_weight <- total_weight + reliability_j
        corrected_time_j <- estimate_j.time - delay_j
    END FOR

    fused_evidence <- fused_evidence / total_weight

    level <- centroid_level * MAX(fused_evidence)
    center_weights <- MAX(fused_evidence - level, 0)

    IF SUM(center_weights) > 0:
        t_peak <- WEIGHTED_MEAN(reference_time, center_weights)
    ELSE:
        t_peak <- reference_time at ARGMAX(fused_evidence)

    spread <- MAX(corrected individual times) - MIN(corrected individual times)
    agreement <- spread <= agreement_tolerance

    base_confidence <- MEAN(individual sensor confidences)
    confidence <- base_confidence IF agreement ELSE 0.5 * base_confidence

    RETURN {
        time: t_peak,
        confidence: confidence,
        agreement: agreement,
        evidence: fused_evidence,
        contributing_sensors: all available sensors
    }
END FUNCTION
```

## Default proof-of-concept parameters

| Parameter | Default | Purpose |
|---|---:|---|
| `median_fraction` | 0.012 | Suppress impulses shorter than the desired event |
| `baseline_quantile` | 0.10 | Estimate the sensor baseline robustly |
| `upper_quantile` | 0.98 | Set a robust normalization scale |
| `broad_scale_fraction` | 0.035 | Integrate evidence at the broad-event scale |
| `min_width_fraction` | 0.035 | Reject very narrow peak candidates |
| `min_prominence` | 0.035 | Reject weak fluctuations |
| `centroid_level` | 0.80 | Locate the center of the high-evidence region |
| `agreement_tolerance` | 0.08 | Flag incompatible sensor estimates |

## Practical deployment notes

1. Calibrate `delay_j` from paired measurements if the sensor locations create
   a systematic propagation delay.
2. Express filter width and minimum event width in seconds once the physical
   sampling rate is known.
3. Validate the meaning of the target timestamp: broad-event center, plateau
   midpoint, or another application-specific landmark.
4. Treat low confidence or sensor disagreement as an abstention condition rather
   than silently averaging incompatible events.
5. Tune the scale parameters on manually or physically labelled recordings,
   including cases with short distractor peaks.

The corresponding implementation is in
[`src/robust_peak_fusion/detectors.py`](../src/robust_peak_fusion/detectors.py)
and [`src/robust_peak_fusion/fusion.py`](../src/robust_peak_fusion/fusion.py).

