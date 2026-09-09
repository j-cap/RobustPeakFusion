import numpy as np

from robust_peak_fusion import (
    DurationAwareDetector,
    GMMPeakDetector,
    fuse_duration_evidence,
    fuse_peak_estimates,
    generate_case,
)


def test_duration_detector_rejects_tall_narrow_spike():
    t = np.linspace(0, 1, 1000)
    broad = np.exp(-0.5 * ((t - 0.55) / 0.09) ** 2)
    tall_spike = 2.5 * np.exp(-0.5 * ((t - 0.18) / 0.004) ** 2)
    estimate = DurationAwareDetector().estimate(t, broad + tall_spike)
    assert abs(estimate.time - 0.55) < 0.04


def test_gmm_detector_returns_valid_component():
    case = generate_case(11, missing_probability=0.0)
    estimate = GMMPeakDetector(pseudo_samples=1000).estimate(
        case.time, case.sensors["blue"]
    )
    assert 0.0 <= estimate.time <= 1.0
    assert estimate.width > 0.0
    assert len(estimate.candidates) >= 1


def test_single_sensor_fusion_is_identity():
    case = generate_case(23, missing_probability=0.0)
    estimate = DurationAwareDetector().estimate(case.time, case.sensors["blue"])
    fused = fuse_duration_evidence({"blue": estimate})
    assert fused.contributing_sensors == ("blue",)
    assert abs(fused.time - estimate.time) < 0.03

    late = fuse_peak_estimates({"blue": estimate})
    assert late.time == estimate.time
    assert late.agreement


def test_generator_never_removes_every_sensor():
    for seed in range(30):
        assert generate_case(seed, missing_probability=0.9).sensors

