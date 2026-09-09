# RobustPeakFusion

Proof of concept for locating the main **broad physical event** in noisy sensor
traces when short, higher-amplitude peaks may occur before or after it. The same
pipeline supports one available sensor or confidence-weighted fusion of several
sensors.

Two approaches are implemented and compared:

1. **GMM baseline:** interpret positive signal amplitude as probability mass in
   time, select the number of Gaussian components by BIC, and use the mean of
   the component with the largest mixture weight.
2. **Duration-aware detector (main candidate):** median de-spiking, robust
   baseline normalization, broad-scale evidence extraction, area/width-based
   event selection, and high-evidence centroid timing.

The complete comparison—including the digitized measurements, missing-sensor
cases, synthetic examples, and a Monte Carlo benchmark—is in
[`notebooks/01_compare_peak_detectors.ipynb`](notebooks/01_compare_peak_detectors.ipynb).
Language-independent pseudocode for the recommended method is available in
[`docs/duration_aware_pseudocode.md`](docs/duration_aware_pseudocode.md).

## Repository layout

```text
data/
  AP_01.csv, AP_02.csv       # Approximate digitized traces
  raw_images/                # Optional local screenshots (gitignored)
figures/                     # Generated plots shown in GitHub's preview
notebooks/
  01_compare_peak_detectors.ipynb
scripts/
  digitize_plots.py          # Reproducible color-based digitization
  build_notebook.py          # Rebuilds the comparison notebook
src/robust_peak_fusion/
  detectors.py               # GMM and duration-aware estimators
  fusion.py                  # Missing-sensor-safe fusion
  preprocessing.py
  synthetic.py               # Dummy-data generator with known ground truth
tests/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[notebook,test]"
pytest
jupyter lab notebooks/01_compare_peak_detectors.ipynb
```

The notebook's complete narrative, code, and generated figures can be read
directly on GitHub. Figures are stored as ordinary PNG files and referenced by
the notebook; large base64 outputs are intentionally not embedded so GitHub's
preview remains reliable.

## Data limitation

The original figures are not published and do not display numerical axis
labels. Consequently, the digitized CSVs use normalized time and image-derived
normalized amplitude. They are appropriate for a shape/timing proof of concept,
but must not be interpreted as calibrated physical measurements.

## Current conclusion

The GMM is useful as an interpretable exploratory baseline, but it often splits
a broad or plateau-shaped event into several Gaussian components. Selecting one
component can then identify only one part of the event. The duration-aware
method encodes the actual requirement directly: narrow impulses are suppressed,
broad high-area regions are preferred, and evidence is fused only from sensors
that are present. In the deterministic 100-case synthetic benchmark, its timing
MAE is 0.0174 versus 0.0355 for the GMM, with fewer large-error cases (4% versus
10% above 0.08 normalized-time units). These figures are generator-dependent;
real-data accuracy still requires manually or physically validated peak-time
labels.
