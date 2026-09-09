# Data

`AP_01.csv` and `AP_02.csv` are approximate traces digitized from supplied
screenshots. The original screenshots are intentionally not published. Each
file contains:

- `time_normalized`: horizontal pixel position mapped to `[0, 1]`;
- `blue`, `orange`: inverted vertical pixel position, also mapped approximately
  to `[0, 1]`.

The screenshots do not show numerical axis labels, so the CSVs preserve curve
shape and relative timing—not physical units. Anti-aliasing, line thickness,
vertical steps, and raster resolution introduce small extraction errors.

To regenerate the CSVs, place local copies of the screenshots under
`data/raw_images/` and run from the repository root:

```bash
python scripts/digitize_plots.py data/raw_images/AP_01.png data/raw_images/AP_02.png
```
