"""Digitize the blue and orange traces from the supplied plot screenshots.

The source plots contain no readable axis values, so output coordinates are
normalized to [0, 1]. This is intended for method prototyping, not measurement
recovery. Trace colors are isolated in HSV space and the median colored pixel
is taken in each image column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


HSV_RANGES = {
    # Hue uses Pillow's [0, 255] convention.
    "blue": ((135, 165), 40),
    "orange": ((5, 30), 35),
}


def extract_trace(hsv: np.ndarray, hue: tuple[int, int], min_saturation: int) -> np.ndarray:
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    mask = (h >= hue[0]) & (h <= hue[1]) & (s >= min_saturation)
    height, width = mask.shape
    trace = np.full(width, np.nan)
    for x in range(width):
        rows = np.flatnonzero(mask[:, x])
        if rows.size:
            # Image y points downwards; invert and normalize amplitude.
            trace[x] = 1.0 - float(np.median(rows)) / (height - 1)
    return trace


def interpolate_internal_gaps(trace: np.ndarray, max_gap: int = 8) -> np.ndarray:
    result = trace.copy()
    finite = np.isfinite(result)
    indices = np.flatnonzero(finite)
    if indices.size < 2:
        return result
    for left, right in zip(indices[:-1], indices[1:]):
        if 1 < right - left <= max_gap + 1:
            result[left : right + 1] = np.linspace(result[left], result[right], right - left + 1)
    return result


def digitize(image_path: Path) -> pd.DataFrame:
    image = Image.open(image_path).convert("RGB")
    hsv = np.asarray(image.convert("HSV"))
    traces = {
        name: interpolate_internal_gaps(extract_trace(hsv, hue, saturation))
        for name, (hue, saturation) in HSV_RANGES.items()
    }
    present = np.logical_or.reduce([np.isfinite(trace) for trace in traces.values()])
    columns = np.flatnonzero(present)
    first, last = int(columns[0]), int(columns[-1])
    time = np.linspace(0.0, 1.0, last - first + 1)
    return pd.DataFrame(
        {
            "time_normalized": time,
            **{name: values[first : last + 1] for name, values in traces.items()},
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        output = args.output_dir / f"{image_path.stem}.csv"
        frame = digitize(image_path)
        frame.to_csv(output, index=False, float_format="%.7f")
        coverage = frame[["blue", "orange"]].notna().mean().to_dict()
        print(f"{image_path.name} -> {output} ({len(frame)} rows, coverage={coverage})")


if __name__ == "__main__":
    main()

