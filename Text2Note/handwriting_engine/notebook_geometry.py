"""Detect writing rules from the currently uploaded blank notebook photo."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks


def detect_notebook_geometry(image: np.ndarray) -> dict:
    if image.ndim != 3:
        raise ValueError("Notebook image must be a color photo.")
    original_height, original_width = image.shape[:2]
    if min(original_height, original_width) < 350:
        raise ValueError("Notebook photo is too small to detect writing lines.")

    scale = min(1.0, 1600.0 / original_width)
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Local lighting normalization preserves faint rules under page shadows.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=9)
    darkness = background - gray
    row_strength = darkness[:, int(width * 0.07):int(width * 0.88)].mean(axis=1)
    peaks, _ = find_peaks(
        row_strength,
        distance=max(14, height // 80),
        prominence=2,
    )
    if len(peaks) == 0:
        raise ValueError("No ruled notebook lines were found.")
    threshold = max(2.0, float(np.quantile(row_strength[peaks], 0.75)) * 0.15)
    positions = [
        int(y) for y in peaks
        if height * 0.10 < y < height * 0.96 and row_strength[y] >= threshold
    ]
    if len(positions) < 2:
        raise ValueError("At least two clear notebook lines are needed.")

    if len(positions) >= 5:
        pitch = float(np.median(np.diff(positions)))
        positions = [
            y for index, y in enumerate(positions)
            if index == 0 or y - positions[index - 1] > pitch * 0.5
        ]

    lines = []
    xs = np.linspace(width * 0.065, width * 0.89, 7)
    for y in positions:
        points = []
        for x in xs:
            x0 = max(0, int(x - width * 0.025))
            x1 = min(width, int(x + width * 0.025))
            y0, y1 = max(0, y - 7), min(height, y + 8)
            local = darkness[y0:y1, x0:x1].mean(axis=1)
            local_y = y0 + int(np.argmax(local))
            points.append({
                "x": round(float(x / width), 6),
                "y": round(float(local_y / height), 6),
            })
        lines.append({"line_number": len(lines) + 1, "points": points})
    return {
        "image_width": original_width,
        "image_height": original_height,
        "lines": lines,
    }
