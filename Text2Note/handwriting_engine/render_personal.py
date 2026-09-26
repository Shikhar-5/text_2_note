"""Place whole generated handwriting words on calibrated notebook curves."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from Text2Note.rendering.render_engine import (
    bezier_derivative,
    bezier_point,
    build_curve,
    distance_to_t,
)


def render_personal_notebook(
    image_path: str | Path,
    text: str,
    word_masks: list[np.ndarray],
    font_size: int,
    ink_color: tuple[int, int, int, int],
    line_data_path: str | Path,
    output_path: str | Path,
    line_geometry: dict | None = None,
    baseline_offset: float = -2,
    left_margin: float = 15,
) -> Path:
    image = Image.open(image_path).convert("RGBA")
    if line_geometry is not None:
        if not isinstance(line_geometry, dict):
            raise ValueError("Notebook line geometry is invalid.")
        lines = line_geometry.get("lines")
        if not isinstance(lines, list) or not 1 <= len(lines) <= 100:
            raise ValueError("Notebook line geometry is invalid.")
        curves = []
        for line in lines:
            points = line.get("points") if isinstance(line, dict) else None
            if not isinstance(points, list) or not 2 <= len(points) <= 50:
                raise ValueError("Notebook line points are invalid.")
            pixels = []
            for point in points:
                if not isinstance(point, dict):
                    raise ValueError("Notebook line point is invalid.")
                try:
                    x, y = float(point["x"]), float(point["y"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("Notebook line point is invalid.") from error
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("Notebook line coordinates must be normalized.")
                pixels.append((x * image.width, y * image.height))
            if any(b[0] <= a[0] for a, b in zip(pixels, pixels[1:])):
                raise ValueError("Notebook line points must run from left to right.")
            lengths = np.hypot(
                np.diff([p[0] for p in pixels]),
                np.diff([p[1] for p in pixels]),
            )
            curves.append({
                "polyline": pixels,
                "distances": np.r_[0, np.cumsum(lengths)],
                "length": float(np.sum(lengths)),
            })
    else:
        with open(line_data_path, encoding="utf-8") as stream:
            line_data = json.load(stream)
        curves = [build_curve(line) for line in line_data]
    if not curves:
        raise ValueError("No calibrated notebook lines found.")

    tokens = re.findall(r"\n|[^\s]+", text)
    if len([token for token in tokens if token != "\n"]) != len(word_masks):
        raise ValueError("The generated words do not match the requested text.")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    scale = max(0.35, font_size / 20.0)
    line_index = 0
    distance = left_margin
    word_index = 0

    for token in tokens:
        if token == "\n":
            line_index += 1
            distance = left_margin
            continue

        alpha = Image.fromarray(word_masks[word_index], mode="L")
        word_index += 1
        word_width = max(1, round(alpha.width * scale))
        word_height = max(1, round(alpha.height * scale))
        advance = word_width + max(6, font_size * 0.80)

        if line_index >= len(curves):
            raise ValueError("The text does not fit on the available notebook lines.")
        curve = curves[line_index]
        if distance + word_width > curve["length"] and distance > left_margin:
            line_index += 1
            distance = left_margin
            if line_index >= len(curves):
                raise ValueError("The text does not fit on the available notebook lines.")
            curve = curves[line_index]
        if distance + word_width > curve["length"]:
            raise ValueError(f"The word {token!r} is too wide for a notebook line.")

        center_distance = distance + word_width / 2
        if "polyline" in curve:
            segment = min(
                len(curve["polyline"]) - 2,
                max(0, int(np.searchsorted(curve["distances"], center_distance) - 1)),
            )
            a, b = curve["polyline"][segment:segment + 2]
            segment_length = curve["distances"][segment + 1] - curve["distances"][segment]
            fraction = (center_distance - curve["distances"][segment]) / max(segment_length, 1e-6)
            x = a[0] + (b[0] - a[0]) * fraction
            y = a[1] + (b[1] - a[1]) * fraction
            dx, dy = b[0] - a[0], b[1] - a[1]
        else:
            parameter = distance_to_t(curve, center_distance)
            x, y = bezier_point(curve["points"], parameter)
            dx, dy = bezier_derivative(curve["points"], parameter)
        tangent = max(1e-6, math.hypot(dx, dy))
        nx, ny = -dy / tangent, dx / tangent
        angle = math.degrees(math.atan2(dy, dx))

        alpha = alpha.resize((word_width, word_height), Image.Resampling.BICUBIC)
        colored = Image.new("RGBA", alpha.size, ink_color[:3] + (0,))
        colored.putalpha(alpha)
        colored = colored.rotate(-angle, Image.Resampling.BICUBIC, expand=True)

        # The source line image is 32 px high; its writing baseline is near y=26.
        center_above_baseline = (26 - 16) * scale
        center_x = x + nx * (baseline_offset - center_above_baseline)
        center_y = y + ny * (baseline_offset - center_above_baseline)
        paste_x = round(center_x - colored.width / 2)
        paste_y = round(center_y - colored.height / 2)
        overlay.alpha_composite(colored, (paste_x, paste_y))
        distance += advance

    result = Image.alpha_composite(image, overlay).convert("RGB")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)
    return output_path
