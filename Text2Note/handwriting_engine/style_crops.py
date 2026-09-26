"""Find usable handwritten word images in an arbitrary photographed page.

The crops are visual style references for a pretrained handwriting generator.
Their text does not need to be transcribed or aligned to characters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class WordCrop:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    line: int
    score: float


def _rotation_angle(gray: np.ndarray) -> float:
    height, width = gray.shape
    edges = cv2.Canny(gray, 60, 180)
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(45, width // 12),
        minLineLength=max(80, int(width * 0.28)),
        maxLineGap=max(12, width // 40),
    )
    if segments is None:
        return 0.0
    angles = []
    weights = []
    for x0, y0, x1, y1 in segments.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        if abs(angle) <= 9:
            angles.append(angle)
            weights.append(float(np.hypot(x1 - x0, y1 - y0)))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(sorted_angles[np.searchsorted(cumulative, cumulative[-1] / 2)])


def _deskew(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.12:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _ruling_positions(line_mask: np.ndarray) -> list[int]:
    height, width = line_mask.shape
    projection = (line_mask > 0).sum(axis=1)
    active = projection > width * 0.14
    change = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    centers = [int((a + b) / 2) for a, b in zip(starts, ends)]
    return [y for y in centers if height * 0.025 < y < height * 0.975]


def _hough_rulings(binary: np.ndarray) -> tuple[np.ndarray, list[int]]:
    height, width = binary.shape
    mask = np.zeros_like(binary)
    segments = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 720,
        threshold=max(45, width // 35),
        minLineLength=int(width * 0.18),
        maxLineGap=int(width * 0.04),
    )
    positions = []
    if segments is None:
        return mask, positions
    for x0, y0, x1, y1 in segments.reshape(-1, 4):
        if x0 == x1:
            continue
        angle = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        if angle > 5:
            continue
        if min(y0, y1) < height * 0.07:
            continue
        cv2.line(mask, (x0, y0), (x1, y1), 255, 3)
        positions.append(int((y0 + y1) / 2))
    positions.sort()
    merged = []
    for y in positions:
        if merged and y - merged[-1][-1] < max(6, height * 0.006):
            merged[-1].append(y)
        else:
            merged.append([y])
    return mask, [int(np.median(group)) for group in merged]


def _components(ink: np.ndarray, pitch: float) -> list[tuple[int, int, int, int, int]]:
    height, width = ink.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    result = []
    for x, y, w, h, area in stats[1:count]:
        if area < max(6, width * height * 0.000004):
            continue
        if w > width * 0.45 or h > max(pitch * 1.8, height * 0.12):
            continue
        if w < 2 or h < 2:
            continue
        if x < width * 0.012 or x + w > width * 0.988:
            continue
        result.append((int(x), int(y), int(w), int(h), int(area)))
    return result


def _line_groups(
    components: list[tuple[int, int, int, int, int]],
    rulings: list[int],
    pitch: float,
) -> list[list[tuple[int, int, int, int, int]]]:
    # Group by upper stroke position. Descenders often cross the next ruling,
    # whereas the tops of letters on one writing line stay close together.
    groups = []
    for component in sorted(components, key=lambda c: c[1] + c[3] * 0.10):
        center = component[1] + component[3] * 0.10
        best = None
        best_distance = pitch * 0.47
        for group in groups:
            group_center = np.median([c[1] + c[3] * 0.10 for c in group])
            distance = abs(center - group_center)
            if distance < best_distance:
                best = group
                best_distance = distance
        if best is None:
            groups.append([component])
        else:
            best.append(component)
    return groups


def extract_word_crops(
    image_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[WordCrop]:
    """Extract word candidates and optionally save them with a debug overlay."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read handwriting image: {image_path}")
    height, width = image.shape[:2]
    if min(height, width) < 350:
        raise ValueError("Handwriting photo is too small; use a clearer page image.")

    angle = _rotation_angle(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    image = _deskew(image, angle)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(12, width / 50))
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    _, binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel_width = max(65, width // 16)
    line_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    hough_mask, hough_positions = _hough_rulings(binary)
    line_mask = cv2.bitwise_or(line_mask, hough_mask)
    positions = sorted(_ruling_positions(line_mask) + hough_positions)
    merged_positions = []
    for y in positions:
        if merged_positions and y - merged_positions[-1][-1] < max(6, height * 0.006):
            merged_positions[-1].append(y)
        else:
            merged_positions.append([y])
    rulings = [int(np.median(group)) for group in merged_positions]
    if len(rulings) >= 3:
        gaps = np.diff(rulings)
        pitch = float(np.median(gaps[gaps > max(12, height * 0.018)]))
    else:
        pitch = max(28.0, height / 22)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bright = gray >= np.quantile(gray, 0.70)
    paper_chroma = np.median(lab[bright, 1:], axis=0)
    chroma_distance = np.linalg.norm(lab[:, :, 1:] - paper_chroma, axis=2)
    bright_distance = chroma_distance[bright]
    bright_median = float(np.median(bright_distance))
    bright_mad = float(np.median(abs(bright_distance - bright_median)))
    chroma_threshold = max(8.0, bright_median + 4 * bright_mad)
    colored_ink = ((chroma_distance > chroma_threshold) & (gray < 235)).astype(np.uint8) * 255
    color_mode = bool(np.count_nonzero(colored_ink) > width * height * 0.003)

    line_mask = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    )
    if color_mode:
        ink = colored_ink
    else:
        ink = cv2.bitwise_and(binary, cv2.bitwise_not(line_mask))
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    )
    components = _components(ink, pitch)
    groups = _line_groups(components, rulings, pitch)

    crops = []
    debug = image.copy()
    for line_index, group in enumerate(groups):
        if not group:
            continue
        group.sort(key=lambda c: c[0])
        word_gap = max(12.0, pitch * 0.43)
        clusters = []
        cluster = []
        right = None
        for component in group:
            x, y, w, h, area = component
            if cluster and right is not None and x - right > word_gap:
                clusters.append(cluster)
                cluster = []
            cluster.append(component)
            right = max(right or 0, x + w)
        if cluster:
            clusters.append(cluster)

        for cluster in clusters:
            x0 = min(c[0] for c in cluster)
            y0 = min(c[1] for c in cluster)
            x1 = max(c[0] + c[2] for c in cluster)
            y1 = max(c[1] + c[3] for c in cluster)
            w, h = x1 - x0, y1 - y0
            if w < pitch * 0.65 or h < pitch * 0.25:
                continue
            if w / max(h, 1) < 0.65 or w / max(h, 1) > 11:
                continue
            if h > pitch * 1.65:
                continue
            padx = max(3, int(pitch * 0.07))
            pady = max(3, int(pitch * 0.12))
            x0, x1 = max(0, x0 - padx), min(width, x1 + padx)
            y0, y1 = max(0, y0 - pady), min(height, y1 + pady)
            patch_mask = ink[y0:y1, x0:x1]
            patch_gray = normalized[y0:y1, x0:x1]
            clean = np.where(patch_mask > 0, patch_gray, 255).astype(np.uint8)
            density = float(np.count_nonzero(patch_mask) / patch_mask.size)
            row_concentration = float(
                np.count_nonzero(patch_mask, axis=1).max()
                / max(1, np.count_nonzero(patch_mask))
            )
            if density < 0.07 or density > 0.6 or row_concentration > 0.15:
                continue
            score = float(
                min(1.0, w / (pitch * 2))
                * min(1.0, len(cluster) / 3)
                * min(1.0, density / 0.14)
            )
            # Printed date/header text often occupies the top strip.
            # Keep it as a fallback when that is all the sample contains.
            if y0 < height * 0.08:
                score *= 0.30
            crops.append(WordCrop(clean, (x0, y0, x1, y1), line_index, score))
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 190, 0), 2)

    crops.sort(key=lambda item: (-item.score, item.line, item.bbox[0]))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, item in enumerate(crops):
            name = f"word_{index:03d}.png"
            cv2.imwrite(str(output_dir / name), item.image)
            records.append(
                {"file": name, "bbox": item.bbox, "line": item.line, "score": item.score}
            )
        cv2.imwrite(str(output_dir / "debug_words.png"), debug)
        cv2.imwrite(str(output_dir / "debug_lines.png"), line_mask)
        cv2.imwrite(str(output_dir / "debug_ink.png"), ink)
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "angle": angle,
                    "pitch": pitch,
                    "color_mode": color_mode,
                    "rulings": rulings,
                    "component_count": len(components),
                    "group_sizes": [len(group) for group in groups],
                    "words": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return crops


def load_word_crops(output_dir: str | Path) -> list[WordCrop]:
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    crops = []
    for record in metadata["words"]:
        image = cv2.imread(str(output_dir / record["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Missing handwriting crop: {record['file']}")
        crops.append(
            WordCrop(
                image=image,
                bbox=tuple(record["bbox"]),
                line=int(record["line"]),
                score=float(record["score"]),
            )
        )
    return crops
