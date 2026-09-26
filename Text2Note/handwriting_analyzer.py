from pathlib import Path

import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLES_DIR = PROJECT_ROOT / "handwriting_samples"
OUTPUT_DIR = PROJECT_ROOT / "handwriting_dataset"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SUPPORTED IMAGE TYPES
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# =========================================================
# FIND HANDWRITING SAMPLES
# =========================================================

sample_files = sorted(
    [
        file
        for file in SAMPLES_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]
)


if not sample_files:
    raise FileNotFoundError(
        f"No handwriting samples found in: {SAMPLES_DIR}"
    )


print()
print("========================================")
print("TEXT2NOTE PERSONAL HANDWRITING ANALYZER")
print("========================================")
print()
print(f"Found {len(sample_files)} handwriting samples.")
print()


# =========================================================
# PROCESS EACH SAMPLE
# =========================================================

for index, sample_path in enumerate(
    sample_files,
    start=1
):

    print("----------------------------------------")
    print(f"Processing sample {index}")
    print(f"File: {sample_path.name}")
    print("----------------------------------------")

    # -----------------------------------------
    # Output directory
    # -----------------------------------------

    sample_output_dir = (
        OUTPUT_DIR /
        f"sample_{index:02d}"
    )

    sample_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # Load image
    # -----------------------------------------

    image = cv2.imread(
        str(sample_path)
    )

    if image is None:
        print(
            f"WARNING: Could not read {sample_path.name}"
        )
        continue

    print(
        "Image size:",
        image.shape
    )

    # -----------------------------------------
    # Grayscale
    # -----------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        str(
            sample_output_dir /
            "01_grayscale.png"
        ),
        gray
    )

    # =====================================================
    # STEP 1 — LIGHTING NORMALIZATION
    # =====================================================

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=255
    )

    # Slight smoothing before thresholding
    normalized = cv2.GaussianBlur(
        normalized,
        (3, 3),
        0
    )

    # =====================================================
    # STEP 2 — EXTRACT DARK INK
    # =====================================================

    _, binary = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    # =====================================================
    # STEP 3 — REMOVE SMALL NOISE
    # =====================================================

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    cv2.imwrite(
        str(
            sample_output_dir /
            "02_binary.png"
        ),
        binary
    )

    # =====================================================
    # STEP 4 — DETECT HORIZONTAL NOTEBOOK LINES
    # =====================================================

    height, width = binary.shape

    horizontal_kernel_width = max(
        30,
        width // 8
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            horizontal_kernel_width,
            1
        )
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # =====================================================
    # STEP 5 — THICKEN DETECTED RULING SLIGHTLY
    # =====================================================

    line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    horizontal_lines = cv2.dilate(
        horizontal_lines,
        line_kernel,
        iterations=1
    )

    # =====================================================
    # STEP 6 — REMOVE ONLY DETECTED LINES
    # =====================================================

    no_lines = cv2.subtract(
        binary,
        horizontal_lines
    )

    # =====================================================
    # STEP 7 — REMOVE LARGE BORDER/BACKGROUND BLOBS
    # =====================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        no_lines,
        connectivity=8
    )

    cleaned = np.zeros_like(
        no_lines
    )

    for label in range(
        1,
        num_labels
    ):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        # -----------------------------------------
        # Ignore extremely large page artifacts
        # -----------------------------------------

        if area > (
            width * height * 0.08
        ):
            continue

        # -----------------------------------------
        # Ignore tiny noise
        # -----------------------------------------

        if area < 8:
            continue

        # -----------------------------------------
        # Keep normal handwriting components
        # -----------------------------------------

        cleaned[
            y:y + h,
            x:x + w
        ][
            labels[
                y:y + h,
                x:x + w
            ] == label
        ] = 255

    # =====================================================
    # STEP 8 — SAVE CLEAN HANDWRITING
    # =====================================================

    cv2.imwrite(
        str(
            sample_output_dir /
            "03_no_lines.png"
        ),
        cleaned
    )

    print(
        "Saved:",
        sample_output_dir /
        "03_no_lines.png"
    )

    print()


# =========================================================
# FINISHED
# =========================================================

print("========================================")
print("ANALYSIS STEP 2 COMPLETE")
print("========================================")
print()
print("Processed:")
print()

for sample_path in sample_files:
    print(
        " ",
        sample_path.name
    )

print()
print("Output directory:")
print(
    OUTPUT_DIR
)
print()