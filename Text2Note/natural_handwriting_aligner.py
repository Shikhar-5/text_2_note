from pathlib import Path
import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# TEST ONLY ON SAMPLE 02 FIRST
SAMPLE_NAME = "sample_02"

DATASET_DIR = PROJECT_ROOT / "handwriting_dataset"
SAMPLE_DIR = DATASET_DIR / SAMPLE_NAME

GRAY_PATH = SAMPLE_DIR / "01_grayscale.png"
CLEAN_PATH = SAMPLE_DIR / "03_no_lines.png"

OUTPUT_DIR = DATASET_DIR / "natural_library" / SAMPLE_NAME
DEBUG_PATH = OUTPUT_DIR / "row_alignment_debug.png"


# ============================================================
# EXPECTED PHYSICAL ROWS
# ============================================================

# This is the actual layout of your calibration sheet.
#
# We are NOT using this to locate handwriting yet.
# It will be used later when we align characters.

EXPECTED_PHYSICAL_ROWS = [
    "Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj",
    "Kk Ll Mm Nn Oo Pp Qq Rr Ss Tt",
    "Uu Vv Ww Xx Yy Zz",

    "0 1 2 3 4 5 6 7 8 9",

    "The quick brown fox jumps over the",
    "lazy dog",

    "Resistance",
    "resistance",
    "RESISTENCE",

    "Experimental",
    "apparatus",
    "galvanometer",
]


# ============================================================
# LOAD IMAGES
# ============================================================

gray = cv2.imread(
    str(GRAY_PATH),
    cv2.IMREAD_GRAYSCALE
)

clean = cv2.imread(
    str(CLEAN_PATH),
    cv2.IMREAD_GRAYSCALE
)

if gray is None:
    raise FileNotFoundError(
        f"Could not open grayscale image:\n{GRAY_PATH}"
    )

if clean is None:
    raise FileNotFoundError(
        f"Could not open cleaned image:\n{CLEAN_PATH}"
    )

print("Image size:", gray.shape)


# ============================================================
# BINARY HANDWRITING MASK
# ============================================================

# 03_no_lines.png:
#   black background
#   white handwriting
#
# Therefore white pixels = handwriting / remaining artifacts.

_, binary = cv2.threshold(
    clean,
    127,
    255,
    cv2.THRESH_BINARY
)


# ============================================================
# REMOVE LONG HORIZONTAL COMPONENTS
# ============================================================

# Notebook remnants are usually long and very thin.
# Detect them separately and subtract them.

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (35, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)

binary = cv2.subtract(
    binary,
    horizontal_lines
)


# ============================================================
# SMALL NOISE CLEANUP
# ============================================================

noise_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2, 2)
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    noise_kernel
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary,
    connectivity=8
)

components = []

for i in range(1, num_labels):

    x, y, w, h, area = stats[i]

    # Ignore tiny noise.
    if area < 12:
        continue

    # Ignore extremely thin horizontal leftovers.
    if h <= 5 and w >= h * 3:
        continue

    # Ignore tiny fragments.
    if w < 2 or h < 4:
        continue

    # Ignore huge page-edge artifacts.
    if w > gray.shape[1] * 0.35 and h < 25:
        continue

    center_x = x + w / 2
    center_y = y + h / 2

    components.append({
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": int(area),
        "cx": float(center_x),
        "cy": float(center_y),
    })


print("Useful connected components:", len(components))


# ============================================================
# ESTIMATE CHARACTER HEIGHT
# ============================================================

heights = [
    c["h"]
    for c in components
    if 7 <= c["h"] <= 80
]

if not heights:
    raise RuntimeError(
        "No usable handwriting components were detected."
    )

median_height = float(np.median(heights))

print("Median component height:", round(median_height, 2))


# ============================================================
# GROUP COMPONENTS INTO PHYSICAL ROWS
# ============================================================

# Sort by vertical center.
components.sort(key=lambda c: c["cy"])


rows = []

for component in components:

    cy = component["cy"]

    best_row = None
    best_distance = None

    for row in rows:

        # Current vertical center of the row.
        row_center = np.mean([
            c["cy"]
            for c in row
        ])

        distance = abs(cy - row_center)

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_row = row

    # Adaptive tolerance.
    #
    # This is deliberately conservative so nearby writing
    # rows don't merge together.

    tolerance = max(
        10,
        median_height * 0.85
    )

    if best_row is not None and best_distance <= tolerance:
        best_row.append(component)
    else:
        rows.append([component])


# ============================================================
# CLEAN ROWS
# ============================================================

# Remove rows containing only a couple of tiny artifacts.

clean_rows = []

for row in rows:

    meaningful = [
        c for c in row
        if c["area"] >= 15 and c["h"] >= 6
    ]

    if len(meaningful) >= 2:
        clean_rows.append(meaningful)


rows = clean_rows


# Sort rows vertically.
rows.sort(
    key=lambda row: np.mean([
        c["cy"]
        for c in row
    ])
)


# ============================================================
# PRINT DETECTED ROWS
# ============================================================

print()
print("========================================")
print("DETECTED PHYSICAL ROWS")
print("========================================")

for index, row in enumerate(rows):

    row.sort(key=lambda c: c["x"])

    x1 = min(c["x"] for c in row)
    x2 = max(c["x"] + c["w"] for c in row)

    y1 = min(c["y"] for c in row)
    y2 = max(c["y"] + c["h"] for c in row)

    print(
        f"ROW {index:02d}: "
        f"components={len(row):3d} "
        f"box=({x1},{y1})→({x2},{y2})"
    )


# ============================================================
# DEBUG IMAGE
# ============================================================

debug = cv2.cvtColor(
    gray,
    cv2.COLOR_GRAY2BGR
)


for index, row in enumerate(rows):

    row.sort(key=lambda c: c["x"])

    x1 = min(c["x"] for c in row)
    x2 = max(c["x"] + c["w"] for c in row)

    y1 = min(c["y"] for c in row)
    y2 = max(c["y"] + c["h"] for c in row)

    # Row bounding box.
    cv2.rectangle(
        debug,
        (x1 - 5, y1 - 5),
        (x2 + 5, y2 + 5),
        (0, 255, 0),
        2
    )

    # Row number.
    cv2.putText(
        debug,
        f"ROW {index}",
        (
            max(0, x1),
            max(20, y1 - 8)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE DEBUG IMAGE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(DEBUG_PATH),
    debug
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("========================================")
print("ROW DETECTION COMPLETE")
print("========================================")

print(
    "Expected physical rows:",
    len(EXPECTED_PHYSICAL_ROWS)
)

print(
    "Detected physical rows:",
    len(rows)
)

print()
print("Expected layout:")

for i, text in enumerate(EXPECTED_PHYSICAL_ROWS):
    print(f"  {i:02d}: {text}")

print()
print("Debug image:")
print(DEBUG_PATH)
print()