from pathlib import Path
import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_NAME = "sample_02"

DATASET_DIR = PROJECT_ROOT / "handwriting_dataset"
SAMPLE_DIR = DATASET_DIR / SAMPLE_NAME

GRAY_PATH = SAMPLE_DIR / "01_grayscale.png"
CLEAN_PATH = SAMPLE_DIR / "03_no_lines.png"

OUTPUT_DIR = DATASET_DIR / "natural_library" / SAMPLE_NAME
WORD_DIR = OUTPUT_DIR / "words"
DEBUG_PATH = OUTPUT_DIR / "word_detection_debug.png"


# ============================================================
# LOAD
# ============================================================

gray = cv2.imread(str(GRAY_PATH), cv2.IMREAD_GRAYSCALE)
clean = cv2.imread(str(CLEAN_PATH), cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError(f"Could not open: {GRAY_PATH}")

if clean is None:
    raise FileNotFoundError(f"Could not open: {CLEAN_PATH}")


# ============================================================
# PREPARE OUTPUT
# ============================================================

WORD_DIR.mkdir(parents=True, exist_ok=True)

# Remove old extracted words so every run starts clean.
for old_file in WORD_DIR.glob("*.png"):
    old_file.unlink()


# ============================================================
# THRESHOLD
# ============================================================

# The cleaned image has:
#   black background
#   white handwriting
#
# Convert it into a binary image where handwriting = 255.

_, binary = cv2.threshold(
    clean,
    127,
    255,
    cv2.THRESH_BINARY
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
    if area < 8:
        continue

    # Ignore extremely thin horizontal remnants.
    if h <= 5 and w > h * 4:
        continue

    # Ignore extremely small components.
    if w < 2 or h < 3:
        continue

    components.append({
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": int(area),
    })


print(f"Detected components: {len(components)}")


# ============================================================
# GROUP COMPONENTS INTO TEXT LINES
# ============================================================

# Sort roughly from top to bottom.
components.sort(key=lambda c: c["y"])

lines = []

for comp in components:

    cy = comp["y"] + comp["h"] / 2

    best_line = None
    best_overlap = 0

    for line in lines:

        line_top = min(c["y"] for c in line)
        line_bottom = max(c["y"] + c["h"] for c in line)

        overlap = min(
            comp["y"] + comp["h"],
            line_bottom
        ) - max(
            comp["y"],
            line_top
        )

        if overlap > 0:
            ratio = overlap / min(
                comp["h"],
                line_bottom - line_top
            )

            if ratio > best_overlap:
                best_overlap = ratio
                best_line = line

    if best_line is not None and best_overlap >= 0.25:
        best_line.append(comp)
    else:
        lines.append([comp])


# Remove tiny accidental lines.
lines = [
    line for line in lines
    if len(line) >= 2
]


# Sort lines vertically.
lines.sort(
    key=lambda line: min(c["y"] for c in line)
)


print(f"Detected text lines: {len(lines)}")


# ============================================================
# GROUP COMPONENTS INTO WORDS
# ============================================================

all_words = []

for line_index, line in enumerate(lines):

    line.sort(key=lambda c: c["x"])

    # Estimate typical character width.
    widths = [
        c["w"]
        for c in line
        if c["w"] > 2
    ]

    if not widths:
        continue

    median_width = float(np.median(widths))

    # Gap larger than this is considered a word gap.
    word_gap = max(
        12,
        median_width * 1.5
    )

    current_word = []
    previous_right = None

    for comp in line:

        left = comp["x"]
        right = comp["x"] + comp["w"]

        if (
            previous_right is not None
            and left - previous_right > word_gap
        ):
            if current_word:
                all_words.append(
                    (line_index, current_word)
                )

            current_word = []

        current_word.append(comp)
        previous_right = right

    if current_word:
        all_words.append(
            (line_index, current_word)
        )


print(f"Detected word regions: {len(all_words)}")


# ============================================================
# SAVE WORD CROPS + DEBUG IMAGE
# ============================================================

debug = cv2.cvtColor(
    gray,
    cv2.COLOR_GRAY2BGR
)

saved_count = 0

for word_index, (line_index, word) in enumerate(all_words):

    if len(word) < 2:
        continue

    # Bounding box.
    x1 = min(c["x"] for c in word)
    y1 = min(c["y"] for c in word)

    x2 = max(
        c["x"] + c["w"]
        for c in word
    )

    y2 = max(
        c["y"] + c["h"]
        for c in word
    )

    width = x2 - x1
    height = y2 - y1

    # Reject obviously tiny fragments.
    if width < 10 or height < 5:
        continue

    # Add padding around handwriting.
    padding_x = 8
    padding_y = 8

    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)

    crop_x2 = min(
        gray.shape[1],
        x2 + padding_x
    )

    crop_y2 = min(
        gray.shape[0],
        y2 + padding_y
    )

    crop = gray[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    filename = (
        f"line_{line_index:02d}"
        f"_word_{saved_count:03d}.png"
    )

    output_path = WORD_DIR / filename

    cv2.imwrite(
        str(output_path),
        crop
    )

    # Draw debug rectangle.
    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        debug,
        str(saved_count),
        (x1, max(15, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    saved_count += 1


cv2.imwrite(
    str(DEBUG_PATH),
    debug
)


# ============================================================
# RESULT
# ============================================================

print()
print("========================================")
print("Natural word extraction complete")
print("========================================")
print(f"Words saved : {saved_count}")
print(f"Word folder : {WORD_DIR}")
print(f"Debug image : {DEBUG_PATH}")
print()