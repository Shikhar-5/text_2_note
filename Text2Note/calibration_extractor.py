from pathlib import Path
import json
import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_PATH = (
    PROJECT_ROOT
    / "handwriting_samples"
    / "sample_04.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "handwriting_dataset"
    / "personal_library"
)

UPPER_DIR = OUTPUT_DIR / "uppercase"
LOWER_DIR = OUTPUT_DIR / "lowercase"
DIGIT_DIR = OUTPUT_DIR / "digits"

DEBUG_DIR = OUTPUT_DIR / "debug"


for directory in [
    UPPER_DIR,
    LOWER_DIR,
    DIGIT_DIR,
    DEBUG_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# CALIBRATION CONTENT
# =========================================================

ALPHABET_ROWS = [
    list("AaBbCcDdEeFfGgHhIi"),
    list("JjKkLlMmNnOoPpQqRr"),
    list("SsTtUuVvWwXxYyZz")
]

DIGITS = list("0123456789")


# =========================================================
# LOAD IMAGE
# =========================================================

if not SAMPLE_PATH.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{SAMPLE_PATH}\n\n"
        "Make sure sample_04.jpg exists inside "
        "Text2Note\\handwriting_samples."
    )


image = cv2.imread(
    str(SAMPLE_PATH)
)

if image is None:

    raise RuntimeError(
        "OpenCV could not read sample_04.jpg."
    )


original = image.copy()

height, width = image.shape[:2]


print()
print("========================================")
print("TEXT2NOTE CALIBRATION EXTRACTOR V4.0")
print("========================================")
print()
print(
    "Image size:",
    width,
    "x",
    height
)
print()


# =========================================================
# CREATE CLEAN INK MASK
# =========================================================
#
# We do NOT use the old 03_no_lines image.
#
# sample_04 was deliberately written on a blank page,
# so we preserve the original image information.
#
# The method below estimates the paper background
# and isolates darker handwriting.
# =========================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Remove slow lighting/background variation.
background = cv2.GaussianBlur(
    gray,
    (0, 0),
    21
)

normalized = cv2.divide(
    gray,
    background,
    scale=255
)

# Slight smoothing.
normalized = cv2.GaussianBlur(
    normalized,
    (3, 3),
    0
)

# Dark ink becomes white.
_, ink = cv2.threshold(
    normalized,
    0,
    255,
    cv2.THRESH_BINARY_INV +
    cv2.THRESH_OTSU
)


# Remove tiny noise.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2, 2)
)

ink = cv2.morphologyEx(
    ink,
    cv2.MORPH_OPEN,
    kernel
)


cv2.imwrite(
    str(
        DEBUG_DIR /
        "01_ink_mask.png"
    ),
    ink
)


# =========================================================
# MANUAL CALIBRATION GRID
# =========================================================
#
# IMPORTANT:
#
# We are using the known structure of sample_04.
#
# These are NORMALIZED coordinates.
#
# That means they work even if the camera image
# resolution changes.
#
# The values describe the actual calibration sheet
# visible in your sample:
#
#   Row 1: Aa ... Ii
#   Row 2: Jj ... Rr
#   Row 3: Ss ... Zz
#   Row 4: 0 ... 9
#
# Each cell contains exactly one calibration unit.
# =========================================================


# These values are deliberately conservative.
# They cover the handwriting region rather than the
# entire photograph.

ALPHABET_GRID = [

    # row 1
    {
        "y1": 0.035,
        "y2": 0.265,
        "x1": 0.015,
        "x2": 0.990,
        "count": 9,
        "letters": list("AaBbCcDdEeFfGgHhIi")
    },

    # row 2
    {
        "y1": 0.125,
        "y2": 0.350,
        "x1": 0.015,
        "x2": 0.990,
        "count": 9,
        "letters": list("JjKkLlMmNnOoPpQqRr")
    },

    # row 3
    {
        "y1": 0.215,
        "y2": 0.445,
        "x1": 0.015,
        "x2": 0.990,
        "count": 8,
        "letters": list("SsTtUuVvWwXxYyZz")
    }
]


DIGIT_GRID = {
    "y1": 0.300,
    "y2": 0.520,
    "x1": 0.015,
    "x2": 0.990,
    "count": 10
}


# =========================================================
# CROP INK FROM CELL
# =========================================================

def crop_ink_from_cell(
    mask,
    x1,
    y1,
    x2,
    y2
):

    cell = mask[
        y1:y2,
        x1:x2
    ]

    if cell.size == 0:
        return None

    # Find pixels containing handwriting.
    ys, xs = np.where(
        cell > 0
    )

    if len(xs) == 0:
        return None

    # Bounding box of actual ink.
    left = max(
        0,
        int(xs.min()) - 6
    )

    right = min(
        cell.shape[1],
        int(xs.max()) + 7
    )

    top = max(
        0,
        int(ys.min()) - 6
    )

    bottom = min(
        cell.shape[0],
        int(ys.max()) + 7
    )

    cropped = cell[
        top:bottom,
        left:right
    ]

    return cropped


# =========================================================
# SAVE NORMALIZED GLYPH
# =========================================================

def save_glyph(
    glyph,
    destination
):

    if glyph is None:
        return False

    # Add black padding.
    padding = 12

    glyph = cv2.copyMakeBorder(
        glyph,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    cv2.imwrite(
        str(destination),
        glyph
    )

    return True


# =========================================================
# DEBUG IMAGE
# =========================================================

debug = original.copy()


# =========================================================
# EXTRACT ALPHABET CELLS
# =========================================================

metadata = {
    "source": str(SAMPLE_PATH),
    "alphabet": {},
    "digits": {}
}


print("Extracting alphabet...")
print()


for row_index, row in enumerate(
    ALPHABET_GRID,
    start=1
):

    y1 = int(
        height *
        row["y1"]
    )

    y2 = int(
        height *
        row["y2"]
    )

    x_start = int(
        width *
        row["x1"]
    )

    x_end = int(
        width *
        row["x2"]
    )

    count = row["count"]

    cell_width = (
        x_end -
        x_start
    ) / count

    letters = row["letters"]

    for index in range(
        count
    ):

        cell_x1 = int(
            x_start +
            index *
            cell_width
        )

        cell_x2 = int(
            x_start +
            (index + 1) *
            cell_width
        )

        # Slightly shrink each cell horizontally
        # so neighboring handwriting does not leak in.
        shrink_x = int(
            cell_width *
            0.08
        )

        crop_x1 = cell_x1 + shrink_x
        crop_x2 = cell_x2 - shrink_x

        glyph = crop_ink_from_cell(
            ink,
            crop_x1,
            y1,
            crop_x2,
            y2
        )

        pair = letters[
            index * 2:
            index * 2 + 2
        ]

        if len(pair) != 2:
            continue

        upper = pair[0]
        lower = pair[1]

        upper_path = (
            UPPER_DIR /
            f"{upper}.png"
        )

        lower_path = (
            LOWER_DIR /
            f"{lower}.png"
        )

        ok_upper = save_glyph(
            glyph,
            upper_path
        )

        ok_lower = save_glyph(
            glyph,
            lower_path
        )

        # -----------------------------------------
        # Debug rectangle
        # -----------------------------------------

        cv2.rectangle(
            debug,
            (crop_x1, y1),
            (crop_x2, y2),
            (0, 255, 0),
            2
        )

        if ok_upper:

            metadata[
                "alphabet"
            ][upper] = {
                "pair": pair,
                "row": row_index,
                "cell": index,
                "source": str(
                    upper_path
                )
            }

            metadata[
                "alphabet"
            ][lower] = {
                "pair": pair,
                "row": row_index,
                "cell": index,
                "source": str(
                    lower_path
                )
            }

            print(
                f"  {pair}"
            )


# =========================================================
# EXTRACT DIGITS
# =========================================================

print()
print("Extracting digits...")
print()


y1 = int(
    height *
    DIGIT_GRID["y1"]
)

y2 = int(
    height *
    DIGIT_GRID["y2"]
)

x_start = int(
    width *
    DIGIT_GRID["x1"]
)

x_end = int(
    width *
    DIGIT_GRID["x2"]
)

count = DIGIT_GRID["count"]

cell_width = (
    x_end -
    x_start
) / count


for index, digit in enumerate(
    DIGITS
):

    cell_x1 = int(
        x_start +
        index *
        cell_width
    )

    cell_x2 = int(
        x_start +
        (index + 1) *
        cell_width
    )

    shrink_x = int(
        cell_width *
        0.08
    )

    crop_x1 = cell_x1 + shrink_x
    crop_x2 = cell_x2 - shrink_x

    glyph = crop_ink_from_cell(
        ink,
        crop_x1,
        y1,
        crop_x2,
        y2
    )

    destination = (
        DIGIT_DIR /
        f"{digit}.png"
    )

    if save_glyph(
        glyph,
        destination
    ):

        metadata[
            "digits"
        ][digit] = {
            "cell": index,
            "source": str(
                destination
            )
        }

        print(
            f"  {digit}"
        )

    cv2.rectangle(
        debug,
        (crop_x1, y1),
        (crop_x2, y2),
        (255, 0, 0),
        2
    )


# =========================================================
# SAVE DEBUG IMAGE
# =========================================================

debug_path = (
    DEBUG_DIR /
    "02_calibration_cells.jpg"
)

cv2.imwrite(
    str(debug_path),
    debug
)


# =========================================================
# SAVE METADATA
# =========================================================

metadata_path = (
    OUTPUT_DIR /
    "calibration_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )


# =========================================================
# SUMMARY
# =========================================================

print()
print("========================================")
print("V4.0 COMPLETE")
print("========================================")
print()

print(
    "Uppercase:",
    len(
        [
            key
            for key in metadata["alphabet"]
            if key.isupper()
        ]
    )
)

print(
    "Lowercase:",
    len(
        [
            key
            for key in metadata["alphabet"]
            if key.islower()
        ]
    )
)

print(
    "Digits:",
    len(
        metadata["digits"]
    )
)

print()
print(
    "Debug image:"
)
print(
    debug_path
)

print()
print(
    "Library:"
)
print(
    OUTPUT_DIR
)

print()