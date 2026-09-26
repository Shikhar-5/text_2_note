import requests
import json
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BACKEND_DIR
    / ".."
    / "Text2Note"
    / "notebook.jpg"
).resolve()

OUTPUT_PATH = (
    BACKEND_DIR
    / ".."
    / "Text2Note"
    / "outputs"
    / "ai_precise_lines_v3_debug.jpg"
).resolve()

JSON_OUTPUT_PATH = (
    BACKEND_DIR
    / ".."
    / "Text2Note"
    / "outputs"
    / "ai_precise_lines_v3.json"
).resolve()


# =========================================================
# FINAL SETTINGS
# =========================================================

# More points = better representation of curvature.
NUM_POINTS = 220

# Vertical search around AI prediction.
SEARCH_RADIUS = 12

# Horizontal neighborhood used to determine whether
# something looks like a notebook ruling.
HORIZONTAL_WINDOW = 24

# Stop before spiral binding.
PAGE_RIGHT_LIMIT = 0.875

# Maximum vertical movement between neighboring points.
MAX_STEP = 4

# Maximum change in movement between neighboring points.
# This prevents sharp unnatural bends.
MAX_ACCELERATION = 2

# Light smoothing only.
SMOOTH_WINDOW = 5


# =========================================================
# GET AI LINE PREDICTIONS
# =========================================================

print("Getting line predictions from OpenAI...")

with open(IMAGE_PATH, "rb") as image_file:

    response = requests.post(
        "http://127.0.0.1:8000/api/detect-lines/",
        files={
            "image": image_file
        }
    )


if not response.ok:

    print("AI request failed:")
    print(response.text)
    raise SystemExit


result = response.json()

if not result.get("success"):

    print("AI returned an error:")
    print(result)
    raise SystemExit


ai_data = result["lines"]

if isinstance(ai_data, str):
    ai_data = json.loads(ai_data)


ai_lines = ai_data.get("lines", [])

print(
    "OpenAI detected",
    len(ai_lines),
    "lines."
)


# =========================================================
# IGNORE FIRST + LAST LINE
# =========================================================

if len(ai_lines) > 2:

    usable_ai_lines = ai_lines[1:-1]

else:

    usable_ai_lines = ai_lines


print(
    "Using",
    len(usable_ai_lines),
    "writing lines."
)

print(
    "Ignoring first and last detected lines."
)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:

    print("Could not load notebook image.")
    raise SystemExit


height, width = image.shape[:2]

print(
    "Image size:",
    width,
    "x",
    height
)


# =========================================================
# PREPROCESS IMAGE
# =========================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Local contrast enhancement.
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)


# Small blur to suppress camera noise.
blurred = cv2.GaussianBlur(
    enhanced,
    (3, 3),
    0
)


# =========================================================
# HORIZONTAL EDGE RESPONSE
# =========================================================

# A notebook ruling is horizontal, therefore its strongest
# local edge response is generally in the Y direction.

sobel_y = cv2.Sobel(
    blurred,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)

edge_strength = np.abs(
    sobel_y
)


# Normalize.
edge_strength = cv2.normalize(
    edge_strength,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


# =========================================================
# HORIZONTAL MORPHOLOGY
# =========================================================

binary = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    7
)


horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (35, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)


horizontal_strength = cv2.GaussianBlur(
    horizontal_lines.astype(np.float32),
    (5, 5),
    0
)


# =========================================================
# COMBINED RULING SCORE
# =========================================================

# Combine:
#
# 1. horizontal morphology
# 2. horizontal-edge strength
#
# This is more reliable than either one alone.

ruling_strength = (
    horizontal_strength * 0.70
    + edge_strength * 0.30
)


# =========================================================
# CURVE SMOOTHING
# =========================================================

def smooth_curve(values, window):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) < 3:
        return values

    if window % 2 == 0:
        window += 1

    half = window // 2

    padded = np.pad(
        values,
        (half, half),
        mode="edge"
    )

    kernel = np.ones(
        window,
        dtype=np.float32
    ) / window

    return np.convolve(
        padded,
        kernel,
        mode="valid"
    )


# =========================================================
# INTERPOLATE AI CURVE
# =========================================================

def interpolate_ai_line(
    ai_points,
    x_positions
):

    ai_x = np.array(
        [
            point["x"] * width
            for point in ai_points
        ],
        dtype=np.float32
    )

    ai_y = np.array(
        [
            point["y"] * height
            for point in ai_points
        ],
        dtype=np.float32
    )

    order = np.argsort(ai_x)

    ai_x = ai_x[order]
    ai_y = ai_y[order]

    return np.interp(
        x_positions,
        ai_x,
        ai_y
    )


# =========================================================
# REFINE ONE LINE
# =========================================================

def refine_line(ai_points):

    ai_x = np.array(
        [
            point["x"] * width
            for point in ai_points
        ],
        dtype=np.float32
    )

    x_start = max(
        0,
        int(ai_x.min())
    )

    x_end = min(
        width - 1,
        int(PAGE_RIGHT_LIMIT * width)
    )


    x_positions = np.linspace(
        x_start,
        x_end,
        NUM_POINTS
    ).astype(np.int32)


    predicted_y = interpolate_ai_line(
        ai_points,
        x_positions
    )


    detected_y = []

    previous_y = None
    previous_step = 0


    # =====================================================
    # TRACK FROM LEFT → RIGHT
    # =====================================================

    for index, (
        x,
        expected_y
    ) in enumerate(
        zip(
            x_positions,
            predicted_y
        )
    ):

        expected_y = int(
            round(expected_y)
        )


        # -------------------------------------------------
        # SEARCH WINDOW
        # -------------------------------------------------

        if previous_y is None:

            center_y = expected_y

        else:

            # The actual line should remain close to both
            # the AI prediction and its previous position.
            center_y = int(
                round(
                    0.70 * expected_y
                    + 0.30 * previous_y
                )
            )


        search_start = max(
            0,
            center_y - SEARCH_RADIUS
        )

        search_end = min(
            height - 1,
            center_y + SEARCH_RADIUS
        )


        best_y = center_y
        best_score = -1e9


        # -------------------------------------------------
        # TEST EVERY POSSIBLE Y
        # -------------------------------------------------

        for candidate_y in range(
            search_start,
            search_end + 1
        ):

            x1 = max(
                0,
                x - HORIZONTAL_WINDOW
            )

            x2 = min(
                width,
                x + HORIZONTAL_WINDOW + 1
            )


            # Horizontal ruling evidence.
            morphology_score = float(
                np.mean(
                    horizontal_strength[
                        candidate_y,
                        x1:x2
                    ]
                )
            )


            # Horizontal edge evidence.
            edge_score = float(
                np.mean(
                    edge_strength[
                        candidate_y,
                        x1:x2
                    ]
                )
            )


            # Distance from AI prediction.
            ai_distance = abs(
                candidate_y - expected_y
            )


            # Movement from previous point.
            if previous_y is None:

                movement = 0

            else:

                movement = abs(
                    candidate_y - previous_y
                )


            # Reject large sudden movement.
            if movement > MAX_STEP:

                continue


            # Estimate direction change.
            if previous_y is not None:

                current_step = (
                    candidate_y - previous_y
                )

                acceleration = abs(
                    current_step
                    - previous_step
                )

            else:

                current_step = 0
                acceleration = 0


            # Reject sharp direction changes.
            if acceleration > MAX_ACCELERATION:

                continue


            # -------------------------------------------------
            # FINAL SCORE
            # -------------------------------------------------

            score = (

                # Strong physical ruling evidence.
                morphology_score * 1.00

                # Edge evidence.
                + edge_score * 0.35

                # Stay near AI prediction.
                - ai_distance * 1.20

                # Stay continuous.
                - movement * 1.50

                # Avoid sudden curvature changes.
                - acceleration * 2.00
            )


            if score > best_score:

                best_score = score
                best_y = candidate_y
                best_step = current_step


        detected_y.append(
            best_y
        )

        previous_y = best_y

        previous_step = (
            best_step
            if previous_y is not None
            else previous_step
        )


    # =====================================================
    # LIGHT SMOOTHING
    # =====================================================

    detected_y = smooth_curve(
        detected_y,
        SMOOTH_WINDOW
    )


    return (
        x_positions,
        detected_y
    )


# =========================================================
# PROCESS ALL LINES
# =========================================================

refined_lines = []


for index, line in enumerate(
    usable_ai_lines
):

    print(
        f"Refining line {index + 1}/"
        f"{len(usable_ai_lines)}..."
    )


    x_positions, y_positions = refine_line(
        line["points"]
    )


    refined_lines.append(
        {
            "line_number":
                line.get(
                    "line_number",
                    index + 2
                ),
            "x_positions":
                x_positions,
            "y_positions":
                y_positions
        }
    )


# =========================================================
# ENFORCE LINE ORDER
# =========================================================

print(
    "Checking neighboring line order..."
)


for i in range(
    1,
    len(refined_lines)
):

    previous = refined_lines[
        i - 1
    ]["y_positions"]

    current = refined_lines[
        i
    ]["y_positions"]


    # Prevent a lower line from crossing above
    # the previous line.

    minimum_gap = 5

    current = np.maximum(
        current,
        previous + minimum_gap
    )


    refined_lines[
        i
    ]["y_positions"] = current


# =========================================================
# BUILD JSON
# =========================================================

json_lines = []


for line in refined_lines:

    points = []


    for x, y in zip(
        line["x_positions"],
        line["y_positions"]
    ):

        points.append(
            {
                "x": float(
                    x / width
                ),
                "y": float(
                    y / height
                )
            }
        )


    json_lines.append(
        {
            "line_number":
                line["line_number"],
            "points":
                points
        }
    )


final_geometry = {
    "image_width": width,
    "image_height": height,
    "ignored_first_line": True,
    "ignored_last_line": True,
    "lines": json_lines
}


# =========================================================
# SAVE GEOMETRY
# =========================================================

with open(
    JSON_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_geometry,
        file,
        indent=2
    )


# =========================================================
# DRAW DEBUG IMAGE
# =========================================================

debug_image = image.copy()


for line in json_lines:

    pixel_points = []


    for point in line["points"]:

        x = int(
            point["x"] * width
        )

        y = int(
            point["y"] * height
        )

        pixel_points.append(
            (x, y)
        )


    if len(pixel_points) >= 2:

        cv2.polylines(
            debug_image,
            [
                np.array(
                    pixel_points,
                    dtype=np.int32
                )
            ],
            False,
            (0, 255, 0),
            2
        )


    # Draw sparse points.
    for point_index, (
        x,
        y
    ) in enumerate(
        pixel_points
    ):

        if point_index % 25 == 0:

            cv2.circle(
                debug_image,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


# =========================================================
# SAVE DEBUG IMAGE
# =========================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    debug_image
)


# =========================================================
# DONE
# =========================================================

print()
print(
    "======================================"
)

print(
    "V3 refinement complete!"
)

print(
    "Writing lines:",
    len(json_lines)
)

print(
    "Points per line:",
    NUM_POINTS
)

print(
    "Ignored first line: YES"
)

print(
    "Ignored last line: YES"
)

print()
print(
    "Debug image:"
)

print(
    OUTPUT_PATH
)

print()
print(
    "Geometry JSON:"
)

print(
    JSON_OUTPUT_PATH
)

print(
    "======================================"
)