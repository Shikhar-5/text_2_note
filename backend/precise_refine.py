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
    / "ai_precise_lines_debug.jpg"
).resolve()


# =========================================================
# SETTINGS
# =========================================================

# Number of points tracked along every notebook line.
NUM_POINTS = 140

# How far vertically OpenCV can search around the AI prediction.
SEARCH_RADIUS = 16

# Horizontal region used to measure whether a pixel belongs
# to a long notebook ruling line.
HORIZONTAL_WINDOW = 22

# Ignore the extreme right side where the spiral binding is.
PAGE_RIGHT_LIMIT = 0.91

# Small smoothing amount.
SMOOTH_WINDOW = 9


# =========================================================
# GET AI PREDICTION
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
# PREPARE IMAGE
# =========================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Improve local contrast.
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)


# Detect faint dark structures.
binary = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    8
)


# =========================================================
# EXTRACT LONG HORIZONTAL STRUCTURES
# =========================================================

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (35, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)


# Give the line image a little blur so tiny broken sections
# don't cause the tracking to jump.
horizontal_strength = cv2.GaussianBlur(
    horizontal_lines.astype(np.float32),
    (5, 5),
    0
)


# =========================================================
# HELPER: SMOOTH A CURVE
# =========================================================

def smooth_curve(values, window):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if window < 3:
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

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    return smoothed


# =========================================================
# REFINE EVERY NOTEBOOK LINE
# =========================================================

refined_lines = []


for line_index, line in enumerate(ai_lines):

    ai_points = line.get(
        "points",
        []
    )

    if len(ai_points) < 2:
        continue


    # -----------------------------------------------------
    # AI POINTS → PIXEL COORDINATES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TRACK ONLY INSIDE PAGE
    # -----------------------------------------------------

    x_start = max(
        0,
        int(ai_x.min())
    )

    x_end = min(
        width - 1,
        int(PAGE_RIGHT_LIMIT * width)
    )


    if x_end <= x_start:
        continue


    # Dense x positions.
    x_positions = np.linspace(
        x_start,
        x_end,
        NUM_POINTS
    ).astype(np.int32)


    # Interpolate AI prediction over dense x positions.
    predicted_y = np.interp(
        x_positions,
        ai_x,
        ai_y
    )


    detected_y = []


    # -----------------------------------------------------
    # FOLLOW THE REAL PIXELS
    # -----------------------------------------------------

    previous_y = None


    for x, expected_y in zip(
        x_positions,
        predicted_y
    ):

        expected_y = int(
            round(expected_y)
        )


        search_start = max(
            0,
            expected_y - SEARCH_RADIUS
        )

        search_end = min(
            height - 1,
            expected_y + SEARCH_RADIUS
        )


        best_y = expected_y
        best_score = -1e9


        for candidate_y in range(
            search_start,
            search_end + 1
        ):

            # Horizontal neighborhood around this candidate.
            x1 = max(
                0,
                x - HORIZONTAL_WINDOW
            )

            x2 = min(
                width,
                x + HORIZONTAL_WINDOW + 1
            )


            row_strength = horizontal_strength[
                candidate_y,
                x1:x2
            ]


            # Average strength of horizontal ruling.
            line_score = float(
                np.mean(row_strength)
            )


            # Prefer candidates close to AI prediction.
            prediction_distance = abs(
                candidate_y - expected_y
            )


            score = (
                line_score
                - prediction_distance * 1.5
            )


            # Also encourage smooth movement from the
            # previous tracked point.
            if previous_y is not None:

                movement = abs(
                    candidate_y - previous_y
                )

                score -= movement * 1.0


            if score > best_score:

                best_score = score
                best_y = candidate_y


        detected_y.append(
            best_y
        )

        previous_y = best_y


    # -----------------------------------------------------
    # SMOOTH WITHOUT DESTROYING CURVATURE
    # -----------------------------------------------------

    smooth_y = smooth_curve(
        detected_y,
        SMOOTH_WINDOW
    )


    # -----------------------------------------------------
    # BUILD DENSE NORMALIZED POINTS
    # -----------------------------------------------------

    final_points = []

    for x, y in zip(
        x_positions,
        smooth_y
    ):

        final_points.append(
            {
                "x": float(x / width),
                "y": float(y / height)
            }
        )


    refined_lines.append(
        {
            "line_number":
                line.get(
                    "line_number",
                    line_index + 1
                ),
            "points":
                final_points
        }
    )


# =========================================================
# DRAW DEBUG IMAGE
# =========================================================

debug_image = image.copy()


for line in refined_lines:

    points = line["points"]

    pixel_points = []

    for point in points:

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


    # Draw only every 10th point so the image remains readable.
    for index, (x, y) in enumerate(
        pixel_points
    ):

        if index % 10 == 0:

            cv2.circle(
                debug_image,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


# =========================================================
# SAVE
# =========================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    debug_image
)


print()
print(
    "Precise refinement complete."
)

print(
    "Tracked points per line:",
    NUM_POINTS
)

print(
    "Output:"
)

print(
    OUTPUT_PATH
)