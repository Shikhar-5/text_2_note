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
    / "ai_opencv_lines_debug.jpg"
).resolve()


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
    "Image:",
    width,
    "x",
    height
)


# =========================================================
# PREPARE IMAGE FOR LINE DETECTION
# =========================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Increase local contrast.
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)


# Detect dark/faint structures.
binary = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    8
)


# =========================================================
# HORIZONTAL LINE EMPHASIS
# =========================================================

# Notebook ruling is long and horizontally continuous.
# This removes most small handwriting strokes.

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (35, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)


# =========================================================
# REFINE EACH AI POINT
# =========================================================

refined_lines = []


# Search radius around the AI prediction.
SEARCH_RADIUS = 14


# Width used to measure horizontal continuity.
WINDOW_X = 18


for line in ai_lines:

    original_points = line.get(
        "points",
        []
    )

    refined_points = []


    for point in original_points:

        predicted_x = int(
            point["x"] * width
        )

        predicted_y = int(
            point["y"] * height
        )


        # Keep coordinates inside image.
        predicted_x = max(
            0,
            min(width - 1, predicted_x)
        )

        predicted_y = max(
            0,
            min(height - 1, predicted_y)
        )


        # -------------------------------------------------
        # SEARCH NEAR AI PREDICTION
        # -------------------------------------------------

        best_y = predicted_y
        best_score = -1


        y_start = max(
            0,
            predicted_y - SEARCH_RADIUS
        )

        y_end = min(
            height - 1,
            predicted_y + SEARCH_RADIUS
        )


        x_start = max(
            0,
            predicted_x - WINDOW_X
        )

        x_end = min(
            width - 1,
            predicted_x + WINDOW_X
        )


        for candidate_y in range(
            y_start,
            y_end + 1
        ):

            row = horizontal_lines[
                candidate_y,
                x_start:x_end + 1
            ]


            # Number of detected horizontal pixels.
            score = int(
                np.count_nonzero(row)
            )


            if score > best_score:

                best_score = score
                best_y = candidate_y


        refined_points.append(
            {
                "x": predicted_x / width,
                "y": best_y / height
            }
        )


    refined_lines.append(
        {
            "line_number":
                line.get(
                    "line_number"
                ),
            "points":
                refined_points
        }
    )


# =========================================================
# DRAW RESULTS
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

        # Green = OpenCV refined line
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


    for x, y in pixel_points:

        cv2.circle(
            debug_image,
            (x, y),
            4,
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
    "OpenCV refined debug image:"
)

print(
    OUTPUT_PATH
)