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
    / "ai_precise_lines_v2_debug.jpg"
).resolve()

JSON_OUTPUT_PATH = (
    BACKEND_DIR
    / ".."
    / "Text2Note"
    / "outputs"
    / "ai_lines_v2.json"
).resolve()
# =========================================================
# SETTINGS
# =========================================================

NUM_POINTS = 160

# Search around AI prediction.
SEARCH_RADIUS = 13

# Horizontal neighborhood for detecting ruling strength.
HORIZONTAL_WINDOW = 20

# Ignore spiral-binding area.
PAGE_RIGHT_LIMIT = 0.885

# Maximum vertical movement between neighboring tracking points.
MAX_STEP = 5

# Curve smoothing.
SMOOTH_WINDOW = 7

# Minimum confidence required for a detected pixel.
MIN_SCORE = 18


# =========================================================
# GET AI PREDICTIONS
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
# REMOVE HEADER + LAST LINE
# =========================================================

if len(ai_lines) > 2:

    ai_lines = ai_lines[1:-1]

    print(
        "Ignoring first and last detected lines."
    )

print(
    "Using",
    len(ai_lines),
    "writing lines."
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
# IMAGE PREPROCESSING
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


# Slight blur removes tiny handwriting noise.
blurred = cv2.GaussianBlur(
    enhanced,
    (3, 3),
    0
)


# Adaptive threshold.
binary = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    7
)


# =========================================================
# HORIZONTAL RULING EXTRACTION
# =========================================================

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (31, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)


# Blur the resulting ruling map.
strength = cv2.GaussianBlur(
    horizontal_lines.astype(np.float32),
    (5, 5),
    0
)


# =========================================================
# HELPER FUNCTIONS
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


def interpolate_ai_line(ai_points, x_positions):

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
# FIRST PASS — TRACK EACH LINE
# =========================================================

all_detected_lines = []


for line_index, line in enumerate(ai_lines):

    ai_points = line.get(
        "points",
        []
    )

    if len(ai_points) < 2:
        continue


    # -----------------------------------------------------
    # X RANGE
    # -----------------------------------------------------

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


    if x_end <= x_start:
        continue


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


    # -----------------------------------------------------
    # TRACK ACROSS IMAGE
    # -----------------------------------------------------

    for point_index, (
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
        # LIMIT SEARCH USING PREVIOUS POINT
        # -------------------------------------------------

        if previous_y is not None:

            center_y = int(
                round(
                    0.65 * expected_y
                    + 0.35 * previous_y
                )
            )

        else:

            center_y = expected_y


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
        # FIND BEST PIXEL
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


            row = strength[
                candidate_y,
                x1:x2
            ]


            pixel_strength = float(
                np.mean(row)
            )


            # Distance from AI prediction.
            ai_distance = abs(
                candidate_y - expected_y
            )


            # Continuity from previous point.
            if previous_y is not None:

                movement = abs(
                    candidate_y - previous_y
                )

            else:

                movement = 0


            # Reject sudden jumps.
            if movement > MAX_STEP:

                continue


            # -------------------------------------------------
            # SCORE
            # -------------------------------------------------

            score = (
                pixel_strength
                - ai_distance * 1.2
                - movement * 2.0
            )


            if score > best_score:

                best_score = score
                best_y = candidate_y


        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if best_score < MIN_SCORE:

            best_y = center_y


        detected_y.append(
            best_y
        )

        previous_y = best_y


    # =====================================================
    # SMOOTH CURVE
    # =====================================================

    smooth_y = smooth_curve(
        detected_y,
        SMOOTH_WINDOW
    )


    all_detected_lines.append(
        {
            "line_number":
                line.get(
                    "line_number",
                    line_index + 1
                ),
            "x_positions":
                x_positions,
            "y_positions":
                smooth_y
        }
    )


# =========================================================
# SECOND PASS — NEIGHBOR LINE SPACING
# =========================================================

print(
    "Applying neighboring-line consistency..."
)


if len(all_detected_lines) > 2:

    # Lines are already ordered vertically.
    for i in range(
        1,
        len(all_detected_lines) - 1
    ):

        previous_line = (
            all_detected_lines[i - 1]
        )

        current_line = (
            all_detected_lines[i]
        )

        next_line = (
            all_detected_lines[i + 1]
        )


        current_y = current_line[
            "y_positions"
        ]

        previous_y = previous_line[
            "y_positions"
        ]

        next_y = next_line[
            "y_positions"
        ]


        # Expected position halfway between
        # neighboring lines.
        expected_spacing_y = (
            previous_y
            + next_y
        ) / 2.0


        # Only apply gentle correction.
        correction = (
            expected_spacing_y
            - current_y
        )


        # Limit correction so we preserve real curvature.
        correction = np.clip(
            correction,
            -3.0,
            3.0
        )


        current_line[
            "y_positions"
        ] = (
            current_y
            + correction * 0.25
        )


# =========================================================
# BUILD FINAL NORMALIZED GEOMETRY
# =========================================================

refined_lines = []


for line in all_detected_lines:

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


    refined_lines.append(
        {
            "line_number":
                line["line_number"],
            "points":
                points
        }
    )


# =========================================================
# DRAW DEBUG IMAGE
# =========================================================

debug_image = image.copy()


for line in refined_lines:

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


    # Show tracking points occasionally.
    for index, (x, y) in enumerate(
        pixel_points
    ):

        if index % 20 == 0:

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
# =========================================================
# SAVE V2 GEOMETRY AS JSON
# =========================================================

json_lines = []

for line in all_detected_lines:

    points = []

    for x, y in zip(
        line["x_positions"],
        line["y_positions"]
    ):

        points.append(
            {
                "x": float(x / width),
                "y": float(y / height)
            }
        )

    json_lines.append(
        {
            "line_number": line["line_number"],
            "points": points
        }
    )


geometry_data = {
    "image_width": width,
    "image_height": height,
    "lines": json_lines
}


with open(
    JSON_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        geometry_data,
        file,
        indent=2
    )


print()
print(
    "V2 geometry saved:"
)

print(
    JSON_OUTPUT_PATH
)


cv2.imwrite(
    str(OUTPUT_PATH),
    debug_image
)


print()
print(
    "V2 refinement complete."
)

print(
    "Points per line:",
    NUM_POINTS
)

print(
    "Output:"
)

print(
    OUTPUT_PATH
)