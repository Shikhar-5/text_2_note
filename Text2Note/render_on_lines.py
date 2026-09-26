import json
import math
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

IMAGE_PATH = "Text2Note/notebook.jpg"
LINE_DATA_PATH = "Text2Note/line_data.json"
OUTPUT_PATH = "Text2Note/final_test.jpg"

FONT_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_SIZE = 32

TEXT = "Hello, this is Text2Note!"

# Small adjustment:
# Positive = move text slightly DOWN
# Negative = move text slightly UP
BASELINE_OFFSET = 20


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

print(
    f"Notebook loaded: "
    f"{image.width} x {image.height}"
)


# ---------------------------------------------------------
# LOAD LINE DATA
# ---------------------------------------------------------

with open(LINE_DATA_PATH, "r") as f:
    data = json.load(f)

lines = data

print(f"Loaded {len(lines)} notebook lines.")


if len(lines) == 0:
    print("No lines found.")
    exit()


# Use first manually calibrated line
line = lines[0]

p0 = (
    line["left"]["x"],
    line["left"]["y"]
)

p1 = (
    line["middle"]["x"],
    line["middle"]["y"]
)

p2 = (
    line["right"]["x"],
    line["right"]["y"]
)


# ---------------------------------------------------------
# QUADRATIC BEZIER
# ---------------------------------------------------------

def bezier_point(t):
    """Return point on the curved notebook line."""

    x = (
        (1 - t) ** 2 * p0[0]
        + 2 * (1 - t) * t * p1[0]
        + t ** 2 * p2[0]
    )

    y = (
        (1 - t) ** 2 * p0[1]
        + 2 * (1 - t) * t * p1[1]
        + t ** 2 * p2[1]
    )

    return x, y


def bezier_derivative(t):
    """Return direction of the curve."""

    dx = (
        2 * (1 - t) * (p1[0] - p0[0])
        + 2 * t * (p2[0] - p1[0])
    )

    dy = (
        2 * (1 - t) * (p1[1] - p0[1])
        + 2 * t * (p2[1] - p1[1])
    )

    return dx, dy


# ---------------------------------------------------------
# BUILD ARC-LENGTH TABLE
# ---------------------------------------------------------
# This is the important improvement.
#
# Instead of assuming:
#
#     t = distance / straight_line_width
#
# we calculate the actual length of the curved line.
# ---------------------------------------------------------

samples = 1000

curve_points = []

for i in range(samples + 1):

    t = i / samples

    x, y = bezier_point(t)

    curve_points.append(
        (x, y, t)
    )


# Calculate distance between samples

distances = [0.0]

total_length = 0.0

for i in range(1, len(curve_points)):

    x1, y1, _ = curve_points[i - 1]
    x2, y2, _ = curve_points[i]

    distance = math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )

    total_length += distance

    distances.append(total_length)


print(
    f"Actual curved line length: "
    f"{total_length:.1f}px"
)


# ---------------------------------------------------------
# FIND t FROM DISTANCE
# ---------------------------------------------------------

def distance_to_t(target_distance):

    if target_distance <= 0:
        return 0

    if target_distance >= total_length:
        return 1

    # Binary search
    low = 0
    high = len(distances) - 1

    while low <= high:

        mid = (low + high) // 2

        if distances[mid] < target_distance:
            low = mid + 1

        else:
            high = mid - 1

    index = max(1, low)

    d1 = distances[index - 1]
    d2 = distances[index]

    t1 = curve_points[index - 1][2]
    t2 = curve_points[index][2]

    if d2 == d1:
        return t1

    ratio = (
        target_distance - d1
    ) / (d2 - d1)

    return t1 + ratio * (t2 - t1)


# ---------------------------------------------------------
# LOAD FONT
# ---------------------------------------------------------

font = ImageFont.truetype(
    FONT_PATH,
    FONT_SIZE
)


# ---------------------------------------------------------
# MEASURE TEXT
# ---------------------------------------------------------

dummy = Image.new(
    "RGBA",
    (10, 10),
    (0, 0, 0, 0)
)

dummy_draw = ImageDraw.Draw(dummy)

text_bbox = dummy_draw.textbbox(
    (0, 0),
    TEXT,
    font=font
)

text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]

print(
    f"Text size: "
    f"{text_width} x {text_height}"
)

print(
    f"Available curved line length: "
    f"{total_length:.1f}px"
)


if text_width > total_length:

    print(
        "WARNING: Text is wider than the notebook line."
    )


# ---------------------------------------------------------
# CREATE TRANSPARENT LAYER
# ---------------------------------------------------------

result = image.convert("RGBA")

layer = Image.new(
    "RGBA",
    result.size,
    (0, 0, 0, 0)
)


# ---------------------------------------------------------
# DRAW CHARACTERS
# ---------------------------------------------------------

draw_layer = ImageDraw.Draw(layer)

current_distance = 0.0


for index, char in enumerate(TEXT):

    # Measure this character
    bbox = draw_layer.textbbox(
        (0, 0),
        char,
        font=font
    )

    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    # Put character center on the curve
    center_distance = (
        current_distance
        + char_width / 2
    )

    # Stop if character goes beyond line
    if center_distance > total_length:
        break

    # Convert actual distance -> Bezier t
    t = distance_to_t(center_distance)

    # Position on curve
    x, y = bezier_point(t)

    # Direction of curve
    dx, dy = bezier_derivative(t)

    angle = math.degrees(
        math.atan2(dy, dx)
    )


    # -----------------------------------------------------
    # CREATE CHARACTER IMAGE
    # -----------------------------------------------------

    padding = 20

    char_image = Image.new(
        "RGBA",
        (
            char_width + padding * 2,
            char_height + padding * 2
        ),
        (0, 0, 0, 0)
    )

    char_draw = ImageDraw.Draw(char_image)

    # Draw using baseline anchor
    char_draw.text(
        (
            padding,
            padding
        ),
        char,
        font=font,
        fill=(30, 30, 30, 255),
        anchor="la"
    )


    # -----------------------------------------------------
    # ROTATE CHARACTER
    # -----------------------------------------------------

    rotated = char_image.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=True
    )


    # -----------------------------------------------------
    # POSITION CHARACTER
    # -----------------------------------------------------

    paste_x = int(
        x - rotated.width / 2
    )

    paste_y = int(
        y - rotated.height / 2
        + BASELINE_OFFSET
    )

    layer.alpha_composite(
        rotated,
        (
            paste_x,
            paste_y
        )
    )


    # Move to next character
    current_distance += char_width


# ---------------------------------------------------------
# COMBINE WITH NOTEBOOK
# ---------------------------------------------------------

result = Image.alpha_composite(
    result,
    layer
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

result = result.convert("RGB")

result.save(
    OUTPUT_PATH,
    quality=95
)

print()
print(
    f"Saved: {OUTPUT_PATH}"
)