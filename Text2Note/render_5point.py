import json
import math
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = "Text2Note/notebook.jpg"
LINE_DATA_PATH = "Text2Note/line_data_5point.json"
OUTPUT_PATH = "Text2Note/final_5point.jpg"

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

FONT_SIZE = 22

BASELINE_OFFSET = -2

LEFT_MARGIN = 15

TEXT_COLOR = (30, 30, 30, 255)

TEXT = (
    "Introduction to Biology. Biology is the scientific study of life "
    "and living organisms. Living organisms are made up of cells, which "
    "are considered the basic structural and functional units of life. "
    "Cells can be classified into two major types: prokaryotic cells "
    "and eukaryotic cells. Prokaryotic cells are generally smaller and "
    "simpler, while eukaryotic cells contain a nucleus and other "
    "membrane-bound organelles. Every cell has a cell membrane that "
    "controls the movement of substances into and out of the cell. "
    "The cytoplasm contains many structures that perform important "
    "functions. The nucleus stores genetic material called DNA and "
    "controls many activities of the cell. Mitochondria are responsible "
    "for producing energy through cellular respiration. Ribosomes help "
    "in the production of proteins. Plants use photosynthesis to produce "
    "food using sunlight, carbon dioxide, and water. Cellular respiration "
    "releases energy from glucose and produces ATP for cellular activities."
)


# =========================================================
# LOAD IMAGE
# =========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGBA")

print(
    f"Notebook loaded: "
    f"{image.width} x {image.height}"
)


# =========================================================
# LOAD 5-POINT DATA
# =========================================================

with open(
    LINE_DATA_PATH,
    "r"
) as f:

    lines = json.load(f)

print(
    f"Loaded {len(lines)} five-point notebook lines."
)

if not lines:
    print("No lines found.")
    exit()


# =========================================================
# FONT
# =========================================================

font = ImageFont.truetype(
    FONT_PATH,
    FONT_SIZE
)


# =========================================================
# MEASUREMENT
# =========================================================

measure_image = Image.new(
    "RGBA",
    (100, 100),
    (0, 0, 0, 0)
)

measure_draw = ImageDraw.Draw(
    measure_image
)


def char_width(char):

    return measure_draw.textlength(
        char,
        font=font
    )


def word_width(word):

    total = 0

    for char in word:
        total += char_width(char)

    return total


SPACE_WIDTH = char_width(" ")


# =========================================================
# 4th DEGREE BEZIER
# =========================================================

def bezier_point(points, t):

    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    p3 = points[3]
    p4 = points[4]

    x = (
        ((1 - t) ** 4) * p0[0]
        + 4 * ((1 - t) ** 3) * t * p1[0]
        + 6 * ((1 - t) ** 2) * (t ** 2) * p2[0]
        + 4 * (1 - t) * (t ** 3) * p3[0]
        + (t ** 4) * p4[0]
    )

    y = (
        ((1 - t) ** 4) * p0[1]
        + 4 * ((1 - t) ** 3) * t * p1[1]
        + 6 * ((1 - t) ** 2) * (t ** 2) * p2[1]
        + 4 * (1 - t) * (t ** 3) * p3[1]
        + (t ** 4) * p4[1]
    )

    return x, y


# =========================================================
# BEZIER DERIVATIVE
# =========================================================

def bezier_derivative(points, t):

    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    p3 = points[3]
    p4 = points[4]

    dx = (
        4 * ((1 - t) ** 3) * (p1[0] - p0[0])
        + 12 * ((1 - t) ** 2) * t * (p2[0] - p1[0])
        + 12 * (1 - t) * (t ** 2) * (p3[0] - p2[0])
        + 4 * (t ** 3) * (p4[0] - p3[0])
    )

    dy = (
        4 * ((1 - t) ** 3) * (p1[1] - p0[1])
        + 12 * ((1 - t) ** 2) * t * (p2[1] - p1[1])
        + 12 * (1 - t) * (t ** 2) * (p3[1] - p2[1])
        + 4 * (t ** 3) * (p4[1] - p3[1])
    )

    return dx, dy


# =========================================================
# BUILD CURVE
# =========================================================

def build_curve(line):

    points = []

    for key in [
        "p0",
        "p1",
        "p2",
        "p3",
        "p4"
    ]:

        points.append(
            (
                line[key]["x"],
                line[key]["y"]
            )
        )

    samples = 2000

    curve_points = []

    distances = [0.0]

    total_length = 0.0

    previous = None

    for i in range(samples + 1):

        t = i / samples

        x, y = bezier_point(
            points,
            t
        )

        curve_points.append(
            (x, y, t)
        )

        if previous is not None:

            px, py = previous

            segment = math.sqrt(
                (x - px) ** 2
                + (y - py) ** 2
            )

            total_length += segment

            distances.append(
                total_length
            )

        previous = (
            x,
            y
        )

    return {
        "points": points,
        "curve_points": curve_points,
        "distances": distances,
        "length": total_length
    }


# =========================================================
# DISTANCE -> T
# =========================================================

def distance_to_t(
    curve,
    target
):

    distances = curve["distances"]

    curve_points = curve["curve_points"]

    if target <= 0:
        return 0

    if target >= curve["length"]:
        return 1

    low = 0
    high = len(distances) - 1

    while low <= high:

        mid = (
            low + high
        ) // 2

        if distances[mid] < target:

            low = mid + 1

        else:

            high = mid - 1

    index = max(
        1,
        low
    )

    d1 = distances[index - 1]
    d2 = distances[index]

    t1 = curve_points[index - 1][2]
    t2 = curve_points[index][2]

    if d2 == d1:
        return t1

    ratio = (
        target - d1
    ) / (
        d2 - d1
    )

    return (
        t1
        + ratio * (t2 - t1)
    )


# =========================================================
# CREATE CURVES
# =========================================================

curves = []

for i, line in enumerate(lines):

    curve = build_curve(line)

    curves.append(curve)

    print(
        f"Line {i + 1}: "
        f"{curve['length']:.1f}px"
    )


# =========================================================
# DRAW CHARACTER
# =========================================================

def draw_character(
    layer,
    curve,
    distance,
    char
):

    advance = char_width(char)

    if advance <= 0:
        return True

    center_distance = (
        distance
        + advance / 2
    )

    if center_distance >= curve["length"]:
        return False

    # -----------------------------------------------------
    # Position on curve
    # -----------------------------------------------------

    t = distance_to_t(
        curve,
        center_distance
    )

    x, y = bezier_point(
        curve["points"],
        t
    )

    # -----------------------------------------------------
    # Tangent
    # -----------------------------------------------------

    dx, dy = bezier_derivative(
        curve["points"],
        t
    )

    tangent_length = math.sqrt(
        dx * dx
        + dy * dy
    )

    if tangent_length == 0:
        return True

    # -----------------------------------------------------
    # Normal
    # -----------------------------------------------------

    nx = -dy / tangent_length
    ny = dx / tangent_length

    # Move perpendicular to curve
    x += (
        nx
        * BASELINE_OFFSET
    )

    y += (
        ny
        * BASELINE_OFFSET
    )

    # -----------------------------------------------------
    # Character angle
    # -----------------------------------------------------

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    # -----------------------------------------------------
    # Character dimensions
    # -----------------------------------------------------

    bbox = measure_draw.textbbox(
        (0, 0),
        char,
        font=font
    )

    glyph_width = max(
        1,
        bbox[2] - bbox[0]
    )

    glyph_height = max(
        1,
        bbox[3] - bbox[1]
    )

    padding = 30

    canvas_width = int(
        max(
            advance,
            glyph_width
        )
        + padding * 2
    )

    canvas_height = int(
        glyph_height
        + padding * 2
        + 10
    )

    # -----------------------------------------------------
    # Character canvas
    # -----------------------------------------------------

    char_image = Image.new(
        "RGBA",
        (
            canvas_width,
            canvas_height
        ),
        (0, 0, 0, 0)
    )

    char_draw = ImageDraw.Draw(
        char_image
    )

    center_x = (
        canvas_width / 2
    )

    center_y = (
        canvas_height / 2
    )

    # Baseline center exactly at canvas center
    char_draw.text(
        (
            center_x,
            center_y
        ),
        char,
        font=font,
        fill=TEXT_COLOR,
        anchor="ms"
    )

    # -----------------------------------------------------
    # Rotate
    # -----------------------------------------------------

    rotated = char_image.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )

    # -----------------------------------------------------
    # Place rotated character center
    # directly on curve
    # -----------------------------------------------------

    paste_x = int(
        x
        - rotated.width / 2
    )

    paste_y = int(
        y
        - rotated.height / 2
    )

    layer.alpha_composite(
        rotated,
        (
            paste_x,
            paste_y
        )
    )

    return True


# =========================================================
# CREATE TEXT LAYER
# =========================================================

layer = Image.new(
    "RGBA",
    image.size,
    (0, 0, 0, 0)
)


# =========================================================
# WORD WRAPPING
# =========================================================

words = TEXT.split()

line_index = 0

distance_on_line = LEFT_MARGIN


# =========================================================
# RENDER
# =========================================================

for word in words:

    if line_index >= len(curves):

        print(
            "WARNING: "
            "Text ran out of notebook lines."
        )

        break

    curve = curves[line_index]

    current_word_width = word_width(
        word
    )

    remaining = (
        curve["length"]
        - distance_on_line
    )

    # Move to next line
    if (
        current_word_width > remaining
        and distance_on_line > LEFT_MARGIN
    ):

        line_index += 1

        distance_on_line = LEFT_MARGIN

        if line_index >= len(curves):

            break

        curve = curves[line_index]

    # -----------------------------------------------------
    # Draw word
    # -----------------------------------------------------

    for char in word:

        advance = char_width(
            char
        )

        success = draw_character(
            layer,
            curve,
            distance_on_line,
            char
        )

        if not success:
            break

        distance_on_line += advance

    # Space
    distance_on_line += SPACE_WIDTH


# =========================================================
# COMBINE
# =========================================================

result = Image.alpha_composite(
    image,
    layer
)


# =========================================================
# SAVE
# =========================================================

result = result.convert(
    "RGB"
)

result.save(
    OUTPUT_PATH,
    quality=95
)

print()
print(
    f"Saved: {OUTPUT_PATH}"
)