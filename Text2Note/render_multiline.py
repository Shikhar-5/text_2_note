import json
import math
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = "Text2Note/notebook.jpg"
LINE_DATA_PATH = "Text2Note/line_data.json"
OUTPUT_PATH = "Text2Note/final_multiline.jpg"

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

# Small text for testing
FONT_SIZE = 22

TEXT = (
    "Introduction to Biology. Biology is the scientific study of life "
    "and living organisms. Living organisms are made up of cells, which "
    "are considered the basic structural and functional units of life. "
    "Cells can be classified into two major types: prokaryotic cells "
    "and eukaryotic cells. Prokaryotic cells are generally smaller and "
    "simpler, while eukaryotic cells contain a nucleus and other "
    "membrane-bound organelles.\n\n"

    "The Cell. Every cell has a cell membrane that controls the movement "
    "of substances into and out of the cell. The cytoplasm contains many "
    "structures that perform important functions. The nucleus stores the "
    "genetic material called DNA and controls many activities of the cell. "
    "Mitochondria are responsible for producing energy through cellular "
    "respiration. Ribosomes help in the production of proteins.\n\n"

    "Photosynthesis. Plants and other photosynthetic organisms can produce "
    "their own food using sunlight. During photosynthesis, carbon dioxide "
    "and water are converted into glucose and oxygen in the presence of "
    "light energy and chlorophyll. This process is important because it "
    "provides food for plants and releases oxygen into the atmosphere.\n\n"

    "Cellular respiration is the process by which cells release energy "
    "from glucose. It mainly occurs in the mitochondria of eukaryotic "
    "cells. Oxygen is used during aerobic respiration, and the products "
    "include carbon dioxide, water, and energy in the form of ATP.\n\n"

    "Important points. Cells are the basic units of life. DNA contains "
    "genetic information. The nucleus controls many cellular activities. "
    "Mitochondria produce usable energy. Ribosomes make proteins. "
    "Photosynthesis converts light energy into chemical energy, while "
    "cellular respiration releases energy from food."
)

# Move text perpendicular to the notebook line
# Try -2, 0, 2, etc.
BASELINE_OFFSET = -2

TEXT_COLOR = (30, 30, 30, 255)

# Distance from left edge of notebook line
LEFT_MARGIN = 15


# =========================================================
# LOAD NOTEBOOK IMAGE
# =========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGBA")

print(
    f"Notebook loaded: "
    f"{image.width} x {image.height}"
)


# =========================================================
# LOAD LINE DATA
# =========================================================

with open(
    LINE_DATA_PATH,
    "r"
) as f:

    lines = json.load(f)

print(
    f"Loaded {len(lines)} notebook lines."
)

if not lines:
    print("No notebook lines found.")
    exit()


# =========================================================
# LOAD FONT
# =========================================================

font = ImageFont.truetype(
    FONT_PATH,
    FONT_SIZE
)


# =========================================================
# FONT MEASUREMENT
# =========================================================

measure_image = Image.new(
    "RGBA",
    (100, 100),
    (0, 0, 0, 0)
)

measure_draw = ImageDraw.Draw(
    measure_image
)


def get_advance(char):
    """
    Actual horizontal advance of the character.
    This is better than using bounding-box width.
    """

    return measure_draw.textlength(
        char,
        font=font
    )


def get_word_width(word):

    total = 0

    for char in word:
        total += get_advance(char)

    return total


SPACE_WIDTH = get_advance(" ")


# =========================================================
# QUADRATIC BEZIER POINT
# =========================================================

def bezier_point(
    p0,
    p1,
    p2,
    t
):

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


# =========================================================
# QUADRATIC BEZIER DERIVATIVE
# =========================================================

def bezier_derivative(
    p0,
    p1,
    p2,
    t
):

    dx = (
        2 * (1 - t) * (p1[0] - p0[0])
        + 2 * t * (p2[0] - p1[0])
    )

    dy = (
        2 * (1 - t) * (p1[1] - p0[1])
        + 2 * t * (p2[1] - p1[1])
    )

    return dx, dy


# =========================================================
# BUILD CURVE
# =========================================================

def build_curve(line):

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

    # Lots of samples gives accurate arc length
    samples = 1500

    points = []
    distances = [0.0]

    total_length = 0.0

    previous_x = None
    previous_y = None

    for i in range(samples + 1):

        t = i / samples

        x, y = bezier_point(
            p0,
            p1,
            p2,
            t
        )

        points.append(
            (x, y, t)
        )

        if previous_x is not None:

            segment = math.sqrt(
                (x - previous_x) ** 2
                + (y - previous_y) ** 2
            )

            total_length += segment

            distances.append(
                total_length
            )

        previous_x = x
        previous_y = y

    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "points": points,
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
    points = curve["points"]

    if target <= 0:
        return 0.0

    if target >= curve["length"]:
        return 1.0

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

    t1 = points[index - 1][2]
    t2 = points[index][2]

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
# BUILD ALL NOTEBOOK CURVES
# =========================================================

curves = []

for index, line in enumerate(lines):

    curve = build_curve(line)

    curves.append(curve)

    print(
        f"Line {index + 1}: "
        f"{curve['length']:.1f}px"
    )


# =========================================================
# DRAW ONE CHARACTER
# =========================================================

def draw_character(
    layer,
    curve,
    distance,
    char
):

    # -----------------------------------------------------
    # Character advance width
    # -----------------------------------------------------

    advance = get_advance(char)

    if advance <= 0:
        return True

    # Character center along curve
    center_distance = (
        distance
        + advance / 2
    )

    if center_distance >= curve["length"]:
        return False

    # -----------------------------------------------------
    # Find exact position on curve
    # -----------------------------------------------------

    t = distance_to_t(
        curve,
        center_distance
    )

    x, y = bezier_point(
        curve["p0"],
        curve["p1"],
        curve["p2"],
        t
    )

    # -----------------------------------------------------
    # Find tangent direction
    # -----------------------------------------------------

    dx, dy = bezier_derivative(
        curve["p0"],
        curve["p1"],
        curve["p2"],
        t
    )

    tangent_length = math.sqrt(
        dx * dx + dy * dy
    )

    if tangent_length == 0:
        return True

    # Unit tangent
    tx = dx / tangent_length
    ty = dy / tangent_length

    # -----------------------------------------------------
    # Normal vector
    # -----------------------------------------------------

    nx = -ty
    ny = tx

    # Move perpendicular to line
    x += nx * BASELINE_OFFSET
    y += ny * BASELINE_OFFSET

    # -----------------------------------------------------
    # Angle of notebook line
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

    # Large enough canvas around character
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
    # Create character canvas
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

    # -----------------------------------------------------
    # IMPORTANT
    #
    # Put the character's BASELINE CENTER exactly
    # at the CENTER of the canvas.
    #
    # "ms" = middle + baseline
    # -----------------------------------------------------

    center_x = canvas_width / 2
    center_y = canvas_height / 2

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
    # Rotate around center
    # -----------------------------------------------------

    rotated = char_image.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )

    # -----------------------------------------------------
    # Put canvas center directly on curve
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
# CREATE TRANSPARENT TEXT LAYER
# =========================================================

layer = Image.new(
    "RGBA",
    image.size,
    (0, 0, 0, 0)
)


# =========================================================
# SPLIT TEXT INTO WORDS
# =========================================================

words = TEXT.split()

line_index = 0

distance_on_line = LEFT_MARGIN


# =========================================================
# RENDER WORDS
# =========================================================

for word in words:

    # -----------------------------------------------------
    # Make sure a notebook line exists
    # -----------------------------------------------------

    if line_index >= len(curves):

        print()
        print(
            "WARNING: "
            "Not enough notebook lines."
        )

        break

    curve = curves[line_index]

    # -----------------------------------------------------
    # Width of word
    # -----------------------------------------------------

    current_word_width = get_word_width(
        word
    )

    remaining_space = (
        curve["length"]
        - distance_on_line
    )

    # -----------------------------------------------------
    # Move to next line if word doesn't fit
    # -----------------------------------------------------

    if (
        current_word_width > remaining_space
        and distance_on_line > LEFT_MARGIN
    ):

        line_index += 1

        distance_on_line = LEFT_MARGIN

        if line_index >= len(curves):

            print(
                "WARNING: "
                "Text ran out of notebook lines."
            )

            break

        curve = curves[line_index]

    # -----------------------------------------------------
    # Draw word character by character
    # -----------------------------------------------------

    for char in word:

        advance = get_advance(
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

    # -----------------------------------------------------
    # Add space
    # -----------------------------------------------------

    distance_on_line += SPACE_WIDTH


# =========================================================
# COMBINE WITH NOTEBOOK
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