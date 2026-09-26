import json
import math

from PIL import Image, ImageDraw, ImageFont


# =========================================================
# BEZIER FUNCTIONS
# =========================================================

def bezier_point(points, t):
    """
    Calculate a point on a 5-point Bezier curve.
    """

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


def bezier_derivative(points, t):
    """
    Calculate the tangent direction of a 5-point Bezier curve.
    """

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
# BUILD ONE CURVE
# =========================================================

def build_curve(line):
    """
    Convert one JSON line into a sampled Bezier curve.
    """

    points = []

    for key in ["p0", "p1", "p2", "p3", "p4"]:

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

    previous_x = None
    previous_y = None

    for i in range(samples + 1):

        t = i / samples

        x, y = bezier_point(
            points,
            t
        )

        curve_points.append(
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
        "points": points,
        "curve_points": curve_points,
        "distances": distances,
        "length": total_length
    }


# =========================================================
# DISTANCE -> T
# =========================================================

def distance_to_t(curve, target_distance):
    """
    Convert real distance along the curve into
    Bezier parameter t.
    """

    distances = curve["distances"]
    curve_points = curve["curve_points"]

    if target_distance <= 0:
        return 0.0

    if target_distance >= curve["length"]:
        return 1.0

    low = 0
    high = len(distances) - 1

    while low <= high:

        mid = (
            low + high
        ) // 2

        if distances[mid] < target_distance:

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
        target_distance - d1
    ) / (
        d2 - d1
    )

    return (
        t1
        + ratio * (t2 - t1)
    )


# =========================================================
# RENDER NOTEBOOK
# =========================================================

def render_notebook(
    image_path,
    text,
    font_path,
    font_size,
    font_color,
    line_data_path,
    output_path,
    baseline_offset=-2,
    left_margin=15
):
    """
    Render text onto a calibrated notebook image.

    Parameters
    ----------
    image_path : str
        Path to notebook image.

    text : str
        Text that should be rendered.

    font_path : str
        Path to .ttf/.otf font.

    font_size : int
        Font size.

    font_color : tuple
        RGBA color, e.g. (30, 30, 30, 255).

    line_data_path : str
        Path to line_data_5point.json.

    output_path : str
        Where final image should be saved.

    baseline_offset : float
        Offset perpendicular to notebook line.

    left_margin : float
        Starting distance from left side.
    """

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = Image.open(
        image_path
    ).convert("RGBA")

    # =====================================================
    # LOAD LINE DATA
    # =====================================================

    with open(
        line_data_path,
        "r"
    ) as f:

        lines = json.load(f)

    if not lines:

        raise ValueError(
            "No calibrated notebook lines found."
        )

    # =====================================================
    # LOAD FONT
    # =====================================================

    font = ImageFont.truetype(
        font_path,
        font_size
    )

    # =====================================================
    # FONT MEASUREMENT
    # =====================================================

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

    space_width = char_width(" ")

    # =====================================================
    # BUILD CURVES
    # =====================================================

    curves = []

    for line in lines:

        curve = build_curve(
            line
        )

        curves.append(
            curve
        )

    # =====================================================
    # CREATE TRANSPARENT LAYER
    # =====================================================

    layer = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0)
    )

    # =====================================================
    # DRAW CHARACTER
    # =====================================================

    def draw_character(
        curve,
        distance,
        char
    ):

        advance = char_width(
            char
        )

        if advance <= 0:

            return True

        center_distance = (
            distance
            + advance / 2
        )

        if center_distance >= curve["length"]:

            return False

        # -----------------------------------------------
        # Find curve position
        # -----------------------------------------------

        t = distance_to_t(
            curve,
            center_distance
        )

        x, y = bezier_point(
            curve["points"],
            t
        )

        # -----------------------------------------------
        # Find curve direction
        # -----------------------------------------------

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

        # Unit normal
        nx = (
            -dy
            / tangent_length
        )

        ny = (
            dx
            / tangent_length
        )

        # -----------------------------------------------
        # Baseline offset
        # -----------------------------------------------

        x += (
            nx
            * baseline_offset
        )

        y += (
            ny
            * baseline_offset
        )

        # -----------------------------------------------
        # Character angle
        # -----------------------------------------------

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        # -----------------------------------------------
        # Character size
        # -----------------------------------------------

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

        # -----------------------------------------------
        # Character canvas
        # -----------------------------------------------

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

        # Baseline center at canvas center
        char_draw.text(
            (
                center_x,
                center_y
            ),
            char,
            font=font,
            fill=font_color,
            anchor="ms"
        )

        # -----------------------------------------------
        # Rotate
        # -----------------------------------------------

        rotated = char_image.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=False
        )

        # -----------------------------------------------
        # Place character
        # -----------------------------------------------

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

    # =====================================================
    # RENDER WORDS
    # =====================================================

    words = text.split()

    line_index = 0

    distance_on_line = left_margin

    for word in words:

        # -----------------------------------------------
        # Check available line
        # -----------------------------------------------

        if line_index >= len(curves):

            print(
                "WARNING: "
                "Text does not fit on notebook."
            )

            break

        curve = curves[line_index]

        word_size = word_width(
            word
        )

        remaining = (
            curve["length"]
            - distance_on_line
        )

        # -----------------------------------------------
        # Move to next line
        # -----------------------------------------------

        if (
            word_size > remaining
            and distance_on_line > left_margin
        ):

            line_index += 1

            distance_on_line = left_margin

            if line_index >= len(curves):

                print(
                    "WARNING: "
                    "Text ran out of notebook lines."
                )

                break

            curve = curves[line_index]

        # -----------------------------------------------
        # Render word
        # -----------------------------------------------

        for char in word:

            advance = char_width(
                char
            )

            success = draw_character(
                curve,
                distance_on_line,
                char
            )

            if not success:

                break

            distance_on_line += advance

        # -----------------------------------------------
        # Add space
        # -----------------------------------------------

        distance_on_line += space_width

    # =====================================================
    # COMBINE
    # =====================================================

    result = Image.alpha_composite(
        image,
        layer
    )

    # =====================================================
    # SAVE
    # =====================================================

    result = result.convert(
        "RGB"
    )

    result.save(
        output_path,
        quality=95
    )

    return output_path