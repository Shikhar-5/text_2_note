import cv2
import numpy as np
import json


# =========================================
# SETTINGS
# =========================================

IMAGE_PATH = "Text2Note/notebook.jpg"

OUTPUT_IMAGE = "Text2Note/manual_lines_curve.jpg"

OUTPUT_JSON = "Text2Note/line_data.json"

MAX_WIDTH = 1000
MAX_HEIGHT = 750


# =========================================
# LOAD IMAGE
# =========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load notebook.jpg")
    exit()

original_height, original_width = image.shape[:2]

print()
print("======================================")
print("TEXT2NOTE MANUAL LINE CALIBRATION")
print("======================================")
print()

print(
    f"Image size: "
    f"{original_width} x {original_height}"
)


# =========================================
# RESIZE FOR DISPLAY
# =========================================

scale = min(
    MAX_WIDTH / original_width,
    MAX_HEIGHT / original_height,
    1.0
)

display_width = int(
    original_width * scale
)

display_height = int(
    original_height * scale
)

display_image = cv2.resize(
    image,
    (
        display_width,
        display_height
    )
)


# =========================================
# DATA
# =========================================

# Every line contains:

# LEFT
# MIDDLE
# RIGHT

lines = []

current_points = []


# =========================================
# COORDINATE CONVERSION
# =========================================

def to_display(point):

    x, y = point

    return (
        int(x * scale),
        int(y * scale)
    )


# =========================================
# DRAW SMOOTH CURVE
# =========================================

def draw_curve(
    canvas,
    p0,
    p1,
    p2
):

    points = []

    # Quadratic Bezier curve

    for t in np.linspace(
        0,
        1,
        100
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

        points.append(
            (
                int(x),
                int(y)
            )
        )

    for i in range(
        1,
        len(points)
    ):

        cv2.line(
            canvas,
            points[i - 1],
            points[i],
            (0, 0, 255),
            2
        )


# =========================================
# REDRAW EVERYTHING
# =========================================

def redraw():

    global display_image

    # Start from original image

    display_image = cv2.resize(
        image,
        (
            display_width,
            display_height
        )
    )

    # -------------------------------------
    # Draw completed lines
    # -------------------------------------

    for index, line in enumerate(lines):

        left = to_display(
            line[0]
        )

        middle = to_display(
            line[1]
        )

        right = to_display(
            line[2]
        )

        # Draw curve

        draw_curve(
            display_image,
            left,
            middle,
            right
        )

        # LEFT point

        cv2.circle(
            display_image,
            left,
            5,
            (0, 0, 255),
            -1
        )

        # MIDDLE point

        cv2.circle(
            display_image,
            middle,
            5,
            (0, 255, 255),
            -1
        )

        # RIGHT point

        cv2.circle(
            display_image,
            right,
            5,
            (0, 0, 255),
            -1
        )

        # Line number

        cv2.putText(
            display_image,
            str(index + 1),
            (
                middle[0] - 8,
                middle[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # -------------------------------------
    # Draw currently selected points
    # -------------------------------------

    names = [
        "LEFT",
        "MIDDLE",
        "RIGHT"
    ]

    for index, point in enumerate(
        current_points
    ):

        p = to_display(point)

        cv2.circle(
            display_image,
            p,
            6,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display_image,
            names[index],
            (
                p[0] + 8,
                p[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Manual Notebook Lines",
        display_image
    )


# =========================================
# MOUSE CALLBACK
# =========================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Maximum 3 points for one line

    if len(current_points) >= 3:
        return

    # Convert display coordinate
    # to original image coordinate

    original_x = int(
        x / scale
    )

    original_y = int(
        y / scale
    )

    point = (
        original_x,
        original_y
    )

    current_points.append(
        point
    )

    names = [
        "LEFT",
        "MIDDLE",
        "RIGHT"
    ]

    current_name = names[
        len(current_points) - 1
    ]

    print(
        f"Line {len(lines) + 1} "
        f"{current_name}: "
        f"x={original_x}, "
        f"y={original_y}"
    )

    # -------------------------------------
    # If 3 points are selected,
    # complete the line
    # -------------------------------------

    if len(current_points) == 3:

        lines.append(
            (
                current_points[0],
                current_points[1],
                current_points[2]
            )
        )

        print(
            f"Line {len(lines)} completed."
        )

        current_points.clear()

    redraw()


# =========================================
# CREATE WINDOW
# =========================================

cv2.namedWindow(
    "Manual Notebook Lines"
)

cv2.setMouseCallback(
    "Manual Notebook Lines",
    mouse_callback
)


# =========================================
# INSTRUCTIONS
# =========================================

print()
print("--------------------------------------")
print("HOW TO USE")
print("--------------------------------------")
print()
print("For EVERY notebook line:")
print()
print("1. Click LEFT")
print("2. Click MIDDLE")
print("3. Click RIGHT")
print()
print("Then move to the next notebook line.")
print()
print("Keyboard controls:")
print()
print("R     = Reset all lines")
print("U     = Undo last line")
print("ENTER = Finish")
print("ESC   = Cancel")
print()
print("--------------------------------------")
print()
print("RED    = notebook line")
print("YELLOW = middle control point")
print("BLUE   = line number")
print("GREEN  = current point")
print()


cv2.imshow(
    "Manual Notebook Lines",
    display_image
)


# =========================================
# MAIN LOOP
# =========================================

while True:

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------
    # RESET
    # -------------------------------------

    if key == ord("r"):

        lines.clear()

        current_points.clear()

        redraw()

        print()
        print("All lines reset.")

    # -------------------------------------
    # UNDO
    # -------------------------------------

    elif key == ord("u"):

        if lines:

            lines.pop()

            current_points.clear()

            redraw()

            print()
            print(
                f"Last line removed."
            )

            print(
                f"Remaining lines: "
                f"{len(lines)}"
            )

        else:

            print(
                "Nothing to undo."
            )

    # -------------------------------------
    # FINISH
    # -------------------------------------

    elif key == 13:

        if current_points:

            print()
            print(
                "Current line is incomplete."
            )

            print(
                "You need LEFT, MIDDLE "
                "and RIGHT."
            )

        elif len(lines) == 0:

            print()
            print(
                "No lines selected."
            )

        else:

            break

    # -------------------------------------
    # CANCEL
    # -------------------------------------

    elif key == 27:

        cv2.destroyAllWindows()

        print()
        print("Cancelled.")

        exit()


# =========================================
# CLOSE WINDOW
# =========================================

cv2.destroyAllWindows()


# =========================================
# PRINT CALIBRATION
# =========================================

print()
print("======================================")
print("CALIBRATION COMPLETE")
print("======================================")
print()

print(
    f"Total notebook lines: "
    f"{len(lines)}"
)

print()


# =========================================
# CREATE JSON DATA
# =========================================

calibration_data = []


for index, line in enumerate(lines):

    left = line[0]

    middle = line[1]

    right = line[2]

    calibration_data.append({

        "line": index + 1,

        "left": {
            "x": left[0],
            "y": left[1]
        },

        "middle": {
            "x": middle[0],
            "y": middle[1]
        },

        "right": {
            "x": right[0],
            "y": right[1]
        }

    })


# =========================================
# SAVE JSON
# =========================================

with open(
    OUTPUT_JSON,
    "w"
) as file:

    json.dump(
        calibration_data,
        file,
        indent=4
    )


print(
    f"Saved calibration data:"
)

print(
    OUTPUT_JSON
)


# =========================================
# SAVE DEBUG IMAGE
# =========================================

# Redraw final image

redraw()

cv2.imwrite(
    OUTPUT_IMAGE,
    display_image
)


print()

print(
    f"Saved debug image:"
)

print(
    OUTPUT_IMAGE
)

print()
print("======================================")
print("DONE")
print("======================================")