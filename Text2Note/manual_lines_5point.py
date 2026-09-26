import cv2
import json
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = "Text2Note/notebook.jpg"

OUTPUT_IMAGE = "Text2Note/manual_lines_5point.jpg"
OUTPUT_JSON = "Text2Note/line_data_5point.json"

WINDOW_NAME = "5-Point Notebook Calibration"


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not load notebook.jpg")
    exit()

original = image.copy()

height, width = image.shape[:2]

print(
    f"Notebook loaded: {width} x {height}"
)


# =========================================================
# FIT IMAGE TO SCREEN
# =========================================================

screen_width = 1200
screen_height = 800

scale = min(
    screen_width / width,
    screen_height / height
)

display_width = int(width * scale)
display_height = int(height * scale)

display_image = cv2.resize(
    original,
    (
        display_width,
        display_height
    )
)


# =========================================================
# CALIBRATION DATA
# =========================================================

all_lines = []

current_points = []


# =========================================================
# BEZIER FUNCTION
# =========================================================

def bezier_point(points, t):

    p0 = np.array(points[0], dtype=float)
    p1 = np.array(points[1], dtype=float)
    p2 = np.array(points[2], dtype=float)
    p3 = np.array(points[3], dtype=float)
    p4 = np.array(points[4], dtype=float)

    # 5-point smooth curve
    #
    # We use a degree-4 Bezier curve.
    #

    point = (
        ((1 - t) ** 4) * p0
        + 4 * ((1 - t) ** 3) * t * p1
        + 6 * ((1 - t) ** 2) * (t ** 2) * p2
        + 4 * (1 - t) * (t ** 3) * p3
        + (t ** 4) * p4
    )

    return (
        int(point[0]),
        int(point[1])
    )


# =========================================================
# DRAW CURVE
# =========================================================

def draw_curve(
    img,
    points
):

    if len(points) != 5:
        return

    previous = None

    for i in range(101):

        t = i / 100

        x, y = bezier_point(
            points,
            t
        )

        if previous is not None:

            cv2.line(
                img,
                previous,
                (x, y),
                (0, 0, 255),
                2
            )

        previous = (
            x,
            y
        )


# =========================================================
# REDRAW DISPLAY
# =========================================================

def redraw():

    canvas = original.copy()

    # -----------------------------------------------------
    # Draw completed curves
    # -----------------------------------------------------

    for line in all_lines:

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

        draw_curve(
            canvas,
            points
        )

    # -----------------------------------------------------
    # Draw current points
    # -----------------------------------------------------

    for point in current_points:

        cv2.circle(
            canvas,
            point,
            7,
            (0, 255, 0),
            -1
        )

    # -----------------------------------------------------
    # Show instructions
    # -----------------------------------------------------

    cv2.putText(
        canvas,
        "Click: LEFT -> 25% -> MIDDLE -> 75% -> RIGHT",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        "R = Reset | U = Undo | ENTER = Finish | ESC = Cancel",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )

    # Resize for display

    display = cv2.resize(
        canvas,
        (
            display_width,
            display_height
        )
    )

    return display


# =========================================================
# MOUSE CALLBACK
# =========================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global current_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Convert display coordinates
    # back to original image coordinates.

    original_x = int(
        x / scale
    )

    original_y = int(
        y / scale
    )

    current_points.append(
        (
            original_x,
            original_y
        )
    )

    print(
        f"Point {len(current_points)}/5: "
        f"({original_x}, {original_y})"
    )

    # -----------------------------------------------------
    # Once 5 points are selected
    # -----------------------------------------------------

    if len(current_points) == 5:

        points = current_points.copy()

        all_lines.append(
            {
                "line": len(all_lines) + 1,

                "p0": {
                    "x": points[0][0],
                    "y": points[0][1]
                },

                "p1": {
                    "x": points[1][0],
                    "y": points[1][1]
                },

                "p2": {
                    "x": points[2][0],
                    "y": points[2][1]
                },

                "p3": {
                    "x": points[3][0],
                    "y": points[3][1]
                },

                "p4": {
                    "x": points[4][0],
                    "y": points[4][1]
                }
            }
        )

        print(
            f"Line {len(all_lines)} saved."
        )

        current_points = []


# =========================================================
# CREATE WINDOW
# =========================================================

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    frame = redraw()

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    key = cv2.waitKey(20) & 0xFF

    # -----------------------------------------------------
    # R = RESET EVERYTHING
    # -----------------------------------------------------

    if key == ord("r"):

        all_lines.clear()

        current_points.clear()

        print(
            "All calibration data reset."
        )

    # -----------------------------------------------------
    # U = UNDO LAST LINE
    # -----------------------------------------------------

    elif key == ord("u"):

        if current_points:

            current_points.clear()

            print(
                "Current line points cleared."
            )

        elif all_lines:

            removed = all_lines.pop()

            print(
                f"Removed line "
                f"{removed['line']}."
            )

            # Renumber lines

            for i, line in enumerate(
                all_lines
            ):

                line["line"] = i + 1

        else:

            print(
                "Nothing to undo."
            )

    # -----------------------------------------------------
    # ENTER = FINISH
    # -----------------------------------------------------

    elif key == 13:

        print()
        print(
            f"Finished calibration."
        )

        print(
            f"Total lines: "
            f"{len(all_lines)}"
        )

        break

    # -----------------------------------------------------
    # ESC = CANCEL
    # -----------------------------------------------------

    elif key == 27:

        print(
            "Calibration cancelled."
        )

        cv2.destroyAllWindows()

        exit()


# =========================================================
# SAVE JSON
# =========================================================

with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        all_lines,
        f,
        indent=4
    )


# =========================================================
# SAVE PREVIEW
# =========================================================

final_image = original.copy()

for line in all_lines:

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

    draw_curve(
        final_image,
        points
    )

    # Draw calibration points

    for point in points:

        cv2.circle(
            final_image,
            point,
            5,
            (0, 255, 0),
            -1
        )


cv2.imwrite(
    OUTPUT_IMAGE,
    final_image
)


# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()


print()
print(
    f"Saved preview: "
    f"{OUTPUT_IMAGE}"
)

print(
    f"Saved coordinates: "
    f"{OUTPUT_JSON}"
)