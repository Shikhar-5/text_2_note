import cv2
import numpy as np

IMAGE_PATH = "Text2Note/notebook.jpg"

# -----------------------------------------
# Load image
# -----------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load notebook.jpg")
    exit()

original_height, original_width = image.shape[:2]

print(
    f"Image size: "
    f"{original_width} x {original_height}"
)

# -----------------------------------------
# Fit image to screen
# -----------------------------------------

MAX_WIDTH = 1000
MAX_HEIGHT = 750

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
    (display_width, display_height)
)

# -----------------------------------------
# Store manually selected lines
#
# Each line contains:
# (left_point, right_point)
# -----------------------------------------

lines = []

# Current line's first point
first_point = None


# -----------------------------------------
# Convert original coordinates
# to display coordinates
# -----------------------------------------

def to_display(point):

    x, y = point

    return (
        int(x * scale),
        int(y * scale)
    )


# -----------------------------------------
# Mouse callback
# -----------------------------------------

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global first_point
    global display_image

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Convert screen coordinates
    # to original image coordinates

    original_x = int(x / scale)
    original_y = int(y / scale)

    current_point = (
        original_x,
        original_y
    )

    # -------------------------------------
    # FIRST CLICK = LEFT SIDE
    # -------------------------------------

    if first_point is None:

        first_point = current_point

        cv2.circle(
            display_image,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_image,
            f"LEFT {len(lines) + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        print()
        print(
            f"Line {len(lines) + 1}: "
            f"LEFT = "
            f"({original_x}, {original_y})"
        )

    # -------------------------------------
    # SECOND CLICK = RIGHT SIDE
    # -------------------------------------

    else:

        second_point = current_point

        # Store complete line
        lines.append(
            (
                first_point,
                second_point
            )
        )

        # Display coordinates
        p1 = to_display(first_point)
        p2 = to_display(second_point)

        # Draw the line
        cv2.line(
            display_image,
            p1,
            p2,
            (0, 0, 255),
            2
        )

        # Draw right point
        cv2.circle(
            display_image,
            p2,
            6,
            (0, 0, 255),
            -1
        )

        # Number
        middle_x = (
            p1[0] + p2[0]
        ) // 2

        middle_y = (
            p1[1] + p2[1]
        ) // 2

        cv2.putText(
            display_image,
            str(len(lines)),
            (middle_x, middle_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        print(
            f"Line {len(lines)}: "
            f"RIGHT = "
            f"({original_x}, {original_y})"
        )

        # Reset for next line
        first_point = None

    cv2.imshow(
        "Manual Notebook Lines",
        display_image
    )


# -----------------------------------------
# Create window
# -----------------------------------------

cv2.namedWindow(
    "Manual Notebook Lines"
)

cv2.setMouseCallback(
    "Manual Notebook Lines",
    mouse_callback
)

# -----------------------------------------
# Instructions
# -----------------------------------------

print()
print("======================================")
print("MANUAL NOTEBOOK LINE DETECTION")
print("======================================")
print()
print("For EVERY notebook line:")
print()
print("1. Click LEFT endpoint")
print("2. Click RIGHT endpoint")
print("3. Move to the NEXT line")
print("4. Click LEFT endpoint")
print("5. Click RIGHT endpoint")
print()
print("Continue until all lines are selected.")
print()
print("Keyboard:")
print("R = Reset everything")
print("U = Undo last completed line")
print("ENTER = Finish")
print("ESC = Cancel")
print()
print("The entire image is fitted to the window.")
print()

cv2.imshow(
    "Manual Notebook Lines",
    display_image
)


# -----------------------------------------
# Keyboard controls
# -----------------------------------------

while True:

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------
    # RESET
    # -------------------------------------

    if key == ord("r"):

        lines.clear()
        first_point = None

        display_image = cv2.resize(
            image,
            (
                display_width,
                display_height
            )
        )

        cv2.imshow(
            "Manual Notebook Lines",
            display_image
        )

        print("All lines reset.")

    # -------------------------------------
    # UNDO
    # -------------------------------------

    elif key == ord("u"):

        if lines:

            removed = lines.pop()

            first_point = None

            display_image = cv2.resize(
                image,
                (
                    display_width,
                    display_height
                )
            )

            # Redraw all existing lines
            for index, line in enumerate(lines):

                p1 = to_display(
                    line[0]
                )

                p2 = to_display(
                    line[1]
                )

                cv2.line(
                    display_image,
                    p1,
                    p2,
                    (0, 0, 255),
                    2
                )

                middle_x = (
                    p1[0] + p2[0]
                ) // 2

                middle_y = (
                    p1[1] + p2[1]
                ) // 2

                cv2.putText(
                    display_image,
                    str(index + 1),
                    (
                        middle_x,
                        middle_y - 5
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

            cv2.imshow(
                "Manual Notebook Lines",
                display_image
            )

            print(
                f"Removed line. "
                f"Remaining: {len(lines)}"
            )

        else:

            print("Nothing to undo.")

    # -------------------------------------
    # FINISH
    # -------------------------------------

    elif key == 13:

        if first_point is not None:

            print(
                "You have selected a LEFT point "
                "without a RIGHT point."
            )

            print(
                "Complete that line first."
            )

        elif len(lines) == 0:

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

        print("Cancelled.")

        exit()


cv2.destroyAllWindows()


# -----------------------------------------
# Print final data
# -----------------------------------------

print()
print("======================================")
print("CALIBRATION COMPLETE")
print("======================================")

print(
    f"Total notebook lines: {len(lines)}"
)

print()

for index, line in enumerate(lines):

    left_point = line[0]
    right_point = line[1]

    print(
        f"Line {index + 1}: "
        f"LEFT={left_point}, "
        f"RIGHT={right_point}"
    )


# -----------------------------------------
# Calculate line information
# -----------------------------------------

line_data = []

for index, line in enumerate(lines):

    left = np.array(
        line[0],
        dtype=float
    )

    right = np.array(
        line[1],
        dtype=float
    )

    dx = right[0] - left[0]
    dy = right[1] - left[1]

    length = np.sqrt(
        dx * dx +
        dy * dy
    )

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    line_data.append({
        "line": index + 1,
        "left": tuple(left.astype(int)),
        "right": tuple(right.astype(int)),
        "length": length,
        "angle": angle
    })


# -----------------------------------------
# Print line measurements
# -----------------------------------------

print()
print("Line measurements:")
print()

for data in line_data:

    print(
        f"Line {data['line']}: "
        f"length={data['length']:.1f}px, "
        f"angle={data['angle']:.2f}°"
    )


# -----------------------------------------
# Save final debug image
# -----------------------------------------

output_path = (
    "Text2Note/manual_lines.jpg"
)

cv2.imwrite(
    output_path,
    display_image
)

print()
print(
    f"Saved: {output_path}"
)