import cv2
import numpy as np

IMAGE_PATH = "Text2Note/notebook.jpg"

OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 1400

# -----------------------------------------
# Load image
# -----------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load notebook.jpg")
    exit()

original_height, original_width = image.shape[:2]

print(
    f"Original image: "
    f"{original_width} x {original_height}"
)

# -----------------------------------------
# Resize image so the WHOLE notebook
# fits on the screen
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

points = []


# -----------------------------------------
# Mouse click
# -----------------------------------------

def select_point(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) >= 4:
        return

    # Convert displayed coordinates
    # back to ORIGINAL image coordinates
    original_x = int(x / scale)
    original_y = int(y / scale)

    points.append(
        (original_x, original_y)
    )

    print(
        f"Point {len(points)}: "
        f"x={original_x}, "
        f"y={original_y}"
    )

    # Draw point on displayed image
    cv2.circle(
        display_image,
        (x, y),
        7,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        display_image,
        str(len(points)),
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # Draw connecting line
    if len(points) > 1:

        previous = points[-2]

        previous_display = (
            int(previous[0] * scale),
            int(previous[1] * scale)
        )

        cv2.line(
            display_image,
            previous_display,
            (x, y),
            (0, 255, 0),
            2
        )

    # Close shape after 4 points
    if len(points) == 4:

        first = points[0]

        first_display = (
            int(first[0] * scale),
            int(first[1] * scale)
        )

        cv2.line(
            display_image,
            (x, y),
            first_display,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Manual Page Calibration",
        display_image
    )


# -----------------------------------------
# Window
# -----------------------------------------

cv2.namedWindow(
    "Manual Page Calibration",
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(
    "Manual Page Calibration",
    select_point
)

print()
print("===================================")
print("MANUAL PAGE CALIBRATION")
print("===================================")
print()
print("The FULL image should now be visible.")
print()
print("Click in this order:")
print()
print("1 = TOP LEFT")
print("2 = TOP RIGHT")
print("3 = BOTTOM RIGHT")
print("4 = BOTTOM LEFT")
print()
print("Press R to reset.")
print("Press ENTER after 4 clicks.")
print("Press ESC to cancel.")
print()

cv2.imshow(
    "Manual Page Calibration",
    display_image
)


# -----------------------------------------
# Wait for clicks
# -----------------------------------------

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        points.clear()

        display_image = cv2.resize(
            image,
            (display_width, display_height)
        )

        cv2.imshow(
            "Manual Page Calibration",
            display_image
        )

        print("Reset.")

    elif key == 13:

        if len(points) == 4:
            break

        print(
            "You must select exactly "
            "4 points."
        )

    elif key == 27:

        cv2.destroyAllWindows()

        print("Cancelled.")

        exit()


cv2.destroyAllWindows()


# -----------------------------------------
# Perspective transformation
# -----------------------------------------

source_points = np.float32(points)

destination_points = np.float32([
    [0, 0],
    [OUTPUT_WIDTH - 1, 0],
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
    [0, OUTPUT_HEIGHT - 1]
])

matrix = cv2.getPerspectiveTransform(
    source_points,
    destination_points
)

warped = cv2.warpPerspective(
    image,
    matrix,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT)
)


# -----------------------------------------
# Save calibrated page
# -----------------------------------------

cv2.imwrite(
    "Text2Note/calibrated_page.jpg",
    warped
)

print()
print("===================================")
print("CALIBRATION COMPLETE")
print("===================================")
print()
print(
    "Saved: "
    "Text2Note/calibrated_page.jpg"
)