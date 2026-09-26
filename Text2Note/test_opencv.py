import cv2
import numpy as np

image_path = "Text2Note/notebook.jpg"

# -----------------------------------------
# 1. Load image
# -----------------------------------------

image = cv2.imread(image_path)

if image is None:
    print("Could not load the image.")
    exit()

height, width = image.shape[:2]

print("Image loaded successfully!")
print(f"Image size: {width} x {height}")

# -----------------------------------------
# 2. Resize for processing
# -----------------------------------------

# Work on a smaller copy.
# This makes processing faster.

scale = 0.5

small = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale
)

small_height, small_width = small.shape[:2]

# -----------------------------------------
# 3. Convert to grayscale
# -----------------------------------------

gray = cv2.cvtColor(
    small,
    cv2.COLOR_BGR2GRAY
)

# -----------------------------------------
# 4. Blur the image
# -----------------------------------------

blurred = cv2.GaussianBlur(
    gray,
    (7, 7),
    0
)

# -----------------------------------------
# 5. Detect edges
# -----------------------------------------

edges = cv2.Canny(
    blurred,
    30,
    100
)

# -----------------------------------------
# 6. Close small gaps in edges
# -----------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (7, 7)
)

closed = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------------------------
# 7. Find contours
# -----------------------------------------

contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(
    f"Contours found: {len(contours)}"
)

# -----------------------------------------
# 8. Look for large rectangular contours
# -----------------------------------------

page_contour = None
page_area = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore small objects
    if area < (
        small_width
        * small_height
        * 0.20
    ):
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    approximation = cv2.approxPolyDP(
        contour,
        0.03 * perimeter,
        True
    )

    # We want approximately four corners
    if len(approximation) == 4:

        if area > page_area:

            page_area = area
            page_contour = approximation

# -----------------------------------------
# 9. Draw detected page
# -----------------------------------------

result = image.copy()

if page_contour is not None:

    # Convert coordinates back to original size
    page_points = (
        page_contour.reshape(4, 2)
        / scale
    ).astype(np.int32)

    print("\nPossible page corners:")

    for index, point in enumerate(
        page_points
    ):

        print(
            f"Corner {index + 1}: "
            f"x = {point[0]}, "
            f"y = {point[1]}"
        )

    # Draw the four corners
    cv2.polylines(
        result,
        [page_points],
        True,
        (0, 0, 255),
        5
    )

    # Draw corner points
    for index, point in enumerate(
        page_points
    ):

        cv2.circle(
            result,
            tuple(point),
            10,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            result,
            str(index + 1),
            tuple(point),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

else:

    print(
        "Could not find a rectangular "
        "page boundary."
    )

# -----------------------------------------
# 10. Save result
# -----------------------------------------

cv2.imwrite(
    "Text2Note/page_boundary.jpg",
    result
)

print(
    "\nSaved Text2Note/page_boundary.jpg"
)