import cv2
import numpy as np

IMAGE_PATH = "Text2Note/calibrated_page.jpg"

# -----------------------------------------
# 1. Load calibrated page
# -----------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load calibrated_page.jpg")
    exit()

height, width = image.shape[:2]

print("Calibrated page loaded!")
print(f"Size: {width} x {height}")

# -----------------------------------------
# 2. Convert to grayscale
# -----------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# -----------------------------------------
# 3. Slight blur
# -----------------------------------------

gray = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# -----------------------------------------
# 4. Detect dark horizontal structures
# -----------------------------------------

binary = cv2.threshold(
    gray,
    190,
    255,
    cv2.THRESH_BINARY_INV
)[1]

# -----------------------------------------
# 5. Extract horizontal lines
# -----------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (100, 1)
)

horizontal = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

# -----------------------------------------
# 6. Calculate strength of every row
# -----------------------------------------

row_scores = np.sum(
    horizontal > 0,
    axis=1
)

# Smooth row scores
row_scores = np.convolve(
    row_scores,
    np.ones(5) / 5,
    mode="same"
)

# -----------------------------------------
# 7. Find candidate rows
# -----------------------------------------

max_score = np.max(row_scores)

threshold = max_score * 0.20

candidate_rows = []

for y in range(
    20,
    height - 20
):

    if row_scores[y] < threshold:
        continue

    start = max(
        0,
        y - 3
    )

    end = min(
        height,
        y + 4
    )

    if row_scores[y] == np.max(
        row_scores[start:end]
    ):

        candidate_rows.append(y)

# -----------------------------------------
# 8. Merge nearby rows
# -----------------------------------------

line_positions = []

for y in candidate_rows:

    if not line_positions:

        line_positions.append(y)

        continue

    if (
        y - line_positions[-1]
        < 12
    ):

        # Keep stronger row
        if row_scores[y] > row_scores[
            line_positions[-1]
        ]:

            line_positions[-1] = y

    else:

        line_positions.append(y)

# -----------------------------------------
# 9. Print detected lines
# -----------------------------------------

print(
    f"Detected notebook lines: "
    f"{len(line_positions)}"
)

print("\nLine positions:")

for index, y in enumerate(
    line_positions
):

    print(
        f"Line {index + 1}: "
        f"y = {y}"
    )

# -----------------------------------------
# 10. Calculate spacing
# -----------------------------------------

if len(line_positions) > 1:

    spacings = []

    for i in range(
        1,
        len(line_positions)
    ):

        difference = (
            line_positions[i]
            - line_positions[i - 1]
        )

        if 20 < difference < 70:

            spacings.append(
                difference
            )

    if spacings:

        print("\nLine spacing:")

        print(
            f"Average: "
            f"{np.mean(spacings):.2f}px"
        )

        print(
            f"Median: "
            f"{np.median(spacings):.2f}px"
        )

# -----------------------------------------
# 11. Draw detected lines
# -----------------------------------------

result = image.copy()

for index, y in enumerate(
    line_positions
):

    cv2.line(
        result,
        (40, y),
        (960, y),
        (0, 0, 255),
        2
    )

    cv2.putText(
        result,
        str(index + 1),
        (10, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

# -----------------------------------------
# 12. Save
# -----------------------------------------

cv2.imwrite(
    "Text2Note/calibrated_lines.jpg",
    result
)

print(
    "\nSaved:"
)

print(
    "Text2Note/calibrated_lines.jpg"
)   