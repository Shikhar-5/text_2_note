from pathlib import Path
import json

import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "handwriting_dataset"

SEGMENTED_DIR = DATASET_DIR / "segmented"

SEGMENTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

# Ignore extremely tiny connected components.
MIN_COMPONENT_AREA = 8

# Maximum vertical distance between components that
# can belong to the same handwriting line.
#
# This is intentionally based on the actual component
# heights rather than a fixed pixel distance.
LINE_VERTICAL_FACTOR = 0.65

# Word separation is calculated dynamically from
# the character/component spacing.
WORD_GAP_FACTOR = 2.0

# Absolute safety limits.
MIN_WORD_GAP = 12
MAX_WORD_GAP = 70


# =========================================================
# FIND PROCESSED SAMPLES
# =========================================================

sample_dirs = sorted(
    [
        directory
        for directory in DATASET_DIR.iterdir()
        if directory.is_dir()
        and directory.name.startswith("sample_")
    ]
)


if not sample_dirs:
    raise FileNotFoundError(
        "No processed handwriting samples found."
    )


print()
print("========================================")
print("TEXT2NOTE HANDWRITING SEGMENTER V3.3")
print("========================================")
print()
print(
    f"Found {len(sample_dirs)} processed samples."
)
print()


# =========================================================
# CONNECTED COMPONENT EXTRACTION
# =========================================================

def get_components(binary):
    """
    Find connected handwriting components.

    Returns bounding boxes:
        x, y, width, height, area
    """

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        # Ignore extremely large artifacts.
        #
        # We don't want page/background blobs
        # becoming handwriting components.
        if area > (
            binary.shape[0] *
            binary.shape[1] *
            0.15
        ):
            continue

        components.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "center_x": x + w / 2,
                "center_y": y + h / 2
            }
        )

    return components


# =========================================================
# GROUP COMPONENTS INTO HANDWRITING LINES
# =========================================================

def group_into_lines(
    components
):
    """
    Group connected components by vertical position.

    This is deliberately NOT based on horizontal
    projection.

    Each component is assigned to the closest
    compatible handwriting line.
    """

    if not components:
        return []

    components = sorted(
        components,
        key=lambda item: item["center_y"]
    )

    lines = []

    for component in components:

        center_y = component["center_y"]

        assigned_line = None

        best_distance = float("inf")

        for line in lines:

            line_center = line["center_y"]

            # Estimate typical character height
            # for this line.
            median_height = np.median(
                [
                    item["h"]
                    for item in line["components"]
                ]
            )

            allowed_distance = max(
                10,
                median_height *
                LINE_VERTICAL_FACTOR
            )

            distance = abs(
                center_y -
                line_center
            )

            if distance <= allowed_distance:

                if distance < best_distance:

                    best_distance = distance
                    assigned_line = line

        if assigned_line is None:

            new_line = {
                "center_y": center_y,
                "components": [
                    component
                ]
            }

            lines.append(
                new_line
            )

        else:

            assigned_line[
                "components"
            ].append(
                component
            )

            # Update line center.
            assigned_line[
                "center_y"
            ] = np.mean(
                [
                    item["center_y"]
                    for item in
                    assigned_line[
                        "components"
                    ]
                ]
            )

    # Sort top → bottom.
    lines.sort(
        key=lambda line: line["center_y"]
    )

    return lines


# =========================================================
# MERGE CLOSE COMPONENTS
# =========================================================

def calculate_word_gap(
    components
):
    """
    Estimate the natural character spacing
    for a handwriting line.
    """

    if len(components) < 2:
        return MIN_WORD_GAP

    sorted_components = sorted(
        components,
        key=lambda item: item["x"]
    )

    gaps = []

    for i in range(
        len(sorted_components) - 1
    ):

        current = sorted_components[i]
        next_item = sorted_components[i + 1]

        current_end = (
            current["x"] +
            current["w"]
        )

        gap = (
            next_item["x"] -
            current_end
        )

        if gap >= 0:
            gaps.append(gap)

    if not gaps:
        return MIN_WORD_GAP

    median_gap = float(
        np.median(gaps)
    )

    word_gap = (
        median_gap *
        WORD_GAP_FACTOR
    )

    word_gap = max(
        MIN_WORD_GAP,
        word_gap
    )

    word_gap = min(
        MAX_WORD_GAP,
        word_gap
    )

    return int(word_gap)


# =========================================================
# GROUP LINE COMPONENTS INTO WORDS
# =========================================================

def group_into_words(
    line_components
):
    """
    Group components horizontally.

    A large gap between components is treated
    as a word boundary.
    """

    if not line_components:
        return []

    components = sorted(
        line_components,
        key=lambda item: item["x"]
    )

    word_gap = calculate_word_gap(
        components
    )

    words = []

    current_word = [
        components[0]
    ]

    current_end = (
        components[0]["x"] +
        components[0]["w"]
    )

    for component in components[1:]:

        gap = (
            component["x"] -
            current_end
        )

        if gap <= word_gap:

            current_word.append(
                component
            )

        else:

            words.append(
                current_word
            )

            current_word = [
                component
            ]

        current_end = max(
            current_end,
            component["x"] +
            component["w"]
        )

    words.append(
        current_word
    )

    return words


# =========================================================
# BOUNDING BOX FOR GROUP
# =========================================================

def group_bbox(
    components,
    image_width,
    image_height
):

    x1 = min(
        item["x"]
        for item in components
    )

    y1 = min(
        item["y"]
        for item in components
    )

    x2 = max(
        item["x"] +
        item["w"]
        for item in components
    )

    y2 = max(
        item["y"] +
        item["h"]
        for item in components
    )

    # Small padding around handwriting.
    padding = 5

    x1 = max(
        0,
        x1 - padding
    )

    y1 = max(
        0,
        y1 - padding
    )

    x2 = min(
        image_width,
        x2 + padding
    )

    y2 = min(
        image_height,
        y2 + padding
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# =========================================================
# PROCESS SAMPLES
# =========================================================

for sample_dir in sample_dirs:

    sample_name = sample_dir.name

    input_path = (
        sample_dir /
        "03_no_lines.png"
    )

    if not input_path.exists():

        print(
            f"Skipping {sample_name}: "
            "03_no_lines.png missing."
        )

        continue

    print("----------------------------------------")
    print(
        f"Processing {sample_name}"
    )
    print("----------------------------------------")

    binary = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if binary is None:

        print(
            "Could not read:",
            input_path
        )

        continue

    # Make sure image is binary.
    _, binary = cv2.threshold(
        binary,
        127,
        255,
        cv2.THRESH_BINARY
    )

    height, width = binary.shape

    # =====================================================
    # FIND COMPONENTS
    # =====================================================

    components = get_components(
        binary
    )

    print(
        "Ink components:",
        len(components)
    )

    # =====================================================
    # GROUP INTO LINES
    # =====================================================

    lines = group_into_lines(
        components
    )

    print(
        "Detected handwriting lines:",
        len(lines)
    )

    # =====================================================
    # OUTPUT DIRECTORIES
    # =====================================================

    sample_output = (
        SEGMENTED_DIR /
        sample_name
    )

    lines_dir = (
        sample_output /
        "lines"
    )

    words_dir = (
        sample_output /
        "words"
    )

    lines_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    words_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # DEBUG IMAGE
    # =====================================================

    debug_image = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )

    metadata = []

    # =====================================================
    # PROCESS EACH LINE
    # =====================================================

    for line_index, line in enumerate(
        lines,
        start=1
    ):

        line_components = sorted(
            line["components"],
            key=lambda item: item["x"]
        )

        # -----------------------------------------
        # Save complete line
        # -----------------------------------------

        x1, y1, x2, y2 = group_bbox(
            line_components,
            width,
            height
        )

        line_image = binary[
            y1:y2,
            x1:x2
        ]

        line_filename = (
            f"line_{line_index:03d}.png"
        )

        cv2.imwrite(
            str(
                lines_dir /
                line_filename
            ),
            line_image
        )

        # Draw BLUE line box.
        cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # -----------------------------------------
        # Detect words inside line
        # -----------------------------------------

        words = group_into_words(
            line_components
        )

        for word_index, word_components in enumerate(
            words,
            start=1
        ):

            wx1, wy1, wx2, wy2 = group_bbox(
                word_components,
                width,
                height
            )

            word_image = binary[
                wy1:wy2,
                wx1:wx2
            ]

            word_filename = (
                f"line_{line_index:03d}"
                f"_word_{word_index:03d}.png"
            )

            cv2.imwrite(
                str(
                    words_dir /
                    word_filename
                ),
                word_image
            )

            # Draw GREEN word box.
            cv2.rectangle(
                debug_image,
                (wx1, wy1),
                (wx2, wy2),
                (0, 255, 0),
                2
            )

            metadata.append(
                {
                    "sample": sample_name,
                    "line": line_index,
                    "word": word_index,
                    "x": wx1,
                    "y": wy1,
                    "width": wx2 - wx1,
                    "height": wy2 - wy1,
                    "filename": word_filename
                }
            )

    # =====================================================
    # SAVE DEBUG IMAGE
    # =====================================================

    debug_path = (
        sample_output /
        "word_detection_debug.png"
    )

    cv2.imwrite(
        str(debug_path),
        debug_image
    )

    # =====================================================
    # SAVE METADATA
    # =====================================================

    metadata_path = (
        sample_output /
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print(
        "Words detected:",
        len(metadata)
    )

    print(
        "Saved:",
        debug_path
    )

    print()


# =========================================================
# COMPLETE
# =========================================================

print("========================================")
print("V3.3 SEGMENTATION COMPLETE")
print("========================================")
print()
print(
    "Output:",
    SEGMENTED_DIR
)
print()