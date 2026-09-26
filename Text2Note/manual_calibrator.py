from pathlib import Path
import json
import cv2


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "handwriting_samples"
    / "sample_04.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "handwriting_dataset"
    / "personal_library"
)

BOXES_PATH = (
    OUTPUT_DIR
    / "calibration_boxes.json"
)


# =========================================================
# CHARACTERS TO CALIBRATE
# =========================================================

CHARACTERS = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not load:\n{IMAGE_PATH}"
    )


original = image.copy()

# Scale the displayed image so it fits on screen.
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800

height, width = image.shape[:2]

scale = min(
    MAX_DISPLAY_WIDTH / width,
    MAX_DISPLAY_HEIGHT / height,
    1.0
)

display_width = int(
    width * scale
)

display_height = int(
    height * scale
)

display_image = cv2.resize(
    image,
    (
        display_width,
        display_height
    )
)


# =========================================================
# STATE
# =========================================================

current_index = 0

boxes = {}

drawing = False

start_x = 0
start_y = 0

current_box = None


# =========================================================
# WINDOW
# =========================================================

WINDOW_NAME = (
    "Text2Note Manual Calibrator"
)


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


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

    global drawing
    global start_x
    global start_y
    global current_box

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        current_box = (
            x,
            y,
            x,
            y
        )

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            current_box = (
                start_x,
                start_y,
                x,
                y
            )

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        current_box = (
            start_x,
            start_y,
            x,
            y
        )


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# =========================================================
# HELPERS
# =========================================================

def normalize_box(
    box
):

    x1, y1, x2, y2 = box

    left = min(
        x1,
        x2
    )

    right = max(
        x1,
        x2
    )

    top = min(
        y1,
        y2
    )

    bottom = max(
        y1,
        y2
    )

    return (
        left,
        top,
        right,
        bottom
    )


def display_to_original(
    box
):

    x1, y1, x2, y2 = normalize_box(
        box
    )

    return (
        int(x1 / scale),
        int(y1 / scale),
        int(x2 / scale),
        int(y2 / scale)
    )


def save_current_box():

    global current_box

    if current_box is None:
        return False

    left, top, right, bottom = normalize_box(
        current_box
    )

    if (
        right - left < 3
        or
        bottom - top < 3
    ):
        return False

    char = CHARACTERS[current_index]

    x1, y1, x2, y2 = display_to_original(
        current_box
    )

    # Small padding around the handwriting.
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
        original.shape[1],
        x2 + padding
    )

    y2 = min(
        original.shape[0],
        y2 + padding
    )

    boxes[char] = {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1
    }

    print(
        f"Saved box for '{char}':",
        boxes[char]
    )

    return True


def save_json():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        BOXES_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            boxes,
            file,
            indent=2
        )


def save_all_glyphs():

    uppercase_dir = (
        OUTPUT_DIR /
        "uppercase"
    )

    lowercase_dir = (
        OUTPUT_DIR /
        "lowercase"
    )

    digits_dir = (
        OUTPUT_DIR /
        "digits"
    )

    uppercase_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    lowercase_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    digits_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for char, box in boxes.items():

        x = box["x"]
        y = box["y"]

        w = box["width"]
        h = box["height"]

        crop = original[
            y:y + h,
            x:x + w
        ]

        if crop.size == 0:
            continue

        if char.isupper():

            destination = (
                uppercase_dir /
                f"{char}.png"
            )

        elif char.islower():

            destination = (
                lowercase_dir /
                f"{char}.png"
            )

        else:

            destination = (
                digits_dir /
                f"{char}.png"
            )

        cv2.imwrite(
            str(destination),
            crop
        )


# =========================================================
# MAIN LOOP
# =========================================================

print()
print("========================================")
print("TEXT2NOTE MANUAL CALIBRATOR")
print("========================================")
print()
print(
    "Total characters:",
    len(CHARACTERS)
)
print()
print("Controls:")
print()
print("  Drag mouse  = draw box")
print("  ENTER       = save current character")
print("  R           = redo current character")
print("  BACKSPACE   = go back one character")
print("  ESC         = save progress and exit")
print()
print(
    "IMPORTANT:"
)
print(
    "Draw a box around ONE character only."
)
print()


while True:

    canvas = display_image.copy()

    # -----------------------------------------
    # Header
    # -----------------------------------------

    if current_index < len(
        CHARACTERS
    ):

        current_char = CHARACTERS[
            current_index
        ]

        status = (
            f"Character "
            f"{current_index + 1}/"
            f"{len(CHARACTERS)}"
            f"   ->   '{current_char}'"
        )

    else:

        status = (
            "ALL CHARACTERS COMPLETE"
        )

    cv2.putText(
        canvas,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        "Drag box | ENTER=save | "
        "R=redo | BACKSPACE=back | ESC=finish",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    # -----------------------------------------
    # Current box
    # -----------------------------------------

    if current_box is not None:

        x1, y1, x2, y2 = normalize_box(
            current_box
        )

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )

    key = cv2.waitKey(20) & 0xFF

    # -----------------------------------------
    # ENTER = save
    # -----------------------------------------

    if key in (
        13,
        10
    ):

        if current_index >= len(
            CHARACTERS
        ):
            continue

        if save_current_box():

            save_json()

            current_index += 1

            current_box = None

            if current_index >= len(
                CHARACTERS
            ):

                save_all_glyphs()

                print()
                print(
                    "========================================"
                )
                print(
                    "ALL CALIBRATION CHARACTERS SAVED"
                )
                print(
                    "========================================"
                )
                print()

                break

    # -----------------------------------------
    # R = redo
    # -----------------------------------------

    elif key in (
        ord("r"),
        ord("R")
    ):

        current_box = None

    # -----------------------------------------
    # BACKSPACE = previous
    # -----------------------------------------

    elif key in (
        8,
        127
    ):

        if current_index > 0:

            current_index -= 1

            previous_char = CHARACTERS[
                current_index
            ]

            boxes.pop(
                previous_char,
                None
            )

            current_box = None

            print(
                "Going back to:",
                previous_char
            )

    # -----------------------------------------
    # ESC = save and exit
    # -----------------------------------------

    elif key == 27:

        save_json()

        print()
        print(
            "Progress saved."
        )

        break


cv2.destroyAllWindows()

save_json()

save_all_glyphs()

print()
print(
    "Calibration data:"
)
print(
    BOXES_PATH
)
print()