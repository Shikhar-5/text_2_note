from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


# =========================================
# SETTINGS
# =========================================

IMAGE_PATH = "Text2Note/notebook.jpg"

OUTPUT_PATH = (
    "Text2Note/text_rendered.jpg"
)

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

FONT_SIZE = 32

TEXT = (
    "Hello, this is Text2Note!"
)


# =========================================
# LOAD IMAGE
# =========================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

draw = ImageDraw.Draw(image)

width, height = image.size

print(
    f"Image loaded: "
    f"{width} x {height}"
)


# =========================================
# LOAD FONT
# =========================================

try:

    font = ImageFont.truetype(
        FONT_PATH,
        FONT_SIZE
    )

except:

    print(
        "Could not load the font."
    )

    print(
        "Using default font instead."
    )

    font = ImageFont.load_default()


# =========================================
# TEST LINE
# =========================================

# We're deliberately using ONE line
# first.
#
# You will change these coordinates
# after seeing the result.

x = 80
y = 150


# =========================================
# DRAW TEXT
# =========================================

draw.text(
    (x, y),
    TEXT,
    font=font,
    fill=(40, 40, 40)
)


# =========================================
# SAVE
# =========================================

image.save(
    OUTPUT_PATH,
    quality=95
)

print(
    f"Saved: {OUTPUT_PATH}"
)