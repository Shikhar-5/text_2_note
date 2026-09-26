import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BACKEND_DIR
    / ".."
    / "Text2Note"
    / "notebook.jpg"
).resolve()

OUTPUT_PATH = (
    BACKEND_DIR
    / ".."
    / "Text2Note"
    / "outputs"
    / "ai_detected_lines_debug.jpg"
).resolve()


# =========================================================
# SEND IMAGE TO AI ENDPOINT
# =========================================================

print("Sending notebook image to AI...")

with open(IMAGE_PATH, "rb") as image_file:

    response = requests.post(
        "http://127.0.0.1:8000/api/detect-lines/",
        files={
            "image": image_file
        }
    )


# =========================================================
# CHECK RESPONSE
# =========================================================

if not response.ok:

    print("AI request failed.")
    print(response.text)
    raise SystemExit


result = response.json()

if not result.get("success"):

    print("AI returned an error.")
    print(result)
    raise SystemExit


# =========================================================
# PARSE AI JSON
# =========================================================

ai_data = result["lines"]

if isinstance(ai_data, str):
    ai_data = json.loads(ai_data)

lines = ai_data.get("lines", [])

print()
print("AI detected", len(lines), "lines.")
print()


# =========================================================
# OPEN ORIGINAL IMAGE
# =========================================================

image = Image.open(IMAGE_PATH).convert("RGB")

width, height = image.size

print("Image size:", width, "x", height)


# =========================================================
# DRAW AI LINES
# =========================================================

draw = ImageDraw.Draw(image)

for line in lines:

    points = line.get("points", [])

    pixel_points = []

    for point in points:

        x = point["x"] * width
        y = point["y"] * height

        pixel_points.append(
            (int(x), int(y))
        )

    # Draw connected line
    if len(pixel_points) >= 2:

        draw.line(
            pixel_points,
            fill=(255, 0, 0),
            width=3
        )

    # Draw points
    for x, y in pixel_points:

        radius = 5

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ),
            fill=(255, 0, 0)
        )


# =========================================================
# SAVE DEBUG IMAGE
# =========================================================

image.save(
    OUTPUT_PATH,
    quality=95
)

print()
print("Debug image created:")
print(OUTPUT_PATH)