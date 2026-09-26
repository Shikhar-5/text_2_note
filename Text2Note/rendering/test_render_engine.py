from Text2Note.rendering.render_engine import render_notebook

# =========================================================
# TEST SETTINGS
# =========================================================

IMAGE_PATH = "Text2Note/notebook.jpg"

TEXT = (
    "Introduction to Biology. Biology is the scientific study "
    "of life and living organisms. Living organisms are made up "
    "of cells, which are considered the basic structural and "
    "functional units of life. Cells can be classified into "
    "prokaryotic and eukaryotic cells. The nucleus stores "
    "genetic material called DNA, while mitochondria help "
    "produce energy for cellular activities. Plants use "
    "photosynthesis to produce food using sunlight, carbon "
    "dioxide, and water. Cellular respiration releases "
    "energy from glucose."
)

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

FONT_SIZE = 22

FONT_COLOR = (
    30,   # Red
    30,   # Green
    30,   # Blue
    255   # Alpha
)

LINE_DATA_PATH = (
    "Text2Note/line_data_5point.json"
)

OUTPUT_PATH = (
    "Text2Note/engine_test.jpg"
)


# =========================================================
# RUN RENDER ENGINE
# =========================================================

output = render_notebook(
    image_path=IMAGE_PATH,
    text=TEXT,
    font_path=FONT_PATH,
    font_size=FONT_SIZE,
    font_color=FONT_COLOR,
    line_data_path=LINE_DATA_PATH,
    output_path=OUTPUT_PATH,
    baseline_offset=-2,
    left_margin=15
)


# =========================================================
# RESULT
# =========================================================

print()
print("Rendering completed!")
print(f"Output: {output}")