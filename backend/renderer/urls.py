from django.urls import path

from .views import (
    health_check,
    render_note,
    get_rendered_image,
    output_pdf,
    ai_test,
    detect_notebook_lines,
    ai_lines,
    detect_font,
    serve_detected_font,
)


urlpatterns = [

    path("health/", health_check),

    path("render/", render_note),

    path("output/", get_rendered_image),

    path("output-pdf/", output_pdf, name="output_pdf"),

    path("ai-test/", ai_test, name="ai_test"),

    path(
        "detect-lines/",
        detect_notebook_lines,
        name="detect_notebook_lines"
    ),

    path(
        "ai-lines/",
        ai_lines,
        name="ai_lines"
    ),

    path(
        "detect-font/",
        detect_font,
        name="detect_font"
    ),
    path(
        "detected-font/<str:filename>/",
        serve_detected_font,
        name="serve_detected_font"
    ),

]