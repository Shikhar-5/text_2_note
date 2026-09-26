"""Upload and inspect a personal handwriting style sample."""

from __future__ import annotations

import uuid
import sys
from pathlib import Path

import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Text2Note.handwriting_engine.style_crops import extract_word_crops
from Text2Note.handwriting_engine.synthesis import model_directory
from Text2Note.handwriting_engine.notebook_geometry import detect_notebook_geometry

PROFILES_DIR = PROJECT_ROOT / "Text2Note" / "handwriting_profiles"


@csrf_exempt
def analyze_handwriting(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Only POST requests are allowed."}, status=405)

    model_root = model_directory()
    if not (model_root / "files" / "iam_model.pth").is_file():
        return JsonResponse(
            {
                "success": False,
                "error": "Personal handwriting model is not installed. Run python scripts/setup_handwriting_model.py.",
            },
            status=503,
        )

    upload = request.FILES.get("image")
    if upload is None:
        return JsonResponse({"success": False, "error": "Handwriting photo is required."}, status=400)
    if upload.size > 12 * 1024 * 1024:
        return JsonResponse({"success": False, "error": "Handwriting photo must be under 12 MB."}, status=400)

    try:
        with Image.open(upload) as opened:
            if opened.width * opened.height > 25_000_000:
                raise ValueError("Handwriting photo is too large; use an image under 25 megapixels.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return JsonResponse({"success": False, "error": str(error)}, status=400)

    style_id = uuid.uuid4().hex
    profile_dir = PROFILES_DIR / style_id
    profile_dir.mkdir(parents=True, exist_ok=False)
    sample_path = profile_dir / "sample.png"
    image.save(sample_path)
    try:
        crops = extract_word_crops(sample_path, profile_dir / "words")
    except ValueError as error:
        return JsonResponse({"success": False, "error": str(error)}, status=422)
    if len(crops) < 5:
        return JsonResponse(
            {
                "success": False,
                "error": "Only a few clear words were found. Upload a page with at least five separated handwritten words.",
            },
            status=422,
        )

    return JsonResponse(
        {
            "success": True,
            "style_id": style_id,
            "reference_count": len(crops),
            "message": "Personal handwriting style is ready.",
        }
    )


@csrf_exempt
def analyze_notebook(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Only POST requests are allowed."}, status=405)
    upload = request.FILES.get("image")
    if upload is None:
        return JsonResponse({"success": False, "error": "Notebook photo is required."}, status=400)
    if upload.size > 12 * 1024 * 1024:
        return JsonResponse({"success": False, "error": "Notebook photo must be under 12 MB."}, status=400)
    try:
        with Image.open(upload) as opened:
            if opened.width * opened.height > 25_000_000:
                raise ValueError("Notebook photo is too large; use an image under 25 megapixels.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
        geometry = detect_notebook_geometry(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return JsonResponse({"success": False, "error": str(error)}, status=422)
    return JsonResponse({"success": True, "geometry": geometry})
