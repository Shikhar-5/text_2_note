import sys
from pathlib import Path
from PIL import Image,ImageDraw, ImageFont
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from django.conf import settings
import base64
import json
import io
import urllib.request
from urllib.request import Request, urlopen


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

client = OpenAI(
    api_key=settings.OPENAI_API_KEY
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# =========================================================
# RENDERING ENGINE
# =========================================================

from Text2Note.rendering.render_engine import render_notebook


# =========================================================
# HEALTH CHECK
# =========================================================

def health_check(request):

    return JsonResponse({
        "status": "ok",
        "message": "Text2Note backend is running!"
    })


# =========================================================
# GENERATED IMAGE
# =========================================================

def get_rendered_image(request):

    output_path = (
        PROJECT_ROOT
        / "Text2Note"
        / "outputs"
        / "rendered_note.jpg"
    )

    if not output_path.exists():

        return JsonResponse(
            {
                "error": "Rendered image not found."
            },
            status=404
        )

    return FileResponse(
        open(output_path, "rb"),
        content_type="image/jpeg"
    )

def ai_lines(request):
    """
    Return the latest V2 notebook line geometry.
    """

    text2note_dir = PROJECT_ROOT / "Text2Note"
    json_path = (
        text2note_dir
        / "outputs"
        / "ai_lines_v2.json"
    )

    if not json_path.exists():
        return JsonResponse(
            {
                "success": False,
                "error": "AI line geometry has not been generated yet."
            },
            status=404
        )

    try:
        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            geometry = json.load(file)

        return JsonResponse(
            {
                "success": True,
                "geometry": geometry
            }
        )

    except Exception as error:

        return JsonResponse(
            {
                "success": False,
                "error": str(error)
            },
            status=500
        )

# ============================================================
# AI HANDWRITING FONT DETECTION
# ============================================================

@csrf_exempt
def detect_font(request):

    print("\n========================================")
    print("FONT MATCH V2 STARTED")
    print("========================================")

    # ----------------------------------------
    # CHECK IMAGE
    # ----------------------------------------

    if request.method != "POST":
        return JsonResponse({
            "success": False,
            "error": "POST request required"
        }, status=405)

    image_file = request.FILES.get("image")

    if not image_file:
        return JsonResponse({
            "success": False,
            "error": "No handwriting image uploaded"
        }, status=400)

    try:

        # ----------------------------------------
        # READ IMAGE
        # ----------------------------------------

        image_bytes = image_file.read()

        if not image_bytes:
            return JsonResponse({
                "success": False,
                "error": "Uploaded image is empty"
            }, status=400)

        # Limit image size
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((1536, 1536))

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Encode image for OpenAI
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="JPEG", quality=90)

        image_base64 = base64.b64encode(
            image_buffer.getvalue()
        ).decode("utf-8")

        image_data_url = (
            "data:image/jpeg;base64,"
            + image_base64
        )

        # ----------------------------------------
        # STEP 1
        # GET HANDWRITING FONTS
        # ----------------------------------------

        print("Loading handwriting font catalog...")

        fonts_url = (
            "https://api.fontsource.org/v1/fonts"
            "?category=handwriting"
            "&type=google"
            "&subsets=latin"
        )

        font_request = urllib.request.Request(
            fonts_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            font_request,
            timeout=20
        ) as response:

            font_catalog = json.loads(
                response.read().decode("utf-8")
            )

        candidates = []

        for font in font_catalog:

            weights = font.get("weights", [])
            styles = font.get("styles", [])

            if 400 not in weights:
                continue

            if "normal" not in styles:
                continue

            candidates.append({
                "id": font.get("id"),
                "family": font.get("family")
            })

        # Remove invalid entries
        candidates = [
            item
            for item in candidates
            if item["id"] and item["family"]
        ]

        print(
            f"Found {len(candidates)} handwriting fonts."
        )

        if not candidates:
            return JsonResponse({
                "success": False,
                "error": "No handwriting fonts found"
            }, status=500)

        # ----------------------------------------
        # STEP 2
        # AI SHORTLIST
        # ----------------------------------------

        print("Asking AI for candidate shortlist...")

        candidate_text = "\n".join(
            f'{item["id"]} | {item["family"]}'
            for item in candidates
        )

        shortlist_prompt = f"""
You are a professional typeface and handwriting matching system.

We need to identify which digital handwriting font most closely
matches the handwriting in the uploaded image.

IMPORTANT:

Do NOT simply choose a font because its name sounds appropriate.

Analyze the actual handwriting:

- letter shapes
- lowercase a
- lowercase e
- lowercase g
- lowercase r
- lowercase t
- lowercase f
- uppercase letters
- ascenders
- descenders
- stroke thickness
- stroke endings
- connected vs disconnected writing
- slant
- width
- spacing
- baseline behavior
- overall rhythm
- rounded vs angular shapes

First transcribe as much readable text from the handwriting as possible.

Then choose the 8 most visually promising fonts from the supplied
candidate list.

The candidate IDs MUST come exactly from the supplied list.

Return ONLY valid JSON:

{{
    "transcription": "text you can read",
    "candidates": [
        {{
            "id": "font-id",
            "family": "Font Family",
            "reason": "why the handwriting may resemble it"
        }}
    ]
}}

Return exactly 8 candidates if possible.

FONT CANDIDATES:

{candidate_text}
"""

        shortlist_response = client.chat.completions.create(

            model="gpt-5.6-luna",

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": shortlist_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
        )

        shortlist_raw = (
            shortlist_response
            .choices[0]
            .message
            .content
        )

        print(
            "AI shortlist response:",
            shortlist_raw
        )

        # ----------------------------------------
        # CLEAN JSON
        # ----------------------------------------

        shortlist_raw = shortlist_raw.strip()

        if shortlist_raw.startswith("```"):
            shortlist_raw = (
                shortlist_raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        shortlist = json.loads(shortlist_raw)

        transcription = (
            shortlist.get("transcription")
            or "The quick brown fox jumps over the lazy dog."
        )

        ai_candidates = shortlist.get(
            "candidates",
            []
        )

        # Validate candidate IDs
        valid_ids = {
            item["id"]: item
            for item in candidates
        }

        shortlist_candidates = []

        for item in ai_candidates:

            font_id = item.get("id")

            if font_id not in valid_ids:
                continue

            shortlist_candidates.append(
                valid_ids[font_id]
            )

        # Remove duplicates
        unique_candidates = []

        seen_ids = set()

        for item in shortlist_candidates:

            if item["id"] in seen_ids:
                continue

            seen_ids.add(item["id"])

            unique_candidates.append(item)

        shortlist_candidates = unique_candidates[:8]

        print(
            "AI shortlisted:",
            [
                item["family"]
                for item in shortlist_candidates
            ]
        )

        if not shortlist_candidates:
            return JsonResponse({
                "success": False,
                "error": "AI could not create a font shortlist"
            }, status=500)

        # ----------------------------------------
        # STEP 3
        # RENDER REAL FONT SPECIMENS
        # ----------------------------------------

        print("Rendering actual font specimens...")

        specimens = []

        specimen_text = transcription.strip()

        # Keep specimen manageable
        if len(specimen_text) > 120:
            specimen_text = specimen_text[:120]

        for candidate in shortlist_candidates:

            font_id = candidate["id"]
            family = candidate["family"]

            try:

                detail_url = (
                    "https://api.fontsource.org/v1/fonts/"
                    + font_id
                )

                detail_request = urllib.request.Request(
                    detail_url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                with urllib.request.urlopen(
                    detail_request, 
                    timeout=20
                ) as response:

                    detail = json.loads(
                        response.read().decode("utf-8")
                    )

                ttf_url = None

                variants = detail.get(
                    "variants",
                    {}
                )

                # Fontsource structure:
                # variants -> weight -> style -> subset -> url -> ttf

                weight_data = variants.get(
                    "400",
                    {}
                )

                normal_data = weight_data.get(
                    "normal",
                    {}
                )

                latin_data = normal_data.get(
                    "latin",
                    {}
                )

                url_data = latin_data.get(
                    "url",
                    {}
                )

                ttf_url = url_data.get("ttf")

                if not ttf_url:
                    print(
                        f"No TTF found for {family}"
                    )
                    continue

                # Download actual font
                ttf_request = urllib.request.Request(
                    ttf_url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                with urllib.request.urlopen(
                    ttf_request,
                    timeout=30
                ) as response:

                    font_bytes = response.read()

                pil_font = ImageFont.truetype(
                    io.BytesIO(font_bytes),
                    48
                )

                # ----------------------------------------
                # CREATE SPECIMEN IMAGE
                # ----------------------------------------

                canvas_width = 900
                canvas_height = 150

                specimen = Image.new(
                    "RGB",
                    (
                        canvas_width,
                        canvas_height
                    ),
                    "white"
                )

                draw = ImageDraw.Draw(
                    specimen
                )

                # Font label
                draw.text(
                    (20, 10),
                    family,
                    fill=(120, 120, 120)
                )

                # Actual rendered text
                draw.text(
                    (20, 55),
                    specimen_text,
                    font=pil_font,
                    fill=(20, 20, 20)
                )

                specimens.append({
                    "id": font_id,
                    "family": family,
                    "image": specimen
                })

                print(
                    f"Rendered specimen: {family}"
                )

            except Exception as font_error:

                print(
                    f"Could not render {family}:",
                    font_error
                )

        if not specimens:

            return JsonResponse({
                "success": False,
                "error": "Could not render candidate fonts"
            }, status=500)

        # ----------------------------------------
        # STEP 4
        # CREATE CONTACT SHEET
        # ----------------------------------------

        print("Creating font comparison sheet...")

        columns = 2
        rows = (
            (len(specimens) + columns - 1)
            // columns
        )

        cell_width = 900
        cell_height = 150

        contact_sheet = Image.new(
            "RGB",
            (
                cell_width * columns,
                cell_height * rows
            ),
            "white"
        )

        for index, specimen in enumerate(
            specimens
        ):

            x = (
                index % columns
            ) * cell_width

            y = (
                index // columns
            ) * cell_height

            contact_sheet.paste(
                specimen["image"],
                (x, y)
            )

        # Encode contact sheet
        sheet_buffer = io.BytesIO()

        contact_sheet.save(
            sheet_buffer,
            format="JPEG",
            quality=90
        )

        sheet_base64 = base64.b64encode(
            sheet_buffer.getvalue()
        ).decode("utf-8")

        sheet_data_url = (
            "data:image/jpeg;base64,"
            + sheet_base64
        )

        # ----------------------------------------
        # STEP 5
        # VISUAL GLYPH COMPARISON
        # ----------------------------------------

        print(
            "AI is comparing actual rendered glyphs..."
        )

        comparison_prompt = f"""
You are the final stage of a handwriting-to-font matching system.

The first image is the ORIGINAL HANDWRITING.

The second image is a CONTACT SHEET containing real rendered
samples from candidate fonts.

Each candidate was rendered using the EXACT SAME TEXT extracted
from the handwriting.

This is extremely important:

Do NOT judge based on font names.

Compare the actual glyph shapes.

Look especially at:

- lowercase a
- lowercase e
- lowercase g
- lowercase r
- lowercase t
- lowercase f
- uppercase letters
- loops
- joins
- stroke endings
- ascenders
- descenders
- letter width
- character spacing
- slant
- stroke thickness
- rounded/angular construction
- overall handwritten rhythm

The contact sheet labels each specimen with its font family.

Choose the candidate whose ACTUAL GLYPH SHAPES are closest to the
original handwriting.

Return ONLY valid JSON:

{{
    "ranking": [
        {{
            "id": "font-id",
            "family": "Font Family",
            "score": 0,
            "reason": "specific visual comparison"
        }}
    ]
}}

Return the candidates from closest to least similar.

Score should be between 0 and 100.

Do not invent font IDs.
"""

        comparison_response = client.chat.completions.create(

            model="gpt-5.6-luna",

            messages=[
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "text",
                            "text": comparison_prompt
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": sheet_data_url
                            }
                        }

                    ]
                }
            ],

        )

        comparison_raw = (
            comparison_response
            .choices[0]
            .message
            .content
        )

        print(
            "AI comparison response:",
            comparison_raw
        )

        comparison_raw = (
            comparison_raw
            .strip()
        )

        if comparison_raw.startswith("```"):
            comparison_raw = (
                comparison_raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        comparison = json.loads(
            comparison_raw
        )

        ranking = comparison.get(
            "ranking",
            []
        )

        # ----------------------------------------
        # STEP 6
        # VALIDATE WINNER
        # ----------------------------------------

        valid_candidate_map = {
            item["id"]: item
            for item in specimens
        }

        winner = None

        for result in ranking:

            font_id = result.get("id")

            if font_id in valid_candidate_map:

                winner = {
                    "id": font_id,
                    "family": result.get(
                        "family",
                        valid_candidate_map[
                            font_id
                        ]["family"]
                    ),
                    "score": float(
                        result.get(
                            "score",
                            0
                        )
                    ),
                    "reason": result.get(
                        "reason",
                        ""
                    )
                }

                break

        if not winner:

            # Fallback to first successfully
            # rendered candidate

            first = specimens[0]

            winner = {
                "id": first["id"],
                "family": first["family"],
                "score": 50,
                "reason": "Fallback candidate"
            }

        print(
            "FINAL FONT:",
            winner["family"]
        )

        print(
            "FINAL SCORE:",
            winner["score"]
        )

        # ----------------------------------------
        # STEP 7
        # DOWNLOAD WINNING TTF
        # ----------------------------------------

        winning_id = winner["id"]

        detail_url = (
            "https://api.fontsource.org/v1/fonts/"
            + winning_id
        )

        detail_request = urllib.request.Request(
            detail_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            detail_request,
            timeout=20
        ) as response:

            winning_detail = json.loads(
                response.read().decode("utf-8")
            )

        winning_ttf_url = (
            winning_detail
            .get("variants", {})
            .get("400", {})
            .get("normal", {})
            .get("latin", {})
            .get("url", {})
            .get("ttf")
        )

        if not winning_ttf_url:

            return JsonResponse({
                "success": False,
                "error": "Winning font TTF could not be found"
            }, status=500)

        winning_ttf_request = urllib.request.Request(
            winning_ttf_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            winning_ttf_request,
            timeout=30
        ) as response:

            winning_font_bytes = response.read()

        uploaded_fonts_dir = (
            PROJECT_ROOT
            / "backend"
            / "uploaded_fonts"
        )

        uploaded_fonts_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        font_filename = (
            f"detected_{winning_id}.ttf"
        )

        font_path = (
            uploaded_fonts_dir
            / font_filename
        )

        with open(
            font_path,
            "wb"
        ) as font_file:

            font_file.write(
                winning_font_bytes
            )

        # ----------------------------------------
        # SUCCESS
        # ----------------------------------------

        print(
            "Saved detected font:",
            font_path
        )

        print(
            "========================================"
        )
        print(
            "FONT MATCH V2 FINISHED"
        )
        print(
            "========================================"
        )

        return JsonResponse({

            "success": True,

            "font_id":
                winner["id"],

            "font_family":
                winner["family"],

            "confidence":
                round(
                    winner["score"] / 100,
                    2
                ),

            "reason":
                winner["reason"],

            "font_filename":
                font_filename,

            "transcription":
                transcription,

            "ranking":
                ranking[:3]

        })

    except Exception as error:

        print(
            "FONT MATCH V2 ERROR:",
            repr(error)
        )

        return JsonResponse({

            "success": False,

            "error":
                str(error)

        }, status=500)
    
# =========================================================
# SERVE DETECTED FONT TO FRONTEND
# =========================================================

def serve_detected_font(request, filename):

    fonts_directory = (
        PROJECT_ROOT
        / "backend"
        / "uploaded_fonts"
    )

    # Prevent path traversal
    safe_filename = Path(filename).name

    font_path = (
        fonts_directory
        / safe_filename
    )

    if not font_path.exists():
        return JsonResponse(
            {
                "success": False,
                "error": "Font file not found."
            },
            status=404
        )

    return FileResponse(
        open(font_path, "rb"),
        content_type="font/ttf"
    )

# =========================================================
# RENDER API
# =========================================================

@csrf_exempt
def render_note(request):

    if request.method != "POST":

        return JsonResponse(
            {
                "error": "Only POST requests are allowed."
            },
            status=405
        )

    try:

        # =================================================
        # GET TEXT SETTINGS
        # =================================================

        text = request.POST.get(
            "text",
            ""
        )

        font_size = int(
            request.POST.get(
                "font_size",
                22
            )
        )

        font_color = request.POST.get(
            "font_color",
            "#202020"
        )


        # =================================================
        # GET UPLOADED IMAGE
        # =================================================

        uploaded_image = request.FILES.get(
            "image"
        )

        if not uploaded_image:

            return JsonResponse(
                {
                    "error": "Notebook image is required."
                },
                status=400
            )


        # =================================================
        # VALIDATE TEXT
        # =================================================

        if not text.strip():

            return JsonResponse(
                {
                    "error": "Text is required."
                },
                status=400
            )


        # =================================================
        # CONVERT HEX COLOR TO RGBA
        # =================================================

        font_color = font_color.lstrip("#")

        if len(font_color) != 6:

            return JsonResponse(
                {
                    "error": "Invalid font color."
                },
                status=400
            )


        r = int(
            font_color[0:2],
            16
        )

        g = int(
            font_color[2:4],
            16
        )

        b = int(
            font_color[4:6],
            16
        )


        rgba_color = (
            r,
            g,
            b,
            255
        )


        # =================================================
        # DIRECTORIES
        # =================================================

        text2note_dir = (
            PROJECT_ROOT
            / "Text2Note"
        )

        uploads_dir = (
            text2note_dir
            / "uploads"
        )

        outputs_dir = (
            text2note_dir
            / "outputs"
        )

        fonts_dir = (
            PROJECT_ROOT
            / "backend"
            / "uploaded_fonts"
        )


        # Create directories if they don't exist

        uploads_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        outputs_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fonts_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # =================================================
        # SAVE UPLOADED IMAGE
        # =================================================

        image_path = (
            uploads_dir
            / uploaded_image.name
        )

        with open(
            image_path,
            "wb+"
        ) as destination:

            for chunk in uploaded_image.chunks():

                destination.write(chunk)


        # =================================================
        # LINE CALIBRATION DATA
        # =================================================

        line_data_path = (
            text2note_dir
            / "line_data_5point.json"
        )

        if not line_data_path.exists():

            return JsonResponse(
                {
                    "error": "5-point line data not found."
                },
                status=500
            )


        # =================================================
        # OUTPUT FILE
        # =================================================

        output_path = (
            outputs_dir
            / "rendered_note.jpg"
        )


        font_name = request.POST.get("font", "Caveat")

        # ---------------------------------------------------------
        # FONT SELECTION
        # ---------------------------------------------------------

        # 1. Check whether the frontend selected an AI-detected font.
        detected_font_filename = request.POST.get(
            "detected_font_filename",
            ""
        ).strip()


        if detected_font_filename:

            # Prevent path traversal.
            safe_filename = Path(
                detected_font_filename
            ).name

            detected_font_path = (
                PROJECT_ROOT
                / "backend"
                / "uploaded_fonts"
                / safe_filename
            )

            if not detected_font_path.exists():

                return JsonResponse(
                    {
                        "success": False,
                        "error": (
                            "AI detected font file not found: "
                            f"{safe_filename}"
                        )
                    },
                    status=400
                )

            # Use the downloaded AI font.
            font_path = detected_font_path

            print(
                "Using AI detected font:",
                font_name
            )

            print(
                "AI font file:",
                font_path
            )


        else:

            # -----------------------------------------------------
            # 2. Custom uploaded font
            # -----------------------------------------------------

            custom_font = request.FILES.get("font_file")

            if custom_font:

                uploaded_fonts_dir = (
                    PROJECT_ROOT
                    / "backend"
                    / "uploaded_fonts"
                )

                uploaded_fonts_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                custom_font_path = (
                    uploaded_fonts_dir
                    / custom_font.name
                )

                with open(
                    custom_font_path,
                    "wb"
                ) as destination:

                    for chunk in custom_font.chunks():
                        destination.write(chunk)

                font_path = custom_font_path


            else:

                # -------------------------------------------------
                # 3. Built-in Text2Note fonts
                # -------------------------------------------------

                font_map = {

                    "Caveat":
                        PROJECT_ROOT
                        / "Text2Note"
                        / "fonts"
                        / "Caveat-Regular.ttf",

                    "Kalam":
                        PROJECT_ROOT
                        / "Text2Note"
                        / "fonts"
                        / "Kalam-Regular.ttf",

                    "Patrick Hand":
                        PROJECT_ROOT
                        / "Text2Note"
                        / "fonts"
                        / "PatrickHand-Regular.ttf",

                    "Dancing Script":
                        PROJECT_ROOT
                        / "Text2Note"
                        / "fonts"
                        / "DancingScript-Regular.ttf",
                }

                if font_name not in font_map:

                    return JsonResponse(
                        {
                            "success": False,
                            "error": (
                                f"Unsupported font: {font_name}"
                            )
                        },
                        status=400
                    )

                font_path = font_map[font_name]

        # =================================================
        # CALL RENDERING ENGINE
        # =================================================

        render_notebook(
            image_path=str(image_path),
            text=text,
            font_path=str(font_path),
            font_size=font_size,
            font_color=rgba_color,
            line_data_path=str(line_data_path),
            output_path=str(output_path),
            baseline_offset=-2,
            left_margin=15
        )
        # Create PDF version of the generated note
        pdf_path = outputs_dir / "rendered_note.pdf"

        rendered_image = Image.open(output_path).convert("RGB")
        rendered_image.save(pdf_path, "PDF", resolution=100.0)


        # =================================================
        # SUCCESS
        # =================================================

        return JsonResponse({

            "status": "success",

            "success": True,

            "message":
                "Note rendered successfully.",

            "output":
                str(output_path),

            "font":
                str(font_path)

        })


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print(
            "RENDER ERROR:",
            repr(e)
        )

        return JsonResponse(
            {
                "status": "error",
                "success": False,
                "error": str(e)
            },
            status=500
        )

def output_pdf(request):
    text2note_dir = PROJECT_ROOT / "Text2Note"
    outputs_dir = text2note_dir / "outputs"

    pdf_path = outputs_dir / "rendered_note.pdf"

    if not pdf_path.exists():
        return JsonResponse(
            {
                "error": "Generated note is not available yet."
            },
            status=404
        )

    return FileResponse(
        open(pdf_path, "rb"),
        content_type="application/pdf",
        as_attachment=False,
        filename="Text2Note-Handwritten-Note.pdf"
    )
def ai_test(request):
    try:
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        response = client.responses.create(
            model="gpt-5.6-luna",
            input="Reply with exactly: Text2Note AI connection successful."
        )

        return JsonResponse({
            "success": True,
            "message": response.output_text
        })

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)

@csrf_exempt
def detect_notebook_lines(request):

    if request.method != "POST":
        return JsonResponse(
            {
                "success": False,
                "error": "Only POST requests are allowed."
            },
            status=405
        )

    try:

        # =====================================================
        # GET IMAGE
        # =====================================================

        uploaded_image = request.FILES.get("image")

        if not uploaded_image:
            return JsonResponse(
                {
                    "success": False,
                    "error": "Notebook image is required."
                },
                status=400
            )

        # =====================================================
        # CHECK IMAGE TYPE
        # =====================================================

        image_name = uploaded_image.name.lower()

        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]

        if not any(
            image_name.endswith(ext)
            for ext in allowed_extensions
        ):
            return JsonResponse(
                {
                    "success": False,
                    "error": "Please upload JPG, JPEG, PNG or WEBP."
                },
                status=400
            )

        # =====================================================
        # READ IMAGE
        # =====================================================

        image_bytes = uploaded_image.read()

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        # =====================================================
        # DETERMINE MIME TYPE
        # =====================================================

        if image_name.endswith(".png"):
            mime_type = "image/png"

        elif image_name.endswith(".webp"):
            mime_type = "image/webp"

        else:
            mime_type = "image/jpeg"

        # =====================================================
        # OPENAI
        # =====================================================

        client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        # =====================================================
        # ASK AI TO DETECT NOTEBOOK LINES
        # =====================================================

        prompt = """
You are analyzing a photograph of a physical notebook page.

Your task is to detect the visible horizontal writing lines
on the notebook page.

IMPORTANT:

1. Detect the actual notebook ruling/writing lines.
2. Ignore the page borders.
3. Ignore shadows and table/background edges.
4. Ignore handwriting, printed text, drawings and objects.
5. The lines may be curved because of perspective or the
   physical bending of the notebook.
6. Represent every detected line as a smooth sequence of
   points.
7. Coordinates must be NORMALIZED from 0 to 1.
8. x=0 means the left edge of the image.
9. x=1 means the right edge of the image.
10. y=0 means the top edge of the image.
11. y=1 means the bottom edge of the image.
12. For every line, provide exactly 7 points from left to right.
13. The points should follow the center of the actual ruling line.
14. Detect as many clear writing lines as possible.
15. Do not invent lines that are not visible.

Return ONLY valid JSON.

Use this exact structure:

{
    "image_width": 1,
    "image_height": 1,
    "lines": [
        {
            "line_number": 1,
            "points": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.16, "y": 0.0},
                {"x": 0.33, "y": 0.0},
                {"x": 0.50, "y": 0.0},
                {"x": 0.66, "y": 0.0},
                {"x": 0.83, "y": 0.0},
                {"x": 1.0, "y": 0.0}
            ]
        }
    ]
}
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime_type};base64,"
                                f"{encoded_image}"
                            ),
                            "detail": "high"
                        }
                    ]
                }
            ]
        )

        # =====================================================
        # GET AI RESPONSE
        # =====================================================

        ai_output = response.output_text

        return JsonResponse(
            {
                "success": True,
                "lines": ai_output
            }
        )

    except Exception as e:

        print(
            "LINE DETECTION ERROR:",
            repr(e)
        )

        return JsonResponse(
            {
                "success": False,
                "error": str(e)
            },
            status=500
        )