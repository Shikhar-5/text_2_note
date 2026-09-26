/* ==========================================
   TEXT2NOTE - FRONTEND MVP
========================================== */

/* ---------- Upload Elements ---------- */
let currentX = 120;
let currentY = 120;

const uploadBox =
    document.getElementById("uploadBox");

const imageInput =
    document.getElementById("imageInput");

const browseButton =
    document.getElementById("browseButton");

const imagePreview =
    document.getElementById("imagePreview");

const previewImage =
    document.getElementById("previewImage");

const replaceButton =
    document.getElementById("replaceButton");

const removeButton =
    document.getElementById("removeButton");


/* ---------- Text ---------- */

const textInput =
    document.getElementById("textInput");

const wordCount =
    document.getElementById("wordCount");

const characterCount =
    document.getElementById("characterCount");


/* ---------- Controls ---------- */

const fontSelect =
    document.getElementById("fontSelect");

const handwritingInput =
    document.getElementById("handwritingInput");

const handwritingUploadButton =
    document.getElementById("handwritingUploadButton");

const handwritingStatus =
    document.getElementById("handwritingStatus");

const fontSize =
    document.getElementById("fontSize");

const fontSizeValue =
    document.getElementById("fontSizeValue");

const fontColor =
    document.getElementById("fontColor");

const colorValue =
    document.getElementById("colorValue");

const lineSpacing =
    document.getElementById("lineSpacing");

const rotation =
    document.getElementById("rotation");


/* ---------- Placement ---------- */

const placementOptions =
    document.querySelectorAll(".placement-option");


/* ---------- Generate ---------- */

const generateButton =
    document.getElementById("generateButton");


/* ---------- Preview ---------- */

const notebookCanvas =
    document.getElementById("notebookPreview");

const downloadButton =
    document.querySelector(".download-button");


/* ==========================================
   STATE
========================================== */

let uploadedImage = null;
let uploadedImageFile = null;

let customFontFile = null;

let placementMode = "automatic";

// Personal handwriting profile
let handwritingStyleId = null;

/* ==========================================
   IMAGE UPLOAD
========================================== */

browseButton?.addEventListener("click", () => {
    imageInput.click();
});


replaceButton?.addEventListener("click", () => {
    imageInput.click();
});


imageInput?.addEventListener("change", (e) => {

    const file =
        e.target.files[0];

    if (!file) return;

    loadImage(file);
});


function loadImage(file) {

    // Keep the actual File object
    // so we can send it to Django later.
    uploadedImageFile = file;
    aiGeometry = null;

    const reader =
        new FileReader();

    reader.onload = function (event) {

        // This is only for displaying the preview.
        uploadedImage =
            event.target.result;

        previewImage.src =
            uploadedImage;

        if (uploadBox) {
            uploadBox.style.display =
                "none";
        }

        if (imagePreview) {
            imagePreview.hidden =
                false;
        }

        updatePreview();
        loadAILines();
    };

    reader.readAsDataURL(file);
}

/* ==========================================
   PERSONAL HANDWRITING ANALYSIS
========================================== */

async function detectHandwritingFont(file) {
    if (!file) return;

    handwritingStyleId = null;
    if (handwritingStatus) {
        handwritingStatus.textContent = "Extracting words from your handwriting sample...";
    }
    if (handwritingUploadButton) {
        handwritingUploadButton.disabled = true;
        handwritingUploadButton.textContent = "Analyzing handwriting...";
    }

    try {
        const formData = new FormData();
        formData.append("image", file);
        const response = await fetch(
            "http://127.0.0.1:8000/api/analyze-handwriting/",
            { method: "POST", body: formData }
        );
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not analyze handwriting.");
        }

        handwritingStyleId = data.style_id;
        if (fontSelect) {
            let option = fontSelect.querySelector('option[value="personal"]');
            if (!option) {
                option = document.createElement("option");
                option.value = "personal";
                option.textContent = "My handwriting";
                fontSelect.prepend(option);
            }
            fontSelect.value = "personal";
        }
        if (handwritingStatus) {
            handwritingStatus.textContent =
                `Ready: ${data.reference_count} handwritten word samples. Generate to see your result.`;
        }
        updatePreview();
    } catch (error) {
        console.error("Handwriting analysis error:", error);
        handwritingStyleId = null;
        if (handwritingStatus) {
            handwritingStatus.textContent =
                "Could not extract enough handwriting. Try a clearer sample.";
        }
        alert("Handwriting analysis failed:\n\n" + error.message);
    } finally {
        if (handwritingUploadButton) {
            handwritingUploadButton.disabled = false;
            handwritingUploadButton.textContent = "✍ Upload handwriting sample";
        }
    }
}
/* ==========================================
   HANDWRITING SAMPLE UPLOAD
========================================== */

handwritingUploadButton?.addEventListener(
    "click",
    () => {

        handwritingInput?.click();

    }
);


handwritingInput?.addEventListener(
    "change",
    () => {

        const file =
            handwritingInput.files[0];

        if (!file) {
            return;
        }

        console.log(
            "Handwriting sample selected:",
            file.name
        );

        detectHandwritingFont(file);

    }
);


/* ==========================================
   REMOVE IMAGE
========================================== */

removeButton?.addEventListener("click", () => {

    uploadedImage = null;
    uploadedImageFile = null;
    aiGeometry = null;

    imageInput.value = "";

    previewImage.src = "";

    imagePreview.hidden = true;

    uploadBox.style.display = "flex";

    updatePreview();
});


/* ==========================================
   DRAG & DROP
========================================== */

uploadBox?.addEventListener("dragover", (e) => {

    e.preventDefault();

    uploadBox.classList.add("dragging");
});


uploadBox?.addEventListener("dragleave", () => {

    uploadBox.classList.remove("dragging");
});


uploadBox?.addEventListener("drop", (e) => {

    e.preventDefault();

    uploadBox.classList.remove("dragging");

    const file =
        e.dataTransfer.files[0];

    if (!file) return;

    loadImage(file);
});


/* ==========================================
   WORD + CHARACTER COUNTER
========================================== */

textInput?.addEventListener("input", () => {

    const text =
        textInput.value.trim();

    const words =
        text.length > 0
            ? text.split(/\s+/).length
            : 0;

    if (wordCount) {

        wordCount.textContent =
            `${words} words`;
    }

    if (characterCount) {

        characterCount.textContent =
            `${text.length} characters`;
    }

    updatePreview();
});


/* ==========================================
   FONT SIZE
========================================== */

fontSize?.addEventListener("input", () => {

    if (fontSizeValue) {

        fontSizeValue.textContent =
            `${fontSize.value}px`;
    }

    updatePreview();
});


/* ==========================================
   COLOR PICKER
========================================== */

fontColor?.addEventListener("input", () => {

    if (colorValue) {

        colorValue.textContent =
            fontColor.value;
    }

    updatePreview();
});


/* ==========================================
   OTHER CONTROLS
========================================== */

fontSelect?.addEventListener(
    "change",
    updatePreview
);


lineSpacing?.addEventListener(
    "input",
    updatePreview
);


rotation?.addEventListener(
    "input",
    updatePreview
);


/* ==========================================
   PLACEMENT SELECTOR
========================================== */

placementOptions.forEach(option => {

    option.addEventListener("click", () => {

        placementOptions.forEach(item => {

            item.classList.remove("active");
        });

        option.classList.add("active");

        placementMode =
            option.dataset.mode ||
            "automatic";

        updatePreview();
    });
});


/* ==========================================
   LIVE NOTEBOOK PREVIEW
========================================== */

function updatePreview() {

    if (!notebookCanvas) {
        return;
    }

    // Show the AI editor when a notebook image exists.
    if (uploadedImage) {

        // Hide the empty placeholder.
        const canvasEmpty =
            document.getElementById("canvasEmpty");

        if (canvasEmpty) {
            canvasEmpty.hidden = true;
        }

        // Show the editor.
        const editorContainer =
            document.getElementById("editorContainer");

        if (editorContainer) {
            editorContainer.hidden = false;
        }

        // Put the uploaded image into the editor.
        const editorImage =
            document.getElementById("editorImage");

        if (editorImage) {
            editorImage.src = uploadedImage;
        }

        // Draw AI lines if they have already loaded.
        if (typeof drawAILines === "function") {
            drawAILines();
        }

        return;
    }

    // No image uploaded.
    const canvasEmpty =
        document.getElementById("canvasEmpty");

    if (canvasEmpty) {
        canvasEmpty.hidden = false;
    }

    const editorContainer =
        document.getElementById("editorContainer");

    if (editorContainer) {
        editorContainer.hidden = true;
    }
}


/* ==========================================
   GENERATE BUTTON
========================================== */

generateButton?.addEventListener(
    "click",
    async (event) => {

        event.preventDefault();


        // ==========================================
        // CHECK IMAGE
        // ==========================================

        if (!uploadedImageFile) {

            alert(
                "Please upload a notebook image first."
            );

            return;
        }


        // ==========================================
        // CHECK TEXT
        // ==========================================

        const text =
            textInput?.value.trim() ||
            "";


        if (!text) {

            alert(
                "Please enter some text first."
            );

            return;
        }
        if (fontSelect?.value === "personal" && !handwritingStyleId) {
            alert("Upload a handwriting sample first.");
            return;
        }


        // ==========================================
        // SAVE ORIGINAL BUTTON
        // ==========================================

        const originalHTML =
            generateButton.innerHTML;


        generateButton.disabled =
            true;

        generateButton.innerHTML =
            "Generating note...";


        // ==========================================
        // CREATE FORM DATA
        // ==========================================

        const formData =
            new FormData();


        formData.append(
            "image",
            uploadedImageFile
        );


        formData.append(
            "text",
            text
        );


        formData.append(
            "font_size",
            fontSize?.value ||
            "22"
        );


        formData.append(
            "font_color",
            fontColor?.value ||
            "#202020"
        );


        formData.append(
            "mode",
            placementMode
        );

        // ==========================================
        // SEND EDITED AI GEOMETRY
        // ==========================================

        if (
            placementMode === "automatic" &&
            aiGeometry
        ) {
            formData.append(
                "ai_geometry",
                JSON.stringify(aiGeometry)
            );

            console.log(
                "Sending edited AI geometry:",
                aiGeometry
            );
        }

        /* ---------- CUSTOM FONT ---------- */

        if (customFontFile) {

            formData.append(
                "custom_font",
                customFontFile
            );
        }


        /* ---------- SELECTED FONT ---------- */

        const fontName =
            fontSelect?.value ||
            "Caveat";


        formData.append(
            "font",
            fontName
        );

        if (fontName === "personal") {
            formData.append(
                "style_id",
                handwritingStyleId
            );
        }

        // ==========================================
        // DEBUG
        // ==========================================

        console.log(
            "Sending Text2Note render request..."
        );


        console.log(
            "Image:",
            uploadedImageFile.name
        );


        console.log(
            "Text:",
            text
        );


        console.log(
            "Font size:",
            fontSize?.value
        );


        console.log(
            "Font color:",
            fontColor?.value
        );


        console.log(
            "Mode:",
            placementMode
        );


        console.log(
            "Font:",
            fontName
        );


        if (customFontFile) {

            console.log(
                "Custom font being sent:",
                customFontFile.name
            );
        }


        // ==========================================
        // SEND TO DJANGO
        // ==========================================

        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/render/",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // ======================================
            // READ RESPONSE
            // ======================================

            const responseText =
                await response.text();


            console.log(
                "Django status:",
                response.status
            );


            console.log(
                "Django response:",
                responseText
            );


            // ======================================
            // PARSE JSON
            // ======================================

            let data;


            try {

                data =
                    JSON.parse(responseText);

            } catch (error) {

                throw new Error(
                    "Django returned an invalid response."
                );
            }


            // ======================================
            // CHECK HTTP ERROR
            // ======================================

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    `Server error: ${response.status}`
                );
            }


            // ======================================
            // CHECK RENDER ERROR
            // ======================================

            if (!data.success) {

                throw new Error(
                    data.error ||
                    "Rendering failed."
                );
            }


            // ======================================
            // RENDER SUCCESS
            // ======================================

            console.log(
                "Rendering successful!"
            );


            // ======================================
            // GENERATED IMAGE URL
            // ======================================

            const generatedImageURL =
                "http://127.0.0.1:8000/api/output/?t=" +
                Date.now();


            console.log(
                "Generated image:",
                generatedImageURL
            );


            // ======================================
            // SHOW GENERATED IMAGE
            // ======================================

            if (notebookCanvas) {

                notebookCanvas.innerHTML =
                    "";


                const generatedImage =
                    document.createElement("img");


                generatedImage.src =
                    generatedImageURL;


                generatedImage.alt =
                    "Generated Text2Note notebook";


                generatedImage.style.width =
                    "100%";


                generatedImage.style.height =
                    "100%";


                generatedImage.style.objectFit =
                    "contain";


                generatedImage.style.position =
                    "absolute";


                generatedImage.style.inset =
                    "0";


                generatedImage.style.zIndex =
                    "10";


                notebookCanvas.appendChild(
                    generatedImage
                );
            }


            // ======================================
            // SUCCESS BUTTON
            // ======================================

            generateButton.innerHTML =
                "✓ Note Created";


            // ======================================
            // RESTORE BUTTON
            // ======================================

            setTimeout(() => {

                generateButton.innerHTML =
                    originalHTML;

                generateButton.disabled =
                    false;

            }, 1500);


        } catch (error) {

            // ======================================
            // ERROR
            // ======================================

            console.error(
                "Text2Note render error:",
                error
            );


            alert(
                "Generate error:\n\n" +
                error.message
            );


            generateButton.innerHTML =
                originalHTML;


            generateButton.disabled =
                false;
        }
    }
);


/* ==========================================
   INITIALIZE
========================================== */

if (fontSizeValue && fontSize) {

    fontSizeValue.textContent =
        `${fontSize.value}px`;
}


if (colorValue && fontColor) {

    colorValue.textContent =
        fontColor.value;
}


updatePreview();


/* ==========================================
   DRAGGABLE TEXT
========================================== */

function makeDraggable(element) {

    let isDragging =
        false;

    let startX;
    let startY;


    element.addEventListener(
        "mousedown",
        startDrag
    );


    function startDrag(e) {

        isDragging =
            true;

        startX =
            e.clientX -
            currentX;

        startY =
            e.clientY -
            currentY;


        document.addEventListener(
            "mousemove",
            drag
        );


        document.addEventListener(
            "mouseup",
            stopDrag
        );
    }


    function drag(e) {

        if (!isDragging) return;


        currentX =
            e.clientX -
            startX;

        currentY =
            e.clientY -
            startY;


        element.style.left =
            currentX +
            "px";


        element.style.top =
            currentY +
            "px";
    }


    function stopDrag() {

        isDragging =
            false;


        document.removeEventListener(
            "mousemove",
            drag
        );


        document.removeEventListener(
            "mouseup",
            stopDrag
        );
    }
}


/* ==========================================
   FULL PREVIEW
========================================== */

const openPreviewBtn =
    document.getElementById(
        "openPreviewBtn"
    );


openPreviewBtn?.addEventListener(
    "click",
    () => {

        const text =
            textInput.value;


        const previewWindow =
            window.open(
                "",
                "_blank"
            );


        previewWindow.document.write(`
            <!DOCTYPE html>

            <html>

            <head>

                <title>
                    Text2Note Preview
                </title>

                <style>

                    body {
                        margin: 0;
                        padding: 40px;
                        background: #f7f3eb;
                        font-family: Arial, sans-serif;
                    }

                    .page {

                        max-width: 900px;
                        min-height: 1200px;

                        margin: auto;

                        background: white;

                        position: relative;

                        border-radius: 16px;

                        box-shadow:
                            0 20px 40px
                            rgba(0,0,0,.08);

                        overflow: hidden;
                    }

                    .page::before {

                        content: "";

                        position: absolute;

                        left: 60px;
                        top: 0;
                        bottom: 0;

                        width: 2px;

                        background:
                            rgba(220,90,90,.3);
                    }

                    .note {

                        padding:
                            80px 100px;

                        white-space: pre-wrap;

                        color:
                            ${fontColor.value};

                        font-size:
                            ${fontSize.value}px;

                        line-height:
                            ${lineSpacing.value};

                        font-family:
                            '${fontSelect.value}', cursive;

                        transform:
                            rotate(${rotation.value}deg);
                    }

                </style>

            </head>

            <body>

                <div class="page">

                    <div class="note">
                        ${text || "No text entered yet"}
                    </div>

                </div>

            </body>

            </html>
        `);


        previewWindow.document.close();
    }
);


/* ==========================================
   ZOOM
========================================== */

let zoom = 1;


const zoomInBtn =
    document.getElementById(
        "zoomInBtn"
    );


const zoomOutBtn =
    document.getElementById(
        "zoomOutBtn"
    );


zoomInBtn?.addEventListener(
    "click",
    () => {

        zoom += 0.1;

        notebookCanvas.style.transform =
            `scale(${zoom})`;
    }
);


zoomOutBtn?.addEventListener(
    "click",
    () => {

        zoom =
            Math.max(
                0.5,
                zoom - 0.1
            );

        notebookCanvas.style.transform =
            `scale(${zoom})`;
    }
);


/* ==========================================
   ADD CUSTOM FONT
========================================== */

const addFontButton =
    document.getElementById(
        "addFontButton"
    );


const fontInput =
    document.getElementById(
        "fontInput"
    );


addFontButton?.addEventListener(
    "click",
    () => {

        fontInput?.click();
    }
);


fontInput?.addEventListener(
    "change",
    () => {

        const file =
            fontInput.files[0];

        if (!file) return;

        customFontFile =
            file;

        console.log(
            "Custom font selected:",
            file.name
        );


        /* ======================================
           ADD CUSTOM FONT TO DROPDOWN
        ====================================== */

        const fontName =
            file.name
                .replace(/\.(ttf|otf)$/i, "")
                .replace(/[-_]/g, " ");


        // Check if this font is already in
        // the dropdown.
        const existingOption =
            Array.from(
                fontSelect.options
            ).find(
                option =>
                    option.value === fontName
            );


        if (!existingOption) {

            const option =
                document.createElement("option");

            option.value =
                fontName;

            option.textContent =
                `${fontName} — Custom`;

            fontSelect.appendChild(
                option
            );
        }


        // Automatically select the
        // newly uploaded font.
        fontSelect.value =
            fontName;


        console.log(
            "Custom font added to dropdown:",
            fontName
        );


        updatePreview();
    }
);

/* ==========================================
   DOWNLOAD GENERATED NOTE
========================================== */

downloadButton?.addEventListener(
    "click",
    async () => {

        try {

            // Get the latest generated image
            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/output/?t=" +
                    Date.now()
                );

            if (!response.ok) {

                throw new Error(
                    "Generated note is not available yet."
                );
            }

            const blob =
                await response.blob();


            // Create a temporary download URL
            const url =
                URL.createObjectURL(blob);


            // Create temporary download link
            const link =
                document.createElement("a");

            link.href =
                url;

            link.download =
                "Text2Note-Handwritten-Note.jpg";


            document.body.appendChild(link);

            link.click();

            document.body.removeChild(link);


            // Clean up temporary URL
            URL.revokeObjectURL(url);


            console.log(
                "Note downloaded successfully!"
            );

        } catch (error) {

            console.error(
                "Download error:",
                error
            );

            alert(
                "Download error:\n\n" +
                error.message
            );
        }
    }
);

const downloadPdfButton =
    document.getElementById("downloadPdfButton");

downloadPdfButton?.addEventListener(
    "click",
    async () => {

        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/output-pdf/?t=" +
                    Date.now()
                );

            if (!response.ok) {
                throw new Error(
                    "PDF is not available yet."
                );
            }

            const blob =
                await response.blob();

            const url =
                URL.createObjectURL(blob);

            const link =
                document.createElement("a");

            link.href =
                url;

            link.download =
                "Text2Note-Handwritten-Note.pdf";

            document.body.appendChild(link);

            link.click();

            document.body.removeChild(link);

            URL.revokeObjectURL(url);

            console.log(
                "PDF downloaded successfully!"
            );

        } catch (error) {

            console.error(
                "PDF download error:",
                error
            );

            alert(
                "PDF download error:\n\n" +
                error.message
            );
        }
    }
);
// ==========================================
// AI LINE EDITOR
// ==========================================

let aiGeometry = null;
let selectedLineIndex = null;

const editorContainer =
    document.getElementById("editorContainer");

const editorImage =
    document.getElementById("editorImage");

const lineEditorCanvas =
    document.getElementById("lineEditorCanvas");

const lineEditorContext =
    lineEditorCanvas?.getContext("2d");

const editorStatus =
    document.getElementById("editorStatus");


async function loadAILines() {
    const sourceFile = uploadedImageFile;
    try {
        if (!sourceFile) return;
        const formData = new FormData();
        formData.append("image", sourceFile);
        const response = await fetch(
            "http://127.0.0.1:8000/api/analyze-notebook/",
            { method: "POST", body: formData }
        );
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(
                data.error || "Failed to detect notebook lines."
            );
        }
        if (sourceFile !== uploadedImageFile) return;

        aiGeometry = data.geometry;

        console.log(
            "AI geometry loaded:",
            aiGeometry
        );

        drawAILines();

    } catch (error) {
        if (sourceFile !== uploadedImageFile) return;
        aiGeometry = null;
        console.error(
            "AI line loading error:",
            error
        );
        if (editorStatus) {
            editorStatus.textContent = error.message;
        }
    }
}

function getImageDisplayBounds() {

    if (!lineEditorCanvas || !editorImage) {
        return null;
    }

    const canvasRect =
        lineEditorCanvas.getBoundingClientRect();

    const imageRect =
        editorImage.getBoundingClientRect();

    const naturalWidth =
        editorImage.naturalWidth;

    const naturalHeight =
        editorImage.naturalHeight;

    if (!naturalWidth || !naturalHeight) {
        return null;
    }

    const containerWidth =
        imageRect.width;

    const containerHeight =
        imageRect.height;

    const imageRatio =
        naturalWidth / naturalHeight;

    const containerRatio =
        containerWidth / containerHeight;

    let displayedWidth;
    let displayedHeight;

    if (imageRatio > containerRatio) {

        displayedWidth =
            containerWidth;

        displayedHeight =
            containerWidth / imageRatio;

    } else {

        displayedHeight =
            containerHeight;

        displayedWidth =
            containerHeight * imageRatio;
    }

    const offsetX =
        (containerWidth - displayedWidth) / 2;

    const offsetY =
        (containerHeight - displayedHeight) / 2;

    return {
        x:
            imageRect.left -
            canvasRect.left +
            offsetX,

        y:
            imageRect.top -
            canvasRect.top +
            offsetY,

        width:
            displayedWidth,

        height:
            displayedHeight
    };
}   

function drawAILines() {
    if (!lineEditorCanvas || !editorContainer) {
        return;
    }

    const rect =
        editorContainer.getBoundingClientRect();

    lineEditorCanvas.width =
        Math.max(1, Math.round(rect.width));

    lineEditorCanvas.height =
        Math.max(1, Math.round(rect.height));

    lineEditorContext.clearRect(
        0,
        0,
        lineEditorCanvas.width,
        lineEditorCanvas.height
    );

    if (!aiGeometry?.lines?.length) {
        return;
    }

    const imageBounds =
    getImageDisplayBounds();

    if (!imageBounds) {
        return;
    }

    aiGeometry.lines.forEach(
        (line, lineIndex) => {
            if (!line.points?.length) {
                return;
            }

           lineEditorContext.beginPath();

            line.points.forEach(
                (point, pointIndex) => {
                    const x =
                imageBounds.x +
                point.x * imageBounds.width;

            const y =
                imageBounds.y +
                point.y * imageBounds.height;

                    if (pointIndex === 0) {
                        lineEditorContext.moveTo(
                            x,
                            y
                        );
                    } else {
                        lineEditorContext.lineTo(
                            x,
                            y
                        );
                    }
                }
            );

            if (
                lineIndex === selectedLineIndex
            ) {
                lineEditorContext.strokeStyle =
                    "#2563eb";

                lineEditorContext.lineWidth = 4;
            } else {
                lineEditorContext.strokeStyle =
                    "rgba(255, 60, 60, 0.65)";

                lineEditorContext.lineWidth = 2;
            }

            lineEditorContext.stroke();

            
            // Draw only a few control points
            // instead of showing every AI geometry point.
            if (lineIndex === selectedLineIndex) {

                const totalPoints =
                    line.points.length;

                const maxControlPoints = 8;

                const step =
                    Math.max(
                        1,
                        Math.floor(
                            (totalPoints - 1) /
                            (maxControlPoints - 1)
                        )
                    );

                line.points.forEach(
                    (point, pointIndex) => {

                        const isControlPoint =
                            pointIndex === 0 ||
                            pointIndex === totalPoints - 1 ||
                            pointIndex % step === 0;

                        if (!isControlPoint) {
                            return;
                        }

                        const pointX =
                            imageBounds.x +
                            point.x * imageBounds.width;

                        const pointY =
                            imageBounds.y +
                            point.y * imageBounds.height;

                        lineEditorContext.beginPath();

                        lineEditorContext.arc(
                            pointX,
                            pointY,
                            6,
                            0,
                            Math.PI * 2
                        );

                        lineEditorContext.fillStyle =
                            "#ffffff";

                        lineEditorContext.fill();

                        lineEditorContext.strokeStyle =
                            "#2563eb";

                        lineEditorContext.lineWidth =
                            2;

                        lineEditorContext.stroke();
                    }
                );
            }

            // Line number
            const firstPoint =
                line.points[0];

            const labelX =
                imageBounds.x +
                firstPoint.x * imageBounds.width +
                8;

            const labelY =
                imageBounds.y +
                firstPoint.y * imageBounds.height -
                6;

            lineEditorContext.font =
                "bold 12px Arial";

            lineEditorContext.fillStyle =
                lineIndex === selectedLineIndex
                    ? "#2563eb"
                    : "rgba(220, 30, 30, 0.9)";

            lineEditorContext.fillText(
                String(lineIndex + 1),
                labelX,
                labelY
            );
        }
    );
    drawTextOnSelectedLine();
}

function selectAILine(event) {
    if (!aiGeometry?.lines?.length) {
        return;
    }

    const rect =
        lineEditorCanvas.getBoundingClientRect();

    const imageBounds =
        getImageDisplayBounds();

    if (!imageBounds) {
        return;
    }

    const mouseX =
        event.clientX - rect.left;

    const mouseY =
        event.clientY - rect.top;

    let closestLine = null;
    let closestDistance = Infinity;

    aiGeometry.lines.forEach(
        (line, lineIndex) => {
            if (!line.points?.length) {
                return;
            }

            for (
                let i = 0;
                i < line.points.length - 1;
                i++
            ) {
                const p1 = line.points[i];
                const p2 = line.points[i + 1];

                const x1 =
                    imageBounds.x +
                    p1.x * imageBounds.width;

                const y1 =
                    imageBounds.y +
                    p1.y * imageBounds.height;

                const x2 =
                    imageBounds.x +
                    p2.x * imageBounds.width;

                const y2 =
                    imageBounds.y +
                    p2.y * imageBounds.height;

                const distance =
                    distanceToSegment(
                        mouseX,
                        mouseY,
                        x1,
                        y1,
                        x2,
                        y2
                    );

                if (
                    distance < closestDistance
                ) {
                    closestDistance = distance;
                    closestLine = lineIndex;
                }
            }
        }
    );

    // Only select if the click was reasonably close
    // to a detected line.
    if (
        closestDistance <= 15
    ) {
        selectedLineIndex =
            closestLine;
    } else {
        selectedLineIndex = null;
    }

    drawAILines();
}

// ==========================================
// DRAG SELECTED AI LINE
// ==========================================

// ==========================================
// DRAG SELECTED AI LINE
// ==========================================

let isDraggingLine = false;
let dragStartX = 0;
let dragStartY = 0;
let originalLinePoints = null;
let draggedControlPointIndex = null;


lineEditorCanvas?.addEventListener(
    "mousedown",
    startLineDrag
);


function startLineDrag(event) {

    if (
        selectedLineIndex === null ||
        !aiGeometry?.lines?.length
    ) {
        return;
    }

    // First check whether the mouse is
    // directly on a visible control point.
    draggedControlPointIndex =
        findControlPoint(event);

    const rect =
        lineEditorCanvas.getBoundingClientRect();

    dragStartX =
        event.clientX - rect.left;

    dragStartY =
        event.clientY - rect.top;

    const selectedLine =
        aiGeometry.lines[selectedLineIndex];

    // Save the original geometry before dragging.
    originalLinePoints =
        selectedLine.points.map(
            point => ({
                x: point.x,
                y: point.y
            })
        );

    isDraggingLine = true;

    lineEditorCanvas.style.cursor =
        "grabbing";

    document.addEventListener(
        "mousemove",
        dragLine
    );

    document.addEventListener(
        "mouseup",
        stopLineDrag
    );

    event.preventDefault();
}


function dragLine(event) {

    if (
        !isDraggingLine ||
        selectedLineIndex === null ||
        !originalLinePoints
    ) {
        return;
    }

    const rect =
        lineEditorCanvas.getBoundingClientRect();

    const currentX =
        event.clientX - rect.left;

    const currentY =
        event.clientY - rect.top;

    const deltaX =
        currentX - dragStartX;

    const deltaY =
        currentY - dragStartY;

    const imageBounds =
        getImageDisplayBounds();

    if (!imageBounds) {
        return;
    }

    const normalizedDeltaX =
        deltaX / imageBounds.width;

    const normalizedDeltaY =
        deltaY / imageBounds.height;

    const selectedLine =
        aiGeometry.lines[selectedLineIndex];


    // ==========================================
    // CONTROL POINT DRAG
    // ==========================================

    if (
        draggedControlPointIndex !== null
    ) {

        const controlIndex =
            draggedControlPointIndex;

        const totalPoints =
            originalLinePoints.length;

        const influenceRadius =
            Math.max(
                3,
                Math.floor(totalPoints / 10)
            );

        selectedLine.points =
            originalLinePoints.map(
                (point, index) => {

                    const distance =
                        Math.abs(
                            index -
                            controlIndex
                        );

                    if (
                        distance >
                        influenceRadius
                    ) {
                        return {
                            x: point.x,
                            y: point.y
                        };
                    }

                    const influence =
                        Math.cos(
                            (
                                distance /
                                influenceRadius
                            ) *
                            Math.PI /
                            2
                        );

                    return {
                        x:
                            point.x +
                            normalizedDeltaX *
                            influence,

                        y:
                            point.y +
                            normalizedDeltaY *
                            influence
                    };
                }
            );

        drawAILines();

        return;
    }


    // ==========================================
    // WHOLE LINE DRAG
    // ==========================================

    selectedLine.points =
        originalLinePoints.map(
            point => ({
                x:
                    point.x +
                    normalizedDeltaX,

                y:
                    point.y +
                    normalizedDeltaY
            })
        );

    drawAILines();
}


function stopLineDrag() {

    if (!isDraggingLine) {
        return;
    }

    isDraggingLine = false;

    originalLinePoints = null;

    draggedControlPointIndex = null;

    lineEditorCanvas.style.cursor =
        "pointer";

    document.removeEventListener(
        "mousemove",
        dragLine
    );

    document.removeEventListener(
        "mouseup",
        stopLineDrag
    );
}

function findControlPoint(event) {

    if (
        selectedLineIndex === null ||
        !aiGeometry?.lines?.length
    ) {
        return null;
    }

    const imageBounds =
        getImageDisplayBounds();

    if (!imageBounds) {
        return null;
    }

    const rect =
        lineEditorCanvas.getBoundingClientRect();

    const mouseX =
        event.clientX - rect.left;

    const mouseY =
        event.clientY - rect.top;

    const line =
        aiGeometry.lines[
            selectedLineIndex
        ];

    const totalPoints =
        line.points.length;

    const maxControlPoints = 8;

    const step =
        Math.max(
            1,
            Math.floor(
                (totalPoints - 1) /
                (maxControlPoints - 1)
            )
        );

    let closestIndex = null;
    let closestDistance = Infinity;

    line.points.forEach(
        (point, pointIndex) => {

            const isControlPoint =
                pointIndex === 0 ||
                pointIndex === totalPoints - 1 ||
                pointIndex % step === 0;

            if (!isControlPoint) {
                return;
            }

            const pointX =
                imageBounds.x +
                point.x * imageBounds.width;

            const pointY =
                imageBounds.y +
                point.y * imageBounds.height;

            const distance =
                Math.hypot(
                    mouseX - pointX,
                    mouseY - pointY
                );

            if (
                distance < closestDistance
            ) {
                closestDistance =
                    distance;

                closestIndex =
                    pointIndex;
            }
        }
    );

    if (
        closestDistance <= 14
    ) {
        return closestIndex;
    }

    return null;
}

function distanceToSegment(
    px,
    py,
    x1,
    y1,
    x2,
    y2
) {
    const dx = x2 - x1;
    const dy = y2 - y1;

    if (dx === 0 && dy === 0) {
        return Math.hypot(
            px - x1,
            py - y1
        );
    }

    const t =
        (
            (px - x1) * dx +
            (py - y1) * dy
        ) /
        (dx * dx + dy * dy);

    const clampedT =
        Math.max(
            0,
            Math.min(1, t)
        );

    const closestX =
        x1 + clampedT * dx;

    const closestY =
        y1 + clampedT * dy;

    return Math.hypot(
        px - closestX,
        py - closestY
    );
}       
lineEditorCanvas?.addEventListener(
    "click",
    selectAILine
);

// ==========================================
// TEXT ON SELECTED AI LINE
// ==========================================

// ==========================================
// TEXT WRAPPING ACROSS AI LINES
// ==========================================

function drawTextOnSelectedLine() {

    if (
        !lineEditorCanvas ||
        !lineEditorContext ||
        selectedLineIndex === null ||
        !aiGeometry?.lines?.length
    ) {
        return;
    }

    const text =
        textInput?.value.trim();

    if (!text) {
        return;
    }
    if (fontSelect?.value === "personal") {
        return;
    }

    const imageBounds =
        getImageDisplayBounds();

    if (!imageBounds) {
        return;
    }

    // ==========================================
    // FONT SETTINGS
    // ==========================================

    const size =
        Number(fontSize?.value || 22);

    const color =
        fontColor?.value || "#202020";

    const fontName =
        fontSelect?.value || "Caveat";

    lineEditorContext.save();

    lineEditorContext.font =
        `${size}px "${fontName}"`;

    lineEditorContext.fillStyle =
        color;

    lineEditorContext.textAlign =
        "left";

    lineEditorContext.textBaseline =
        "alphabetic";


    // ==========================================
    // SPLIT TEXT INTO WORDS
    // ==========================================

    const words =
        text.split(/\s+/);

    let wordIndex = 0;


    // ==========================================
    // GO THROUGH AI LINES
    // ==========================================

    for (
        let lineIndex = selectedLineIndex;
        lineIndex < aiGeometry.lines.length;
        lineIndex++
    ) {

        if (wordIndex >= words.length) {
            break;
        }

        const line =
            aiGeometry.lines[lineIndex];

        if (
            !line?.points ||
            line.points.length < 2
        ) {
            continue;
        }


        // ======================================
        // CONVERT LINE TO SCREEN COORDINATES
        // ======================================

        const points =
            line.points.map(point => ({
                x:
                    imageBounds.x +
                    point.x * imageBounds.width,

                y:
                    imageBounds.y +
                    point.y * imageBounds.height
            }));


        // ======================================
        // CALCULATE LINE LENGTH
        // ======================================

        const distances = [0];

        let totalLength = 0;

        for (
            let i = 1;
            i < points.length;
            i++
        ) {

            const dx =
                points[i].x -
                points[i - 1].x;

            const dy =
                points[i].y -
                points[i - 1].y;

            totalLength +=
                Math.hypot(dx, dy);

            distances.push(
                totalLength
            );
        }


        // ======================================
        // BUILD TEXT THAT FITS THIS LINE
        // ======================================

        let lineText = "";

        let testText = "";

        for (
            let i = wordIndex;
            i < words.length;
            i++
        ) {

            const candidate =
                lineText
                    ? lineText +
                      " " +
                      words[i]
                    : words[i];

            const candidateWidth =
                lineEditorContext.measureText(
                    candidate
                ).width;

            // Leave a small margin at the end.
            if (
                candidateWidth >
                totalLength - 10
            ) {
                break;
            }

            lineText =
                candidate;

            wordIndex = i + 1;
        }


        // ======================================
        // HANDLE A SINGLE WORD THAT IS TOO LONG
        // ======================================

        if (!lineText && wordIndex < words.length) {

            const word =
                words[wordIndex];

            let partialText = "";

            for (
                let charIndex = 0;
                charIndex < word.length;
                charIndex++
            ) {

                const candidate =
                    partialText +
                    word[charIndex];

                const width =
                    lineEditorContext.measureText(
                        candidate
                    ).width;

                if (
                    width >
                    totalLength - 10
                ) {
                    break;
                }

                partialText =
                    candidate;
            }

            if (partialText) {

                lineText =
                    partialText;

                wordIndex++;

                // Put the remaining part of the
                // word back into the word list.
                const remaining =
                    word.slice(
                        partialText.length
                    );

                if (remaining) {
                    words.splice(
                        wordIndex,
                        0,
                        remaining
                    );
                }
            }
        }


        if (!lineText) {
            continue;
        }


        // ======================================
        // DRAW TEXT ALONG THIS LINE
        // ======================================

        drawTextAlongLine(
            lineText,
            points,
            distances,
            totalLength
        );
    }

    lineEditorContext.restore();
}


// ==========================================
// DRAW TEXT ALONG ONE CURVED LINE
// ==========================================

function drawTextAlongLine(
    text,
    points,
    distances,
    totalLength
) {

    let currentDistance = 0;

    for (
        let charIndex = 0;
        charIndex < text.length;
        charIndex++
    ) {

        const character =
            text[charIndex];

        const charWidth =
            lineEditorContext.measureText(
                character
            ).width;

        const targetDistance =
            currentDistance +
            charWidth / 2;


        // Don't draw outside the line.
        if (
            targetDistance >
            totalLength
        ) {
            break;
        }


        // ======================================
        // FIND SEGMENT
        // ======================================

        let segmentIndex = 0;

        for (
            let i = 0;
            i < distances.length - 1;
            i++
        ) {

            if (
                targetDistance >= distances[i] &&
                targetDistance <= distances[i + 1]
            ) {

                segmentIndex = i;
                break;
            }
        }


        const start =
            points[segmentIndex];

        const end =
            points[
                Math.min(
                    segmentIndex + 1,
                    points.length - 1
                )
            ];


        const segmentLength =
            distances[segmentIndex + 1] -
            distances[segmentIndex];


        const localDistance =
            targetDistance -
            distances[segmentIndex];


        const ratio =
            segmentLength > 0
                ? localDistance /
                  segmentLength
                : 0;


        // ======================================
        // POSITION
        // ======================================

        const x =
            start.x +
            (end.x - start.x) *
            ratio;

        const y =
            start.y +
            (end.y - start.y) *
            ratio;


        // ======================================
        // ROTATION
        // ======================================

        const angle =
            Math.atan2(
                end.y - start.y,
                end.x - start.x
            );


        // ======================================
        // DRAW CHARACTER
        // ======================================

        lineEditorContext.save();

        lineEditorContext.translate(
            x,
            y
        );

        lineEditorContext.rotate(
            angle
        );

        lineEditorContext.fillText(
            character,
            -charWidth / 2,
            0
        );

        lineEditorContext.restore();


        currentDistance +=
            charWidth;
    }
}
