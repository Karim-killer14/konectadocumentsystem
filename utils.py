# utils.py
from PIL import Image, ImageDraw
import pytesseract
import numpy as np
from pdf2image import convert_from_bytes
import re

# ------------------------------------------------------------
# 1. PDF → Images
# ------------------------------------------------------------
def pdf_to_images(pdf_bytes_io):
    """
    Convert PDF bytes to list of PIL images (RGB).
    Works on Streamlit Cloud because poppler-utils is installed via packages.txt.
    """
    pages = convert_from_bytes(
        pdf_bytes_io.read() if hasattr(pdf_bytes_io, "read") else pdf_bytes_io
    )
    return [p.convert("RGB") for p in pages]

# ------------------------------------------------------------
# 2. OCR + lightweight layout segmentation (Tesseract-based)
# ------------------------------------------------------------
def ocr_and_layout(image: Image.Image):
    """
    Performs OCR and extracts bounding boxes using Tesseract only.
    Returns:
        list of dicts: { "text": str, "bbox": [x1,y1,x2,y2], "conf": float }
    This is Streamlit Cloud–safe (no Detectron2).
    """

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    blocks = []
    n = len(data["level"])

    for i in range(n):
        text = data["text"][i]
        if text.strip() == "":
            continue

        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else None
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        blocks.append({
            "text": clean_text(text),
            "bbox": [x, y, x + w, y + h],
            "conf": conf,
        })

    return blocks

# ------------------------------------------------------------
# 3. Layout Visualization Helper
# ------------------------------------------------------------
def visualize_layout_on_image(image: Image.Image, blocks):
    """
    Draw bounding boxes around detected text blocks.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for b in blocks:
        x1, y1, x2, y2 = b["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return img

# ------------------------------------------------------------
# 4. Prepare LayoutLM inputs (tokens + normalized bboxes)
# ------------------------------------------------------------
def prepare_layoutlm_inputs(image: Image.Image, blocks):
    """
    Converts OCR blocks into:
        - words (token list)
        - boxes (normalized 0–1000 LayoutLMv3 format)
    """

    width, height = image.size
    words = []
    boxes = []

    for block in blocks:
        text = block["text"]
        if not text:
            continue

        # Tokenize by whitespace
        tokens = text.split()
        x1, y1, x2, y2 = block["bbox"]

        # Normalize bbox into 0-1000 scale
        norm_box = [
            int(1000 * (x1 / width)),
            int(1000 * (y1 / height)),
            int(1000 * (x2 / width)),
            int(1000 * (y2 / height)),
        ]

        for t in tokens:
            words.append(t)
            boxes.append(norm_box)

    return {
        "image": image,
        "words": words,
        "boxes": boxes,
    }

# ------------------------------------------------------------
# 5. Text cleaner for noisy OCR output
# ------------------------------------------------------------
def clean_text(text: str):
    """Removes non-printable characters and artifacts."""
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    return text.strip()
