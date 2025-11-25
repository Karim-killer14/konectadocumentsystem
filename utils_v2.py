# utils_v2.py
import io
import re
import pytesseract
import numpy as np
from PIL import Image, ImageOps
import cv2
from dateutil import parser as dateparser

# -------- image preprocessing helpers --------
def pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def auto_deskew_and_enhance(pil_img: Image.Image) -> Image.Image:
    """
    Deskew + denoise + contrast stretch to improve OCR quality.
    Safe defaults so it does not over-process.
    """
    try:
        img = pil_to_cv(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # threshold for noise reduction
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # compute skew via moments
        coords = np.column_stack(np.where(gray > 0))
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                (h, w) = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # contrast/brightness stretch
        pil_out = cv_to_pil(img)
        pil_out = ImageOps.autocontrast(pil_out, cutoff=1)
        return pil_out
    except Exception:
        return pil_img

# -------- tesseract OCR fallback --------
def ocr_text_full(image: Image.Image) -> str:
    """
    Return full-page OCR text (fast).
    """
    try:
        txt = pytesseract.image_to_string(image)
        return txt or ""
    except Exception:
        return ""

def ocr_text_boxes(image: Image.Image):
    """
    Return Tesseract word-level dict with boxes.
    """
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []
        n = len(data['level'])
        for i in range(n):
            w = data['text'][i].strip()
            if w == "":
                continue
            x = int(data['left'][i]); y = int(data['top'][i]); w0 = int(data['width'][i]); h0 = int(data['height'][i])
            results.append({
                "text": w,
                "conf": float(data['conf'][i]) if data['conf'][i] != '-1' else None,
                "bbox": [x, y, x + w0, y + h0]
            })
        return results
    except Exception:
        return []

# -------- simple normalizers & validators --------
_amount_re = re.compile(r'(?<!\S)([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{1,2})?|\d+\.\d{1,2})(?=\s*(AED|USD|EUR|SAR|OMR)?\b)', re.IGNORECASE)
_date_candidates_re = re.compile(r'((?:\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})|(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}))')

def extract_amount_from_text(text: str):
    m = _amount_re.search(text)
    if not m:
        return None
    val = m.group(1)
    # clean separators
    v = val.replace(',', '').replace(' ', '')
    try:
        return float(v)
    except:
        return None

def extract_currency_from_text(text: str):
    m = re.search(r'\b(AED|USD|EUR|SAR|OMR)\b', text, re.IGNORECASE)
    return (m.group(1).upper() if m else None)

def extract_date_from_text(text: str):
    # try dateparser safely
    m = _date_candidates_re.search(text)
    if not m:
        # fallback to dateutil parse attempt
        try:
            dt = dateparser.parse(text, fuzzy=True, dayfirst=False)
            return dt.date().isoformat() if dt else None
        except Exception:
            return None
    try:
        dt = dateparser.parse(m.group(1), fuzzy=True, dayfirst=False)
        return dt.date().isoformat() if dt else None
    except:
        return None

def extract_invoice_id(text: str):
    # common patterns: INV-2024-0001, PO-2024-0001, APV-2024-0001, Invoice #12345
    m = re.search(r'\b(?:INV|INVOICE|PO|APV|APV|REF)[:\s\-#]*([A-Z0-9\-\_\/]{4,40})\b', text, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    m2 = re.search(r'\b(?:#|No\.)\s*([0-9]{3,12})\b', text)
    return m2.group(0).strip() if m2 else None

# vendor heuristics: look for capitalized company-like tokens
def guess_vendor_from_text(text: str):
    # look for common business suffixes or long capitalized spans
    m = re.search(r'\b([A-Z][A-Za-z&\.\s]{3,60}(?:LLC|LTD|FZ|CO|COMPANY|SERVICES|SOLUTIONS|TRADING))\b', text)
    if m:
        return m.group(1).strip()
    # fallback: long capitalized line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines[:6]:
        if len(ln) > 8 and sum(1 for c in ln if c.isupper()) > max(1, len(ln)*0.2):
            return ln
    return None

# department heuristic
def guess_department(text: str):
    for d in ["Finance", "Procurement", "Operations", "HR", "IT", "Facilities"]:
        if re.search(r'\b' + re.escape(d) + r'\b', text, re.IGNORECASE):
            return d
    return None
