# utils_v2.py
import io
import re
import pytesseract
import numpy as np
from PIL import Image, ImageOps
import cv2
from dateutil import parser as dateparser

# ---------- image helper conversions ----------
def pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# ---------- preprocessing ----------
def auto_deskew_and_enhance(pil_img: Image.Image) -> Image.Image:
    try:
        img = pil_to_cv(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
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
        pil_out = cv_to_pil(img)
        pil_out = ImageOps.autocontrast(pil_out, cutoff=1)
        return pil_out
    except Exception:
        return pil_img

# ---------- Tesseract fallback ----------
def ocr_text_full(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""

def ocr_text_boxes(image: Image.Image):
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []
        n = len(data['level'])
        for i in range(n):
            w = data['text'][i].strip()
            if w == "":
                continue
            x = int(data['left'][i]); y = int(data['top'][i]); w0 = int(data['width'][i]); h0 = int(data['height'][i])
            conf = None
            try:
                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else None
            except:
                conf = None
            results.append({"text": w, "conf": conf, "bbox": [x, y, x + w0, y + h0]})
        return results
    except Exception:
        return []

# ---------- cleaners & extractors ----------
def clean_token_text(txt: str) -> str:
    txt = txt.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
    # HF tokenization may include '▁' markers; replace them with spaces
    txt = txt.replace("▁", " ")
    return re.sub(r'\s+', ' ', txt).strip()

_amount_re = re.compile(r'(?<!\S)([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)(?:\s*(AED|USD|EUR|SAR|OMR))?', re.IGNORECASE)
_date_candidates_re = re.compile(r'(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})')

def extract_amount_from_text(text: str):
    if not text:
        return None
    m = _amount_re.search(text)
    if not m:
        return None
    num = m.group(1)
    num = num.replace(',', '').replace(' ', '')
    try:
        val = float(num)
        # sanity checks
        if val < 1:  # tiny values likely erroneous in invoices
            return None
        if val > 10_000_000:
            return None
        return val
    except:
        return None

def extract_currency_from_text(text: str):
    if not text:
        return None
    m = re.search(r'\b(AED|USD|EUR|SAR|OMR)\b', text, re.IGNORECASE)
    return m.group(1).upper() if m else None

def extract_date_from_text(text: str):
    if not text:
        return None
    m = _date_candidates_re.search(text)
    try:
        if m:
            dt = dateparser.parse(m.group(1), fuzzy=True, dayfirst=False)
            return dt.date().isoformat()
        dt = dateparser.parse(text, fuzzy=True, dayfirst=False)
        return dt.date().isoformat()
    except:
        return None

def extract_invoice_id(text: str):
    if not text:
        return None
    m = re.search(r'\b(?:INV|INVOICE|PO|APV|REF|APV)[:\s\-#]*([A-Z0-9\-\_\/]{3,40})\b', text, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    m2 = re.search(r'\b(?:#|No\.)\s*([0-9]{3,12})\b', text)
    return m2.group(0).strip() if m2 else None

def guess_vendor_from_text(text: str):
    if not text:
        return None
    m = re.search(r'\b([A-Z][A-Za-z&\.\s]{3,60}(?:LLC|LTD|FZ|CO|COMPANY|SERVICES|SOLUTIONS|TRADING))\b', text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines[:6]:
        if len(ln) > 6 and sum(1 for c in ln if c.isupper()) >= max(1, int(len(ln)*0.18)):
            return ln
    return None

def guess_department(text: str):
    if not text:
        return None
    for d in ["Finance", "Procurement", "Operations", "HR", "IT", "Facilities"]:
        if re.search(r'\b' + re.escape(d) + r'\b', text, re.IGNORECASE):
            return d
    return None
