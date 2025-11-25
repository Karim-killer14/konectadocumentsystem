# extract_kv.py
from collections import defaultdict
import re
from utils_v2 import (
    auto_deskew_and_enhance,
    ocr_text_full,
    ocr_text_boxes,
    extract_amount_from_text,
    extract_currency_from_text,
    extract_date_from_text,
    extract_invoice_id,
    guess_vendor_from_text,
    guess_department
)

def tokens_to_text(tokens_list):
    # tokens_list is list of {"token": "...", "label_id": ...}
    # We just join tokens conservatively into a single text blob
    txt = " ".join([t.get("token","") for t in tokens_list])
    return txt

class KeyValueExtractor:
    def __init__(self, layoutlm_inferencer, image_preprocess=True):
        self.model = layoutlm_inferencer
        self.preprocess = image_preprocess

    def extract_from_image(self, pil_image):
        """
        Full pipeline:
         - optional preprocessing
         - run LayoutLM to get tokens
         - heuristic parsing of tokens->fields
         - fallback to Tesseract where needed
        Returns: dict of fields
        """
        img = pil_image
        if self.preprocess:
            img = auto_deskew_and_enhance(img)

        # 1) LayoutLM inference (tokens)
        tokens = self.model.infer(img)  # tokens = [{"token": "...", "label_id": ...}, ...]
        text_blob = tokens_to_text(tokens)

        # 2) heuristic extraction from tokens-first
        kv = {}

        # invoice/po/approval id
        inv = extract_invoice_id(text_blob)
        if inv:
            kv['document_id'] = inv

        # date
        dt = extract_date_from_text(text_blob)
        if dt:
            kv['date'] = dt

        # amount & currency
        amt = extract_amount_from_text(text_blob)
        if amt is not None:
            kv['amount'] = float(round(amt,2))
            curr = extract_currency_from_text(text_blob)
            if curr:
                kv['currency'] = curr

        # vendor
        vendor = guess_vendor_from_text(text_blob)
        if vendor:
            kv['vendor'] = vendor

        # department
        dept = guess_department(text_blob)
        if dept:
            kv['department'] = dept

        # purpose / approver / status heuristics (search for keywords)
        m_purpose = re.search(r'Purpose[:\s\-]*(.+?)(?:Approver|Status|$)', text_blob, re.IGNORECASE)
        if m_purpose:
            kv['purpose'] = m_purpose.group(1).strip()

        m_approver = re.search(r'Approver[:\s\-]*([A-Z][A-Za-z\s]{2,40})', text_blob, re.IGNORECASE)
        if m_approver:
            kv['approver'] = m_approver.group(1).strip()

        m_status = re.search(r'\b(Status|STATUS)[:\s\-]*(Approved|Pending|Rejected)\b', text_blob, re.IGNORECASE)
        if m_status:
            kv['status'] = m_status.group(2).strip()

        # 3) fallback: if some key is missing, run Tesseract full-page OCR and re-try patterns
        missing = [k for k in ['date','amount','vendor','document_id'] if k not in kv]
        if missing:
            ocr_txt = ocr_text_full(img)
            if 'date' in missing:
                dt2 = extract_date_from_text(ocr_txt)
                if dt2: kv['date'] = dt2
            if 'amount' in missing:
                amt2 = extract_amount_from_text(ocr_txt)
                if amt2 is not None:
                    kv['amount'] = float(round(amt2,2))
                    curr2 = extract_currency_from_text(ocr_txt)
                    if curr2: kv['currency'] = curr2
            if 'vendor' in missing:
                v2 = guess_vendor_from_text(ocr_txt)
                if v2: kv['vendor'] = v2
            if 'document_id' in missing:
                id2 = extract_invoice_id(ocr_txt)
                if id2: kv['document_id'] = id2

        # 4) final validation & normalization
        if 'amount' in kv:
            try:
                kv['amount'] = float(kv['amount'])
            except:
                kv.pop('amount', None)
        if 'date' in kv:
            # leave as ISO string (YYYY-MM-DD)
            pass

        # 5) return structured result with confidence placeholders
        out = {
            "fields": kv,
            "raw_text": text_blob,
        }
        return out
