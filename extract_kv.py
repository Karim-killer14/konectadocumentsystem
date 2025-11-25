# extract_kv.py
from utils_v2 import (
    auto_deskew_and_enhance,
    ocr_text_full,
    extract_amount_from_text,
    extract_currency_from_text,
    extract_date_from_text,
    extract_invoice_id,
    guess_vendor_from_text,
    guess_department,
    clean_token_text
)
import re
import logging

logger = logging.getLogger(__name__)

def tokens_to_text(tokens_list):
    # join tokens after cleaning tokens from LayoutLM
    toks = [clean_token_text(t.get("token","")) for t in tokens_list]
    toks = [t for t in toks if t]
    return " ".join(toks)

class KeyValueExtractor:
    def __init__(self, layoutlm_inferencer, image_preprocess=True):
        self.model = layoutlm_inferencer
        self.preprocess = image_preprocess

    def extract_from_image(self, pil_image):
        img = pil_image
        if self.preprocess:
            img = auto_deskew_and_enhance(img)

        # run LayoutLM for tokens
        try:
            tokens = self.model.infer(img)
        except Exception as e:
            logger.exception("LayoutLM inference failed, falling back to OCR-only.")
            tokens = []

        text_blob = tokens_to_text(tokens)

        kv = {}
        confidence = {}

        # document id (invoice/po/approval)
        docid = extract_invoice_id(text_blob)
        if docid:
            kv["document_id"] = docid
            confidence["document_id"] = 0.9

        # date
        dt = extract_date_from_text(text_blob)
        if dt:
            kv["date"] = dt
            confidence["date"] = 0.85

        # amount & currency
        amt = extract_amount_from_text(text_blob)
        if amt is not None:
            kv["amount"] = round(float(amt), 2)
            curr = extract_currency_from_text(text_blob)
            if curr:
                kv["currency"] = curr
            confidence["amount"] = 0.8

        # vendor
        vendor = guess_vendor_from_text(text_blob)
        if vendor:
            kv["vendor"] = vendor
            confidence["vendor"] = 0.7

        # department
        dept = guess_department(text_blob)
        if dept:
            kv["department"] = dept
            confidence["department"] = 0.7

        # purpose
        m_purpose = re.search(r'Purpose[:\s\-]+(.{3,120}?)(?:Approver|Status|$)', text_blob, re.IGNORECASE)
        if m_purpose:
            kv["purpose"] = m_purpose.group(1).strip()
            confidence["purpose"] = 0.6

        # approver & status
        m_approver = re.search(r'Approver[:\s\-]*([A-Z][A-Za-z\s]{2,60})', text_blob, re.IGNORECASE)
        if m_approver:
            kv["approver"] = m_approver.group(1).strip()
            confidence["approver"] = 0.6

        m_status = re.search(r'\b(Status)[:\s\-]*(Approved|Pending|Rejected)\b', text_blob, re.IGNORECASE)
        if m_status:
            kv["status"] = m_status.group(2).strip()
            confidence["status"] = 0.8

        # Fallback: if critical fields missing, run full Tesseract OCR and try again
        critical_missing = [k for k in ["amount","date","document_id","vendor"] if k not in kv]
        if critical_missing:
            ocr_txt = ocr_text_full(img)
            if not ocr_txt:
                logger.warning("Tesseract OCR returned empty. Tesseract may not be installed.")
            else:
                if "amount" in critical_missing:
                    amt2 = extract_amount_from_text(ocr_txt)
                    if amt2 is not None:
                        kv["amount"] = round(float(amt2),2)
                        confidence["amount"] = 0.6
                        curr2 = extract_currency_from_text(ocr_txt)
                        if curr2: kv["currency"] = curr2
                if "date" in critical_missing:
                    dt2 = extract_date_from_text(ocr_txt)
                    if dt2:
                        kv["date"] = dt2
                        confidence["date"] = 0.6
                if "vendor" in critical_missing:
                    v2 = guess_vendor_from_text(ocr_txt)
                    if v2:
                        kv["vendor"] = v2
                        confidence["vendor"] = 0.5
                if "document_id" in critical_missing:
                    id2 = extract_invoice_id(ocr_txt)
                    if id2:
                        kv["document_id"] = id2
                        confidence["document_id"] = 0.6

        # final validation: tight checks
        if "amount" in kv:
            try:
                if not (1 <= float(kv["amount"]) <= 10_000_000):
                    del kv["amount"]
                    confidence.pop("amount", None)
            except:
                kv.pop("amount", None)
                confidence.pop("amount", None)

        # build output
        out = {
            "fields": kv,
            "confidence": confidence,
            "raw_text": text_blob
        }
        return out
