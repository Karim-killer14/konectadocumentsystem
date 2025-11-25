# kv_extractor_final.py
"""
Key-Value Extractor (final)
Uses LayoutLM token output + fallback OCR + heuristics to extract
invoice / PO / approval fields and compute per-field confidence.
"""

from collections import defaultdict
import re
import logging
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

logger = logging.getLogger(__name__)

# Document schema we target
REQUIRED_FIELDS = {
    "invoice": ["document_id", "date", "amount", "currency"],
    "purchase_order": ["document_id", "date", "amount", "currency"],
    "approval": ["document_id", "date", "amount", "currency", "approver", "status"]
}

# helper to join tokens into readable blob
def tokens_to_text(tokens_list):
    toks = [clean_token_text(t.get("token","")) for t in tokens_list]
    toks = [t for t in toks if t]
    return " ".join(toks)

# small utility: pick best non-empty value
def pick_first_nonempty(values):
    for v in values:
        if v:
            return v
    return None

class FinalKVExtractor:
    def __init__(self, inferencer, preprocess=True, tesseract_allowed=True):
        """
        inferencer: LayoutLMInferencer instance
        preprocess: run auto enhancement (deskew/denoise)
        tesseract_allowed: if False, skip OCR fallback (useful when tesseract not installed)
        """
        self.model = inferencer
        self.preprocess = preprocess
        self.tesseract_allowed = tesseract_allowed

    def _initial_tokens_blob(self, image):
        try:
            tokens = self.model.infer(image)
            text_blob = tokens_to_text(tokens)
        except Exception as e:
            logger.exception("LayoutLM inference failed: %s", e)
            tokens = []
            text_blob = ""
        return tokens, text_blob

    def _fallback_ocr(self, image):
        if not self.tesseract_allowed:
            return ""
        try:
            txt = ocr_text_full(image)
            return txt or ""
        except Exception as e:
            logger.warning("Tesseract OCR failed: %s", e)
            return ""

    def _classify_doc_type(self, text_blob):
        tb = (text_blob or "").lower()
        if re.search(r'\binvoice\b|\binv\b', tb):
            return "invoice"
        if re.search(r'\bpurchase order\b|\bpo\b', tb):
            return "purchase_order"
        if re.search(r'\bapproval\b|\bapprov\b|\brequest id\b', tb):
            return "approval"
        # fallback heuristics
        if "approver" in tb or "requested by" in tb:
            return "approval"
        return "invoice"  # default conservative

    def extract_from_image(self, pil_image):
        """
        Full pipeline for a single page image.
        Returns: {
            fields: {k: v},
            confidence: {k: score},
            issues: [messages],
            raw_texts: { "layoutlm": "...", "ocr": "..." },
            doc_type: one of ('invoice','purchase_order','approval')
        }
        """
        img = pil_image
        if self.preprocess:
            img = auto_deskew_and_enhance(img)

        tokens, layout_text = self._initial_tokens_blob(img)
        ocr_text = ""
        fields = {}
        conf = {}
        issues = []

        # classify doc type early (affects required fields)
        doc_type = self._classify_doc_type(layout_text)

        # 1) try to extract from LayoutLM text first
        # document id
        did = extract_invoice_id(layout_text)
        if did:
            fields["document_id"] = did
            conf["document_id"] = 0.88

        # date
        dt = extract_date_from_text(layout_text)
        if dt:
            fields["date"] = dt
            conf["date"] = 0.85

        # amount & currency
        amt = extract_amount_from_text(layout_text)
        if amt is not None:
            fields["amount"] = round(float(amt), 2)
            cur = extract_currency_from_text(layout_text)
            if cur:
                fields["currency"] = cur
            conf["amount"] = 0.82

        # vendor
        v = guess_vendor_from_text(layout_text)
        if v:
            fields["vendor"] = v
            conf["vendor"] = 0.7

        # department
        d = guess_department(layout_text)
        if d:
            fields["department"] = d
            conf["department"] = 0.7

        # purpose/approver/status
        m_purpose = re.search(r'Purpose[:\s\-]+(.{3,160}?)(?:Approver|Status|$)', layout_text, re.IGNORECASE)
        if m_purpose:
            p = m_purpose.group(1).strip()
            fields["purpose"] = p
            conf["purpose"] = 0.6

        m_approver = re.search(r'Approver[:\s\-]*([A-Z][A-Za-z\s,]{2,80})', layout_text, re.IGNORECASE)
        if m_approver:
            fields["approver"] = m_approver.group(1).strip()
            conf["approver"] = 0.65

        m_status = re.search(r'\bStatus[:\s\-]*(Approved|Pending|Rejected)\b', layout_text, re.IGNORECASE)
        if m_status:
            fields["status"] = m_status.group(1).strip()
            conf["status"] = 0.8

        # 2) detect missing critical fields and run OCR fallback if allowed
        needed = REQUIRED_FIELDS.get(doc_type, [])
        missing = [k for k in needed if k not in fields]
        if missing and self.tesseract_allowed:
            ocr_text = self._fallback_ocr(img)
            # re-check missing
            if "document_id" in missing and "document_id" not in fields:
                did2 = extract_invoice_id(ocr_text)
                if did2:
                    fields["document_id"] = did2
                    conf["document_id"] = 0.6
            if "date" in missing and "date" not in fields:
                dt2 = extract_date_from_text(ocr_text)
                if dt2:
                    fields["date"] = dt2
                    conf["date"] = 0.6
            if "amount" in missing and "amount" not in fields:
                amt2 = extract_amount_from_text(ocr_text)
                if amt2 is not None:
                    fields["amount"] = round(float(amt2),2)
                    c2 = extract_currency_from_text(ocr_text)
                    if c2: fields["currency"] = c2
                    conf["amount"] = 0.6
            if "vendor" in missing and "vendor" not in fields:
                v2 = guess_vendor_from_text(ocr_text)
                if v2:
                    fields["vendor"] = v2
                    conf["vendor"] = 0.55

        # 3) validation rules & sanity checks
        # amount boundary checks
        if "amount" in fields:
            try:
                a = float(fields["amount"])
                if not (1 <= a <= 10_000_000):
                    issues.append("amount_out_of_range")
                    del fields["amount"]
                    conf.pop("amount", None)
            except Exception:
                issues.append("amount_invalid")
                fields.pop("amount", None)
                conf.pop("amount", None)

        # date sanity
        if "date" in fields:
            # simple ISO check length
            if not isinstance(fields["date"], str) or len(fields["date"]) < 6:
                issues.append("date_invalid")
                fields.pop("date", None)
                conf.pop("date", None)

        # vendor sanity: too short / token garble
        if "vendor" in fields:
            vval = fields["vendor"]
            if len(vval) < 4 or re.search(r'[^ -~]', vval):  # non-ascii garbage
                issues.append("vendor_suspect")
                # keep vendor but lower confidence
                conf["vendor"] = min(conf.get("vendor", 0.5), 0.5)

        # final confidence normalization (0.0 - 1.0)
        for k in list(conf.keys()):
            try:
                conf[k] = float(max(0.0, min(1.0, conf[k])))
            except:
                conf[k] = 0.5

        # produce final output
        out = {
            "doc_type": doc_type,
            "fields": fields,
            "confidence": conf,
            "issues": issues,
            "raw_texts": {
                "layoutlm": layout_text,
                "ocr": ocr_text
            }
        }
        return out
