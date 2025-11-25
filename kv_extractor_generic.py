# kv_extractor_generic.py
import re
import logging
from collections import defaultdict
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

# Universal set of target fields
TARGET_FIELDS = [
    "document_id", "date", "amount", "currency", "vendor", "department",
    "line_items", "subtotal", "vat", "approver", "requested_by", "status",
    "purpose", "delivery_date", "po_number"
]

# helper: tokens -> cleaned text
def tokens_to_text(tokens):
    toks = [clean_token_text(t.get("token", "")) for t in tokens]
    toks = [t for t in toks if t]
    return " ".join(toks)


class KVExtractorGeneric:
    def __init__(self, inferencer, preprocess=True, tesseract_allowed=True):
        """
        inferencer: LayoutLMInferencer instance
        preprocess: whether to run image enhancements
        tesseract_allowed: whether to use pytesseract fallback
        """
        self.model = inferencer
        self.preprocess = preprocess
        self.tesseract_allowed = tesseract_allowed

    def _layout_text_and_tokens(self, img):
        try:
            tokens = self.model.infer(img)
            text = tokens_to_text(tokens)
        except Exception as e:
            logger.exception("LayoutLM infer failed: %s", e)
            tokens = []
            text = ""
        return tokens, text

    def _ocr_fallback(self, img):
        if not self.tesseract_allowed:
            return ""
        try:
            return ocr_text_full(img) or ""
        except Exception:
            logger.exception("Tesseract OCR fallback failed")
            return ""

    def extract(self, pil_img):
        """
        Main extraction entrypoint.
        Returns dict:
          {
            "doc_type": <invoice|purchase_order|approval|unknown>,
            "fields": {...},      # may omit missing fields
            "confidence": {...},  # 0.0-1.0
            "issues": [...],
            "raw": {"layout": "...", "ocr": "..."}
          }
        """
        img = pil_img
        if self.preprocess:
            img = auto_deskew_and_enhance(img)

        tokens, layout_text = self._layout_text_and_tokens(img)

        # classify doc type heuristically
        doc_type = self._classify_doc_type(layout_text)

        # store outputs
        fields = {}
        conf = {}
        issues = []

        # 1. Attempt extraction from LayoutLM text
        self._extract_basic(layout_text, fields, conf)
        self._extract_special(layout_text, fields, conf)

        # 2. If critical fields missing, run OCR fallback and re-extract
        critical = ["amount", "date", "document_id", "currency"]
        missing_critical = [k for k in critical if k not in fields]
        ocr_text = ""
        if missing_critical and self.tesseract_allowed:
            ocr_text = self._ocr_fallback(img)
            if ocr_text:
                self._extract_basic(ocr_text, fields, conf, source="ocr")
                self._extract_special(ocr_text, fields, conf, source="ocr")

        # 3. Post-process line-items detection (attempt naive table parse from layout or OCR)
        line_items = self._extract_line_items(tokens, layout_text, ocr_text)
        if line_items:
            fields["line_items"] = line_items
            conf["line_items"] = 0.6

        # 4. Final validation & sanity checks
        self._validate_and_flag(fields, conf, issues)

        out = {
            "doc_type": doc_type,
            "fields": fields,
            "confidence": conf,
            "issues": issues,
            "raw": {"layout": layout_text, "ocr": ocr_text}
        }
        return out

    # ----------------------
    # Helper extraction routines
    # ----------------------
    def _classify_doc_type(self, text):
        t = (text or "").lower()
        if re.search(r'\binvoice\b|\binv\b', t):
            return "invoice"
        if re.search(r'\bpurchase order\b|\bpo\b', t):
            return "purchase_order"
        if re.search(r'\bapproval\b|\bapprov\b|\brequest id\b', t) or re.search(r'\bapprover\b|\brequested by\b', t):
            return "approval"
        return "unknown"

    def _extract_basic(self, text, fields, conf, source="layout"):
        if not text:
            return
        # document id
        if "document_id" not in fields:
            did = extract_invoice_id(text)
            if did:
                fields["document_id"] = did
                conf["document_id"] = 0.9 if source == "layout" else 0.6

        # date
        if "date" not in fields:
            d = extract_date_from_text(text)
            if d:
                fields["date"] = d
                conf["date"] = 0.85 if source == "layout" else 0.6

        # amount & currency
        if "amount" not in fields:
            a = extract_amount_from_text(text)
            if a is not None:
                fields["amount"] = round(float(a), 2)
                conf["amount"] = 0.85 if source == "layout" else 0.6
                cur = extract_currency_from_text(text)
                if cur:
                    fields["currency"] = cur

        # vendor
        if "vendor" not in fields:
            v = guess_vendor_from_text(text)
            if v:
                fields["vendor"] = v
                conf["vendor"] = 0.7 if source == "layout" else 0.5

        # department
        if "department" not in fields:
            dept = guess_department(text)
            if dept:
                fields["department"] = dept
                conf["department"] = 0.7

    def _extract_special(self, text, fields, conf, source="layout"):
        if not text:
            return
        # purpose
        if "purpose" not in fields:
            m = re.search(r'Purpose[:\s\-]+(.{3,160}?)(?:Approver|Status|$)', text, re.IGNORECASE)
            if m:
                fields["purpose"] = m.group(1).strip()
                conf["purpose"] = 0.6

        # approver/requested_by/status
        if "approver" not in fields:
            ma = re.search(r'Approver[:\s\-]*([A-Z][A-Za-z\s,]{2,80})', text, re.IGNORECASE)
            if ma:
                fields["approver"] = ma.group(1).strip()
                conf["approver"] = 0.65
        if "requested_by" not in fields:
            mr = re.search(r'Requested By[:\s\-]*([A-Z][A-Za-z\s,]{2,80})', text, re.IGNORECASE)
            if mr:
                fields["requested_by"] = mr.group(1).strip()
                conf["requested_by"] = 0.65
        if "status" not in fields:
            ms = re.search(r'\bStatus[:\s\-]*(Approved|Pending|Rejected)\b', text, re.IGNORECASE)
            if ms:
                fields["status"] = ms.group(1).strip()
                conf["status"] = 0.8

        # PO-specific: delivery_date, po_number
        if "delivery_date" not in fields:
            md = re.search(r'Delivery Date[:\s\-]*([0-9A-Za-z\-\./\s]{4,40})', text, re.IGNORECASE)
            if md:
                dd = extract_date_from_text(md.group(1))
                if dd:
                    fields["delivery_date"] = dd
                    conf["delivery_date"] = 0.6
        if "po_number" not in fields:
            mpo = re.search(r'\bPO[:\s\-#]*([A-Z0-9\-\_\/]{3,40})\b', text, re.IGNORECASE)
            if mpo:
                fields["po_number"] = mpo.group(0)
                conf["po_number"] = 0.7

    def _extract_line_items(self, tokens, layout_text, ocr_text):
        """
        Naive attempt to extract line items by looking for table-like patterns:
        If LayoutLM tokens or OCR contain header words like Description, Qty, Unit, Total,
        attempt to parse following lines in OCR text into items.
        This is a best-effort fallback; returns [] if nothing found.
        """
        txt = ocr_text or layout_text or ""
        if not txt:
            return []
        if re.search(r'\bDescription\b.*\bQty\b.*\bUnit\b.*\bTotal\b', txt, re.IGNORECASE):
            # try a simple line-based parse after header
            lines = txt.splitlines()
            items = []
            header_idx = None
            for i, ln in enumerate(lines):
                if re.search(r'\bDescription\b.*\bQty\b.*\bUnit\b.*\bTotal\b', ln, re.IGNORECASE):
                    header_idx = i
                    break
            if header_idx is None:
                return []
            for ln in lines[header_idx+1: header_idx+20]:
                ln = ln.strip()
                if not ln: continue
                # naive: split by multiple spaces or tabs
                parts = re.split(r'\s{2,}|\t', ln)
                if len(parts) >= 3:
                    desc = parts[0]
                    qty = None
                    unit = None
                    total = None
                    # heuristics: last part numeric = total, second last = unit or qty
                    try:
                        total = float(parts[-1].replace(',', '').strip())
                    except:
                        total = None
                    try:
                        unit = float(parts[-2].replace(',', '').strip())
                    except:
                        unit = None
                    items.append({"description": desc, "qty": None, "unit_price": unit, "total": total})
            return items
        return []

    def _validate_and_flag(self, fields, conf, issues):
        # Amount sanity
        if "amount" in fields:
            try:
                a = float(fields["amount"])
                # reject tiny amounts likely to be token noise
                if a < 5:
                    issues.append("amount_too_small")
                    # keep the value but reduce confidence
                    conf["amount"] = min(conf.get("amount", 0.5), 0.5)
                if a > 10_000_000:
                    issues.append("amount_too_large")
                    fields.pop("amount", None)
                    conf.pop("amount", None)
            except:
                issues.append("amount_invalid")
                fields.pop("amount", None)
                conf.pop("amount", None)
        # Vendor heuristic: length + weird chars
        if "vendor" in fields:
            v = fields["vendor"]
            if len(v) < 3 or re.search(r'[^ -~]', v):
                issues.append("vendor_suspect")
                conf["vendor"] = min(conf.get("vendor", 0.5), 0.5)
        # date check
        if "date" in fields:
            d = fields["date"]
            if not isinstance(d, str) or len(d) < 6:
                issues.append("date_invalid")
                fields.pop("date", None)
                conf.pop("date", None)

        # ensure currency is standardized if present
        if "currency" in fields:
            cur = fields["currency"]
            if isinstance(cur, str):
                cur = cur.upper()
                if not re.match(r'^[A-Z]{3}$', cur):
                    # attempt to detect from amount text presence
                    conf.pop("currency", None)
                    fields.pop("currency", None)
                else:
                    fields["currency"] = cur
