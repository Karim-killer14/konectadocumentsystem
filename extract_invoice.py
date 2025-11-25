# extract_invoice.py
import re

class InvoiceExtractor:
    """
    Extract key invoice fields from clean OCR text.
    Returns a dict: {doc_type, fields, confidence, issues}
    """
    def extract(self, text: str, tokens=None):
        fields = {}
        conf = {}
        issues = []

        # invoice number
        m = re.search(r"(invoice\s*(no|#|number)?[:\s-]*([A-Z0-9\-\/]+))", text, re.I)
        if m:
            fields["invoice_number"] = m.group(3).strip()
            conf["invoice_number"] = 0.9

        # vendor: try heuristics (top lines)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            fields.setdefault("vendor", lines[0])
            conf["vendor"] = 0.6

        # date
        md = re.search(r"(\d{2}[\/\-]\d{2}[\/\-]\d{4})", text)
        if md:
            fields["date"] = md.group(1)
            conf["date"] = 0.8

        # amounts: take largest numeric
        nums = re.findall(r"\d[\d,\.]+\d", text)
        if nums:
            try:
                vals = [float(n.replace(",", "")) for n in nums]
                fields["total"] = max(vals)
                conf["total"] = 0.85
            except:
                pass
        else:
            issues.append("missing_total")

        return {"doc_type": "invoice", "fields": fields, "confidence": conf, "issues": issues}
