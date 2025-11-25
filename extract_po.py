# extract_po.py
import re

class POExtractor:
    def extract(self, text: str, tokens=None):
        fields = {}
        conf = {}
        issues = []

        m = re.search(r"(po(?:#| number)?[:\s-]*([A-Z0-9\-\/]+))", text, re.I)
        if m:
            fields["po_number"] = m.group(2).strip()
            conf["po_number"] = 0.9

        md = re.search(r"delivery date[:\s-]*([0-9\/\-]+)", text, re.I)
        if md:
            fields["delivery_date"] = md.group(1).strip()
            conf["delivery_date"] = 0.7

        # vendor heuristic
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            fields.setdefault("vendor", lines[0])
            conf["vendor"] = 0.6

        # total amount
        nums = re.findall(r"\d[\d,\.]+\d", text)
        if nums:
            try:
                vals = [float(n.replace(",", "")) for n in nums]
                fields["total"] = max(vals)
                conf["total"] = 0.8
            except:
                pass

        return {"doc_type": "po", "fields": fields, "confidence": conf, "issues": issues}
