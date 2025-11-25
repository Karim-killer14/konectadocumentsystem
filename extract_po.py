# extract_po.py
import re

class POExtractor:
    """
    Purchase Order extraction:
    - PO Number
    - Vendor
    - Date
    - Total
    """

    def extract(self, ocr_text: str, tokens):
        text = ocr_text.lower()
        fields = {}
        confidence = {}
        issues = []

        # --- PO Number ---
        m = re.search(r"(po|p\.o\.)[^\d]*(\d{3,})", text)
        if m:
            fields["po_number"] = m.group(2)
            confidence["po_number"] = 0.9
        else:
            fields["po_number"] = None
            issues.append("missing_po_number")

        # --- Vendor ---
        vendor = self._extract_vendor(text)
        fields["vendor"] = vendor
        confidence["vendor"] = 0.7 if vendor else 0.3
        if not vendor:
            issues.append("missing_vendor")

        # --- Date ---
        date = self._extract_date(text)
        fields["date"] = date
        confidence["date"] = 0.75 if date else 0.3
        if not date:
            issues.append("missing_date")

        # --- Total ---
        total = self._extract_amount(text)
        fields["total"] = total
        confidence["total"] = 0.8 if total else 0.3
        if not total:
            issues.append("missing_total")

        return fields, confidence, issues


    # -----------------
    # Helper Functions
    # -----------------
    def _extract_vendor(self, text):
        lines = text.split("\n")
        for line in lines[:5]:
            if len(line.strip()) > 3:
                return line.strip().title()
        return None

    def _extract_date(self, text):
        m = re.search(r"(\d{2}[\/\-]\d{2}[\/\-]\d{4})", text)
        return m.group(1) if m else None

    def _extract_amount(self, text):
        m = re.findall(r"\d[\d,\.]{2,}", text)
        if not m:
            return None
        nums = [float(x.replace(",", "")) for x in m]
        return max(nums)
