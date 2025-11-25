# extract_invoice.py
import re

class InvoiceExtractor:
    """
    Extracts invoice fields using:
    - regex patterns
    - keyword matching
    - numeric heuristics
    """

    def extract(self, ocr_text: str, tokens):
        text = ocr_text.lower()
        fields = {}
        confidence = {}
        issues = []

        # --- Document ID ---
        m = re.search(r"(invoice|inv)[^\d]*(\d{3,})", text)
        if m:
            fields["document_id"] = m.group(2)
            confidence["document_id"] = 0.9
        else:
            fields["document_id"] = None
            issues.append("missing_document_id")

        # --- Vendor ---
        vendor = self._extract_vendor(text)
        fields["vendor"] = vendor
        confidence["vendor"] = 0.7 if vendor else 0.3
        if not vendor:
            issues.append("missing_vendor")

        # --- Date ---
        date = self._extract_date(text)
        fields["date"] = date
        confidence["date"] = 0.8 if date else 0.3
        if not date:
            issues.append("missing_date")

        # --- Amounts ---
        amounts = re.findall(r"\d[\d,\.]{2,}", text)
        clean_amounts = [float(a.replace(",", "")) for a in amounts if self._is_valid_amount(a)]

        if clean_amounts:
            total = max(clean_amounts)  # largest number tends to be total
            fields["total"] = total
            confidence["total"] = 0.85
        else:
            fields["total"] = None
            issues.append("missing_total")

        return fields, confidence, issues

    # -----------------------------
    # Helper methods
    # -----------------------------
    def _is_valid_amount(self, amt):
        return re.match(r"\d[\d,\.]+\d", amt)

    def _extract_vendor(self, text):
        # naive but effective vendor detection
        lines = text.split("\n")
        for line in lines[:5]:  # top of invoice usually has vendor
            if len(line.strip()) > 3:
                return line.strip().title()
        return None

    def _extract_date(self, text):
        m = re.search(r"(\d{2}[\/\-]\d{2}[\/\-]\d{4})", text)
        if m:
            return m.group(1)
        return None
