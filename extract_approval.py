# extract_approval.py
import re

class ApprovalExtractor:
    """
    Handles approval forms:
    - Request ID
    - Requested By
    - Department
    - Purpose
    - Amount
    """

    def extract(self, ocr_text: str, tokens):
        text = ocr_text.lower()
        fields = {}
        confidence = {}
        issues = []

        # --- Request ID ---
        m = re.search(r"(request id|req id|id)[^\d]*(\d{3,})", text)
        if m:
            fields["request_id"] = m.group(2)
            confidence["request_id"] = 0.9
        else:
            fields["request_id"] = None
            issues.append("missing_request_id")

        # --- Requested By ---
        requested = self._extract_requested_by(text)
        fields["requested_by"] = requested
        confidence["requested_by"] = 0.8 if requested else 0.4

        # --- Department ---
        dept = self._extract_department(text)
        fields["department"] = dept
        confidence["department"] = 0.75 if dept else 0.3

        # --- Purpose ---
        purpose = self._extract_purpose(text)
        fields["purpose"] = purpose
        confidence["purpose"] = 0.7 if purpose else 0.3

        # --- Amount ---
        amount = self._extract_amount(text)
        fields["amount"] = amount
        confidence["amount"] = 0.85 if amount else 0.3

        if not amount:
            issues.append("missing_amount")

        return fields, confidence, issues

    # ---------------------------
    # Helper Functions
    # ---------------------------
    def _extract_requested_by(self, text):
        m = re.search(r"requested by[:\s]*([a-z\s]+)", text)
        return m.group(1).title().strip() if m else None

    def _extract_department(self, text):
        m = re.search(r"department[:\s]*([a-z\s]+)", text)
        return m.group(1).title().strip() if m else None

    def _extract_purpose(self, text):
        m = re.search(r"purpose[:\s]*([a-z\s]+)", text)
        return m.group(1).title().strip() if m else None

    def _extract_amount(self, text):
        m = re.findall(r"\d[\d,\.]{2,}", text)
        if not m:
            return None
        nums = [float(x.replace(",", "")) for x in m]
        return max(nums)
