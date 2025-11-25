# extract_approval.py

import re

class ApprovalExtractor:
    def extract(self, text: str):
        fields = {}
        conf = {}
        issues = []

        # --- Request ID ---
        m = re.search(r"(APV[-\s]*\d+[-\s]*\d+)", text, re.I)
        if m:
            fields["request_id"] = m.group(1)
            conf["request_id"] = 0.9

        # --- Requested by ---
        m = re.search(r"requested by[:\s]+(.+)", text, re.I)
        if m:
            fields["requested_by"] = m.group(1).strip()
            conf["requested_by"] = 0.8

        # --- Department ---
        m = re.search(r"department[:\s]+(.+)", text, re.I)
        if m:
            fields["department"] = m.group(1).strip()
            conf["department"] = 0.75

        # --- Amount ---
        m = re.search(r"amount[:\s]+([0-9\.,]+)", text, re.I)
        if m:
            try:
                fields["amount"] = float(m.group(1).replace(",", ""))
                conf["amount"] = 0.85
            except:
                pass

        # --- Purpose ---
        m = re.search(r"purpose[:\s]+(.+)", text, re.I)
        if m:
            fields["purpose"] = m.group(1).strip()
            conf["purpose"] = 0.7

        return fields, conf, issues
