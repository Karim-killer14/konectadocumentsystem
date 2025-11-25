# extract_po.py

import re

class POExtractor:
    def extract(self, text: str):
        fields = {}
        conf = {}
        issues = []

        # --- PO Number ---
        m = re.search(r"(PO[-\s]*\d+[-\s]*\d+)", text, re.I)
        if m:
            fields["po_number"] = m.group(1)
            conf["po_number"] = 0.9

        # --- Delivery date ---
        m = re.search(r"delivery date[:\s]+([0-9\-\/]+)", text, re.I)
        if m:
            fields["delivery_date"] = m.group(1)
            conf["delivery_date"] = 0.7

        return fields, conf, issues
