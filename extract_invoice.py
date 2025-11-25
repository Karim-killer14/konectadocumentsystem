# extract_invoice.py

import re

class InvoiceExtractor:
    def extract(self, text: str):
        fields = {}
        conf = {}
        issues = []

        # --- Invoice number ---
        m = re.search(r"(INV[-\s]*\d+[-\s]*\d+)", text, re.I)
        if m:
            fields["invoice_number"] = m.group(1)
            conf["invoice_number"] = 0.9

        # --- Vendor ---
        m = re.search(r"vendor[:\s]+(.+?)\n", text, re.I)
        if m:
            fields["vendor"] = m.group(1).strip()
            conf["vendor"] = 0.75

        # --- Amount ---
        m = re.search(r"total[:\s]+([0-9\.,]+)", text, re.I)
        if m:
            try:
                fields["total_amount"] = float(m.group(1).replace(",", ""))
                conf["total_amount"] = 0.85
            except:
                pass

        return fields, conf, issues
