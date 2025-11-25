import re

class ApprovalExtractor:
    def extract(self, text: str):
        fields = {}
        conf = {}

        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', text).strip()

        # --- Request ID ---
        m = re.search(r'\b(APV[- ]?\d{4}[- ]?\d{3,4})\b', clean, re.I)
        if m:
            fields["request_id"] = m.group(1).replace(" ", "")
            conf["request_id"] = 0.9

        # --- Requested By ---
        m = re.search(r'Requested By[: ]+([A-Za-z ]+)', clean, re.I)
        if m:
            fields["requested_by"] = m.group(1).strip()
            conf["requested_by"] = 0.8

        # --- Department ---
        m = re.search(r'Department[: ]+([A-Za-z ]+)', clean, re.I)
        if m:
            fields["department"] = m.group(1).strip()
            conf["department"] = 0.75

        # --- Amount ---
        m = re.search(r'Amount[: ]+([\d,.]+)', clean, re.I)
        if m:
            num = m.group(1).replace(",", "")
            try:
                fields["amount"] = float(num)
                conf["amount"] = 0.85
            except:
                pass

        # --- Purpose ---
        m = re.search(r'Purpose[: ]+([A-Za-z0-9 ,.-]+)', clean, re.I)
        if m:
            fields["purpose"] = m.group(1).strip()
            conf["purpose"] = 0.7

        # --- Approver (NEW) ---
        m = re.search(r'Approver[: ]+([A-Za-z ]+)', clean, re.I)
        if m:
            fields["approver"] = m.group(1).strip()
            conf["approver"] = 0.75

        # --- Status (NEW) ---
        m = re.search(r'Status[: ]+([A-Za-z ]+)', clean, re.I)
        if m:
            fields["status"] = m.group(1).strip()
            conf["status"] = 0.7

        return {
            "doc_type": "approval",
            "fields": fields,
            "confidence": conf,
            "issues": []
        }
