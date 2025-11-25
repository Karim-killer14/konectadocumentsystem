# extract_approval.py
import re

class ApprovalExtractor:
    def extract(self, text: str, tokens=None):
        fields = {}
        conf = {}
        issues = []

        clean = re.sub(r'\s+', ' ', text).strip()

        # request id
        m = re.search(r'\b(APV[-\s]?\d{4}[-\s]?\d{3,4})\b', clean, re.I)
        if m:
            fields["request_id"] = m.group(1).replace(" ", "")
            conf["request_id"] = 0.9

        # requested by
        m = re.search(r'Requested By[:\s]*([A-Za-z\s,.-]{2,80})', clean, re.I)
        if m:
            fields["requested_by"] = m.group(1).strip()
            conf["requested_by"] = 0.85

        # department
        m = re.search(r'Department[:\s]*([A-Za-z\s\-]{2,60})', clean, re.I)
        if m:
            fields["department"] = m.group(1).strip()
            conf["department"] = 0.75

        # amount
        m = re.search(r'Amount[:\s]*([\d,\.]+)', clean, re.I)
        if m:
            try:
                fields["amount"] = float(m.group(1).replace(",", ""))
                conf["amount"] = 0.9
            except:
                pass

        # purpose
        m = re.search(r'Purpose[:\s]*([A-Za-z0-9\s,\-\.]{3,120})', clean, re.I)
        if m:
            fields["purpose"] = m.group(1).strip()
            conf["purpose"] = 0.7

        # approver (new)
        m = re.search(r'Approver[:\s]*([A-Za-z\s,.-]{2,80})', clean, re.I)
        if m:
            fields["approver"] = m.group(1).strip()
            conf["approver"] = 0.8

        # status (new)
        m = re.search(r'Status[:\s]*([A-Za-z\s]{3,20})', clean, re.I)
        if m:
            fields["status"] = m.group(1).strip()
            conf["status"] = 0.8

        return {"doc_type": "approval", "fields": fields, "confidence": conf, "issues": issues}
