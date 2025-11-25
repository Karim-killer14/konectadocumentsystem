# doc_classifier.py
import re

class DocumentClassifier:
    """
    Simple hybrid classifier using OCR text (preferred) and token text fallback.
    """
    def classify(self, text: str) -> str:
        if not text:
            return "unknown"
        t = text.lower()

        if any(k in t for k in ["approval", "request id", "requested by", "approver", "status"]):
            return "approval"
        if any(k in t for k in ["invoice", "inv", "bill to", "vat", "subtotal", "grand total"]):
            return "invoice"
        if any(k in t for k in ["purchase order", "po ", "delivery date", "supplier"]):
            return "po"
        return "unknown"
