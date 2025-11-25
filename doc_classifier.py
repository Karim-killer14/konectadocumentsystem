# doc_classifier.py
import re

class DocumentClassifier:
    """
    Lightweight regex-based classifier.
    Looks at extracted tokens or raw OCR text.
    """

    def classify(self, text: str) -> str:
        text_low = text.lower()

        # ---- APPROVAL ----
        if any(k in text_low for k in [
            "approval", "request id", "requested by", "approver", "status"
        ]):
            return "approval"

        # ---- INVOICE ----
        if any(k in text_low for k in [
            "invoice", "inv-", "vat", "subtotal", "grand total", "bill to"
        ]):
            return "invoice"

        # ---- PURCHASE ORDER ----
        if any(k in text_low for k in [
            "purchase order", "po-", "delivery date", "supplier"
        ]):
            return "po"

        # ---- UNKNOWN ----
        return "unknown"
