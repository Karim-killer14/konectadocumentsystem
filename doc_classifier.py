# doc_classifier.py
import re

class DocumentClassifier:
    """
    Hybrid classifier:
    - Uses OCR text BEFORE LayoutLM (fast)
    - Uses token patterns AFTER LayoutLM (more accurate)
    """

    KEYWORDS = {
        "invoice": ["invoice", "inv#", "invoice no", "billed to", "bill to", "tax invoice"],
        "po": ["purchase order", "po#", "p.o.", "supplier ref", "delivery date"],
        "approval": ["approval", "approved", "request id", "requested by", "approver", "status"]
    }

    def classify(self, ocr_text: str, tokens=None):
        text = ocr_text.lower()

        # --- Pre-LayoutLM Quick Detection ---
        for doc_type, kws in self.KEYWORDS.items():
            if any(k in text for k in kws):
                return doc_type

        # --- Token-Based Fallback ---
        if tokens:
            token_str = " ".join(t["token"].lower() for t in tokens)

            for doc_type, kws in self.KEYWORDS.items():
                if any(k in token_str for k in kws):
                    return doc_type

        return "unknown"
