# kv_extractor_generic.py

import pytesseract
from doc_classifier import DocumentClassifier
from extract_invoice import InvoiceExtractor
from extract_po import POExtractor
from extract_approval import ApprovalExtractor

class KVExtractorGeneric:
    def __init__(self, inferencer, preprocess=True, tesseract_allowed=True):
        self.model = inferencer
        self.tesseract_allowed = tesseract_allowed

        # Light rule-based classifier
        self.cls = DocumentClassifier()

        # Format-specific extractors
        self.inv_ex = InvoiceExtractor()
        self.po_ex = POExtractor()
        self.apv_ex = ApprovalExtractor()

    # ------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------
    def extract(self, image):

        # --------------------------------------------------------
        # 1. Real OCR text (clean, no Ġ, no <pad>)
        # --------------------------------------------------------
        if self.tesseract_allowed:
            ocr_text = pytesseract.image_to_string(image)
        else:
            # fallback: join tokens from LayoutLM (not recommended)
            tokens = self.model.infer(image)
            ocr_text = " ".join(t["token"] for t in tokens)

        clean_text = self._clean_ocr_text(ocr_text)

        # --------------------------------------------------------
        # 2. Classify document based on OCR text
        # --------------------------------------------------------
        doc_type = self.cls.classify(clean_text)

        # --------------------------------------------------------
        # 3. Format-specific extraction
        # --------------------------------------------------------
        if doc_type == "invoice":
            fields, conf, issues = self.inv_ex.extract(clean_text)

        elif doc_type == "po":
            fields, conf, issues = self.po_ex.extract(clean_text)

        elif doc_type == "approval":
            fields, conf, issues = self.apv_ex.extract(clean_text)

        else:
            fields, conf, issues = {}, {}, ["unknown_document"]

        # --------------------------------------------------------
        # 4. Add model confidence (optional)
        # --------------------------------------------------------
        # LayoutLM not used for text extraction; we only take confidence boost
        for k in fields:
            if k not in conf:
                conf[k] = 0.7  # reasonable default

        return {
            "doc_type": doc_type,
            "fields": fields,
            "confidence": conf,
            "issues": issues
        }

    # --------------------------------------------------------
    # OCR cleaner
    # --------------------------------------------------------
    def _clean_ocr_text(self, text: str):
        text = text.replace("\x0c", "")  # remove form-feed
        text = text.replace("\n\n", "\n")
        text = " ".join(text.split())
        return text
