# kv_extractor_generic.py
import pytesseract
from doc_classifier import DocumentClassifier
from extract_invoice import InvoiceExtractor
from extract_po import POExtractor
from extract_approval import ApprovalExtractor

class KVExtractorGeneric:
    def __init__(self, inferencer, preprocess=True, tesseract_allowed=True):
        self.inferencer = inferencer
        self.preprocess = preprocess
        self.tesseract_allowed = tesseract_allowed

        self.classifier = DocumentClassifier()
        self.inv = InvoiceExtractor()
        self.po = POExtractor()
        self.appr = ApprovalExtractor()

    def extract(self, pil_image):
        # --- Step 1: OCR ---
        ocr_text = pytesseract.image_to_string(pil_image)

        # --- Step 2: Layout Tokens ---
        tokens = self.inferencer.infer(pil_image)

        # --- Step 3: Document classification ---
        doc_type = self.classifier.classify(ocr_text, tokens)

        # --- Step 4: Route to the correct extractor ---
        if doc_type == "invoice":
            fields, conf, issues = self.inv.extract(ocr_text, tokens)
        elif doc_type == "po":
            fields, conf, issues = self.po.extract(ocr_text, tokens)
        elif doc_type == "approval":
            fields, conf, issues = self.appr.extract(ocr_text, tokens)
        else:
            # unknown = fallback
            fields = {"raw_text": ocr_text}
            conf = {"raw_text": 0.5}
            issues = ["unknown_document_type"]

        # Standard output package
        return {
            "doc_type": doc_type,
            "fields": fields,
            "confidence": conf,
            "issues": issues
        }
