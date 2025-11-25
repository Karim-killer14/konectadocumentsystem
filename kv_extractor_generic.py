# kv_extractor_generic.py

from doc_classifier import DocumentClassifier
from extract_invoice import InvoiceExtractor
from extract_po import POExtractor
from extract_approval import ApprovalExtractor

class KVExtractorGeneric:
    def __init__(self, inferencer, preprocess=True, tesseract_allowed=True):
        self.model = inferencer
        self.cls = DocumentClassifier()

        self.inv_ex = InvoiceExtractor()
        self.po_ex = POExtractor()
        self.apv_ex = ApprovalExtractor()

    def extract(self, image):
        # Step 1 — raw tokens from LayoutLM
        tokens = self.model.infer(image)

        # Combine tokens into a long text string
        full_text = " ".join(t["token"] for t in tokens)

        # Step 2 — classify document
        doc_type = self.cls.classify(full_text)

        # Step 3 — extract fields using the correct extractor
        if doc_type == "invoice":
            fields, conf, issues = self.inv_ex.extract(full_text)

        elif doc_type == "po":
            fields, conf, issues = self.po_ex.extract(full_text)

        elif doc_type == "approval":
            fields, conf, issues = self.apv_ex.extract(full_text)

        else:
            fields, conf, issues = {}, {}, ["unknown_document"]

        return {
            "doc_type": doc_type,
            "fields": fields,
            "confidence": conf,
            "issues": issues
        }
