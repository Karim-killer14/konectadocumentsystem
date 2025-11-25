# kv_extractor_generic.py
import pytesseract
from normalize import clean_layout_tokens
from doc_classifier import DocumentClassifier
from extract_invoice import InvoiceExtractor
from extract_po import POExtractor
from extract_approval import ApprovalExtractor

class KVExtractorGeneric:
    """
    Hybrid dispatcher:
     - Gets clean OCR text via pytesseract (primary text input)
     - Uses LayoutLM tokens ONLY for optional confidence boosts / debugging
     - Classifies the document
     - Routes to format-specific extractor
     - Falls back to generic extraction when unknown
    """
    def __init__(self, inferencer, preprocess=True, tesseract_allowed=True):
        self.inferencer = inferencer
        self.tesseract_allowed = tesseract_allowed
        self.classifier = DocumentClassifier()
        self.inv = InvoiceExtractor()
        self.po = POExtractor()
        self.appr = ApprovalExtractor()

    def extract(self, pil_image):
        # 1) OCR (Tesseract)
        ocr_text = ""
        if self.tesseract_allowed:
            try:
                ocr_text = pytesseract.image_to_string(pil_image)
            except Exception:
                ocr_text = ""

        # 2) Clean OCR text
        clean_text = self._clean_text(ocr_text)

        # 3) Get LayoutLM tokens (for diagnostics/confidence)
        try:
            tokens = self.inferencer.infer(pil_image)
            token_text = " ".join(t.get("token", "") for t in tokens)
            token_text = clean_layout_tokens(token_text)
        except Exception:
            tokens = []
            token_text = ""

        # 4) Hybrid classification: prefer OCR-based classification
        doc_type = self.classifier.classify(clean_text if clean_text else token_text)

        # 5) Route to extractor
        out = None
        if doc_type == "invoice":
            out = self.inv.extract(clean_text, tokens)
        elif doc_type == "po":
            out = self.po.extract(clean_text, tokens)
        elif doc_type == "approval":
            out = self.appr.extract(clean_text, tokens)
        else:
            # fallback generic extraction using OCR heuristics
            out = self._generic_extract(clean_text, tokens)

        # 6) Ensure standard output shape (dict)
        result = {
            "doc_type": out.get("doc_type", doc_type if doc_type else "unknown"),
            "fields": out.get("fields", {}),
            "confidence": out.get("confidence", {}),
            "issues": out.get("issues", []),
            "raw": {
                "ocr_text": clean_text,
                "layout_text": token_text
            }
        }

        return result

    def _clean_text(self, text: str):
        if not text:
            return ""
        return " ".join(text.replace("\x0c", " ").split())

    def _generic_extract(self, text, tokens):
        # Very small fallback: try to grab amount/date/vendor heuristics
        fields = {}
        conf = {}
        issues = []

        import re
        # amount: largest number in text
        nums = re.findall(r"\d[\d,\.]+\d", text)
        if nums:
            try:
                vals = [float(n.replace(",", "")) for n in nums]
                fields["amount"] = max(vals)
                conf["amount"] = 0.6
            except:
                pass

        # date
        m = re.search(r"\d{2}[\/\-]\d{2}[\/\-]\d{4}", text)
        if m:
            fields["date"] = m.group(0)
            conf["date"] = 0.6

        # vendor (first non-empty line)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            fields["vendor"] = lines[0]
            conf["vendor"] = 0.5

        return {"doc_type": "unknown", "fields": fields, "confidence": conf, "issues": issues}
