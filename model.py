import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image
import io

class LayoutLMInferencer:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # LayoutLMv3 OCR-enabled processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            apply_ocr=True
        )

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ----------------------------------------------------------------------
    # UNIVERSAL IMAGE/PDF LOADER
    # ----------------------------------------------------------------------
    def load_image(self, file_obj):
        """
        Safely loads PIL image from:
        - Streamlit UploadedFile
        - raw bytes
        - PDF (first page)
        """

        # CASE 1 — Streamlit UploadedFile
        if hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
            filename = file_obj.name.lower()

        # CASE 2 — Bytes
        elif isinstance(file_obj, (bytes, bytearray)):
            file_bytes = file_obj
            filename = ""

        else:
            raise ValueError("load_image(): unsupported input type.")

        # If PDF
        if filename.endswith(".pdf"):
            import pymupdf as fitz
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        # Otherwise: PNG/JPG/etc
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # ----------------------------------------------------------------------
    # MAIN INFERENCE
    # ----------------------------------------------------------------------
    def infer(self, input_data):
        """
        Accepts:
        - PIL.Image.Image
        - Streamlit UploadedFile
        - bytes
        """

        # CASE 1 — Already a PIL Image
        if isinstance(input_data, Image.Image):
            image = input_data

        # CASE 2 — UploadedFile
        elif hasattr(input_data, "read"):
            image = self.load_image(input_data)

        # CASE 3 — Raw bytes
        elif isinstance(input_data, (bytes, bytearray)):
            image = self.load_image(input_data)

        else:
            raise ValueError(f"infer(): unsupported input type {type(input_data)}")

        # ---------------------------------------------------------
        # FEED DIRECTLY TO LayoutLMv3 PROCESSOR (NO BOXES)
        # ---------------------------------------------------------
        enc = self.processor(
            image,
            return_tensors="pt",
            padding="max_length"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()

        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            enc["input_ids"].squeeze()
        )

        # Clean output
        results = []
        for token, pred in zip(tokens, pred_ids):
            if token not in ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]:
                results.append({"token": token, "label_id": pred})

        return results
