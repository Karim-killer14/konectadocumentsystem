import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image
import io

class LayoutLMInferencer:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=None):
        """
        LayoutLMv3 inference wrapper
        """

        # Automatic device selection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # LayoutLMv3 runs its own OCR
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
        Converts:
        - Streamlit UploadedFile
        - bytes
        - PDF → first page converted to PNG
        - PNG/JPG → loaded as PIL Image
        """

        # CASE 1 — Streamlit UploadedFile
        if hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
            filename = file_obj.name

        # CASE 2 — Raw bytes
        elif isinstance(file_obj, (bytes, bytearray)):
            file_bytes = file_obj
            filename = ""

        else:
            raise ValueError("load_image() received unsupported input type.")

        # ---- PDF case ----
        if filename.lower().endswith(".pdf"):
            import pymupdf as fitz
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        # ---- Image case ----
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # ----------------------------------------------------------------------
    # MAIN INFERENCE PIPELINE
    # ----------------------------------------------------------------------
    def infer(self, input_data, task="key_value"):
        """
        Accepts:
        - PIL Image
        - Streamlit UploadedFile
        - bytes
        """

        # CASE 1 — Already a PIL Image
        if isinstance(input_data, Image.Image):
            image = input_data

        # CASE 2 — UploadedFile
        elif hasattr(input_data, "read"):
            image = self.load_image(input_data)

        # CASE 3 — Raw bytes (PDF or image)
        elif isinstance(input_data, (bytes, bytearray)):
            image = self.load_image(input_data)

        else:
            raise ValueError(f"infer() received unsupported input: {type(input_data)}")

        # -------- Run the LayoutLMv3 processor (OCR is done internally) -------
        enc = self.processor(
            image,
            padding="max_length",
            return_tensors="pt"
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            enc["input_ids"].squeeze()
        )

        results = []
        for token, pred in zip(tokens, predictions):
            if token not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]:
                results.append({"token": token, "label_id": pred})

        return results
