# model.py
import io
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image

class LayoutLMInferencer:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Use apply_ocr=True so the processor performs OCR internally.
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_image(self, file_obj):
        """
        Accept: streamlit UploadedFile, bytes, bytearray
        For PDFs convert first page to PNG using pymupdf.
        Returns a PIL.Image (RGB).
        """
        if hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
            filename = getattr(file_obj, "name", "").lower()
        elif isinstance(file_obj, (bytes, bytearray)):
            file_bytes = file_obj
            filename = ""
        else:
            raise ValueError("Unsupported input to load_image()")

        # PDF first-page rasterization
        if filename.endswith(".pdf"):
            import pymupdf as fitz
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            imgbytes = pix.tobytes("png")
            return Image.open(io.BytesIO(imgbytes)).convert("RGB")

        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    def infer(self, input_data):
        """
        Accepts:
        - PIL.Image.Image
        - Streamlit UploadedFile
        - bytes / bytearray
        Returns list of token dicts: [{"token": "...", "label_id": N}, ...]
        """
        # detect input type
        if isinstance(input_data, Image.Image):
            image = input_data
        elif hasattr(input_data, "read") or isinstance(input_data, (bytes, bytearray)):
            image = self.load_image(input_data)
        else:
            raise ValueError(f"infer() received unsupported input: {type(input_data)}")

        enc = self.processor(image, return_tensors="pt", padding="max_length")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()

        token_ids = enc["input_ids"].squeeze().cpu().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(token_ids)

        results = []
        for t, pid in zip(tokens, pred_ids):
            # filter special tokens
            if t in ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]:
                continue
            results.append({"token": t, "label_id": int(pid)})

        return results
