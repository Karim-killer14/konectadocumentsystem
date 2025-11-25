import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image
import io

class LayoutLMInferencer:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Processor performs internal OCR, so no boxes should be given
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            apply_ocr=True
        )

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ----------------------------------------------------------------------
    # UNIVERSAL IMAGE/PDF LOADER (only definition)
    # ----------------------------------------------------------------------
    def load_image(self, file_obj):
        """
        Accepts:
        - Streamlit UploadedFile
        - bytes
        - PDF → returns first-page image
        - PNG/JPG → returns PIL image
        """

        # Streamlit UploadedFile
        if hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
            filename = file_obj.name

        # Raw bytes
        elif isinstance(file_obj, (bytes, bytearray)):
            file_bytes = file_obj
            filename = ""

        else:
            raise ValueError("load_image() received unsupported input.")

        # PDF case
        if filename.lower().endswith(".pdf"):
            import pymupdf as fitz
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            imgbytes = pix.tobytes("png")
            return Image.open(io.BytesIO(imgbytes)).convert("RGB")

        # Image case
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # ----------------------------------------------------------------------
    # MAIN INFERENCE
    # ----------------------------------------------------------------------
    def infer(self, input_data):

        # Already a PIL Image
        if isinstance(input_data, Image.Image):
            image = input_data

        # UploadedFile
        elif hasattr(input_data, "read"):
            image = self.load_image(input_data)

        # Raw bytes
        elif isinstance(input_data, (bytes, bytearray)):
            image = self.load_image(input_data)

        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Run LayoutLM (internal OCR)
        enc = self.processor(
            image,
            padding="max_length",
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            enc["input_ids"].squeeze()
        )

        results = []
        for t, pid in zip(tokens, pred_ids):
            if t not in ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]:
                results.append({"token": t, "label_id": pid})

        return results
