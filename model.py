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

        # IMPORTANT:
        # apply_ocr=True → LayoutLMv3 extracts text + boxes internally
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            apply_ocr=True
        )

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ----------------------------------------------------------------------
    # LOADING IMAGE OR PDF PAGE
    # ----------------------------------------------------------------------
    def load_image(self, uploaded_file):
        """
        Converts uploaded_file (pdf or image) into a PIL image
        """

        file_bytes = uploaded_file.read()

        # PDF case → convert first page to image
        if uploaded_file.name.lower().endswith(".pdf"):
            import fitz  # PyMuPDF
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")  
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Image case
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # ----------------------------------------------------------------------
    # MAIN INFERENCE
    # ----------------------------------------------------------------------
    def infer(self, uploaded_file, task="key_value"):
        """
        Runs LayoutLMv3 inference on input document.
        WARNING: LayoutLMv3 Base does token classification, NOT full form extraction.
        """
        image = self.load_image(uploaded_file)

        # ❗ IMPORTANT:
        # DO NOT supply boxes/words → LayoutLMv3 handles its own OCR
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
