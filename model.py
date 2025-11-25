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
    def infer(self, input_data, task="key_value"):
        """
        Accepts either:
        - PIL Image
        - Uploaded file (pdf, jpg, png)
        """
        # If PIL Image is passed → use directly
        if isinstance(input_data, Image.Image):
            image = input_data
        else:
            image = self.load_image(input_data)
    
        # run LayoutLM processor (no boxes, no words)
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
    
    
    def load_image(self, uploaded_file):
        """
        Support uploading PDFs or raw bytes.
        """
        if isinstance(uploaded_file, bytes):
            file_bytes = uploaded_file
        else:
            file_bytes = uploaded_file.read()
    
        # PDF?
        if hasattr(uploaded_file, "name") and uploaded_file.name.lower().endswith(".pdf"):
            import pymupdf as fitz
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    
        # Image?
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
