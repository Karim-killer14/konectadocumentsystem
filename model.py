# model.py
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification, AutoModelForQuestionAnswering

class LayoutLMInferencer:
    """
    Lightweight LayoutLMv3 inference wrapper.
    """
    def __init__(self, model_name="microsoft/layoutlmv3-base", device="cpu"):
        self.device = torch.device("cpu")
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)

        try:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
            self.mode = "tc"
        except:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            self.mode = "qa"

    def infer(self, inputs, task="key_value"):
        image = inputs["image"]
        words = inputs["words"]
        boxes = inputs["boxes"]

        enc = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        if self.mode == "tc":
            outputs = self.model(**enc)
            sample_text = " ".join(words[:40])
            return {"parsed_text": sample_text}

        else:
            fields = ["Invoice Number", "Date", "Total", "Vendor"]
            results = {}

            for q in fields:
                q_enc = self.processor(image, q, return_tensors="pt").to(self.device)
                out = self.model(**q_enc)

                start = torch.argmax(out.start_logits)
                end = torch.argmax(out.end_logits)
                tokens = self.processor.tokenizer.convert_ids_to_tokens(
                    q_enc["input_ids"][0][start:end+1]
                )

                results[q] = self.processor.tokenizer.convert_tokens_to_string(tokens)

            return results
