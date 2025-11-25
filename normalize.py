# normalize.py
import re

def clean_layout_tokens(text: str) -> str:
    if not text:
        return ""
    # remove special layout markers from tokenized output
    text = text.replace("Ġ", " ")
    text = re.sub(r"<pad>|</s>|<s>|<unk>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
