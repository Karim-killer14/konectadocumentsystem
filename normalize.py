# normalize.py
import re

def clean_layout_tokens(text: str) -> str:
    if not text:
        return ""
    text = text.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
    text = text.replace("▁", " ")
    # remove repeated spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stray non-printable
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text

def normalize_currency_code(cur: str):
    if not cur: return None
    cur = cur.strip().upper()
    if len(cur) == 3 and cur.isalpha():
        return cur
    # common mapping
    mapping = {"AED": "AED", "A E D": "AED", "USD": "USD", "US$":"USD"}
    return mapping.get(cur, None)
