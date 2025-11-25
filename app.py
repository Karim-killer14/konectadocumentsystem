# app.py (final generic)
import streamlit as st
from PIL import Image
import io
import json
import pandas as pd
import pymupdf
from model import LayoutLMInferencer
from kv_extractor_generic import KVExtractorGeneric
from validator import validate_doc
from normalize import clean_layout_tokens

st.set_page_config(page_title="Automated Document Processor", layout="wide")
st.title("Automated Document Processor — Generic Extractor")
st.write("Uploads auto-processed. Handles invoices, POs, approvals and tolerates missing fields.")

@st.cache_resource
def load_inferencer():
    return LayoutLMInferencer(model_name="microsoft/layoutlmv3-base", device="cpu")

inferencer = load_inferencer()
extractor = KVExtractorGeneric(inferencer, preprocess=True, tesseract_allowed=True)

uploaded = st.file_uploader("Upload PDF or image", type=["pdf","png","jpg","jpeg"])
st.markdown("Tip: test using workspace file `/mnt/data/approval_001_native.pdf`")

def pil_from_bytes(file_bytes, filename):
    if filename.endswith(".pdf"):
        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages = []
        for p in range(pdf.page_count):
            page = pdf.load_page(p)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2,2))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pages.append(img)
        return pages
    else:
        return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

if uploaded:
    st.info("Processing automatically...")
    file_bytes = uploaded.read()
    filename = uploaded.name.lower()

    pages = pil_from_bytes(file_bytes, filename)

    outputs = []
    for i, page in enumerate(pages):
        st.subheader(f"Page {i+1}")
        st.image(page, use_column_width=True)
        with st.spinner(f"Extracting page {i+1}..."):
            out = extractor.extract(page)
        st.markdown("**Extracted fields (page)**")
        st.json(out["fields"])
        st.markdown("**Issues & Confidence (page)**")
        st.json({"issues": out["issues"], "confidence": out["confidence"]})
        outputs.append(out)

    # Merge pages (first non-empty wins)
    merged = {"fields": {}, "confidence": {}, "issues": []}
    for out in outputs:
        merged["issues"].extend(out.get("issues", []))
        for k, v in out.get("fields", {}).items():
            if k not in merged["fields"] or not merged["fields"][k]:
                merged["fields"][k] = v
                merged["confidence"][k] = out.get("confidence", {}).get(k, 0.5)

    st.subheader("Merged structured output")
    st.json({"fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]})

    # Validation and suggestions
    st.subheader("Validation & Suggestions")
    doc_type = outputs[0].get("doc_type", "unknown") if outputs else "unknown"
    validation = validate_doc({"doc_type": doc_type, "fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]})
    st.json(validation)

    # tokens view
    st.subheader("Raw tokens (first page)")
    tokens = inferencer.infer(pages[0])
    df = pd.DataFrame(tokens)
    st.dataframe(df)

    # Exports
    st.download_button("Download JSON", json.dumps({"fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]}, indent=2), file_name=f"{filename}_extracted.json")
    st.download_button("Download CSV", pd.DataFrame([merged["fields"]]).to_csv(index=False), file_name=f"{filename}_extracted.csv")

    # If not valid, show warning (but never error)
    if not validation["valid"]:
        st.warning("Document validation failed or is incomplete. See suggestions above.")
    else:
        st.success("Document validated successfully.")
