# app.py
import streamlit as st
from PIL import Image
import io, json, pandas as pd, os
import pymupdf

from model import LayoutLMInferencer
from kv_extractor_generic import KVExtractorGeneric
from validator import validate_doc

st.set_page_config(page_title="Universal Document Processor", layout="wide")
st.title("Universal Document Processor — Hybrid (LayoutLM + OCR + Extractors)")
st.write("Uploads auto-processed. Handles invoices, POs, approvals and unknown documents.")

# cache model & extractor
@st.cache_resource
def load_resources():
    infer = LayoutLMInferencer(model_name="microsoft/layoutlmv3-base", device="cpu")
    extractor = KVExtractorGeneric(inferencer=infer, tesseract_allowed=True)
    return infer, extractor

inferencer, extractor = load_resources()

st.markdown("Tip: test using workspace file `/mnt/data/approval_001_native.pdf`")

uploaded = st.file_uploader("Upload PDF or Image", type=["pdf","png","jpg","jpeg"])

def pdf_to_images(file_bytes):
    pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages = []
    for p in range(pdf.page_count):
        page = pdf.load_page(p)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2,2))
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(img)
    return pages

if uploaded:
    st.info("Processing automatically...")
    file_bytes = uploaded.read()
    filename = uploaded.name

    if filename.lower().endswith(".pdf"):
        pages = pdf_to_images(file_bytes)
    else:
        pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    page_outputs = []
    for i, page in enumerate(pages):
        st.subheader(f"Page {i+1}")
        st.image(page, use_column_width=True)
        with st.spinner(f"Extracting page {i+1}..."):
            out = extractor.extract(page)
        page_outputs.append(out)

        st.markdown("**Extracted fields (page)**")
        st.json(out.get("fields", {}))
        st.markdown("**Issues & Confidence (page)**")
        st.json({"issues": out.get("issues", []), "confidence": out.get("confidence", {})})

    # Merge pages (first non-empty wins)
    merged = {"fields": {}, "confidence": {}, "issues": [], "doc_type": page_outputs[0].get("doc_type", "unknown")}
    for out in page_outputs:
        merged["issues"].extend(out.get("issues", []))
        for k, v in out.get("fields", {}).items():
            if not merged["fields"].get(k):
                merged["fields"][k] = v
                merged["confidence"][k] = out.get("confidence", {}).get(k, 0.5)

    st.header("Merged structured output")
    st.json({"fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"], "doc_type": merged["doc_type"]})

    # Validation & suggestions
    st.header("Validation & Suggestions")
    validation = validate_doc({"doc_type": merged["doc_type"], "fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]})
    st.json(validation)

    # Raw token view (first page) - helpful for debugging
    st.header("Raw tokens (first page) — LayoutLM")
    tokens = inferencer.infer(pages[0])
    df = pd.DataFrame(tokens)
    st.dataframe(df)

    # Downloads
    st.download_button("Download merged JSON", json.dumps({"fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"], "doc_type": merged["doc_type"]}, indent=2), file_name=f"{filename}_extracted.json")
    st.download_button("Download merged CSV", pd.DataFrame([merged["fields"]]).to_csv(index=False), file_name=f"{filename}_extracted.csv")

    if not validation["valid"]:
        st.warning("Document validation incomplete or has issues. See suggestions above.")
    else:
        st.success("Document validated successfully.")
