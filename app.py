# app.py
import streamlit as st
from PIL import Image
import io, json, pandas as pd
import pymupdf

from model import LayoutLMInferencer
from kv_extractor_generic import KVExtractorGeneric
from validator import validate_doc

st.set_page_config(page_title="Universal Document Processor", layout="wide")

st.title("Universal Document Processor (Invoices • POs • Approvals • Unknown)")
st.write("Uploads auto-processed. Missing fields never cause errors.")

@st.cache_resource
def load_inf():
    return LayoutLMInferencer(device="cpu")

inferencer = load_inf()
extractor = KVExtractorGeneric(inferencer)

# --- Upload ---
uploaded = st.file_uploader("Upload PDF or Image", type=["pdf","png","jpg","jpeg"])

def pdf_to_images(file_bytes):
    pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages = []
    for p in range(pdf.page_count):
        pix = pdf.load_page(p).get_pixmap(matrix=pymupdf.Matrix(2,2))
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(img)
    return pages

if uploaded:
    st.info("Processing document...")
    raw = uploaded.read()

    if uploaded.name.lower().endswith(".pdf"):
        imgs = pdf_to_images(raw)
    else:
        imgs = [Image.open(io.BytesIO(raw)).convert("RGB")]

    results = []

    for i, page in enumerate(imgs):
        st.subheader(f"Page {i+1}")
        st.image(page, use_column_width=True)

        with st.spinner("Extracting..."):
            out = extractor.extract(page)

        results.append(out)

        st.markdown("### Extracted fields")
        st.json(out)

    # Merge fields
    merged = {"fields": {}, "confidence": {}, "issues": [], "doc_type": results[0]["doc_type"]}

    for r in results:
        merged["issues"] += r["issues"]
        for k, v in r["fields"].items():
            if k not in merged["fields"] or not merged["fields"][k]:
                merged["fields"][k] = v
                merged["confidence"][k] = r["confidence"].get(k, 0.5)

    st.header("Merged Output")
    st.json(merged)

    # Validation
    validation = validate_doc(merged)
    st.header("Validation & Suggestions")
    st.json(validation)

    st.download_button("Download JSON", json.dumps(merged, indent=2), file_name=uploaded.name+"_extracted.json")
    st.download_button("Download CSV", pd.DataFrame([merged["fields"]]).to_csv(index=False), file_name=uploaded.name+"_extracted.csv")
