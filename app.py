# app.py (final)
import streamlit as st
from PIL import Image
import io
import json
import pandas as pd
import pymupdf
from model import LayoutLMInferencer
from extract_kv import FinalKVExtractor
from validator import validate_fields

st.set_page_config(page_title="Automated Document Processor", layout="wide")
st.title("📄 Automated Document Processor — Final Extractor")
st.write("Auto-extract invoices / POs / approvals (local-only). Shows validation and suggestions.")

@st.cache_resource
def load_inferencer():
    return LayoutLMInferencer(model_name="microsoft/layoutlmv3-base", device="cpu")

inferencer = load_inferencer()
kv = FinalKVExtractor(inferencer, preprocess=True, tesseract_allowed=True)

uploaded = st.file_uploader("Upload PDF or image", type=["pdf","png","jpg","jpeg"])
st.markdown("Tip: to test a workspace file use `/mnt/data/approval_001_native.pdf`")

if uploaded:
    st.info("Processing file...")
    file_bytes = uploaded.read()
    filename = uploaded.name.lower()

    pages = []
    if filename.endswith(".pdf"):
        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
        for p in range(pdf.page_count):
            page = pdf.load_page(p)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2,2))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pages.append(img)
    else:
        pages.append(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

    # process each page
    page_outputs = []
    for i, img in enumerate(pages):
        st.subheader(f"Page {i+1} preview")
        st.image(img, use_column_width=True)
        with st.spinner(f"Extracting page {i+1}..."):
            out = kv.extract_from_image(img)
        st.json(out["fields"])
        page_outputs.append(out)

    # merge simple: take first non-empty for each field and combine issues/confidence
    merged = {"fields": {}, "confidence": {}, "issues": []}
    for po in page_outputs:
        merged["issues"].extend(po.get("issues", []))
        for k, v in po.get("fields", {}).items():
            if k not in merged["fields"] or not merged["fields"][k]:
                merged["fields"][k] = v
                merged["confidence"][k] = po.get("confidence", {}).get(k, 0.5)

    st.subheader("Merged structured output")
    st.json({"fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]})

    # validation
    st.subheader("Validation & Suggestions")
    val = validate_fields({"doc_type": page_outputs[0].get("doc_type", "invoice"), "fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]})
    st.json(val)

    # Raw tokens for debugging (first page)
    st.subheader("Raw tokens (first page)")
    tokens = inferencer.infer(pages[0])
    df_tokens = pd.DataFrame(tokens)
    st.dataframe(df_tokens)

    # Export & Save
    st.subheader("Export")
    st.download_button("Download JSON", json.dumps({"fields": merged["fields"], "confidence": merged["confidence"], "issues": merged["issues"]}, indent=2), file_name=f"{filename}_extracted.json")
    st.download_button("Download CSV", pd.DataFrame([merged["fields"]]).to_csv(index=False), file_name=f"{filename}_extracted.csv")

    # If invalid, show step-by-step suggestions
    if not val["valid"]:
        st.warning("Extraction is incomplete or invalid. See suggestions above for fixes.")
