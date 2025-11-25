# app.py
import streamlit as st
from PIL import Image
import io
import json
import pandas as pd
import pymupdf  # PyMuPDF (correct import for Streamlit Cloud)

from model import LayoutLMInferencer

st.set_page_config(page_title="Automated Document Processor", layout="wide")

st.title("📄 Automated Document Processing System (LayoutLMv3 + Streamlit)")
st.write("Upload a document (PDF or image). The system will automatically extract structured data.")

# ------------------------------------------------------------------
# Load LayoutLMv3 inferencer
# ------------------------------------------------------------------
@st.cache_resource
def load_inferencer():
    return LayoutLMInferencer(
        model_name="microsoft/layoutlmv3-base",
        device="cpu"
    )

inferencer = load_inferencer()

# ------------------------------------------------------------------
# File Upload
# ------------------------------------------------------------------
uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    st.info("📥 File uploaded. Processing automatically...")

    file_bytes = uploaded.read()
    filename = uploaded.name.lower()

    # ------------------------------------------------------------------
    # Convert input → PIL Image (1st page for PDF)
    # ------------------------------------------------------------------
    if filename.endswith(".pdf"):
        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
        page = pdf.load_page(0)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    st.subheader("📄 Document Preview")
    st.image(img, use_column_width=True)

    # ------------------------------------------------------------------
    # AUTOMATIC LAYOUTLM INFERENCE
    # ------------------------------------------------------------------
    with st.spinner("🔍 Running LayoutLMv3 model…"):
        results = inferencer.infer(img)

    # ------------------------------------------------------------------
    # Convert results → dataframe
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)

    st.subheader("🔎 Extracted Tokens")
    st.dataframe(df, use_container_width=True)

    # Editable version for manual correction
    st.subheader("✏️ Editable Extraction (Optional)")
    editable = st.data_editor(df)

    # ------------------------------------------------------------------
    # Export section
    # ------------------------------------------------------------------
    st.subheader("📤 Export Results")

    st.download_button(
        "Download JSON",
        editable.to_json(orient="records", indent=2),
        file_name=f"{filename}_tokens.json",
        mime="application/json"
    )

    st.download_button(
        "Download CSV",
        editable.to_csv(index=False),
        file_name=f"{filename}_tokens.csv",
        mime="text/csv"
    )

    st.success("🎉 Processing complete!")
