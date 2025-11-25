# app.py
import streamlit as st
from PIL import Image
import io
import json
import pandas as pd
from model import LayoutLMInferencer

st.set_page_config(page_title="Automated Document Processor", layout="wide")

st.title("📄 Automated Document Processing System (LayoutLMv3 + Streamlit)")
st.write("Upload a document (PDF or image). The system will automatically extract structured data.")

# ------------------------------------------------------------------
# Load inferencer
# ------------------------------------------------------------------
@st.cache_resource
def load_inferencer():
    return LayoutLMInferencer(model_name="microsoft/layoutlmv3-base", device="cpu")

inferencer = load_inferencer()

# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------
uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    st.info("📥 File uploaded. Processing automatically...")

    raw = uploaded.read()
    filename = uploaded.name

    # --------------------------------------------------------------
    # Convert uploaded file into a PIL image
    # --------------------------------------------------------------
    if filename.lower().endswith(".pdf"):
        import fitz  # PyMuPDF
        pdf = fitz.open(stream=raw, filetype="pdf")
        page = pdf.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        image = Image.open(io.BytesIO(raw)).convert("RGB")

    st.subheader("📄 Document Preview")
    st.image(image, use_column_width=True)

    # --------------------------------------------------------------
    # AUTOMATIC LAYOUTLM INFERENCE
    # --------------------------------------------------------------
    with st.spinner("🔍 Running LayoutLMv3 model..."):
        results = inferencer.infer(image)

    # --------------------------------------------------------------
    # Convert raw tokens → structured table (very simple)
    # --------------------------------------------------------------
    df = pd.DataFrame(results)

    st.subheader("🔎 Extracted Tokens")
    st.dataframe(df, use_container_width=True)

    # Let the user edit token labels if desired
    st.subheader("✏️ Editable Extraction (Optional)")
    editable = st.data_editor(df)

    # --------------------------------------------------------------
    # Export section
    # --------------------------------------------------------------
    st.subheader("📤 Export Results")

    json_out = editable.to_json(orient="records", indent=2)
    csv_out = editable.to_csv(index=False)

    st.download_button(
        "Download JSON",
        json_out,
        file_name=f"{filename}_tokens.json",
        mime="application/json"
    )

    st.download_button(
        "Download CSV",
        csv_out,
        file_name=f"{filename}_tokens.csv",
        mime="text/csv"
    )

    st.success("🎉 Processing complete!")
