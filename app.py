# app.py
import streamlit as st
from PIL import Image
import io
import json
import pandas as pd

from utils import (
    pdf_to_images,
    ocr_and_layout,
    visualize_layout_on_image,
    prepare_layoutlm_inputs,
)

from model import LayoutLMInferencer

st.set_page_config(page_title="Automated Document Processor", layout="wide")

st.title("📄 Automated Document Processing System (LayoutLM + Streamlit)")
st.write(
    "Upload PDFs or images. The system performs OCR, extracts layout, runs LayoutLM, "
    "and allows you to validate & export structured data."
)

# Load the model (cached)
@st.cache_resource
def load_inferencer():
    return LayoutLMInferencer(model_name="microsoft/layoutlmv3-base", device="cpu")

inferencer = load_inferencer()

uploaded = st.file_uploader(
    "Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded:
    raw = uploaded.read()
    filename = uploaded.name

    # Convert PDF or image
    if filename.lower().endswith(".pdf"):
        pages = pdf_to_images(io.BytesIO(raw))
    else:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        pages = [img]

    page_index = st.sidebar.number_input(
        "Page", min_value=1, max_value=len(pages), value=1
    ) - 1

    image = pages[page_index]
    st.subheader(f"Page {page_index+1}")
    st.image(image, use_column_width=True)

    # OCR + Layout
    if st.button("Run OCR + Layout"):
        with st.spinner("Extracting text and layout using Tesseract…"):
            layout_blocks = ocr_and_layout(image)
            st.session_state["layout"] = layout_blocks

        st.success("OCR completed.")
        st.image(
            visualize_layout_on_image(image, layout_blocks),
            caption="Detected text regions",
        )

    # LayoutLM
    if "layout" in st.session_state:
        if st.button("Run LayoutLM Extraction"):
            with st.spinner("Running LayoutLM model inference…"):
                inputs = prepare_layoutlm_inputs(image, st.session_state["layout"])
                results = inferencer.infer(inputs, task="key_value")

            st.session_state["results"] = results
            st.success("Extraction finished.")

    # Editable results
    if "results" in st.session_state:
        st.subheader("Extracted Fields (Editable)")
        edited = {}

        for key, val in st.session_state["results"].items():
            edited[key] = st.text_input(key, value=val)

        # Export
        if st.button("Save Export Files"):
            json_data = json.dumps(edited, indent=2)
            csv_data = pd.DataFrame([edited]).to_csv(index=False)

            st.download_button(
                "Download JSON", json_data, file_name=f"{filename}_extracted.json"
            )
            st.download_button(
                "Download CSV", csv_data, file_name=f"{filename}_extracted.csv"
            )

            st.success("Files ready to download!")

