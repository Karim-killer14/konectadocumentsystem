# app.py
import streamlit as st
from PIL import Image
import io
import json
import pandas as pd
import pymupdf
from model import LayoutLMInferencer
from extract_kv import KeyValueExtractor

st.set_page_config(page_title="Automated Document Processor", layout="wide")
st.title("📄 Automated Document Processing System (LayoutLMv3 + Heuristics)")
st.write("Auto-extract invoices / POs / approvals (local-only, no cloud).")

@st.cache_resource
def load_inferencer():
    return LayoutLMInferencer(model_name="microsoft/layoutlmv3-base", device="cpu")

inferencer = load_inferencer()
kv_extractor = KeyValueExtractor(inferencer, image_preprocess=True)

uploaded = st.file_uploader("Upload PDF or image", type=["pdf","png","jpg","jpeg"])

# quick local test tip
st.markdown("**Tip:** test a workspace file (example): `/mnt/data/approval_001_native.pdf`")

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

    page_results = []
    for i, img in enumerate(pages):
        st.subheader(f"Page {i+1}")
        st.image(img, use_column_width=True)
        with st.spinner(f"Extracting page {i+1}..."):
            res = kv_extractor.extract_from_image(img)
        st.json(res["fields"])
        page_results.append(res)

    # Merge page-level fields (prefer first non-empty)
    merged = {}
    merged_conf = {}
    for r in page_results:
        for k, v in r["fields"].items():
            if k not in merged or not merged[k]:
                merged[k] = v
                merged_conf[k] = r["confidence"].get(k, 0.5)

    st.subheader("Merged structured output")
    st.json({"fields": merged, "confidence": merged_conf})

    # tokens
    st.subheader("Raw tokens (first page)")
    tokens = inferencer.infer(pages[0])
    df_tokens = pd.DataFrame(tokens)
    st.dataframe(df_tokens)

    # Export
    st.download_button("Download Extracted JSON", json.dumps({"fields": merged, "confidence": merged_conf}, indent=2), file_name=f"{filename}_extracted.json")
    st.download_button("Download Extracted CSV", pd.DataFrame([merged]).to_csv(index=False), file_name=f"{filename}_extracted.csv")

    st.success("Done.")
