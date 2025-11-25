# app.py  (UPGRADE)
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
# Example local test quick-run using the file you uploaded earlier:
st.markdown("**Tip:** to test a local file in the environment use the path: `/mnt/data/approval_001_native.pdf`")

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

    all_results = []
    for i, img in enumerate(pages):
        st.subheader(f"Page {i+1} preview")
        st.image(img, use_column_width=True)
        with st.spinner(f"Extracting page {i+1}..."):
            res = kv_extractor.extract_from_image(img)
        st.json(res["fields"])
        all_results.append(res)

    # Merge page-level fields (simple strategy: prefer first non-empty)
    merged = {}
    for r in all_results:
        for k,v in r["fields"].items():
            if k not in merged or not merged[k]:
                merged[k] = v

    st.subheader("Merged structured output")
    st.json(merged)

    # show raw tokens table for inspection (first page)
    st.subheader("Raw tokens (first page)")
    tokens = inferencer.infer(pages[0])
    df_tokens = pd.DataFrame(tokens)
    st.dataframe(df_tokens)

    # Export
    st.download_button("Download JSON", json.dumps(merged, indent=2), file_name=f"{filename}_extracted.json")
    st.download_button("Download CSV", pd.DataFrame([merged]).to_csv(index=False), file_name=f"{filename}_extracted.csv")

    st.success("Done.")
