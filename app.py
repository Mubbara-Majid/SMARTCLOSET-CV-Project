import streamlit as st
import cv2
import numpy as np
from analyzer import analyze_item
from recommender import recommend_outfit

st.set_page_config(page_title="AI Wardrobe Recommender", layout="wide")

st.title("👗 AI Wardrobe Outfit Recommender")

uploaded_files = st.file_uploader(
    "Upload wardrobe items (ONE cloth per image)",
    type=["jpg","png"],
    accept_multiple_files=True
)

items = []

if uploaded_files:
    st.subheader("Uploaded Wardrobe Items")

    for file in uploaded_files:
        image = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        analyzed = analyze_item(image, file.name)
        items.append(analyzed)

        st.image(image, width=150)
        st.write(
            f"**{analyzed['category'].upper()}** | Color: {analyzed['color']}"
        )

    if len(items) >= 3:
        outfit = recommend_outfit(items)

        if outfit:
            st.subheader("✅ Recommended Outfit")
            cols = st.columns(3)
            for col, item in zip(cols, outfit):
                col.image(item["image"], width=200)
                col.write(item["category"].upper())
                col.write(item["color"])
        else:
            st.warning("Not enough compatible items.")
    else:
        st.warning("Upload at least 1 Top, 1 Bottom, and 1 Shoe.")
