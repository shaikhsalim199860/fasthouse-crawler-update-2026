import os
import shutil
from pathlib import Path

from PIL import Image
import streamlit as st
import pandas as pd

from fasthouse.scrape import fetch_text_and_images
from seven.scrape import start


# -----------------------
# Config
# -----------------------

ASSETS_DIR = Path("./assets")
ARTIFACTS_DIR = Path("./artifacts")

st.set_page_config(
    page_title="Fasthouse & Seven Crawler",
    layout="wide"
)


# -----------------------
# Sidebar
# -----------------------

website = st.sidebar.radio(
    "Select Website to crawl",
    ["Fasthouse", "Seven"]
)

crawl_options = ["Data", "Images"]
if website == "Fasthouse":
    crawl_options.append("A+ Images")

crawl_type = st.sidebar.radio(
    "Select Crawling Type",
    crawl_options
)


# -----------------------
# Banner
# -----------------------

image_path = ARTIFACTS_DIR / f"{website}.jpg"
if image_path.exists():
    st.image(Image.open(image_path), width="stretch")


st.title(f"{website} Crawler")
st.subheader(f"Mode: {crawl_type}")

st.info("""
**Required Columns**
- Seller SKU  
- URL  
- No of bullets *(Not required for Seven)*  

**Optional**
- ASIN *(Required for image modes if available)*
""")


# -----------------------
# Helpers
# -----------------------

@st.cache_data
def return_mode(crawl_type: str) -> str:
    return {
        "Data": "fetch_data",
        "Images": "fetch_images",
        "A+ Images": "A_Plus_fetch_images"
    }[crawl_type]


@st.cache_data
def convert_df(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8-sig")


def clean_assets():
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def zip_assets(website: str):
    return shutil.make_archive(website, "zip", ASSETS_DIR)


# -----------------------
# Upload
# -----------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
run = st.button("Run Scraper", type="primary")

crawler = fetch_text_and_images if website == "Fasthouse" else start


# -----------------------
# Required Columns
# -----------------------

if crawl_type == "A+ Images":
    fasthouse_required_columns = ["Seller SKU", "URL"]
else:
    fasthouse_required_columns = ["Seller SKU", "URL", "No of bullets"]

seven_required_columns = ["Seller SKU", "URL"]

required_columns = (
    fasthouse_required_columns
    if website == "Fasthouse"
    else seven_required_columns
)


# -----------------------
# Main Logic
# -----------------------

if run:

    if not uploaded_file:
        st.warning("Please upload a CSV file first.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns. Required: {required_columns}")
        st.stop()

    try:

        # Clean assets for image modes
        if crawl_type in ["Images", "A+ Images"]:
            clean_assets()

        with st.spinner("Running scraper..."):
            out = crawler(df, return_mode(crawl_type), progress_bar=True)

        st.success(f"{crawl_type} scraping completed successfully 🎉")
        st.balloons()

        # -----------------------
        # DATA MODE
        # -----------------------
        if crawl_type == "Data":
            st.download_button(
                label="Download Data CSV",
                data=convert_df(out),
                file_name=f"{website}.csv",
                mime="text/csv"
            )

        # -----------------------
        # IMAGE MODES
        # -----------------------
        if crawl_type in ["Images", "A+ Images"]:

            csv_name = f"{website}__images.csv"
            source_csv = Path(f"./{csv_name}")
            target_csv = ASSETS_DIR / csv_name

            if source_csv.exists():
                shutil.move(source_csv, target_csv)

            zip_path = zip_assets(website)

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Download Images ZIP",
                    data=f,
                    file_name=f"{website}.zip",
                    mime="application/zip"
                )

    except Exception as e:
        st.exception(e)