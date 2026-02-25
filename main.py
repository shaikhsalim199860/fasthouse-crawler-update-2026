import os
import shutil
from pathlib import Path

from PIL import Image
import streamlit as st
import pandas as pd

from fasthouse.scrape import fetch_text_and_images
from seven.scrape import start


# -----------------------
# Configuration
# -----------------------

ASSETS_DIR = Path("./assets")
ARTIFACTS_DIR = Path("./artifacts")


# -----------------------
# Sidebar Controls
# -----------------------

website = st.sidebar.radio(
    label="Select Website to crawl",
    options=["Fasthouse", "Seven"]
)

crawl_options = ["Data", "Images"]
if website == "Fasthouse":
    crawl_options.append("A+ Images")

crawl_type = st.sidebar.radio(
    label="Select Crawling Type",
    options=crawl_options
)


# -----------------------
# Display Banner Image
# -----------------------

image_path = ARTIFACTS_DIR / f"{website}.jpg"
if image_path.exists():
    st.image(Image.open(image_path))


st.header(f"{website} Crawler: {crawl_type} mode")

st.info("""
Required Columns:
- Seller SKU
- URL
- No of bullets (Not required for Seven)

Optional:
- ASIN (for images)
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
    """Delete and recreate assets folder safely."""
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def zip_assets(website: str):
    """Create ZIP archive from assets directory."""
    zip_path = shutil.make_archive(website, "zip", ASSETS_DIR)
    return zip_path


# -----------------------
# File Upload
# -----------------------

uploaded_file = st.file_uploader("Upload a CSV File...", type=["csv"])

run = st.button("Run Scraper")


# -----------------------
# Required Columns
# -----------------------

def get_required_columns(website, crawl_type):
    if website == "Seven":
        return ["Seller SKU", "URL"]

    if crawl_type == "A+ Images":
        return ["Seller SKU", "URL"]

    return ["Seller SKU", "URL", "No of bullets"]


crawler = fetch_text_and_images if website == "Fasthouse" else start


# -----------------------
# Main Logic
# -----------------------

if run:

    if not uploaded_file:
        st.warning("Please upload a CSV file before running the scraper.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    required_columns = get_required_columns(website, crawl_type)

    if not all(col in df.columns for col in required_columns):
        st.error(f"Required columns missing. Needed: {required_columns}")
        st.stop()

    try:
        # Clean assets for image modes
        if crawl_type in ["Images", "A+ Images"]:
            clean_assets()

        with st.spinner("Running scraper..."):
            output = crawler(df, return_mode(crawl_type), progress_bar=True)

        st.success(f"{crawl_type} scraping completed successfully 🎉")
        st.balloons()

        # -----------------------
        # Data Mode
        # -----------------------
        if crawl_type == "Data":
            st.download_button(
                label="Download Data",
                data=convert_df(output),
                file_name=f"{website}.csv",
                mime="text/csv"
            )

        # -----------------------
        # Image Modes
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
                    label="Download Images",
                    data=f,
                    file_name=f"{website}.zip",
                    mime="application/zip"
                )

    except Exception as e:
        st.exception(e)