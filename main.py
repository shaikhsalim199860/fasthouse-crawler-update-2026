import shutil
import inspect
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
# Page Config
# -----------------------

st.set_page_config(
    page_title="Fasthouse & Seven Crawler",
    layout="wide"
)


# -----------------------
# Sidebar Controls
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
# Display Banner
# -----------------------

image_path = ARTIFACTS_DIR / f"{website}.jpg"
if image_path.exists():
    st.image(Image.open(image_path), use_container_width=True)


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


def get_required_columns(website, crawl_type):
    if website == "Seven":
        return ["Seller SKU", "URL"]

    if crawl_type == "A+ Images":
        return ["Seller SKU", "URL"]

    return ["Seller SKU", "URL", "No of bullets"]


def run_crawler_safe(crawler, df, mode, progress_bar, status_text):
    """
    Calls crawler safely depending on whether it supports progress arguments.
    """
    sig = inspect.signature(crawler)

    if "progress_bar" in sig.parameters:
        return crawler(
            df,
            mode,
            progress_bar=progress_bar,
            status_text=status_text
        )
    else:
        # fallback if scraper not updated yet
        status_text.text("Running scraper...")
        result = crawler(df, mode)
        progress_bar.progress(1.0)
        return result


# -----------------------
# File Upload
# -----------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
run = st.button("Run Scraper", type="primary")

crawler = fetch_text_and_images if website == "Fasthouse" else start


# -----------------------
# Main Logic
# -----------------------

if run:

    if not uploaded_file:
        st.warning("Please upload a CSV file first.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    required_columns = get_required_columns(website, crawl_type)

    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns. Needed: {required_columns}")
        st.stop()

    try:

        if crawl_type in ["Images", "A+ Images"]:
            clean_assets()

        progress_bar = st.progress(0)
        status_text = st.empty()

        output = run_crawler_safe(
            crawler,
            df,
            return_mode(crawl_type),
            progress_bar,
            status_text
        )

        progress_bar.empty()
        status_text.empty()

        st.success(f"{crawl_type} scraping completed successfully 🎉")
        st.balloons()

        # -----------------------
        # DATA MODE
        # -----------------------
        if crawl_type == "Data":
            st.download_button(
                label="Download Data CSV",
                data=convert_df(output),
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