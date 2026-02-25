import os
import shutil
from PIL import Image

import streamlit as st
import pandas as pd

from fasthouse.scrape import fetch_text_and_images
from seven.scrape import start


website = st.sidebar.radio(label="Select Website to crawl", options=["Fasthouse", "Seven"])

if website == "Fasthouse":
    crawl_type = st.sidebar.radio(label="Select Crawling Type", options=["Data", "Images","A+ Images"])
else:
    crawl_type = st.sidebar.radio(label="Select Crawling Type", options=["Data", "Images"])


img = Image.open(f"./artifacts/{website}.jpg")

st.image(img)

st.header(f"{website} Crawler: {crawl_type} mode")
st.info(
    """
    Required Columns are:
    - Seller SKU
    - URL
    - No of bullets (Not required for Seven)

    Optional Columns are:
    - ASIN (for images)
"""
)


@st.cache
def return_mode(crawl_type: str) -> str:
    return {"Data": "fetch_data", "Images": "fetch_images","A+ Images":"A_Plus_fetch_images"}[crawl_type]


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8-sig")


def clean_assets():
    pass


uploaded_file = st.file_uploader("Upload a CSV File...", type=".csv")


if crawl_type == "A+ Images":
    fasthouse_required_columns = ["Seller SKU", "URL"]
else:
    fasthouse_required_columns = ["Seller SKU", "URL", "No of bullets"]

seven_required_columns = ["Seller SKU", "URL"]

required_columns = fasthouse_required_columns if website == "Fasthouse" else seven_required_columns

if website == "Fasthouse":
    crawler = fetch_text_and_images
else:
    crawler = start

run = st.button("Run Scraper")

if run and not uploaded_file:
    st.warning("Need to upload a file and then click Run Scraper.")

if uploaded_file and run:
    df = pd.read_csv(uploaded_file)

    all_present = all([col in df.columns for col in required_columns])

    if not all_present:
        st.error("Required Columns do not match. Please check and upload again.")
        st.stop()

    try:

        if crawl_type == "A+ Images":
            if os.path.exists("./assets"):
                shutil.rmtree("./assets")


        # Clean assets directory.
        if crawl_type == "Images":
            if os.path.exists("./assets"):
                shutil.rmtree("./assets")

        out = crawler(df, return_mode(crawl_type), progress_bar=True)
        st.success(f"Done. {crawl_type} available for download.")
        st.balloons()

        if crawl_type == "Data":
            st.download_button(
                label="Download Data", data=convert_df(out), file_name=f"{website}.csv", mime="text/csv"
            )

        if crawl_type == "Images":
            # Move the fasthouse.csv file inside the assets folder.
            shutil.move(f"./{website}__images.csv", f"./assets/{website}__images.csv")
            shutil.make_archive(f"{website}", "zip", "./assets/")

            with open(f"{website}.zip", "rb") as fp:
                st.download_button(
                    label="Download Images", data=fp, file_name=f"{website}.zip", mime="application/zip"
                )

        if crawl_type == "A+ Images":
            # Move the fasthouse.csv file inside the assets folder.
            shutil.move(f"./{website}__images.csv", f"./assets/{website}__images.csv")
            shutil.make_archive(f"{website}", "zip", "./assets/")

            with open(f"{website}.zip", "rb") as fp:
                st.download_button(
                    label="Download Images", data=fp, file_name=f"{website}.zip", mime="application/zip"
                )

    except Exception as e:
        st.exception(e)
