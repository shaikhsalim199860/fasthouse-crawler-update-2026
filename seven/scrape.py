import enum
import logging
import os
from typing import Optional, List, Tuple, Dict
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

from crawler_app.images import ensure_min_size, is_background_white, to_rgb
from crawler_app.netutil import fetch_bytes, fetch_image, make_session
from crawler_app.runner import run_rows, streamlit_progress_callback

log = logging.getLogger(__name__)

"""
Data:
4. Additional Checks.
5. Bullet Checks?


Images:
1. Image URLs
2. Downloaded Images
3. Resize images.
4. Some checks.
"""


class SevenScraper:
    def __init__(
        self,
        outputs_folder: str = "./outputs",
        assets_folder: str = "./assets",
        min_img_size: Tuple[int, int] = (550, 550),
    ) -> None:
        self.assets_folder = assets_folder
        self.outputs_folder = outputs_folder
        self.min_img_size = min_img_size
        self.session = make_session()
        self.soup: Optional[BeautifulSoup] = None

        if not os.path.exists(self.outputs_folder):
            os.makedirs(self.outputs_folder, exist_ok=True)
        if not os.path.exists(self.assets_folder):
            os.makedirs(self.assets_folder, exist_ok=True)

    def release(self) -> None:
        self.soup = None

    def _get_page_source(self, url: str) -> Optional[bytes]:
        return fetch_bytes(self.session, url)

    def make_soup_obj(self, url: str) -> Optional[bool]:
        try:
            content = self._get_page_source(url)
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", "?")
            log.error("Page fetch FAILED (status %s) for %s: %s", status, url, e)
            return None

        self.soup = BeautifulSoup(content, "html.parser")

        return True

    def get_title(self) -> str:
        title = self.soup.select_one(".product-name-normal")

        return title.text if title else ""

    def get_price(self) -> str:
        raw_price = self.soup.select_one("#ProductPrice")

        if not raw_price:
            return ""

        return raw_price.text.strip().replace("$","")

    def get_bullets(self) -> List[str]:
        raw_bullets = self.soup.select_one(".panel-body")

        if not raw_bullets:
            return []

        split_bullets = raw_bullets.text.strip().strip("//").strip().split("//")

        return [unicodedata.normalize("NFKD", b.strip()) for b in split_bullets]

    def bullet_checks(self, expected_bullets: int):
        return len(self.get_bullets()) == expected_bullets

    def get_description(self) -> str:
        raw_desc = self.soup.select_one(".top-description")

        if not raw_desc:
            return ""

        raw_desc = raw_desc.text.strip().replace("\n", " ")

        # Unicode noramalize
        raw_desc = unicodedata.normalize("NFKD", raw_desc)

        # Add html tags
        raw_desc += "<BR><BR>"
        # Add bullets
        desc = raw_desc + "\n\n" + "<BR>\n".join(self.get_bullets()).rstrip("<BR>\n")

        return desc

    def _write_images_sep_folders(self, img_name: str, img: Image.Image):
        img_name_wout_extension = img_name.split(".")[0].strip()
        img_folder = f"{self.assets_folder}/{img_name_wout_extension}"
        img_path = f"{img_folder}/{img_name}"

        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)

        img.save(img_path)

    def _write_images_same_folder(self, img_name: str, img: Image.Image):
        img_path = f"{self.assets_folder}/{img_name}"

        img.save(img_path)

    def _image_download_and_save(self, url: str, img_name: str, folderize: bool) -> Image.Image:
        """Download, normalise and save one image; returns the in-memory
        image so the caller can inspect it without a second download."""
        img = ensure_min_size(to_rgb(fetch_image(self.session, url)), self.min_img_size)

        (
            self._write_images_sep_folders(img_name=img_name, img=img)
            if folderize
            else self._write_images_same_folder(img_name=img_name, img=img)
        )
        return img

    @staticmethod
    def _get_image_metadata(asin: str, index: int) -> dict:
        is_main_image = index == 0
        img_name = f"{asin}.main.jpg" if is_main_image else f"{asin}.pt0{index}.jpg"
        image_url_col_name = "main" if is_main_image else f"pt0{index}"

        return {"img_name": img_name, "image_url_col_name": image_url_col_name}

    def get_images(self, row: Dict[str, str], folderize: bool = False) -> Optional[Dict[str, str]]:
        image_meta = {}
        asin = row.get("ASIN")
        if asin is None or pd.isna(asin) or not str(asin).strip():
            asin = row["Seller SKU"]

        log.info("Fetching images for ASIN: %s", asin)

        images = self.soup.select(".image-large")

        if not images:
            return None

        for index, div_tag in enumerate(images):

            # Check if the no. of images exceeds 9.
            if index == 9:
                image_meta["Exceeded 9 images"] = True

                break

            img_tag = div_tag.img

            if not img_tag:
                continue

            img_src = f"https:{img_tag['src']}"

            meta = self._get_image_metadata(asin=asin, index=index)
            img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]

            # Add the url to the dict
            image_meta[image_url_col_name] = img_src

            try:
                img = self._image_download_and_save(url=img_src, img_name=img_name, folderize=folderize)
            except Exception as e:  # noqa: BLE001 - keep the other images
                log.error("Image failed for %s index %s: %s", asin, index, e)
                image_meta["Image Errors"] = (image_meta.get("Image Errors", "") + ", " + image_url_col_name).strip(", ")
                continue

            if image_url_col_name == "main":
                # Previously this downloaded the main image a second time
                # just for the colour check.
                image_meta["Is Main Image Background White"] = is_background_white(img)

        return image_meta


class RunType(enum.Enum):
    fetch_data = "fetch_data"
    fetch_images = "fetch_images"


def start(
    df: pd.DataFrame,
    mode: str,
    progress_bar: bool = False,
    *,
    on_progress=None,
    cancel_event=None,
    workers: int = 1,
    assets_folder: str = "./assets",
    output_dir: Optional[str] = "./outputs",
):
    """Crawl every row of `df` for Seven; see fasthouse.scrape.fetch_text_and_images
    for the meaning of the keyword arguments."""
    RunType(mode)

    df = df.astype(object).where(df.notna(), '')
    data = df.to_dict("records")

    # Check if image download can even be carried out or not
    if mode == RunType.fetch_images.value and not any(col in df.columns for col in {"ASIN", "Seller SKU"}):
        raise ValueError("ASIN name required for image download to start.")

    if progress_bar and on_progress is None:
        on_progress = streamlit_progress_callback(len(data))

    def process(index: int, row: dict, s: SevenScraper) -> dict:
        resp = s.make_soup_obj(row["URL"])

        if resp is None:
            return {"status": "error", "message": f"could not fetch {row['URL']}"}

        if mode == RunType.fetch_data.value:
            row["Title"] = s.get_title()
            log.info("%s %s", index, row["Title"])

            row["Price"] = s.get_price()
            # TODO: Max Bullets logic is remaining.
            row["Description"] = s.get_description()

            bullets = s.get_bullets()
            for i, b in enumerate(bullets, start=1):
                row[f"Bullet{i}"] = b
            return {"status": "ok", "message": f"{len(bullets)} bullets"}

        if mode == RunType.fetch_images.value:
            image_meta = s.get_images(row=row)

            if not image_meta:
                return {"status": "ok", "images": 0, "message": "no images found on page"}

            row.update(image_meta)
            failed = image_meta.get("Image Errors", "")
            n_images = sum(1 for k in image_meta if k == "main" or k.startswith("pt0"))
            n_images -= len(failed.split(", ")) if failed else 0
            note = f"{n_images} images, failed: {failed}" if failed else f"{n_images} images"
            return {"status": "ok", "images": n_images, "message": note}

        return {"status": "ok"}

    run_rows(
        data,
        process,
        make_scraper=lambda: SevenScraper(assets_folder=assets_folder),
        workers=workers,
        on_progress=on_progress,
        cancel_event=cancel_event,
    )

    out = pd.DataFrame(data)

    # Rearrange the columns to have all image name cols together
    for col in ("Is Main Image Background White", "Exceeded 9 images", "Image Errors", "Crawl Error"):
        if col in out.columns:
            out.insert(len(out.columns) - 1, col, out.pop(col))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        suffix = "__data.csv" if mode == RunType.fetch_data.value else "__images.csv"
        out.to_csv(os.path.join(output_dir, "Seven" + suffix), sep=",", index=False)

    return out


if __name__ == "__main__":
    df = pd.read_csv("./seven/inputs/test.csv")

    start(df, "fetch_data")
