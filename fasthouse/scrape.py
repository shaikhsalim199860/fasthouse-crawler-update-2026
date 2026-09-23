import os
import copy
import enum
import html as html_lib
import json
import logging
import unicodedata
from typing import List, Optional, Set
from typing import Tuple, Text, Union
import requests
import bs4
import re
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image

from crawler_app.images import (
    center_square,
    ensure_min_size,
    is_background_white,
    save_jpeg,
    to_rgb,
)
from crawler_app.netutil import fetch_bytes, fetch_image, make_session
from crawler_app.runner import (
    asin_of,
    copy_asset_files,
    copy_scraped_columns,
    run_rows,
    streamlit_progress_callback,
)
from fasthouse.sizechart import SizeChartCache, fetch_size_charts, parse_kiwi_data

log = logging.getLogger(__name__)


def _describe_error(e: Exception) -> str:
    """Short human-readable reason for a failed request."""
    if isinstance(e, requests.exceptions.ReadTimeout):
        return "read timed out"
    if isinstance(e, requests.exceptions.ConnectTimeout):
        return "connect timed out"
    if isinstance(e, requests.exceptions.SSLError):
        return "SSL error"
    if isinstance(e, requests.exceptions.ConnectionError):
        return "connection error / stalled after retries"
    return type(e).__name__


class BaseScraper:
    # Base scraper class creates the basic required things for any scraper to work well.
    # 1. Outputs folder: For any csv/df/text output generated.
    # 2. Assets folder: For any image assets generated.
    def __init__(self, outputs_folder: str = "./outputs", assets_folder: str = "./assets") -> None:
        self.assets_folder = assets_folder
        self.outputs_folder = outputs_folder
        # One pooled session per scraper instance (== per worker thread):
        # keeps connections alive across the ~7 requests made per ASIN.
        self.session = make_session()
        self.soup: Optional[BeautifulSoup] = None
        self.page_html = ""
        self.current_url = ""
        self.last_fetch_error = ""

        if not os.path.exists(self.outputs_folder):
            os.makedirs(self.outputs_folder, exist_ok=True)
        if not os.path.exists(self.assets_folder):
            os.makedirs(self.assets_folder, exist_ok=True)

    def release(self) -> None:
        """Drop the parsed page so its (large) tree can be garbage
        collected before the next row."""
        self.soup = None
        self.page_html = ""

    def loader(self, fpath: str) -> pd.DataFrame:
        if not os.path.exists(fpath):
            raise ValueError("Given path does not exist")

        return pd.read_csv(fpath)

    def _get_page_source(self, url: str) -> Optional[bytes]:
        # Retries/backoff for 429/5xx and connection errors live on the
        # session adapter (see crawler_app.netutil.make_session).
        return fetch_bytes(self.session, url)


    def make_soup_obj(self, url: str) -> Optional[BeautifulSoup]:
        self.current_url = url
        self.last_fetch_error = ""
        try:
            if url:
                content = self._get_page_source(url)
                self.page_html = content.decode("utf-8", errors="ignore")
                try:
                    # lxml handles malformed real-world pages much better;
                    # falls back to the builtin parser if not installed.
                    self.soup = BeautifulSoup(content, "lxml")
                except Exception:
                    self.soup = BeautifulSoup(content, "html.parser")
            else:
                return None
        except requests.exceptions.RequestException as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                self.last_fetch_error = f"HTTP {status}" if status else _describe_error(e)
                log.error("Page fetch FAILED (%s) for %s: %s", self.last_fetch_error, url, e)
                return None

        if not content:
            self.last_fetch_error = "empty response"
            return None
        return True

    def get_title(self):
        return self.soup.title.text


class FasthouseScraper(BaseScraper):
    # ------------------------------------------------------------------
    # SELECTOR UPDATE (2026 theme), confirmed from the live page HTML:
    # - Description lives in: <div class="pdp-short-description">
    #     <div class="metafield-rich_text_field"><p>...</p></div></div>
    # - Features are in: <details class="fw-accordion"><summary>Features
    #   </summary><div class="fw-accordion-content">... with the bullets
    #   as ONE <p> separated by <br>, each line prefixed with "- "
    #   (no <ul> anymore).
    # - Materials / Returns and Exchanges are sibling fw-accordion blocks
    #   and must not be picked up.
    # .rte is kept last in the chain for any old-format pages.
    # ------------------------------------------------------------------
    description_identifier = ".pdp-short-description, .product__description, .rte"

    def __init__(self, min_img_size: Tuple[int, int] = (550, 550), assets_folder: str = "./assets") -> None:
        super().__init__(assets_folder=assets_folder)
        self.min_img_size = min_img_size

    def get_video_list(self):
        return self.soup.find_all('iframe')

    @staticmethod
    def clean_string(value: str):
        return value.strip().rstrip().lstrip()

    def get_price(self):
        price= self.soup.find("span", class_="sale-price")
        if price:
            return self.clean_string(price.get_text()).replace("$","")
        else:
            price= self.soup.find("span", class_="price")
            if price:
                price = price.get_text().replace("Sale price","")
                return self.clean_string(price).replace("$","")
            # New theme fallback: Shopify always renders this meta tag.
            meta = self.soup.find("meta", attrs={"property": "product:price:amount"})
            if meta and meta.get("content"):
                return self.clean_string(meta["content"])
            return ""

    def get_title_v2(self):
        # We have to override this, as fasthouse has changed their title strategy.
        return self.clean_string(self.soup.title.text.split("\n")[0])

    def matchWord(self,q):
            return re.compile(r'\b({0})\b'.format(q), flags=re.IGNORECASE).search

    def _find_features_bullets(self) -> list:
        """Return the Features bullets as a list of clean strings.

        New format: <details class="fw-accordion"><summary>Features</summary>
        with the bullets as <br>-separated "- " lines inside
        .fw-accordion-content. Handles a real <ul> too if a product has one.
        Falls back to a <ul> inside the description container (old format).
        """
        # --- New format: Features accordion ---
        for summary in self.soup.find_all("summary"):
            if re.match(r"^\s*Features\s*:?\s*$", summary.get_text()):
                details = summary.find_parent("details") or summary.parent
                content = details.select_one(".fw-accordion-content") or details
                content = copy.copy(content)
                for br in content.find_all("br"):
                    br.replace_with("\n")

                ul = content.find("ul")
                if ul:
                    return [
                        li.get_text(" ", strip=True)
                        for li in ul.find_all("li")
                        if li.get_text(strip=True)
                    ]

                return [
                    re.sub(r"^[-\u2022\u00b7*]+\s*", "", line.strip())
                    for line in content.get_text("\n").split("\n")
                    if line.strip() and not re.match(r"^\s*Features\s*:?\s*$", line)
                ]

        # --- Raw HTML fallback: parser-independent regex extraction ---
        # If the parse tree got mangled (malformed markup on the real page),
        # read the Features accordion straight from the raw page bytes.
        raw = getattr(self, "page_html", "")
        if raw:
            m = re.search(
                r"<summary[^>]*>\s*Features\s*:?\s*</summary>(.*?)</details>",
                raw, re.S | re.I,
            )
            if m:
                block = m.group(1)
                lis = re.findall(r"<li[^>]*>(.*?)</li>", block, re.S)
                if lis:
                    bullets = []
                    for li in lis:
                        text = re.sub(r"<[^>]+>", " ", li)
                        text = html_lib.unescape(text)
                        text = re.sub(r"\s+", " ", text).strip()
                        if text:
                            bullets.append(text)
                    if bullets:
                        log.info("Bullets extracted via raw-HTML regex")
                        return bullets
                # br-dash style inside the accordion, no <li>
                text = re.sub(r"<br\s*/?>", "\n", block, flags=re.I)
                text = re.sub(r"<[^>]+>", " ", text)
                text = html_lib.unescape(text)
                bullets = [
                    re.sub(r"^[-\u2022\u00b7*]+\s*", "", line.strip())
                    for line in text.split("\n")
                    if line.strip()
                ]
                if bullets:
                    log.info("Bullets extracted via raw-HTML regex (br style)")
                    return bullets

        # --- Old format: <ul> inside the description container ---
        desc_node = self.soup.select_one(self.description_identifier)
        if desc_node is not None:
            ul = desc_node.find("ul")
            if ul:
                return [
                    li.get_text(" ", strip=True)
                    for li in ul.find_all("li")
                    if li.get_text(strip=True)
                ]

        return []


    def _get_json_fallback(self):
        """When the HTML page does not contain the description/Features
        (theme renders them with JavaScript), fetch the Shopify product
        JSON endpoint ({url}.js) and parse them from body_html instead.
        Returns (desc_text, bullets_list); ('', []) on any failure."""
        try:
            base = self.current_url.split("?")[0].rstrip("/")
            content = self._get_page_source(base + ".js")
            payload = json.loads(content)
        except Exception as e:
            log.warning("JSON fallback failed for %s: %s", getattr(self, 'current_url', '?'), e)
            return "", []

        desc_html = payload.get("description") or ""
        if not desc_html:
            return "", []

        dsoup = BeautifulSoup(desc_html, "html.parser")
        for br in dsoup.find_all("br"):
            br.replace_with("\n")

        # bullets: first <ul>, else "- " lines after "Features"
        bullets = []
        ul = dsoup.find("ul")
        if ul:
            bullets = [
                li.get_text(" ", strip=True)
                for li in ul.find_all("li")
                if li.get_text(strip=True)
            ]

        temp = copy.copy(dsoup)
        for u in temp.find_all("ul"):
            u.decompose()

        # Bullets from raw text (keeps the <br>-inserted newlines intact)
        raw_text = temp.get_text("\n")
        if not bullets and "Features" in raw_text:
            tail = raw_text.split("Features", 1)[1]
            bullets = [
                re.sub(r"^[-\u2022\u00b7*]+\s*", "", l.strip())
                for l in tail.split("\n")
                if l.strip()
            ]

        # Description from per-paragraph text (inline tags don't split lines)
        paragraphs = temp.find_all("p")
        if paragraphs:
            lines = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in paragraphs]
            text = "\n".join([l for l in lines if l])
        else:
            text = raw_text

        desc_text = text.split("Features")[0]
        desc_lines = [l.strip() for l in desc_text.split("\n") if l.strip()]
        return "\n".join(desc_lines), bullets

    def getRedText(self):
        list_q=['size','sizes','ordering']
        desc = self.soup.select_one(self.description_identifier)  # was: '.rte'
        if desc is None:
            return ''
        desc=desc.text.strip("")
        list_text=desc.split("\n")
        for text_desc in list_text:
                if text_desc != '' and "*" not in text_desc:
                    for q in list_q:
                        if self.matchWord(q)(text_desc):
                            return text_desc
        return ''

    def get_description_and_bullets_v2(self, max_bullets: int):
        result = {
            "Description": "",
            "Bullet check": 0,
        }
        description_identifier = self.description_identifier  # was: ".rte"
        bullet_character = "\u2022"

        # Add the remaining bullet headers: Bullet{1} -> Bullet{n}
        result.update({f"Bullet{i + 1}": "" for i in range(max_bullets)})

        # Main description body
        # Raw-HTML fallback for the description if the parse tree missed it
        if not self.soup.select_one(description_identifier):
            raw = getattr(self, "page_html", "")
            m = re.search(
                r'class="pdp-short-description"(.*?)</div>\s*</div>',
                raw, re.S | re.I,
            )
            if m:
                text = re.sub(r"<br\s*/?>", "\n", m.group(1), flags=re.I)
                text = re.sub(r"<[^>]+>", " ", text)
                text = html_lib.unescape(text)
                text = re.sub(r"[ \t]+", " ", text).strip().lstrip(">").strip()
                if text:
                    log.info("Description extracted via raw-HTML regex")
                    result["Description"] = text + "<BR><BR>"

        if self.soup.select_one(description_identifier):
            temp_description: bs4.element.Tag = copy.copy(self.soup.select_one(description_identifier))
            if temp_description.find("ul"):
                # Removing <ul> from the description section
                temp_description.find("ul").replace_with("")

            # If features is present as part of the description text?
            if bullet_character in temp_description.text:
                desc_text = temp_description.text.split("Features")[0]
            else:
                desc_text = temp_description.text

            desc_text=desc_text.replace("CCSizeChartLaunchLocationBefore","")
            desc_text=desc_text.replace("CCSizeChartLaunchLocationAfter","")
            desc_text=desc_text.replace("Swing by the Fasthouse Service Department where the crew will get you geared up and ready to haul ass. At Fasthouse we are all about Speed, Style and Good Times.","")
            desc_text=desc_text.replace("Looking for a good time? Dial 661-775-5963 and let the fun begin. The House of Good Times is ready to accept your call Monday through Friday, 9 AM to 5 PM. International rates may apply.","")
            result["Description"] = desc_text.strip().rstrip().lstrip() + "<BR><BR>"

            if not desc_text.strip():
                result["Description"] = desc_text.strip().rstrip().lstrip()


        # Features bullets (new fw-accordion format, ul formats, or old .rte ul)
        # (was: self.soup.select_one(f"{description_identifier} ul"))
        bullets_list = self._find_features_bullets()

        # JSON fallback: page HTML did not contain the content (rendered by
        # JavaScript on the live site) -> read it from {url}.js body_html.
        if not bullets_list:
            json_desc, json_bullets = self._get_json_fallback()
            if json_bullets:
                log.info("Bullets taken from product JSON endpoint")
                bullets_list = json_bullets
            if json_desc and not result["Description"]:
                for junk in ["CCSizeChartLaunchLocationBefore", "CCSizeChartLaunchLocationAfter"]:
                    json_desc = json_desc.replace(junk, "")
                result["Description"] = json_desc.strip() + "<BR><BR>"

        if bullets_list:
            start_v=1
            if self.getRedText():
                start_v=2
                result['Bullet1']= self.getRedText()

            features = {
                f"Bullet{index}": value
                for index, value in enumerate(bullets_list, start=start_v)
            }


            log.debug("features: %s", features)
            result.update(features)
            # Also add the number of bullets captured
            result["Bullet check"] = len(features)

            # Update the description only when features exists
            result["Description"] += "\n\n" + "\n".join([f"{feature}<BR>" for feature in features.values()]).rstrip(
                "<BR>"
            )

        # If the site has hardcoded bullet values instead of <ul> tags.
        elif (
            self.soup.select_one(description_identifier)
            and bullet_character in self.soup.select_one(description_identifier).text
        ):
            bullet_text = self.soup.select_one(description_identifier).text.split("Features")[1]
            bullet_text = bullet_text.strip().lstrip(":").lstrip().rstrip()
            bullets = bullet_text.split("\n")
            start_v=1
            if self.getRedText():
                start_v=2
                result['Bullet1']= self.getRedText()

            features = {
                f"Bullet{index}": value
                for index, value in enumerate([b.strip().lstrip(bullet_character).lstrip() for b in bullets if b.strip()], start=start_v)
            }


            result.update(features)
            # Also add the number of bullets captured
            result["Bullet check"] = len(features)

            # Update the description only when features exists
            result["Description"] += "\n\n" + "\n".join([f"{feature}<BR>" for feature in features.values()]).rstrip(
                "<BR>"
            )

        else:
            # Copy the description over to the features as well
            result["Bullet1"] = result["Description"].rstrip("<BR><BR>")
            result["Bullet check"] = 1

        for key, val in result.items():
            if isinstance(val, str):
                result[key] = unicodedata.normalize("NFKD", val)

        return result

    def get_description_and_bullets(self, max_bullets: int):
        result = {
            "Description": "",
            "Bullet check": 0,
        }

        # Add the remaining bullet headers: Bullet{1} -> Bullet{n}
        result.update({f"Bullet{i + 1}": "" for i in range(max_bullets)})

        if self.soup.select_one(".description.content"):
            result["Description"] = (
                self.soup.select_one(".description.content").text.strip().split("\n")[0].strip() + "<BR><BR>"
            )

        if self.soup.select_one(".description.content ul"):
            start_v=1
            if self.getRedText():
                result['Bullet1']= self.getRedText()
                start_v=2

            features = {
                f"Bullet{index}": value
                for index, value in enumerate(
                    [
                        val.strip()
                        for val in self.soup.select_one(".description.content ul").text.strip().split("\n")
                        if val
                    ],
                    start=start_v,
                )
            }

            result.update(features)
            # Also add the number of bullets captured
            result["Bullet check"] = len(features)


            # Update the description only when features exists
            result["Description"] += "\n\n" + "\n".join([f"{feature}<BR>" for feature in features.values()]).rstrip(
                "<BR>"
            )
        else:
            # Copy the description over to the features as well
            result["Bullet1"] = result["Description"].rstrip("<BR><BR>")
            result["Bullet check"] = 1

        for key, val in result.items():
            if isinstance(val, str):
                result[key] = unicodedata.normalize("NFKD", val)

        return result

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

    def _image_download_and_save(self, url: str, img_name: str, folderize: bool) -> None:
        img = to_rgb(fetch_image(self.session, url))
        img = ensure_min_size(img, self.min_img_size)

        (
            self._write_images_sep_folders(img_name=img_name, img=img)
            if folderize
            else self._write_images_same_folder(img_name=img_name, img=img)
        )

    def _image_path(self, img_name: str, folderize: bool) -> str:
        if folderize:
            return f"{self.assets_folder}/{img_name.split('.')[0].strip()}/{img_name}"
        return f"{self.assets_folder}/{img_name}"

    def _download_square_1500(self, url: str, img_name: str, folderize: bool) -> Image.Image:
        """Download -> RGB -> centre-crop -> 1500x1500 -> save, in ONE pass.

        The old flow saved the raw download, re-opened it from disk,
        cropped/resized and saved again (double decode + double write per
        image). Returns the in-memory image so callers can run the
        background check without touching disk again.
        """
        img = center_square(to_rgb(fetch_image(self.session, url)))
        save_jpeg(img, self._image_path(img_name, folderize), quality=95, subsampling=0)
        return img

    @staticmethod
    def _get_a_plus_image_metadata(url: str,index : str) -> dict:
            img_name=f"{url.split('/')[4]}.{index}.jpg"
            image_url_col_name = f"A_Plus_pt0{index}"
            return {"img_name": img_name, "image_url_col_name": image_url_col_name}

    @staticmethod
    def _get_image_metadata(asin: str, index: int) -> dict:
        is_main_image = index == 0
        img_name = f"{asin}.main.jpg" if is_main_image else f"{asin}.pt0{index}.jpg"
        image_url_col_name = "main" if is_main_image else f"pt0{index}"

        return {"img_name": img_name, "image_url_col_name": image_url_col_name}

    @staticmethod
    def _gallery_slots(count: int, reserved: Set[int]) -> List[Optional[int]]:
        """Listing slot for each gallery image: 0 = MAIN, 1-8 = PT01-PT08.

        Slots reserved for the size chart are skipped, so the gallery keeps
        its order and shifts down past the chart. Amazon has nine slots in
        total, so when the gallery no longer fits its trailing images are
        dropped (None) rather than the chart being pushed to the end.
        """
        free = [slot for slot in range(1, 9) if slot not in reserved]
        slots: List[Optional[int]] = []
        for position in range(count):
            if position == 0:
                slots.append(0)          # MAIN is never reserved
            else:
                slots.append(free.pop(0) if free else None)
        return slots

    @staticmethod
    def _hi_res_url(src: str) -> str:
        """Turn a thumbnail src into the 1500px rendition.

        Old theme: `.../file_140x140.jpg` -> `.../file_1500x.jpg`.
        New theme: `.../file.jpg?v=1&width=1200` -> `...&width=1500`
        (Shopify never upscales, so this is a no-op for small originals).
        """
        url = src if src.startswith("http") else f"https:{src}"
        url = url.replace("140x140", "1500x")
        return re.sub(r"([?&])width=\d+", r"\g<1>width=1500", url)


    def _a_plus_image_download_and_save(self, url: str, img_name: str, folderize: bool) -> None:
        img = to_rgb(fetch_image(self.session, url))

        (
            self._write_images_sep_folders(img_name=img_name, img=img)
            if folderize
            else self._write_images_same_folder(img_name=img_name, img=img)
        )

    def get_A_Plus_images(self,product_row: dict, folderize: bool = True) -> list:
        image_urls = []
        url=product_row['URL']
        log.info("Fetching A Plus Images for %s", url)
        # Handling normal images
        imageLayout=self.soup.find("div",id="shopify-section-product")
        if imageLayout:
            for index, img in enumerate(imageLayout.findAll("div", {'class':['duo', 'single','small--one-half']})):
                img_t = img.find('img')
                image_url=img_t['src']
                if image_url:
                    meta = self._get_a_plus_image_metadata(url,""+str(index+1))
                    log.debug("A+ image %s -> %s", image_url, meta)
                    img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]
                    image_urls.append({image_url_col_name: img_name})
                    # Read the content, resize if necessary and send the img itself to be saved.
                    try:
                        self._a_plus_image_download_and_save(url=image_url, img_name=img_name, folderize=folderize)
                    except Exception as e:  # noqa: BLE001 - keep the other A+ images
                        log.error("A+ image failed for %s: %s", image_url, e)
                        image_urls.append({f"{image_url_col_name}_error": str(e)})
                else:
                    log.warning("A+ image url not found")
        else:
            for index, img in enumerate(self.soup.findAll('img',class_='slideshow__image')):
                image_url=img['src']
                image_url="https:"+image_url
                if image_url:
                    meta = self._get_a_plus_image_metadata(url,""+str(index+1))
                    log.debug("A+ image %s -> %s", image_url, meta)
                    img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]
                    image_urls.append({image_url_col_name: img_name})
                    # Read the content, resize if necessary and send the img itself to be saved.
                    try:
                        self._a_plus_image_download_and_save(url=image_url, img_name=img_name, folderize=folderize)
                    except Exception as e:  # noqa: BLE001 - keep the other A+ images
                        log.error("A+ image failed for %s: %s", image_url, e)
                        image_urls.append({f"{image_url_col_name}_error": str(e)})

        return image_urls

    def get_images(self, product_row: dict, folderize: bool = True) -> list:
        # productView-thumbnail is the place to start with.
        # 540x is an option. If not available, go for the others, else drop down to the defaulkt one available as thumbnail.
        # Assumption:
        # - 140 px will always be the thumbnail size
        # - That thumbnails will always exist?
        #       - If they do not fall back to -> productView-image
        asin = product_row["ASIN"]
        if pd.isna(asin):
            asin = product_row["Seller SKU"]

        image_urls = []
        log.info("Fetching images for ASIN: %s", asin)
        # Handling normal images
        for index, img in enumerate(
            self.soup.select(".product-gallery.product-gallery--bottom-thumbnails img.lazyload--fade-in")
            + self.soup.select(".description.content img")
        ):
            meta = self._get_image_metadata(asin=asin, index=index)
            img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]

            url: Union[str, Text] = f'https:{img["data-src"]}' if img.get("data-src") else img["src"]

            image_urls.append({image_url_col_name: url})
            # Read the content, resize if necessary and send the img itself to be saved.
            self._image_download_and_save(url=url, img_name=img_name, folderize=folderize)

            return image_urls

    def is_prominent_bg_col_white(self, img_path: str) -> dict:
        with Image.open(img_path) as im:
            return {"Is Main Image Background White": is_background_white(im)}

    def get_images_v2(
        self,
        product_row: dict,
        folderize: bool = True,
        reserved_slots: Optional[Set[int]] = None,
    ) -> list:
        """Download the product gallery into the listing image slots.

        `reserved_slots` holds slots already claimed by the size chart
        (PT05 by default); the gallery skips them and its trailing images
        are dropped if nine slots are not enough.
        """
        asin = product_row.get("ASIN")
        if pd.isna(asin) or not str(asin).strip():
            asin = product_row["Seller SKU"]

        log.info("Fetching images for ASIN via v2: %s", asin)

        thumbnails = self.soup.select(".product__thumbnail")
        image_urls = []
        main_image: Optional[Image.Image] = None
        failed_images = []
        dropped = 0

        if thumbnails:
            slots = self._gallery_slots(len(thumbnails), set(reserved_slots or ()))

            for index, (thumbnail, slot) in enumerate(zip(thumbnails, slots)):

                if slot is None:
                    dropped += 1
                    continue

                meta = self._get_image_metadata(asin=asin, index=slot)
                img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]

                try:
                    if not thumbnail or not thumbnail.img or not thumbnail.img.get("src"):
                        log.warning("Missing thumbnail image for %s index %s", asin, index)
                        continue

                    url = self._hi_res_url(thumbnail.img["src"])

                    image_urls.append({image_url_col_name: url})
                    img = self._download_square_1500(url=url, img_name=img_name, folderize=folderize)
                    if slot == 0:
                        main_image = img

                except Exception as e:
                    log.error("Thumbnail failed for %s index %s: %s", asin, index, e)
                    failed_images.append(image_url_col_name)
                    continue

            if dropped:
                log.info("%s: dropped %s trailing gallery image(s) - all 9 slots used", asin, dropped)
                image_urls.append({"Exceeded 9 images": True, "Gallery Images Dropped": dropped})

        else:
            img_name = f"{asin}.main.jpg"
            image_url_col_name = "main"

            try:
                container = self.soup.select_one('.productView-image')

                if not container or not container.img or not container.img.get("src"):
                    log.error("Main image not found for %s", asin)
                    image_urls.append({"main_image_missing": True})
                else:
                    url = f"https:{container.img['src']}"
                    url = url.replace("300x", "1500x")

                    image_urls.append({image_url_col_name: url})
                    main_image = self._download_square_1500(url=url, img_name=img_name, folderize=folderize)

            except Exception as e:
                log.error("Main image extraction failed for %s: %s", asin, e)
                image_urls.append({"main_image_error": True})
                failed_images.append("main")

        if failed_images:
            # Surfaced in the CSV so the affected SKUs can be re-run.
            image_urls.append({"Image Errors": ", ".join(failed_images)})

        # ---- BACKGROUND CHECK (on the in-memory main image, no re-read) ----
        try:
            if main_image is not None:
                image_urls.append({"Is Main Image Background White": is_background_white(main_image)})
            else:
                image_urls.append(self.is_prominent_bg_col_white(f"{self.assets_folder}/{asin}.main.jpg"))
        except Exception as e:
            log.warning("Background check skipped for %s: %s", asin, e)
            image_urls.append({"bg_check_failed": True})

        return image_urls


    def get_fit_guide_url(self) -> str:
        """URL of the theme's 'Fit' accordion graphic (fit style bars), if any."""
        img = self.soup.find("img", alt=re.compile(r"fit guide", re.I))
        if img is None:
            fit_div = self.soup.find("div", class_="fit_guide_pc")
            img = fit_div.find("img") if fit_div else None
        if img is None or not img.get("src"):
            return ""
        src = img["src"]
        return src if src.startswith("http") else f"https:{src}"

    def get_size_charts_v2(
        self,
        product_row: dict,
        units: str,
        cache: SizeChartCache,
        folderize: bool = False,
        first_slot: Optional[int] = None,
    ) -> dict:
        """Render every Kiwi Sizing chart for the product to 2000x2000 PNGs.

        Standalone mode (first_slot is None): files are named
        `{ASIN}.SIZE-CHART.png` (`.SIZE-CHART-2.png`, ... for extra charts
        such as bikini top/bottom).

        Listing-slot mode (first_slot given, e.g. 5 for PT05): the chart
        claims that slot (and the following ones when a product has several
        charts) and is named for it, e.g. `{ASIN}.pt05.png`. The gallery is
        laid out around the reserved slots afterwards - see
        `get_images_v2(reserved_slots=...)`. Returns the output columns."""
        asin = product_row.get("ASIN")
        if asin is None or pd.isna(asin) or not str(asin).strip():
            asin = product_row["Seller SKU"]

        result = {
            "Size Chart Status": "",
            "Size Chart Name": "",
            "Size Chart File": "",
            "Size Chart Slot": "",
            "Size Chart Sizes": "",
            "Size Chart Measurements": "",
            "How to Measure": "",
            "Size Chart Diagram URL": "",
            "Fit Guide URL": self.get_fit_guide_url(),
            "Size Chart Count": 0,
        }

        kiwi = parse_kiwi_data(self.page_html)
        if not kiwi:
            result["Size Chart Status"] = "No size chart (page has no Kiwi Sizing data)"
            return result

        charts = fetch_size_charts(self.session, kiwi)
        if not charts:
            result["Size Chart Status"] = "No size chart"
            return result

        files, names, sizes, meas, htm, diagrams, slots = [], [], [], [], [], [], []
        next_slot = int(first_slot) if first_slot is not None else None
        for index, chart in enumerate(charts):
            suffix = "" if index == 0 else f"-{index + 1}"
            img_name = f"{asin}.SIZE-CHART{suffix}.png"
            if next_slot is not None:
                if next_slot <= 8:
                    slot_name = f"pt0{next_slot}"
                    img_name = f"{asin}.{slot_name}.png"
                    slots.append(slot_name.upper())
                    result[slot_name] = img_name
                    next_slot += 1
                else:
                    slots.append("no free slot")
            png = cache.png(self.session, chart, units)
            path = self._image_path(img_name, folderize)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(png)
            files.append(img_name)
            names.append(chart.name)
            sizes.append(", ".join(chart.sizes))
            meas.append(", ".join(chart.measurements))
            htm.append(chart.how_to_measure_sentence)
            if chart.diagram_url:
                diagrams.append(chart.diagram_url)

        result.update({
            "Size Chart Status": "Found",
            "Size Chart Name": " | ".join(names),
            "Size Chart File": " | ".join(files),
            "Size Chart Slot": " | ".join(slots),
            "Size Chart Sizes": " | ".join(sizes),
            "Size Chart Measurements": " | ".join(meas),
            "How to Measure": " | ".join(h for h in htm if h),
            "Size Chart Diagram URL": " | ".join(diagrams),
            "Size Chart Count": len(charts),
        })
        return result

    def get_size_chart(self, product_row: dict, folderize: bool = True) -> str:
        asin = product_row.get("ASIN") or product_row.get("Seller SKU")
        if pd.isna(asin):
            asin = product_row.get("Seller SKU")

        log.info("Fetching size chart for ASIN: %s", asin)

        # --- Case 1: Old layout (collapsible-content) ---
        content_image_size = self.soup.find(
            "collapsible-content",
            id="block-template--24935003390025__product-content-image_irrATL-content"
        )
        if content_image_size:
            img_tag = content_image_size.find("img")
            if img_tag and img_tag.get("src"):
                content_image_size_img = f"https:{img_tag['src']}"
                log.info("Found size chart (collapsible-content): %s", content_image_size_img)
                img_name = f"{asin}.SIZE-CHART.jpg"
                self._image_download_and_save(
                    url=content_image_size_img, img_name=img_name, folderize=folderize
                )
                return content_image_size_img

        # --- Case 2: New layout (fit_guide_pc) ---
        fit_guide_div = self.soup.find("div", class_="fit_guide_pc")
        if fit_guide_div:
            img_tag = fit_guide_div.find("img")
            if img_tag and img_tag.get("src"):
                content_image_size_img = f"https:{img_tag['src']}"
                log.info("Found size chart (fit_guide_pc): %s", content_image_size_img)
                img_name = f"{asin}.SIZE-CHART.jpg"
                self._image_download_and_save(
                    url=content_image_size_img, img_name=img_name, folderize=folderize
                )
                return content_image_size_img

        # No size chart found
        log.info("No size chart found for: %s", asin)
        return ""



class RunType(enum.Enum):
    fetch_data = "fetch_data"
    fetch_images = "fetch_images"
    A_Plus_fetch_images="A_Plus_fetch_images"
    fetch_size_charts = "fetch_size_charts"


def _max_bullets(df: pd.DataFrame) -> int:
    if "No of bullets" not in df.columns:
        return 5
    values = pd.to_numeric(df["No of bullets"], errors="coerce")
    top = values.max()
    return int(top) if pd.notna(top) and top > 0 else 5


def fetch_text_and_images(
    df: pd.DataFrame,
    mode: str,
    progress_bar: bool = False,
    *,
    on_progress=None,
    cancel_event=None,
    workers: int = 1,
    assets_folder: str = "./assets",
    output_dir: Optional[str] = "./outputs",
    units: str = "Inches",
    include_size_chart: bool = False,
    size_chart_slot: int = 5,
):
    """Crawl every row of `df` in `mode` (fetch_data / fetch_images /
    A_Plus_fetch_images / fetch_size_charts) and return the enriched DataFrame.

    on_progress   optional callback receiving one event dict per row
    cancel_event  optional threading.Event; remaining rows are skipped once set
    workers       parallel worker threads (each with its own scraper/session)
    output_dir    where the legacy Fasthouse__*.csv is written; None to skip
    progress_bar  legacy flag: draw an st.progress bar (Streamlit thread only)
    units         size charts: "Inches", "Centimetres" or "Both"
    include_size_chart  images mode: also render the Kiwi size chart into the
                  listing image slot `size_chart_slot` (PT05), or the first
                  free slot after the gallery when that one is taken
    """
    website_format = "new"  # Allowed values are old and new.

    # Check if the mode is acceptable.
    RunType(mode)

    df = df.astype(object).where(df.notna(), '')
    data = df.to_dict("records")
    # Check if image download can even be carried out or not
    if mode == RunType.fetch_images.value and not any(col in df.columns for col in {"ASIN", "Seller SKU"}):
        raise ValueError("ASIN name required for image download to start.")

    max_bullets = _max_bullets(df) if mode == RunType.fetch_data.value else 0
    # Shared across worker threads: one render per distinct chart.
    chart_cache = SizeChartCache()

    if progress_bar and on_progress is None:
        on_progress = streamlit_progress_callback(len(data))

    input_columns = list(df.columns)

    def process(index: int, row: dict, scraper: "FasthouseScraper") -> dict:
        resp = scraper.make_soup_obj(row["URL"])
        if resp is None:
            reason = scraper.last_fetch_error or "unknown error"
            return {"status": "error", "message": f"could not fetch ({reason}) {row['URL']}"}

        if mode == RunType.fetch_data.value:
            row["Title"] = scraper.get_title_v2()
            row["Price"] = scraper.get_price()
            log.info("%s %s", index, row["Title"])

            row.update(
                scraper.get_description_and_bullets_v2(max_bullets=max_bullets)
                if website_format == "new"
                else scraper.get_description_and_bullets(max_bullets=max_bullets)
            )
            # Bullets and match
            row["Bullet match"] = row["No of bullets"] == row["Bullet check"]
            if row["Bullet check"] == 1 and not str(row.get("Bullet1", "")).strip():
                log.warning("Row %s: NO description/bullets found for %s", index, row["URL"])
                return {"status": "ok", "message": "no description/bullets found"}
            return {"status": "ok", "message": f"{row['Bullet check']} bullets"}

        # Fetch images
        if mode == RunType.fetch_images.value:
            # Keep everything in the same folder
            chart_note = ""
            n_charts = 0
            reserved: Set[int] = set()

            if include_size_chart:
                # The chart is rendered FIRST so it can claim PT05; the
                # gallery is then laid out around the reserved slot(s). The
                # theme's fit-guide graphic is only recorded as a URL.
                row['Fit Guide URL'] = scraper.get_fit_guide_url()
                info = scraper.get_size_charts_v2(
                    product_row=row, units=units, cache=chart_cache, folderize=False,
                    first_slot=int(size_chart_slot),
                )
                row.update(info)
                n_charts = int(info.get("Size Chart Count") or 0)
                reserved = {s for s in range(int(size_chart_slot), int(size_chart_slot) + n_charts) if s <= 8}
                chart_note = f", size chart -> {info['Size Chart Slot']}" if n_charts else ", no size chart"
            else:
                row['Size-Chart'] = scraper.get_size_chart(product_row=row, folderize=False)
            row['Is Video Available'] = len(scraper.get_video_list()) > 0

            image_urls = (
                scraper.get_images_v2(product_row=row, folderize=False, reserved_slots=reserved)
                if website_format == "new"
                else scraper.get_images(product_row=row, folderize=False)
            )
            for col_url_map in image_urls:
                row.update(col_url_map)
            failed = next((m["Image Errors"] for m in image_urls if "Image Errors" in m), "")
            gallery = sum(1 for m in image_urls if any(k == "main" or k.startswith("pt0") for k in m))
            n_images = gallery - (len(failed.split(", ")) if failed else 0) + n_charts
            n_images += 1 if row.get('Size-Chart') else 0

            dropped = int(next((m["Gallery Images Dropped"] for m in image_urls if "Gallery Images Dropped" in m), 0))
            if dropped:
                chart_note += f", {dropped} trailing gallery image(s) dropped"

            if any("main_image_missing" in m for m in image_urls):
                note = "main image missing on page"
            elif failed:
                note = f"{n_images} images, failed: {failed}"
            else:
                note = f"{n_images} images"
            return {"status": "ok", "images": n_images, "message": note + chart_note}

        # Size charts (Kiwi Sizing -> rendered PNG)
        if mode == RunType.fetch_size_charts.value:
            info = scraper.get_size_charts_v2(product_row=row, units=units, cache=chart_cache, folderize=False)
            row.update(info)
            n = int(info.get("Size Chart Count") or 0)
            note = info["Size Chart Status"] if n == 0 else (f"{n} size charts" if n > 1 else info["Size Chart Name"])
            return {"status": "ok", "images": n, "message": note}

        # Fetch A+ images
        if mode == RunType.A_Plus_fetch_images.value:
            image_urls = scraper.get_A_Plus_images(product_row=row, folderize=False)
            for col_url_map in image_urls:
                row.update(col_url_map)
            return {"status": "ok", "images": len(image_urls), "message": f"{len(image_urls)} A+ images"}

        return {"status": "ok"}

    def replicate(src: dict, dst: dict) -> dict:
        """Apply a crawled row's result to another row with the same URL
        (a size variant): copy the scraped columns and duplicate the files
        under the other row's ASIN."""
        copy_scraped_columns(src, dst, input_columns)
        if mode == RunType.fetch_data.value:
            dst["Bullet match"] = dst.get("No of bullets") == dst.get("Bullet check")
            return {"message": "result copied"}
        copied = copy_asset_files(assets_folder, asin_of(src), asin_of(dst))
        return {"images": copied, "message": f"{copied} files copied"}

    run_rows(
        data,
        process,
        make_scraper=lambda: FasthouseScraper(assets_folder=assets_folder),
        workers=workers,
        on_progress=on_progress,
        cancel_event=cancel_event,
        replicate=replicate,
    )

    out = pd.DataFrame(data)
    if "Size Chart Count" in out.columns:
        out["Size Chart Count"] = pd.to_numeric(out["Size Chart Count"], errors="coerce").fillna(0).astype(int)

    # Rearrange the columns to have all image name cols together
    for col in ("Is Main Image Background White", "Exceeded 9 images", "Gallery Images Dropped",
                "Is Video Available", "Image Errors", "Crawl Error"):
        if col in out.columns:
            out.insert(len(out.columns) - 1, col, out.pop(col))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        suffix = {
            RunType.fetch_data.value: "__data.csv",
            RunType.fetch_size_charts.value: "__sizecharts.csv",
        }.get(mode, "__images.csv")
        out.to_csv(os.path.join(output_dir, "Fasthouse" + suffix), sep=",", index=False)

    return out


if __name__ == "__main__":
    df = pd.read_csv("./fasthouse/may.csv")
    mode = "fetch_data"
    fetch_text_and_images(df, mode)
