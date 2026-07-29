import os
import copy
import enum
import html as html_lib
import json
import unicodedata
from typing import Optional
from typing import Tuple, Text, Union
import backoff
import requests
import bs4
import re
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
# -*- coding: cp1252 -*-

print("[scrape.py] VERSION 6 loaded (lxml + raw-regex + JSON fallback)")


class BaseScraper:
    # Base scraper class creates the basic required things for any scraper to work well.
    # 1. Outputs folder: For any csv/df/text output generated.
    # 2. Assets folder: For any image assets generated.
    def __init__(self, outputs_folder: str = "./outputs", assets_folder: str = "./assets") -> None:
        self.assets_folder = assets_folder
        self.outputs_folder = outputs_folder

        if not os.path.exists(self.outputs_folder):
            os.makedirs(self.outputs_folder, exist_ok=True)
        if not os.path.exists(self.assets_folder):
            os.makedirs(self.assets_folder, exist_ok=True)

    def loader(self, fpath: str) -> pd.DataFrame:
        if not os.path.exists(fpath):
            raise ValueError("Given path does not exist")

        return pd.read_csv(fpath)

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=5,
    )
    def _get_page_source(self, url: str) -> Optional[bytes]:
        resp = requests.get(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        resp.raise_for_status()
        return resp.content
    

    def make_soup_obj(self, url: str) -> Optional[BeautifulSoup]:
        self.current_url = url
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
                status = getattr(getattr(e, "response", None), "status_code", "?")
                print(f"[ERROR] Page fetch FAILED (status {status}) for {url}: {e}")
                return None

        if not content:
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

    def __init__(self, min_img_size: Tuple[int, int] = (550, 550)) -> None:
        super().__init__()
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
                        print("[INFO] Bullets extracted via raw-HTML regex")
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
                    print("[INFO] Bullets extracted via raw-HTML regex (br style)")
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
            print(f"[WARN] JSON fallback failed for {getattr(self, 'current_url', '?')}: {e}")
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
                    print("[INFO] Description extracted via raw-HTML regex")
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
                print("[INFO] Bullets taken from product JSON endpoint")
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


            print(features)
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
        img = Image.open(requests.get(url, stream=True).raw)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if any(sz <= 500 for sz in img.size):
            img = img.resize(self.min_img_size).convert("RGB")

        (
            self._write_images_sep_folders(img_name=img_name, img=img)
            if folderize
            else self._write_images_same_folder(img_name=img_name, img=img)
        )

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


    def _a_plus_image_download_and_save(self, url: str, img_name: str, folderize: bool) -> None:
        img = Image.open(requests.get(url, stream=True).raw)
        if img.mode != "RGB":
            img = img.convert("RGB")

        (
            self._write_images_sep_folders(img_name=img_name, img=img)
            if folderize
            else self._write_images_same_folder(img_name=img_name, img=img)
        )

    def get_A_Plus_images(self,product_row: dict, folderize: bool = True) -> list:
        image_urls = []
        url=product_row['URL']
        print(f"Fetching A Plus Images:")
        # Handling normal images
        imageLayout=self.soup.find("div",id="shopify-section-product")
        if imageLayout:
            for index, img in enumerate(imageLayout.findAll("div", {'class':['duo', 'single','small--one-half']})):
                img_t = img.find('img')
                image_url=img_t['src']
                if image_url:
                    print(image_url)
                    meta = self._get_a_plus_image_metadata(url,""+str(index+1))
                    print(meta)
                    img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]
                    image_urls.append({image_url_col_name: img_name})
                    # Read the content, resize if necessary and send the img itself to be saved.
                    self._a_plus_image_download_and_save(url=image_url, img_name=img_name, folderize=folderize)
                else:
                    print("image url not found")
        else:
            for index, img in enumerate(self.soup.findAll('img',class_='slideshow__image')):
                image_url=img['src']
                image_url="https:"+image_url
                if image_url:
                    print(image_url)
                    meta = self._get_a_plus_image_metadata(url,""+str(index+1))
                    print(meta)
                    img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]
                    image_urls.append({image_url_col_name: img_name})
                    # Read the content, resize if necessary and send the img itself to be saved.
                    self._a_plus_image_download_and_save(url=image_url, img_name=img_name, folderize=folderize)

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
        print(f"Fetching images for ASIN: {asin}")
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
        from PIL import Image

        im = Image.open(img_path)
        prominent_color = max(im.getcolors(im.size[0] * im.size[1]))[1]
        if prominent_color == (255, 255, 255):
            return {"Is Main Image Background White": True}

        return {"Is Main Image Background White": False}

    def get_images_v2(self, product_row: dict, folderize: bool = True) -> list:
        asin = product_row.get("ASIN")
        if pd.isna(asin):
            asin = product_row["Seller SKU"]

        print(f"Fetching images for ASIN via v2: {asin}")

        def force_1500_square(path):
            try:
                if not os.path.exists(path):
                    return

                img = Image.open(path).convert("RGB")
                w, h = img.size

                # center crop
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim
                img = img.crop((left, top, right, bottom))

                # resize
                img = img.resize((1500, 1500), Image.LANCZOS)

                img.save(path, "JPEG", quality=95, subsampling=0)

            except Exception as e:
                print(f"[WARN] Resize failed for {path}: {e}")

        thumbnails = self.soup.select(".product__thumbnail")
        image_urls = []

        if thumbnails:
            for index, thumbnail in enumerate(thumbnails):

                if index == 9:
                    image_urls.append({"Exceeded 9 images": True})
                    break

                meta = self._get_image_metadata(asin=asin, index=index)
                img_name, image_url_col_name = meta["img_name"], meta["image_url_col_name"]

                try:
                    if not thumbnail or not thumbnail.img or not thumbnail.img.get("src"):
                        print(f"[WARN] Missing thumbnail image for {asin} index {index}")
                        continue

                    thumbnail_url: str = thumbnail.img["src"]
                    url = f"https:{thumbnail_url.replace('140x140', '1500x')}"

                    image_urls.append({image_url_col_name: url})
                    self._image_download_and_save(url=url, img_name=img_name, folderize=folderize)

                    # ---- RESIZE AFTER DOWNLOAD ----
                    force_1500_square(f"{self.assets_folder}/{img_name}")

                except Exception as e:
                    print(f"[ERROR] Thumbnail failed for {asin} index {index}: {e}")
                    continue

        else:
            img_name = f"{asin}.main.jpg"
            image_url_col_name = "main"

            try:
                container = self.soup.select_one('.productView-image')

                if not container or not container.img or not container.img.get("src"):
                    print(f"[ERROR] Main image not found for {asin}")
                    image_urls.append({"main_image_missing": True})
                else:
                    url = f"https:{container.img['src']}"
                    url = url.replace("300x", "1500x")

                    image_urls.append({image_url_col_name: url})
                    self._image_download_and_save(url=url, img_name=img_name, folderize=folderize)

                    # ---- RESIZE AFTER DOWNLOAD ----
                    force_1500_square(f"{self.assets_folder}/{img_name}")

            except Exception as e:
                print(f"[ERROR] Main image extraction failed for {asin}: {e}")
                image_urls.append({"main_image_error": True})

        # ---- SAFE BACKGROUND CHECK ----
        try:
            image_urls.append(self.is_prominent_bg_col_white(f"{self.assets_folder}/{asin}.main.jpg"))
        except Exception as e:
            print(f"[WARN] Background check skipped for {asin}: {e}")
            image_urls.append({"bg_check_failed": True})

        return image_urls


    def get_size_chart(self, product_row: dict, folderize: bool = True) -> str:
        asin = product_row.get("ASIN") or product_row.get("Seller SKU")
        if pd.isna(asin):
            asin = product_row.get("Seller SKU")

        print(f"Fetching images for ASIN (Size chart): {asin}")

        # --- Case 1: Old layout (collapsible-content) ---
        content_image_size = self.soup.find(
            "collapsible-content",
            id="block-template--24935003390025__product-content-image_irrATL-content"
        )
        if content_image_size:
            img_tag = content_image_size.find("img")
            if img_tag and img_tag.get("src"):
                content_image_size_img = f"https:{img_tag['src']}"
                print("Found size chart (collapsible-content):", content_image_size_img)
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
                print("Found size chart (fit_guide_pc):", content_image_size_img)
                img_name = f"{asin}.SIZE-CHART.jpg"
                self._image_download_and_save(
                    url=content_image_size_img, img_name=img_name, folderize=folderize
                )
                return content_image_size_img

        # No size chart found
        print("No size chart found for:", asin)
        return ""



class RunType(enum.Enum):
    fetch_data = "fetch_data"
    fetch_images = "fetch_images"
    A_Plus_fetch_images="A_Plus_fetch_images"


def fetch_text_and_images(df: pd.DataFrame, mode: str, progress_bar: bool = False):
    scraper = FasthouseScraper()
    # mode = 'fetch_images' # Allowed values are: fetch_data and fetch_images
    # input_fpath = './inputs/fh_may.csv'
    # image_urls_op_file = './outputs/'
    website_format = "new"  # Allowed values are old and new.

    # Check if the mode is acceptable.
    RunType(mode)

    # df = scraper.loader(fpath=input_fpath)

    # Smaller sample
    # data = df.iloc[-5:, :].to_dict('records')
    df = df.astype(object).where(df.notna(), '')
    data = df.to_dict("records")
    # Check if image download can even be carried out or not
    if mode == RunType.fetch_images.value and not any(col in df.columns for col in {"ASIN", "Seller SKU"}):
        raise ValueError("ASIN name required for image download to start.")

    try:
        if progress_bar:
            import streamlit as st

            status_bar = st.progress(0)
            step = 100 / len(data)

        for index, row in enumerate(data):
            if progress_bar:
                status_bar.progress(int((index + 1) * step))

            resp = scraper.make_soup_obj(row["URL"])
            if resp is None:
                print(f"[ERROR] Skipping row {index} - could not fetch {row['URL']}")
                continue

            if mode == RunType.fetch_data.value:
                row["Title"] = scraper.get_title_v2()
                row["Price"] = scraper.get_price()
                print(index, row["Title"])

                max_bullets = int(max(df["No of bullets"]))
                row.update(
                    scraper.get_description_and_bullets_v2(max_bullets=max_bullets)
                    if website_format == "new"
                    else scraper.get_description_and_bullets(max_bullets=max_bullets)
                )
                # Bullets and match
                row["Bullet match"] = row["No of bullets"] == row["Bullet check"]
                if row["Bullet check"] == 1 and not str(row.get("Bullet1", "")).strip():
                    print(f"[WARN] Row {index}: NO description/bullets found for {row['URL']}")
                else:
                    print(f"[OK] Row {index}: {row['Bullet check']} bullets captured")
            # Fetch images
            if mode == RunType.fetch_images.value:
                # Keep everything in the same folder
                row['Size-Chart']=scraper.get_size_chart(product_row=row, folderize=False)

                if(len(scraper.get_video_list()) > 0):
                    [row.update({'Is Video Available':True})]
                else:
                    [row.update({'Is Video Available':False})]
                image_urls = (
                    scraper.get_images_v2(product_row=row, folderize=False)
                    if website_format == "new"
                    else scraper.get_images(product_row=row, folderize=False)
                )
                [row.update(col_url_map) for col_url_map in image_urls]

             # Fetch A+ images
            if mode == RunType.A_Plus_fetch_images.value:
                    # Keep everything in the same folder
                image_urls = (scraper.get_A_Plus_images(product_row=row, folderize=False))

                [row.update(col_url_map) for col_url_map in image_urls]

    except Exception as e:
        if progress_bar:
            import streamlit as st
            st.exception(e)
        else:
            raise

    finally:
        out = pd.DataFrame(data)

        # Rearrange the columns to have all image name cols together
        if "Is Main Image Background White" in out.columns:
            out.insert(
                len(out.columns) - 1, "Is Main Image Background White", out.pop("Is Main Image Background White")
            )
        if "Exceeded 9 images" in out.columns:
            out.insert(len(out.columns) - 1, "Exceeded 9 images", out.pop("Exceeded 9 images"))
        if "Is Video Available" in out.columns:
             out.insert(len(out.columns) - 1, "Is Video Available", out.pop("Is Video Available"))

        if mode == RunType.fetch_data.value:
            fname = "Fasthouse.csv".replace(".csv", "__data.csv")
        elif mode == RunType.fetch_images.value:
            fname = "Fasthouse.csv".replace(".csv", "__images.csv")
        elif mode == RunType.A_Plus_fetch_images.value:
            fname = "Fasthouse.csv".replace(".csv", "__images.csv")

        out.to_csv(fname, sep=",", index=False)

    return out


if __name__ == "__main__":
    df = pd.read_csv("./fasthouse/may.csv")
    mode = "fetch_data"
    fetch_text_and_images(df, mode)
