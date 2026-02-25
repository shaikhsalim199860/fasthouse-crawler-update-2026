import os
import copy
import enum
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
############################
# How can I change this code such that image download or scraping is a pluggable component?
############################


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
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.content

    def make_soup_obj(self, url: str) -> Optional[BeautifulSoup]:
        try:
            if url:
                content = self._get_page_source(url)
                self.soup = BeautifulSoup(content, "html.parser")
            else:
                return None
        except requests.exceptions.RequestException:
                return None

        if not content:
            return None
        return True

    def get_title(self):
        return self.soup.title.text


class FasthouseScraper(BaseScraper):
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
            price= self.soup.find("span", class_="price").get_text().replace("Sale price","")
            return self.clean_string(price).replace("$","")

    def get_title_v2(self):
        # We have to override this, as fasthouse has changed their title strategy.
        return self.clean_string(self.soup.title.text.split("\n")[0])

    def matchWord(self,q):
            return re.compile(r'\b({0})\b'.format(q), flags=re.IGNORECASE).search
    
    def getRedText(self):
        list_q=['size','sizes','ordering']
        desc = self.soup.select_one('.rte')
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
        description_identifier = ".rte"  # Old: .description.content
        bullet_character = "•"

        # Add the remaining bullet headers: Bullet{1} -> Bullet{n}
        result.update({f"Bullet{i + 1}": "" for i in range(max_bullets)})

        # Main description body
        # Problematic cases:
        # - "Features" word is present in the main body.
        # - <ul> present without "features" word.
        # - Bullets directly present.

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


        # If the description content contains bullets
        if self.soup.select_one(f"{description_identifier} ul"):
            start_v=1
            if self.getRedText():
                start_v=2
                result['Bullet1']= self.getRedText()

            features = {
                f"Bullet{index}": value
                for index, value in enumerate(
                    [
                        val.strip()
                        for val in self.soup.select_one(f"{description_identifier} ul").text.strip().split("\n")
                        if val
                    ],
                    start=start_v,
                )
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
                for index, value in enumerate([b.strip().lstrip(bullet_character).lstrip() for b in bullets], start=start_v)
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

        # ? Instead of this logic:
        # // self.soup.select_one('.description.content').text.strip().split('\n')[0].strip() + '<BR><BR>'
        # We can go for finding features and the using that:
        # *
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
    df.fillna('', inplace=True)
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
                print(row)
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
        st.exception(e)

            
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
    mode = "fetch_images"
    fetch_text_and_images(df, mode)
