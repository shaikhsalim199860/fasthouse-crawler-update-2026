"""Helpers for reviewing a finished crawl: asset thumbnails, an
"issues only" filter over the output, and the running build version.

Kept out of the page so main.py stays readable; no Streamlit import here
so these can be tested on their own.
"""
import io
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)

# ASIN.main.jpg / ASIN.pt05.png / ASIN.SIZE-CHART-2.png
ASSET_RE = re.compile(r"^(?P<asin>.+?)\.(?P<slot>main|pt0\d|SIZE-CHART(?:-\d+)?|A_Plus_pt0\d+)\.(?:jpg|jpeg|png)$", re.I)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _slot_rank(slot: str) -> tuple:
    """MAIN first, then PT01-PT08, then anything else, so a product's
    files appear in the order the listing will show them."""
    low = slot.lower()
    if low == "main":
        return (0, 0, slot)
    if low.startswith("pt0"):
        return (1, int(low[3:] or 0), slot)
    return (2, 0, slot)


def group_assets(assets_dir: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """{ASIN: [(slot, path), ...]} for everything the crawl produced."""
    assets_dir = Path(assets_dir)
    grouped: Dict[str, List[Tuple[str, Path]]] = {}
    if not assets_dir.is_dir():
        return grouped
    for path in assets_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = ASSET_RE.match(path.name)
        if match:
            grouped.setdefault(match.group("asin"), []).append((match.group("slot"), path))
        else:
            grouped.setdefault(path.stem, []).append(("", path))
    for files in grouped.values():
        files.sort(key=lambda item: _slot_rank(item[0]))
    return grouped


def thumbnail_bytes(path: Path, size: int = 360) -> bytes:
    """A small PNG for the preview grid.

    Size charts are 2000x2000; sending those to the browser at full size
    would make the results page crawl, so they are shrunk server side.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, "PNG", optimize=True)
        return buffer.getvalue()


# ---------------------------------------------------------------- issues

def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _nonblank(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan") & text.str.lower().ne("none")


def issue_report(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """(mask of rows needing a look, per-row reason text).

    Everything the crawl already flags in its own columns, collected into
    one filter so the rows worth checking can be found without scrolling
    a 28-column sheet.
    """
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

    def add(condition: pd.Series, label: str) -> None:
        for position in df.index[condition.fillna(False)]:
            reasons.at[position].append(label)

    if "Crawl Error" in df:
        add(_nonblank(df["Crawl Error"]), "crawl error")
    if "SKU Status" in df:
        add(df["SKU Status"].astype(str).str.startswith("Not found"), "SKU not in catalogue")
    if "Available" in df:
        add(df["Available"].astype(str).str.lower().eq("false"), "out of stock")
    if "Image Errors" in df:
        add(_nonblank(df["Image Errors"]), "image download failed")
    if "Bullet match" in df:
        add(df["Bullet match"].astype(str).str.lower().eq("false"), "bullet count differs")
    if "Is Main Image Background White" in df:
        add(df["Is Main Image Background White"].astype(str).str.lower().eq("false"),
            "main image not on white")
    if "Exceeded 9 images" in df:
        add(_truthy(df["Exceeded 9 images"]), "more than 9 images")
    if "Gallery Images Dropped" in df:
        numeric = pd.to_numeric(df["Gallery Images Dropped"], errors="coerce").fillna(0)
        add(numeric.gt(0), "gallery image dropped")
    if "Size Chart Status" in df:
        add(df["Size Chart Status"].astype(str).str.startswith("No size chart"), "no size chart")
    for column, label in (("main_image_missing", "main image missing"),
                          ("main_image_error", "main image failed")):
        if column in df:
            add(_truthy(df[column]), label)

    text = reasons.apply(lambda items: ", ".join(items))
    return text.ne(""), text


# ---------------------------------------------------------------- version

def build_version(root: Path) -> Dict[str, str]:
    """Commit and date of the running code, so "did my deploy land?" is
    answerable from the page. Falls back quietly outside a git checkout."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - git absent or not a checkout
            return ""

    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return {"commit": "", "date": "", "subject": ""}
    return {
        "commit": commit,
        "date": git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"),
        "subject": git("log", "-1", "--format=%s")[:80],
    }
