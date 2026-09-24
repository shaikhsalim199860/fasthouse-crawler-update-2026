"""Resolve Fasthouse Seller SKUs to product URLs.

Fasthouse publishes its whole catalogue at `/products.json`, and every
variant carries the same SKU used in the Amazon feed (`400013-01-07`).
Building an index of those once turns a SKU list into product URLs, so a
file with no `URL` column can still be crawled - and SKUs that are no
longer on the site are reported rather than silently skipped.

The index covers ~5,300 SKUs across ~1,100 products and costs five
requests, so it is fetched once and reused for the whole run.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from crawler_app.netutil import fetch_bytes

log = logging.getLogger(__name__)

PRODUCTS_URL = "https://www.fasthouse.com/products.json"
PRODUCT_URL = "https://www.fasthouse.com/products/{handle}"
PAGE_SIZE = 250
MAX_PAGES = 20
CACHE_SECONDS = 30 * 60

# Columns the lookup adds to the output.
SKU_COLUMNS = ("URL", "SKU Status", "Matched Product", "Variant", "Available", "Catalogue Price")
STATUS_FOUND = "Found"
STATUS_MISSING = "Not found on fasthouse.com"


def normalise(sku: object) -> str:
    """SKUs are compared case-insensitively and without padding, since
    spreadsheets love to add both."""
    return str(sku or "").strip().upper()


@dataclass
class SkuEntry:
    sku: str
    handle: str
    product_title: str = ""
    variant_title: str = ""
    available: bool = True
    price: str = ""
    product_type: str = ""

    @property
    def url(self) -> str:
        return PRODUCT_URL.format(handle=self.handle)


@dataclass
class SkuIndex:
    entries: Dict[str, SkuEntry]
    products: int = 0
    built_at: float = 0.0

    def get(self, sku: object) -> Optional[SkuEntry]:
        return self.entries.get(normalise(sku))

    @property
    def age_seconds(self) -> float:
        return time.time() - self.built_at

    def __len__(self) -> int:
        return len(self.entries)


_cache_lock = threading.Lock()
_cached: Optional[SkuIndex] = None


def build_sku_index(session: requests.Session, max_pages: int = MAX_PAGES) -> SkuIndex:
    """Walk /products.json and map every variant SKU to its product."""
    import json

    entries: Dict[str, SkuEntry] = {}
    products = 0
    for page in range(1, max_pages + 1):
        raw = fetch_bytes(session, f"{PRODUCTS_URL}?limit={PAGE_SIZE}&page={page}")
        batch = json.loads(raw).get("products", [])
        if not batch:
            break
        products += len(batch)
        for product in batch:
            handle = product.get("handle") or ""
            if not handle:
                continue
            for variant in product.get("variants", []):
                key = normalise(variant.get("sku"))
                if not key or key in entries:
                    continue
                entries[key] = SkuEntry(
                    sku=str(variant.get("sku") or "").strip(),
                    handle=handle,
                    product_title=product.get("title") or "",
                    variant_title=variant.get("title") or "",
                    available=bool(variant.get("available")),
                    price=str(variant.get("price") or ""),
                    product_type=product.get("product_type") or "",
                )
    index = SkuIndex(entries=entries, products=products, built_at=time.time())
    log.info("SKU index built: %d SKUs across %d products", len(entries), products)
    return index


def get_sku_index(session: requests.Session, max_age: float = CACHE_SECONDS,
                  force: bool = False) -> SkuIndex:
    """Cached index. Rebuilt when older than `max_age`, so a long-running
    server picks up newly published products without a restart."""
    global _cached
    with _cache_lock:
        if not force and _cached is not None and _cached.age_seconds < max_age:
            return _cached
    index = build_sku_index(session)
    with _cache_lock:
        _cached = index
    return index


def resolve_dataframe(df: pd.DataFrame, index: SkuIndex,
                      sku_column: str = "Seller SKU") -> Tuple[pd.DataFrame, Dict]:
    """Fill in `URL` (and the lookup columns) from the SKU index.

    Rows that already have a URL are left alone, so a file that mixes both
    still works. Returns the frame and a summary for the UI/log.
    """
    if sku_column not in df.columns:
        raise ValueError(f"The file needs a {sku_column!r} column to look SKUs up.")

    df = df.copy()
    if "URL" not in df.columns:
        df["URL"] = ""
    existing = df["URL"].astype(str).str.strip()
    df["URL"] = existing.where(existing.str.lower() != "nan", "")

    statuses, titles, variants, available, prices = [], [], [], [], []
    resolved = kept = missing = 0
    missing_skus: List[str] = []

    for position in range(len(df)):
        current_url = str(df.iloc[position]["URL"]).strip()
        sku = df.iloc[position][sku_column]
        entry = index.get(sku)

        if current_url:
            kept += 1
            statuses.append("URL supplied")
            titles.append(entry.product_title if entry else "")
            variants.append(entry.variant_title if entry else "")
            available.append(entry.available if entry else "")
            prices.append(entry.price if entry else "")
            continue

        if entry is None:
            missing += 1
            missing_skus.append(str(sku))
            statuses.append(STATUS_MISSING)
            titles.append(""); variants.append(""); available.append(""); prices.append("")
            continue

        resolved += 1
        df.iat[position, df.columns.get_loc("URL")] = entry.url
        statuses.append(STATUS_FOUND)
        titles.append(entry.product_title)
        variants.append(entry.variant_title)
        available.append(entry.available)
        prices.append(entry.price)

    df["SKU Status"] = statuses
    df["Matched Product"] = titles
    df["Variant"] = variants
    df["Available"] = available
    df["Catalogue Price"] = prices

    summary = {
        "rows": len(df),
        "resolved": resolved,
        "url_supplied": kept,
        "missing": missing,
        "missing_skus": missing_skus,
        "unique_urls": int(df.loc[df["URL"].astype(str).str.strip() != "", "URL"].nunique()),
        "index_size": len(index),
        "index_products": index.products,
    }
    return df, summary
