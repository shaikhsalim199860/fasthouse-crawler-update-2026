"""Publish crawled images to S3 so Amazon can fetch them by URL.

Two things shape the design:

* **Content-addressed keys.** The six size variants of a product share
  byte-identical images, and a size chart is shared by every product in
  its sizing group - in a real 135-ASIN batch, 1,159 files on disk are
  only 163 distinct images. Naming by content hash uploads each one once
  and lets every row point at the same URL.
* **A changed image must get a new URL.** Amazon ingests a URL once and
  caches it; re-uploading to the same key would leave the listing showing
  the old picture. A content hash changes exactly when the image does,
  which is the behaviour we want.

Nothing here imports Streamlit, so it can be tested with a fake client.
"""
import hashlib
import logging
import mimetypes
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

HASH_LENGTH = 16          # 64 bits - collision-safe for millions of objects
UPLOAD_WORKERS = 8
GALLERY_PREFIX = "images"
CHART_PREFIX = "sizecharts"
# A year; the key changes whenever the image does, so this is always safe.
CACHE_CONTROL = "public, max-age=31536000, immutable"

# ASIN.slot.ext produced by the crawl
ASSET_RE = re.compile(
    r"^(?P<asin>.+?)\.(?P<slot>main|pt0\d|SIZE-CHART(?:-\d+)?|A_Plus_pt0\d+)\.(?P<ext>jpg|jpeg|png)$",
    re.I,
)

# Slot -> Amazon flat-file column, so the output can be pasted straight in.
FLAT_FILE_COLUMNS = {"main": "main_image_url", **{f"pt0{i}": f"other_image_url{i}" for i in range(1, 9)}}


@dataclass
class HostConfig:
    bucket: str = ""
    region: str = "us-east-1"
    prefix: str = ""                    # optional folder inside the bucket
    public_base_url: str = ""           # override, e.g. a CloudFront domain
    access_key_id: str = ""
    secret_access_key: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)

    @property
    def base_url(self) -> str:
        if self.public_base_url:
            return self.public_base_url.rstrip("/")
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com"

    def url_for(self, key: str) -> str:
        return f"{self.base_url}/{key}"


@dataclass
class UploadResult:
    uploaded: int = 0
    skipped: int = 0                    # already in the bucket
    failed: int = 0
    bytes_uploaded: int = 0
    errors: List[str] = field(default_factory=list)
    # local file name -> public URL
    urls: Dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.uploaded + self.skipped + self.failed


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:HASH_LENGTH]


def parse_asset_name(name: str) -> Optional[Tuple[str, str, str]]:
    """(asin, slot, ext) for a crawled file name."""
    match = ASSET_RE.match(name)
    if not match:
        return None
    return match.group("asin"), match.group("slot"), match.group("ext").lower()


class ImageHost:
    def __init__(self, config: HostConfig, client=None) -> None:
        self.config = config
        self._client = client
        self._lock = threading.Lock()
        self._existing: Optional[Set[str]] = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            kwargs = {"region_name": self.config.region}
            if self.config.access_key_id and self.config.secret_access_key:
                kwargs["aws_access_key_id"] = self.config.access_key_id
                kwargs["aws_secret_access_key"] = self.config.secret_access_key
            self._client = boto3.client("s3", **kwargs)
        return self._client

    # -- keys ----------------------------------------------------------
    def key_for(self, path: Path, slot: str = "") -> str:
        """Content-addressed key. Charts live under their own prefix purely
        so the bucket is easy to browse."""
        extension = path.suffix.lower().lstrip(".")
        extension = "jpg" if extension == "jpeg" else extension
        folder = CHART_PREFIX if "size-chart" in slot.lower() or extension == "png" else GALLERY_PREFIX
        key = f"{folder}/{file_hash(path)}.{extension}"
        return f"{self.config.prefix.strip('/')}/{key}" if self.config.prefix else key

    # -- remote state --------------------------------------------------
    def existing_keys(self, refresh: bool = False) -> Set[str]:
        """One listing of the bucket beats a HEAD per file: a re-run of an
        unchanged batch then costs a single request and uploads nothing."""
        with self._lock:
            if self._existing is not None and not refresh:
                return self._existing
        found: Set[str] = set()
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.config.bucket,
                                           Prefix=self.config.prefix.strip("/") if self.config.prefix else ""):
                for item in page.get("Contents", []):
                    found.add(item["Key"])
        except Exception as e:  # noqa: BLE001 - fall back to uploading everything
            log.warning("Could not list %s: %s", self.config.bucket, e)
        with self._lock:
            self._existing = found
        return found

    # -- upload --------------------------------------------------------
    def upload_file(self, path: Path, key: str) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with open(path, "rb") as handle:
            self.client.put_object(
                Bucket=self.config.bucket, Key=key, Body=handle,
                ContentType=content_type, CacheControl=CACHE_CONTROL,
            )

    def upload_assets(self, assets_dir: Path, on_progress=None) -> UploadResult:
        """Upload every crawled image, skipping anything already hosted."""
        assets_dir = Path(assets_dir)
        result = UploadResult()
        if not self.config.enabled or not assets_dir.is_dir():
            return result

        files = [p for p in sorted(assets_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not files:
            return result

        # Distinct content -> one upload, shared by every file with it.
        by_key: Dict[str, List[Path]] = {}
        for path in files:
            parsed = parse_asset_name(path.name)
            slot = parsed[1] if parsed else ""
            by_key.setdefault(self.key_for(path, slot), []).append(path)

        already = self.existing_keys()
        todo = [(key, paths) for key, paths in by_key.items() if key not in already]
        log.info("Hosting %d distinct image(s) from %d file(s); %d already in the bucket",
                 len(by_key), len(files), len(by_key) - len(todo))

        done = 0
        lock = threading.Lock()

        def send(item: Tuple[str, List[Path]]) -> None:
            nonlocal done
            key, paths = item
            try:
                self.upload_file(paths[0], key)
                with lock:
                    result.uploaded += 1
                    result.bytes_uploaded += paths[0].stat().st_size
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the rest
                log.error("Upload failed for %s: %s", paths[0].name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"{paths[0].name}: {e}")
            finally:
                with lock:
                    done += 1
                if on_progress:
                    on_progress(done, len(todo))

        if todo:
            with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
                list(pool.map(send, todo))

        failed_keys = {e.split(":")[0] for e in result.errors}
        for key, paths in by_key.items():
            if paths[0].name in failed_keys:
                continue
            if key in already:
                result.skipped += len(paths)
            url = self.config.url_for(key)
            for path in paths:
                result.urls[path.name] = url
        return result


def apply_hosted_urls(df, urls: Dict[str, str], mode: str = "replace",
                      asin_column: str = "ASIN", fallback_column: str = "Seller SKU"):
    """Put the hosted URLs into the output.

    mode="replace" (default) overwrites the `main` / `pt01`-`pt08` columns,
    which held the Shopify source URL, with the URL of the image actually
    uploaded - those are what goes to Amazon, and the source URL points at
    a different (1200px, differently encoded) picture.

    mode="flatfile" instead adds Amazon's own column names
    (`main_image_url`, `other_image_url1`-`8`) and leaves the source URLs
    alone.

    A slot with no hosted URL - upload failed, or hosting was off - keeps
    whatever was there, so nothing is silently lost.
    """
    by_asin: Dict[str, Dict[str, str]] = {}
    for name, url in urls.items():
        parsed = parse_asset_name(name)
        if not parsed:
            continue
        asin, slot, _ = parsed
        by_asin.setdefault(asin, {})[slot.lower()] = url

    def key_of(row) -> str:
        value = row.get(asin_column)
        if value is None or str(value).strip() in ("", "nan"):
            value = row.get(fallback_column, "")
        return str(value).strip()

    records = df.to_dict("records")
    hosted_flags: List[bool] = []
    chart_urls: List[str] = []
    columns: Dict[str, List[str]] = {}

    for row in records:
        slots = by_asin.get(key_of(row), {})
        hosted_flags.append(bool(slots))
        for slot, flat_column in FLAT_FILE_COLUMNS.items():
            target = slot if mode == "replace" else flat_column
            existing = row.get(slot if mode == "replace" else flat_column, "")
            value = slots.get(slot) or ("" if mode == "flatfile" else existing)
            columns.setdefault(target, []).append("" if value is None else value)

        charts = [url for slot, url in sorted(slots.items()) if "size-chart" in slot]
        chart_slot = str(row.get("Size Chart Slot") or "").split(" | ")[0].lower()
        if not charts and chart_slot.startswith("pt") and slots.get(chart_slot):
            charts = [slots[chart_slot]]
        chart_urls.append(" | ".join(u for u in charts if u))

    for column, values in columns.items():
        # Do not invent empty slot columns the crawl never produced.
        if any(values) or column in df.columns:
            df[column] = values
    if any(chart_urls):
        df["Size Chart URL"] = chart_urls
    if any(hosted_flags) and not all(hosted_flags):
        df["Images Hosted"] = hosted_flags
    return df


def add_hosted_columns(df, urls: Dict[str, str], asin_column: str = "ASIN",
                       fallback_column: str = "Seller SKU"):
    """Add Amazon flat-file image URL columns from the upload mapping.

    Kept for callers that want Amazon's own column names; equivalent to
    apply_hosted_urls(..., mode="flatfile").
    """
    return apply_hosted_urls(df, urls, mode="flatfile",
                             asin_column=asin_column, fallback_column=fallback_column)
