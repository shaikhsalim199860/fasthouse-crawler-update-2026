"""Row iteration shared by both crawlers: worker pool, cancellation,
per-row error isolation, URL de-duplication, retry pass and progress
events."""
import gc
import logging
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

MAX_WORKERS = 6
GC_EVERY_ROWS = 25
RETRY_DELAY_SECONDS = 8.0
CANCELLED_MESSAGE = "cancelled before processing"
# Columns that mark a row as only partially successful (some images failed);
# such rows are re-run in the retry pass as well.
PARTIAL_FAILURE_COLUMNS = ("Image Errors",)

ProgressCallback = Callable[[Dict[str, Any]], None]
Replicate = Callable[[dict, dict], Optional[dict]]

# Columns that identify a row and must never be copied between rows that
# share a URL.
IDENTITY_COLUMNS = {"Seller SKU", "ASIN", "URL", "No of bullets", "Crawl Error"}
# Scraped columns that may also exist in the input file (an older export
# re-uploaded, for instance) and therefore must still be copied.
_SCRAPED_NAME = re.compile(
    r"^(Title|Price|Description|Bullet check|Bullet match|Bullet\d+|Size-Chart|Is Video Available|"
    r"Is Main Image Background White|Exceeded 9 images|Image Errors|main|pt0\d|A_Plus_pt0\d+|"
    r"main_image_missing|main_image_error|bg_check_failed|Size Chart .*|How to Measure|Fit Guide URL)$"
)


def row_label(row: dict, index: int) -> str:
    for key in ("Seller SKU", "ASIN"):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return f"row {index + 1}"


def asin_of(row: dict) -> str:
    """The ASIN used to name a row's files, falling back to the SKU."""
    asin = row.get("ASIN")
    if asin is None or not str(asin).strip() or str(asin).lower() == "nan":
        asin = row.get("Seller SKU", "")
    return str(asin).strip()


def url_key(row: dict) -> Optional[str]:
    url = str(row.get("URL", "") or "").strip()
    return url.split("#")[0].rstrip("/").lower() or None


def copy_scraped_columns(src: dict, dst: dict, input_columns: Iterable[str]) -> None:
    """Copy everything the crawl added to `src` onto `dst` (same URL)."""
    inputs = set(input_columns)
    for key, value in src.items():
        if key in IDENTITY_COLUMNS:
            continue
        if key not in inputs or _SCRAPED_NAME.match(str(key)):
            dst[key] = value
    # A leader whose retry succeeded no longer carries partial-failure
    # markers; drop stale ones copied during the first pass.
    for col in PARTIAL_FAILURE_COLUMNS:
        if col not in src:
            dst.pop(col, None)


def copy_asset_files(assets_folder: str, src_asin: str, dst_asin: str) -> int:
    """Duplicate `<src_asin>.*` files as `<dst_asin>.*`; returns the count."""
    if not src_asin or not dst_asin or src_asin == dst_asin or not os.path.isdir(assets_folder):
        return 0
    copied = 0
    prefix = f"{src_asin}."
    for name in os.listdir(assets_folder):
        if name.startswith(prefix):
            shutil.copyfile(
                os.path.join(assets_folder, name),
                os.path.join(assets_folder, dst_asin + name[len(src_asin):]),
            )
            copied += 1
    return copied


def run_rows(
    data: List[dict],
    process: Callable[[int, dict, Any], Optional[dict]],
    make_scraper: Callable[[], Any],
    *,
    workers: int = 1,
    on_progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    group_key: Optional[Callable[[dict], Optional[str]]] = url_key,
    replicate: Optional[Replicate] = None,
    retry_failed: bool = True,
) -> None:
    """Run `process(index, row, scraper)` for every row, in place.

    - Each worker thread gets its own scraper instance (the scrapers keep
      per-page state such as the parsed soup, so they must not be shared).
    - Rows with the same `group_key` (the URL by default) are crawled once:
      the first row is processed and `replicate(leader_row, other_row)`
      copies the result onto the others. Size variants of one product
      therefore cost one page fetch instead of six.
    - One bad row never aborts the crawl: the exception is recorded in the
      row's "Crawl Error" column and reported through `on_progress`.
    - Rows that failed get one more sequential attempt after a pause
      (`retry_failed`), which recovers transient stalls / rate limits.
    - `cancel_event` makes the remaining rows return immediately as
      "cancelled" so partial results can still be packaged.
    """
    workers = max(1, min(int(workers or 1), MAX_WORKERS))
    local = threading.local()

    def scraper() -> Any:
        instance = getattr(local, "scraper", None)
        if instance is None:
            instance = local.scraper = make_scraper()
        return instance

    def emit(event: Dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    def base_event(index: int, row: dict, retry: bool) -> Dict[str, Any]:
        event: Dict[str, Any] = {"index": index, "sku": row_label(row, index), "url": row.get("URL", "")}
        if retry:
            event["retry"] = True
        return event

    def run_one(index: int, row: dict, retry: bool = False) -> Dict[str, Any]:
        event = base_event(index, row, retry)
        if cancel_event is not None and cancel_event.is_set():
            event["status"] = "cancelled"
            row["Crawl Error"] = CANCELLED_MESSAGE
            return event

        if retry:
            for col in PARTIAL_FAILURE_COLUMNS:
                row.pop(col, None)

        instance = scraper()
        try:
            result = process(index, row, instance) or {}
            event.update(result)
            event.setdefault("status", "ok")
            row["Crawl Error"] = "" if event["status"] == "ok" else str(event.get("message", ""))
        except Exception as e:  # noqa: BLE001 - isolate per-row failures
            log.exception("Row %s (%s) failed", index, event["sku"])
            event["status"] = "error"
            event["message"] = f"{type(e).__name__}: {e}"
            row["Crawl Error"] = event["message"]
        finally:
            release = getattr(instance, "release", None)
            if callable(release):
                release()
        if index % GC_EVERY_ROWS == 0:
            gc.collect()
        return event

    def run_followers(leader_event: Dict[str, Any], leader_row: dict, followers: List[int], retry: bool) -> None:
        for f_index in followers:
            f_row = data[f_index]
            event = base_event(f_index, f_row, retry)
            status = leader_event.get("status", "ok")
            if status == "ok" and replicate is not None:
                try:
                    extra = replicate(leader_row, f_row) or {}
                    event.update(extra)
                    event["status"] = "ok"
                    event["message"] = f"same URL as {leader_event['sku']} - " + str(extra.get("message") or "result copied")
                    f_row["Crawl Error"] = ""
                except Exception as e:  # noqa: BLE001
                    log.exception("Replicating row %s failed", f_index)
                    event["status"] = "error"
                    event["message"] = f"copy from {leader_event['sku']} failed: {type(e).__name__}: {e}"
                    f_row["Crawl Error"] = event["message"]
            else:
                event["status"] = status
                event["message"] = leader_event.get("message", "")
                f_row["Crawl Error"] = leader_row.get("Crawl Error", "")
            emit(event)

    # ---- group rows by URL -------------------------------------------------
    groups: List[Tuple[int, List[int]]] = []
    seen: Dict[str, int] = {}
    for index, row in enumerate(data):
        key = group_key(row) if group_key else None
        if key is not None and key in seen:
            groups[seen[key]][1].append(index)
        else:
            if key is not None:
                seen[key] = len(groups)
            groups.append((index, []))

    def run_group(group: Tuple[int, List[int]], retry: bool = False) -> None:
        leader, followers = group
        event = run_one(leader, data[leader], retry)
        emit(event)
        run_followers(event, data[leader], followers, retry)

    # ---- main pass ----------------------------------------------------------
    if workers == 1:
        for group in groups:
            run_group(group)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="crawl") as pool:
            for _ in pool.map(run_group, groups):
                pass

    # ---- retry pass (sequential, after a pause) ------------------------------
    if not retry_failed or (cancel_event is not None and cancel_event.is_set()):
        return
    def needs_retry(row: dict) -> bool:
        if row.get("Crawl Error") not in ("", None, CANCELLED_MESSAGE):
            return True
        return any(str(row.get(col) or "").strip() for col in PARTIAL_FAILURE_COLUMNS)

    failed = [g for g in groups if needs_retry(data[g[0]])]
    if not failed:
        return
    log.info("Retrying %d failed URL(s) after %.0fs pause", len(failed), RETRY_DELAY_SECONDS)
    deadline = time.time() + RETRY_DELAY_SECONDS
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            return
        time.sleep(0.5)
    for group in failed:
        if cancel_event is not None and cancel_event.is_set():
            return
        run_group(group, retry=True)


def streamlit_progress_callback(total: int) -> ProgressCallback:
    """Legacy shim for `progress_bar=True`: draws an st.progress bar from
    the calling (Streamlit) thread. Only valid when the crawler runs inside
    a script run, not from the background job runner."""
    import streamlit as st

    bar = st.progress(0)
    counter = {"done": 0}

    def on_progress(event: Dict[str, Any]) -> None:
        if event.get("retry"):
            return
        counter["done"] += 1
        bar.progress(min(100, int(counter["done"] * 100 / max(total, 1))))

    return on_progress
