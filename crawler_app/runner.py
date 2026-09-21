"""Row iteration shared by both crawlers: worker pool, cancellation,
per-row error isolation and progress events."""
import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

MAX_WORKERS = 6
GC_EVERY_ROWS = 25

ProgressCallback = Callable[[Dict[str, Any]], None]


def row_label(row: dict, index: int) -> str:
    for key in ("Seller SKU", "ASIN"):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return f"row {index + 1}"


def run_rows(
    data: List[dict],
    process: Callable[[int, dict, Any], Optional[dict]],
    make_scraper: Callable[[], Any],
    *,
    workers: int = 1,
    on_progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Run `process(index, row, scraper)` for every row, in place.

    - Each worker thread gets its own scraper instance (the scrapers keep
      per-page state such as the parsed soup, so they must not be shared).
    - One bad row never aborts the crawl: the exception is recorded in the
      row's "Crawl Error" column and reported through `on_progress`.
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

    def task(index: int, row: dict) -> None:
        event: Dict[str, Any] = {
            "index": index,
            "sku": row_label(row, index),
            "url": row.get("URL", ""),
        }
        if cancel_event is not None and cancel_event.is_set():
            event["status"] = "cancelled"
            row["Crawl Error"] = "cancelled before processing"
            if on_progress:
                on_progress(event)
            return

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

        if on_progress:
            on_progress(event)
        if index % GC_EVERY_ROWS == 0:
            gc.collect()

    if workers == 1:
        for index, row in enumerate(data):
            task(index, row)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="crawl") as pool:
            for _ in pool.map(lambda pair: task(*pair), enumerate(data)):
                pass


def streamlit_progress_callback(total: int) -> ProgressCallback:
    """Legacy shim for `progress_bar=True`: draws an st.progress bar from
    the calling (Streamlit) thread. Only valid when the crawler runs inside
    a script run, not from the background job runner."""
    import streamlit as st

    bar = st.progress(0)
    counter = {"done": 0}

    def on_progress(event: Dict[str, Any]) -> None:
        counter["done"] += 1
        bar.progress(min(100, int(counter["done"] * 100 / max(total, 1))))

    return on_progress
