"""Background job runner.

A crawl runs in its own daemon thread and reports into a `Job` object that
the Streamlit page polls. This decouples the crawl from the browser
session: if the websocket drops during a long run (tab sleeps, network
blip) the crawl keeps going and the page can reconnect to it on reload.
"""
import logging
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import pandas as pd

from crawler_app.archive import DEFAULT_PART_BYTES, build_zip_parts, remove_matching

log = logging.getLogger(__name__)

MODE_BY_CRAWL_TYPE = {
    "Data": "fetch_data",
    "Images": "fetch_images",
    "A+ Images": "A_Plus_fetch_images",
    "Size Charts": "fetch_size_charts",
}
# Crawl types whose output is a folder of files delivered as ZIP parts.
IMAGE_CRAWL_TYPES = {"Images", "A+ Images", "Size Charts"}
CSV_SUFFIX = {"Data": "data", "Images": "images", "A+ Images": "images", "Size Charts": "sizecharts"}

ACTIVE_STATUSES = {"queued", "running", "packaging"}


@dataclass
class Job:
    id: str
    website: str
    crawl_type: str
    total: int
    workers: int
    status: str = "queued"
    stage: str = "Queued"
    done: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    images: int = 0
    current: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=400))
    failures: List[dict] = field(default_factory=list)
    result_df: Optional[pd.DataFrame] = None
    csv_path: Optional[Path] = None
    xlsx_path: Optional[Path] = None
    zip_parts: List[Path] = field(default_factory=list)
    options: Dict = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -- derived -------------------------------------------------------
    @property
    def mode(self) -> str:
        return MODE_BY_CRAWL_TYPE[self.crawl_type]

    @property
    def is_image_job(self) -> bool:
        return self.crawl_type in IMAGE_CRAWL_TYPES

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.done == 0 or self.total == 0 or not self.is_active:
            return None
        per_row = self.elapsed / self.done
        return max(0.0, per_row * (self.total - self.done))

    @property
    def progress(self) -> float:
        return 0.0 if self.total == 0 else min(1.0, self.done / self.total)

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self._log("Cancel requested - finishing the rows already in flight...")

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {message}")

    # Called from worker threads via the crawler's progress callback.
    def on_progress(self, event: Dict) -> None:
        status = event.get("status", "ok")
        sku = str(event.get("sku") or "")
        with self.lock:
            self.done += 1
            self.current = sku
            self.images += int(event.get("images") or 0)
            if status == "ok":
                self.ok += 1
                extra = f" - {event['message']}" if event.get("message") else ""
                self._log(f"OK  {sku}{extra}")
            elif status == "cancelled":
                self.skipped += 1
            else:
                self.failed += 1
                msg = event.get("message") or "unknown error"
                self.failures.append({
                    "Row": event.get("index"),
                    "Seller SKU": sku,
                    "URL": event.get("url", ""),
                    "Error": msg,
                })
                self._log(f"ERR {sku} - {msg}")


class JobRegistry:
    """Holds the single active/most recent job for the process.

    Only one crawl may run at a time because the image modes share the
    `assets/` folder on disk.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: Optional[Job] = None

    def start(
        self,
        *,
        website: str,
        crawl_type: str,
        df: pd.DataFrame,
        crawler: Callable,
        workers: int,
        assets_dir: Path,
        outputs_dir: Path,
        downloads_dir: Path,
        part_bytes: int = DEFAULT_PART_BYTES,
        options: Optional[Dict] = None,
    ) -> Job:
        with self._lock:
            if self.current is not None and self.current.is_active:
                raise RuntimeError("A crawl is already running. Wait for it to finish or cancel it.")
            job = Job(
                id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
                website=website,
                crawl_type=crawl_type,
                total=len(df),
                workers=workers,
                options=dict(options or {}),
            )
            self.current = job

        thread = threading.Thread(
            target=_run_job,
            kwargs=dict(
                job=job,
                df=df,
                crawler=crawler,
                assets_dir=Path(assets_dir),
                outputs_dir=Path(outputs_dir),
                downloads_dir=Path(downloads_dir),
                part_bytes=part_bytes,
            ),
            name=f"crawl-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def clear(self) -> None:
        with self._lock:
            if self.current is not None and self.current.is_active:
                raise RuntimeError("Cannot clear a running crawl. Cancel it first.")
            self.current = None


REGISTRY = JobRegistry()


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _run_job(
    *,
    job: Job,
    df: pd.DataFrame,
    crawler: Callable,
    assets_dir: Path,
    outputs_dir: Path,
    downloads_dir: Path,
    part_bytes: int,
) -> None:
    try:
        job.status = "running"
        job.stage = "Preparing"
        job._log(f"Starting {job.website} / {job.crawl_type} for {job.total} rows with {job.workers} worker(s)")

        outputs_dir.mkdir(parents=True, exist_ok=True)
        if job.is_image_job:
            _clean_dir(assets_dir)
            remove_matching(downloads_dir, job.website)

        job.stage = "Crawling"
        out = crawler(
            df,
            job.mode,
            on_progress=job.on_progress,
            cancel_event=job.cancel_event,
            workers=job.workers,
            assets_folder=str(assets_dir),
            output_dir=None,
            **job.options,
        )

        # Drop the "Crawl Error" column when nothing failed to keep the
        # output identical to before for clean runs.
        if "Crawl Error" in out.columns and not out["Crawl Error"].astype(str).str.strip().any():
            out = out.drop(columns=["Crawl Error"])
        job.result_df = out

        job.stage = "Writing CSV / Excel"
        suffix = CSV_SUFFIX.get(job.crawl_type, "images")
        csv_name = f"{job.website}__{suffix}.csv"
        csv_path = outputs_dir / csv_name
        out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        job.csv_path = csv_path
        try:
            xlsx_path = outputs_dir / f"{job.website}__{suffix}.xlsx"
            out.to_excel(xlsx_path, index=False)
            job.xlsx_path = xlsx_path
        except Exception as e:  # noqa: BLE001 - Excel is a convenience copy
            log.warning("Excel export skipped: %s", e)

        if job.is_image_job:
            job.status = "packaging"
            job.stage = "Packaging images into ZIP"
            job._log("Crawl finished - packaging images...")
            shutil.copyfile(csv_path, assets_dir / csv_name)
            if job.xlsx_path:
                shutil.copyfile(job.xlsx_path, assets_dir / job.xlsx_path.name)
            job.zip_parts = build_zip_parts(
                assets_dir, downloads_dir, f"{job.website}_{job.id}", max_part_bytes=part_bytes
            )
            job._log(f"Packaged {job.images} images into {len(job.zip_parts)} archive(s)")

        job.status = "cancelled" if job.cancel_event.is_set() else "done"
        job.stage = "Cancelled" if job.status == "cancelled" else "Finished"
        job._log(f"{job.stage} in {job.elapsed:.0f}s - ok={job.ok} failed={job.failed} skipped={job.skipped}")

    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        log.exception("Job %s failed", job.id)
        job.status = "failed"
        job.stage = "Failed"
        job.error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        job._log(f"FAILED: {e}")
    finally:
        job.finished_at = time.time()
