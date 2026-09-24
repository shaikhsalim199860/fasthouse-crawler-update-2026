import io
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

from crawler_app.archive import MB
from crawler_app.bigfiles import raise_static_file_limit
from crawler_app import skucheck_ui
from crawler_app.jobs import IMAGE_CRAWL_TYPES, REGISTRY, Job
from crawler_app.review import build_version, group_assets, issue_report, thumbnail_bytes
from crawler_app.runner import MAX_WORKERS
from fasthouse.scrape import fetch_text_and_images
from fasthouse.sizechart import UNIT_CHOICES
from seven.scrape import start as seven_start

# -----------------------
# Config
# -----------------------

ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"
# Served by Streamlit straight from disk at app/static/downloads/<file>
# (server.enableStaticServing in .streamlit/config.toml), so a 500 MB image
# archive never has to be held in RAM.
DOWNLOADS_DIR = ROOT / "static" / "downloads"
STATIC_SERVING = bool(st.get_option("server.enableStaticServing"))
# Ceiling for a single served archive; far above any realistic batch.
SINGLE_FILE_LIMIT = 20 * 1024 * MB


@st.cache_resource(show_spinner=False)
def _enable_big_downloads() -> bool:
    """Raise Streamlit's 200 MB static-file cap once per process so a whole
    batch can be served as one ZIP streamed from disk."""
    return STATIC_SERVING and raise_static_file_limit(SINGLE_FILE_LIMIT) > 0


BIG_DOWNLOADS = _enable_big_downloads()

CRAWLERS = {"Fasthouse": fetch_text_and_images, "Seven": seven_start}

# Rough per-row cost used only for the "estimated time" hint. `.get` keeps a
# new crawl type from crashing the page before it is added here.
SECONDS_PER_ROW = {"Data": 2.0, "Images": 7.0, "A+ Images": 5.0,
                   "Size Charts": 2.0, "SKU Lookup": 0.05}
DEFAULT_SECONDS_PER_ROW = 2.0

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Fasthouse & Seven Crawler",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------
# Helpers
# -----------------------

def required_columns(website: str, crawl_type: str) -> List[str]:
    # Fasthouse product URLs can be looked up from the Seller SKU, so a
    # file exported from Amazon (SKU + ASIN, no URL) is accepted as is.
    if website == "Fasthouse":
        return ["Seller SKU", "No of bullets"] if crawl_type == "Data" else ["Seller SKU"]
    return ["Seller SKU", "URL"]


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "–"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def fmt_size(num_bytes: int) -> str:
    return f"{num_bytes / MB:.0f} MB" if num_bytes >= MB else f"{num_bytes / 1024:.0f} KB"


@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def thumb(path: str, mtime: float, size: int = 360) -> bytes:
    """Cached by path+mtime so a re-crawl of the same ASIN refreshes it."""
    return thumbnail_bytes(Path(path), size)


@st.cache_resource(show_spinner=False)
def version_info() -> dict:
    return build_version(ROOT)


@st.cache_data(show_spinner=False, ttl=900)
def sku_precheck(raw: bytes) -> Optional[dict]:
    """Resolve the file's SKUs before a crawl is started, so a missing
    product is spotted in seconds rather than after a long run."""
    from crawler_app.netutil import make_session
    from fasthouse.catalog import get_sku_index, resolve_dataframe

    session = make_session()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
        _, summary = resolve_dataframe(frame, get_sku_index(session))
        return summary
    except Exception as e:  # noqa: BLE001 - the crawl will report it properly
        log.warning("SKU pre-check failed: %s", e)
        return None
    finally:
        session.close()


def validate(df: pd.DataFrame, website: str, crawl_type: str):
    """Return (blocking problems, non-blocking warnings) for the upload."""
    problems: List[str] = []
    warnings: List[str] = []

    missing = [c for c in required_columns(website, crawl_type) if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): **{', '.join(missing)}**. Found: {', '.join(map(str, df.columns))}")
    if len(df) == 0:
        problems.append("The CSV has no data rows.")
    if problems:
        return problems, warnings

    if "URL" not in df.columns:
        warnings.append("No **URL** column - product URLs will be looked up from the Seller SKU.")
    else:
        urls = df["URL"].astype(str).str.strip()
        blank_urls = int((df["URL"].isna() | (urls == "") | (urls.str.lower() == "nan")).sum())
        if blank_urls and website == "Fasthouse":
            warnings.append(f"{blank_urls} row(s) have no URL - it will be looked up from the Seller SKU.")
        elif blank_urls:
            problems.append(f"{blank_urls} row(s) have an empty URL.")
        bad_urls = int((~urls.str.lower().str.startswith(("http://", "https://")) & (urls != "")).sum())
        if bad_urls:
            warnings.append(f"{bad_urls} URL(s) do not start with http(s):// and will fail to fetch.")

    dup = int(df["Seller SKU"].duplicated().sum())
    if dup:
        warnings.append(f"{dup} duplicate Seller SKU value(s); later rows overwrite earlier image files with the same name.")

    if crawl_type in IMAGE_CRAWL_TYPES:
        if "ASIN" not in df.columns:
            warnings.append("No **ASIN** column - image files will be named after the Seller SKU instead.")
        else:
            blank_asin = int((df["ASIN"].isna() | (df["ASIN"].astype(str).str.strip() == "")).sum())
            if blank_asin:
                warnings.append(f"{blank_asin} row(s) have a blank ASIN - those images will be named after the Seller SKU.")

    if website == "Fasthouse" and crawl_type == "Data":
        bad_bullets = int(pd.to_numeric(df["No of bullets"], errors="coerce").isna().sum())
        if bad_bullets:
            warnings.append(f"{bad_bullets} row(s) have a non-numeric 'No of bullets' value.")

    return problems, warnings


# -----------------------
# Sidebar
# -----------------------

section = st.sidebar.radio(
    "Section", ["Crawler", "SKU checker"], horizontal=True, label_visibility="collapsed"
)

if section == "SKU checker":
    skucheck_ui.render()
    st.stop()

with st.sidebar:
    st.header("Crawl settings")
    website = st.radio("Website", ["Fasthouse", "Seven"], horizontal=True)

    crawl_options = ["Data", "Images"]
    if website == "Fasthouse":
        crawl_options += ["A+ Images", "Size Charts", "SKU Lookup"]
    crawl_type = st.radio("Crawling type", crawl_options)

    units = "Inches"
    include_size_chart = False
    if website == "Fasthouse" and crawl_type == "Images":
        include_size_chart = st.checkbox(
            "Add size chart as PT05",
            value=True,
            help="Renders the product's size chart (2000x2000 PNG) into listing slot PT05. "
                 "The gallery shifts down around it (old PT05 becomes PT06, ...); when all nine "
                 "slots are needed, the last gallery image is dropped.",
        )
    if crawl_type == "Size Charts" or include_size_chart:
        units = st.radio(
            "Size chart units",
            UNIT_CHOICES,
            horizontal=True,
            help="Which measurement table(s) to put in the 2000x2000 PNG. "
                 "'Both' stacks an inches table and a centimetres table.",
        )

    with st.expander("Advanced", expanded=False):
        workers = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=MAX_WORKERS,
            value=3,
            help="Rows crawled at the same time. 3 is a safe default; raise it for speed, "
                 "lower it to 1 if the site starts returning 429 errors.",
        )
        single_zip = BIG_DOWNLOADS and st.checkbox(
            "Download as one ZIP file",
            value=True,
            help="Serves the whole batch as a single archive, streamed from disk. "
                 "Untick to split it into smaller parts, which is safer on a flaky connection.",
        )
        part_mb = 150
        if not single_zip:
            part_mb = st.slider(
                "Max ZIP part size (MB)",
                min_value=50,
                max_value=190,
                value=150,
                step=10,
                help="Large image sets are split into several ZIP files of at most this size "
                     "so downloads stay reliable and the server never holds a huge archive in memory.",
            )

    st.divider()
    with st.expander("Help", expanded=False):
        st.markdown(
            """
**Input columns**

- `Seller SKU` - required
- `No of bullets` - required for Fasthouse *Data* mode
- `URL` - optional for Fasthouse; looked up from the Seller SKU when missing
- `ASIN` - optional; names the image files

**SKU Lookup** turns a Seller SKU list into product URLs from Fasthouse's
own catalogue (~5,300 SKUs) without fetching any product page, so it is
quick - and doubles as a check for which SKUs are still listed and in
stock. Every other Fasthouse mode does this automatically when the file
has no `URL` column, so an Amazon export can be crawled directly.

**Size charts** come from each product's Kiwi Sizing chart (the "What's
My Size?" pop-up), rendered to a 2000 x 2000 PNG: heading, measurement
diagram, *How to Measure* as one sentence, and the table. In *Images*
mode, *Add size chart as PT05* puts it in that listing slot and shifts
the gallery down; the standalone *Size Charts* mode renders charts only.
            """
        )
    if not STATIC_SERVING:
        st.caption("Static file serving is off - ZIP parts are served through download buttons one part at a time.")
    elif not BIG_DOWNLOADS:
        st.caption("This Streamlit version caps served files at 200 MB, so large batches are still split into parts.")

    build = version_info()
    if build.get("commit"):
        st.caption(f"Build `{build['commit']}` · {build['date']}",
                   help=build.get("subject") or "The commit this app is running.")


# -----------------------
# Header
# -----------------------

banner = ARTIFACTS_DIR / f"{website}.jpg"
if banner.exists():
    st.image(Image.open(banner), width="stretch")

st.title(f"{website} Crawler")
st.caption(f"Mode: **{crawl_type}**")


# -----------------------
# 1. Upload & validate
# -----------------------

st.subheader("1 · Upload input CSV")
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    label_visibility="collapsed",
    help="One product per row. See the sidebar for the expected columns.",
)

df: Optional[pd.DataFrame] = None
problems: List[str] = []
if uploaded_file is not None:
    try:
        df = parse_csv(uploaded_file.getvalue())
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the CSV: {e}")

if df is not None:
    problems, warnings = validate(df, website, crawl_type)

    has_urls = "URL" in df.columns and df["URL"].astype(str).str.strip().replace("nan", "").any()
    unique_urls = int(df["URL"].astype(str).str.strip().str.lower().nunique()) if has_urls else None
    # Without URLs the products are not known until the SKU lookup runs, so
    # the estimate falls back to the row count - an upper bound, since rows
    # sharing a product are crawled once.
    basis = unique_urls if unique_urls is not None else len(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric(
        "Unique URLs", f"{unique_urls:,}" if unique_urls is not None else "–",
        help="Each URL is crawled once; rows sharing a URL reuse the result."
             if unique_urls is not None else
             "No URL column - product URLs are looked up from the Seller SKU when the crawl starts.",
    )
    m3.metric("Columns", len(df.columns))
    est = basis * SECONDS_PER_ROW.get(crawl_type, DEFAULT_SECONDS_PER_ROW) / max(1, workers)
    m4.metric("Estimated time", f"≈ {fmt_duration(est)}", help=f"Rough guess with {workers} worker(s).")

    for p in problems:
        st.error(p)
    for w in warnings:
        st.warning(w)

    if not problems and website == "Fasthouse" and not has_urls:
        with st.spinner("Checking the SKUs against the Fasthouse catalogue..."):
            check = sku_precheck(uploaded_file.getvalue())
        if check is None:
            st.info("Could not pre-check the SKUs; the crawl will report any that are missing.")
        elif check["missing"]:
            st.warning(
                f"**{check['missing']} of {check['rows']} SKU(s) are not in the Fasthouse "
                f"catalogue** and will be skipped: "
                + ", ".join(check["missing_skus"][:12])
                + ("…" if len(check["missing_skus"]) > 12 else "")
            )
            st.caption(f"{check['resolved']} resolved to {check['unique_urls']} product page(s).")
        else:
            st.success(f"All {check['resolved']} SKUs matched - {check['unique_urls']} product page(s) to crawl.")

    if not problems:
        st.success("File looks good.")

    with st.expander(f"Preview first {min(20, len(df))} rows", expanded=False):
        st.dataframe(df.head(20), width="stretch", hide_index=True)


# -----------------------
# 2. Run / cancel
# -----------------------

st.subheader("2 · Run")
job: Optional[Job] = REGISTRY.current
running = job is not None and job.is_active

run_col, cancel_col, _ = st.columns([1, 1, 3])
run_clicked = run_col.button(
    "🚀 Run scraper",
    type="primary",
    width="stretch",
    disabled=(df is None or bool(problems) or running),
    help="Upload a valid CSV first." if df is None or problems else None,
)
if running:
    if cancel_col.button("⛔ Cancel crawl", width="stretch"):
        job.request_cancel()
        st.rerun()

if run_clicked and df is not None:
    try:
        job = REGISTRY.start(
            website=website,
            crawl_type=crawl_type,
            df=df,
            crawler=CRAWLERS[website],
            workers=workers,
            assets_dir=ASSETS_DIR,
            outputs_dir=OUTPUTS_DIR,
            downloads_dir=DOWNLOADS_DIR,
            part_bytes=None if single_zip else part_mb * MB,
            options=(
                {"units": units} if crawl_type == "Size Charts"
                else {"units": units, "include_size_chart": True} if include_size_chart
                else {}
            ),
        )
        st.rerun()
    except RuntimeError as e:
        st.error(str(e))

if job is None:
    st.info("Upload a CSV and press **Run scraper**. The crawl runs in the background - you can "
            "close or refresh this tab and come back; progress and downloads will still be here.")
    st.stop()


# -----------------------
# 3. Live progress
# -----------------------

def render_progress(current: Job) -> None:
    st.progress(current.progress, text=f"{current.stage} - {current.done:,} / {current.total:,} rows")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Succeeded", f"{current.ok:,}")
    c2.metric("Failed", f"{current.failed:,}")
    c3.metric("Images saved", f"{current.images:,}")
    c4.metric("Workers", current.workers)
    c5.metric("Elapsed", fmt_duration(current.elapsed))
    c6.metric("ETA", fmt_duration(current.eta_seconds))
    if current.current:
        st.caption(f"Last processed: `{current.current}`")
    with st.expander("Live log", expanded=True):
        st.code("\n".join(list(current.log)[-20:]) or "Waiting for the first row...", language="text")


@st.fragment(run_every=1.0)
def live_progress() -> None:
    current = REGISTRY.current
    if current is None:
        return
    if not current.is_active:
        # Crawl finished while we were polling: rebuild the whole page so
        # the results section replaces this one.
        st.rerun()
    render_progress(current)


if running:
    st.subheader(f"3 · Crawling {job.website} / {job.crawl_type}")
    st.info("You can leave this page open, switch tabs, or refresh - the crawl keeps running on the server.")
    live_progress()
    st.stop()


# -----------------------
# 4. Results
# -----------------------

st.subheader(f"3 · Results - {job.website} / {job.crawl_type}")

if job.status == "done":
    st.success(f"Finished {job.total:,} rows in {fmt_duration(job.elapsed)}.")
    if st.session_state.get("celebrated_job") != job.id:
        st.session_state["celebrated_job"] = job.id
        st.balloons()
elif job.status == "cancelled":
    st.warning(f"Cancelled after {job.done:,} of {job.total:,} rows. Partial results are available below.")
elif job.status == "failed":
    st.error("The crawl stopped because of an unexpected error.")
    with st.expander("Error details", expanded=True):
        st.code(job.error or "unknown error", language="text")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{job.total:,}")
c2.metric("Succeeded", f"{job.ok:,}")
c3.metric("Failed", f"{job.failed:,}", delta=f"-{job.failed}" if job.failed else None, delta_color="inverse")
c4.metric("Images saved", f"{job.images:,}")
c5.metric("Elapsed", fmt_duration(job.elapsed))

out = job.result_df
if out is not None:
    flags = []
    if "Is Main Image Background White" in out.columns:
        non_white = int((out["Is Main Image Background White"] == False).sum())  # noqa: E712
        if non_white:
            flags.append(f"{non_white} main image(s) do not have a white background")
    if "Exceeded 9 images" in out.columns:
        flags.append(f"{int(out['Exceeded 9 images'].eq(True).sum())} product(s) had more than 9 images (extra ones skipped)")
    if "main_image_missing" in out.columns or "main_image_error" in out.columns:
        n = 0
        for col in ("main_image_missing", "main_image_error"):
            if col in out.columns:
                n += int(out[col].eq(True).sum())
        if n:
            flags.append(f"{n} product(s) had no main image on the page")
    if "Image Errors" in out.columns:
        n = int(out["Image Errors"].astype(str).str.strip().replace("nan", "").ne("").sum())
        if n:
            flags.append(f"{n} product(s) had one or more images that failed to download - see the 'Image Errors' column")
    if "Bullet match" in out.columns:
        mismatch = int((out["Bullet match"] == False).sum())  # noqa: E712
        if mismatch:
            flags.append(f"{mismatch} row(s) where the captured bullet count differs from 'No of bullets'")
    if "SKU Status" in out.columns:
        missing_skus = int((out["SKU Status"] == "Not found on fasthouse.com").sum())
        if missing_skus:
            flags.append(f"{missing_skus} SKU(s) are not in the Fasthouse catalogue - "
                         "they were skipped (see the 'SKU Status' column)")
        if "Available" in out.columns:
            sold_out = int((out["Available"] == False).sum())  # noqa: E712
            if sold_out:
                flags.append(f"{sold_out} SKU(s) are currently out of stock on fasthouse.com")
    if "Size Chart Status" in out.columns:
        missing = int(out["Size Chart Status"].astype(str).str.startswith("No size chart").sum())
        if missing:
            flags.append(f"{missing} product(s) have no size chart on the website (see 'Size Chart Status')")
        multi = int((pd.to_numeric(out.get("Size Chart Count"), errors="coerce").fillna(0) > 1).sum())
        if multi:
            flags.append(f"{multi} product(s) have more than one size chart (they take consecutive slots / files)")
        if "Gallery Images Dropped" in out.columns:
            dropped = pd.to_numeric(out["Gallery Images Dropped"], errors="coerce").fillna(0)
            n = int((dropped > 0).sum())
            if n:
                flags.append(
                    f"{n} product(s) had a full gallery - their last gallery image(s) were dropped "
                    "to make room for the size chart (see 'Gallery Images Dropped')"
                )
    for f in flags:
        st.warning(f)

if job.failures:
    with st.expander(f"Failed rows ({len(job.failures)})", expanded=False):
        failed_df = pd.DataFrame(job.failures)
        st.dataframe(failed_df, width="stretch", hide_index=True)

        retry_rows = None
        if job.source_df is not None and "Seller SKU" in job.source_df.columns:
            wanted = {str(f.get("Seller SKU")) for f in job.failures}
            retry_rows = job.source_df[job.source_df["Seller SKU"].astype(str).isin(wanted)]

        action, download = st.columns([1, 1])
        if retry_rows is not None and len(retry_rows):
            # Most failures are transient CDN stalls, so re-running just
            # these rows is usually all that is needed.
            if action.button(f"↻ Re-run these {len(retry_rows)} row(s)", type="primary",
                             width="stretch", disabled=job.is_active):
                try:
                    REGISTRY.start(
                        website=job.website, crawl_type=job.crawl_type, df=retry_rows,
                        crawler=CRAWLERS[job.website], workers=job.workers,
                        assets_dir=ASSETS_DIR, outputs_dir=OUTPUTS_DIR,
                        downloads_dir=DOWNLOADS_DIR, part_bytes=None if single_zip else part_mb * MB,
                        options=job.options,
                    )
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
            action.caption("Runs the same mode and options on the failed rows only.")
        download.download_button(
            "⬇ Download failed rows CSV",
            data=to_csv_bytes(failed_df),
            file_name=f"{job.website}__failed_rows.csv",
            mime="text/csv",
            on_click="ignore",
            width="stretch",
        )

if out is not None:
    issues, reasons = issue_report(out)
    n_issues = int(issues.sum())
    with st.expander(
        f"Output table - {n_issues:,} of {len(out):,} row(s) need a look" if n_issues
        else f"Output table ({len(out):,} rows, nothing flagged)",
        expanded=bool(n_issues),
    ):
        only_issues = st.toggle(
            "Show only rows with issues", value=bool(n_issues), disabled=not n_issues,
            help="Rows the crawl flagged: failures, bullet mismatches, dropped gallery "
                 "images, missing size charts, non-white backgrounds, unknown SKUs.",
        )
        table = out.copy()
        table.insert(0, "Issues", reasons)
        if only_issues and n_issues:
            table = table[issues]
        st.dataframe(table.head(200), width="stretch", hide_index=True)
        if len(table) > 200:
            st.caption(f"Showing the first 200 of {len(table):,} rows - the CSV has them all.")
        st.download_button(
            "⬇ Download flagged rows CSV", data=to_csv_bytes(out[issues].assign(Issues=reasons[issues])),
            file_name=f"{job.website}__flagged_rows.csv", mime="text/csv",
            on_click="ignore", disabled=not n_issues,
        )

    # ---- visual check of what was produced, without downloading the ZIP
    grouped = group_assets(ASSETS_DIR)
    if grouped:
        flagged_asins = set()
        for column in ("ASIN", "Seller SKU"):
            if column in out.columns:
                flagged_asins |= set(out.loc[issues, column].astype(str))
        with st.expander(f"Preview images ({len(grouped)} product(s))", expanded=False):
            names = sorted(grouped)
            default = next((i for i, n in enumerate(names) if n in flagged_asins), 0)
            picked = st.selectbox(
                "Product", names, index=default,
                format_func=lambda n: f"⚠ {n}" if n in flagged_asins else n,
                help="Files are shown in listing order: MAIN, PT01, PT02 ...",
            )
            files = grouped.get(picked, [])
            st.caption(f"{len(files)} file(s) for `{picked}`")
            columns = st.columns(min(4, max(1, len(files))))
            for position, (slot, path) in enumerate(files):
                with columns[position % len(columns)]:
                    try:
                        st.image(thumb(str(path), path.stat().st_mtime),
                                 caption=f"{(slot or '?').upper()} · {path.suffix.lstrip('.')}")
                    except Exception as e:  # noqa: BLE001 - one bad file must not break the page
                        st.caption(f"{slot}: could not preview ({e})")

st.markdown("#### Downloads")
dl_cols = st.columns([1, 1, 2])

if job.csv_path and job.csv_path.exists():
    dl_cols[0].download_button(
        "⬇ Download CSV",
        data=job.csv_path.read_bytes(),
        file_name=job.csv_path.name,
        mime="text/csv",
        type="primary" if not job.is_image_job else "secondary",
        on_click="ignore",
        width="stretch",
    )
if job.xlsx_path and job.xlsx_path.exists():
    dl_cols[1].download_button(
        "⬇ Download Excel",
        data=job.xlsx_path.read_bytes(),
        file_name=job.xlsx_path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click="ignore",
        width="stretch",
    )

if job.is_image_job:
    parts = [p for p in job.zip_parts if p.exists()]
    if not parts:
        st.info("No files were produced, so there is no ZIP to offer.")
    else:
        total_bytes = sum(p.stat().st_size for p in parts)
        if len(parts) > 1:
            st.info(
                f"The images are split into **{len(parts)} ZIP parts** ({fmt_size(total_bytes)} total). "
                "Each part is a normal ZIP - extract all of them into the same folder."
            )
        elif total_bytes > 200 * MB:
            st.caption(f"One archive of {fmt_size(total_bytes)}, streamed from disk - keep the tab open until it finishes.")
        if STATIC_SERVING:
            for i, p in enumerate(parts):
                label = f"⬇ Download images ZIP ({fmt_size(p.stat().st_size)})" if len(parts) == 1 \
                    else f"⬇ Part {i + 1} of {len(parts)} ({fmt_size(p.stat().st_size)})"
                st.link_button(label, f"app/static/downloads/{p.name}", type="primary")
        else:
            # Without static serving the bytes must go through Streamlit's
            # in-memory media store, so only one part is loaded at a time.
            if len(parts) == 1:
                chosen = parts[0]
            else:
                idx = st.selectbox(
                    "Choose ZIP part",
                    options=range(len(parts)),
                    format_func=lambda i: f"Part {i + 1} of {len(parts)} ({fmt_size(parts[i].stat().st_size)})",
                )
                chosen = parts[idx]
            st.download_button(
                f"⬇ Download {chosen.name} ({fmt_size(chosen.stat().st_size)})",
                data=chosen.read_bytes(),
                file_name=f"{job.website}{chosen.name[len(job.website) + 1 + len(job.id):]}",
                mime="application/zip",
                type="primary",
                on_click="ignore",
            )

st.divider()
if st.button("🔄 Start a new crawl"):
    REGISTRY.clear()
    st.rerun()
