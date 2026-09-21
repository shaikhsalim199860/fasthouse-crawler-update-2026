# Streamlit Fasthouse + Seven Scraper

[Streamlit deployment](https://piyush-daga-streamlit-fasthouse-main-qcrsjr.streamlit.app/) for the Fasthouse and Seven scrapers.

## Usage

1. Pick the website and crawl type in the sidebar (Data / Images / A+ Images).
2. Upload a CSV with `Seller SKU`, `URL` and (Fasthouse Data mode only) `No of bullets`.
   An optional `ASIN` column is used to name image files.
3. Press **Run scraper**. The crawl runs in a background thread on the server:
   you can refresh or close the tab and the progress/downloads are still there
   when you come back.
4. Download the CSV and, for image modes, the ZIP part(s).

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Streamlit UI (upload → validate → run → live progress → results/downloads) |
| `crawler_app/jobs.py` | Background job runner + single-job registry |
| `crawler_app/runner.py` | Row loop shared by both crawlers: worker pool, cancel, per-row error isolation |
| `crawler_app/archive.py` | Splits `assets/` into ZIP parts of bounded size (JPEGs stored, CSV deflated) |
| `crawler_app/netutil.py` / `images.py` | Pooled HTTP session with timeouts + retries; image helpers |
| `fasthouse/scrape.py`, `seven/scrape.py` | Site-specific parsing |

## Why the split ZIPs / static serving

A 1500×1500 JPEG is ~0.4 MB and a product has up to 9 of them, so 150+ ASINs
produce a 300–500 MB archive. `st.download_button` has to hold its payload in
RAM, which is what used to crash the app past ~150 ASINs. Archives are now
written to `static/downloads/` and served straight from disk
(`server.enableStaticServing` in `.streamlit/config.toml`), split into parts of
at most 190 MB (Streamlit's static-file limit is 200 MB). If static serving is
disabled the UI falls back to download buttons, one part at a time.

## Running locally

```bash
pip install -r requirements.txt
streamlit run main.py
```
