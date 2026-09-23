# Streamlit Fasthouse + Seven Scraper

[Streamlit deployment](https://piyush-daga-streamlit-fasthouse-main-qcrsjr.streamlit.app/) for the Fasthouse and Seven scrapers.

## Usage

1. Pick the website and crawl type in the sidebar (Data / Images / A+ Images / Size Charts).
2. Upload a CSV with `Seller SKU`, `URL` and (Fasthouse Data mode only) `No of bullets`.
   An optional `ASIN` column is used to name image files.
3. Press **Run scraper**. The crawl runs in a background thread on the server:
   you can refresh or close the tab and the progress/downloads are still there
   when you come back.
4. Download the CSV/Excel and, for image modes, the ZIP part(s).

Each product URL is fetched once: size-variant rows that share a URL get the
result (and their own ASIN-named copies of the files) without another
request. Rows that fail or lose an image to a transient stall are retried
once, sequentially, after the main pass; anything still failing is listed
in the *Failed rows* table and the `Crawl Error` / `Image Errors` columns.

## Size chart as PT05 (Fasthouse Images mode)

With *Add size chart as PT05* ticked (default), Images mode also renders the
product's size chart and saves it as `ASIN.pt05.png` next to the gallery JPGs.

The chart claims PT05 and the **gallery shifts down around it**: what the site
shows 6th becomes PT06, 7th becomes PT07 and so on. Amazon has nine slots
(MAIN + PT01-PT08), and Fasthouse galleries often fill all nine, so the
trailing gallery image(s) are dropped to make room - the chart is never pushed
to the end of the listing. Products with two charts (bikini top/bottom) take
PT05 + PT06.

Per-row columns: `Size Chart Slot` (e.g. `PT05`), `Size Chart Status`
(Found / No size chart) and `Gallery Images Dropped`.

## Size Charts mode (Fasthouse)

Fasthouse's size charts come from the Kiwi Sizing app and are injected by
JavaScript, so they are not in the page HTML. The crawler reads the
`KiwiSizing.data` block from each product page, calls Kiwi's
`getSizingChart` API and renders the result with Pillow to a
**2000 x 2000 PNG** named `ASIN.SIZE-CHART.png` (`-2`, `-3` for products
with several charts, e.g. bikini top/bottom): heading, measurement diagram,
"How to Measure" as one step-by-step sentence, the table in inches, cm or
both, and the notes. Only `Label: instruction` lines become numbered steps -
other prose (such as the red "All measurements are garment measurements"
warning) is kept as a note, in its position and colour from the site. Charts shared by many products are rendered once.
The CSV/Excel output records the status per row (Found / No size chart),
chart name, sizes, measurements, the How-to-Measure sentence, the diagram
URL and the theme's fit-guide image URL. Fonts: bundled Inter (OFL) in
`crawler_app/fonts/`.

## SKU checker (AWS)

The **SKU checker** section (sidebar switch) maintains the plain-text SKU
list that the `FH_New_SKU_Checker` Lambda reads from S3, and can run the
function on demand. The EventBridge schedule (Mon/Fri) is untouched.

- Shows the current list (`inputs/fasthouse/sku-available/SKUS.txt`): SKU
  count, last-modified time and contents.
- Updates it from the last crawl result, an uploaded CSV/TXT, or pasted
  text - either **adding** new SKUs or **replacing** the list. A diff of
  added/removed SKUs is shown before anything is written, and the previous
  version is copied to `backup_prefix` first. That prefix deliberately sits
  outside the folder the Lambda reads, so an old copy can never be mistaken
  for input.
- **Run SKU checker** invokes the Lambda and shows its response.
- **Schedule** shows when AWS runs the checker by itself, and lets the days,
  time and (for EventBridge Scheduler) timezone be changed, or the schedule
  paused and resumed. Both EventBridge **Rules** and **Scheduler** are
  supported and auto-detected from the function ARN. Saving requires an
  explicit confirmation, and an expression the app does not model (rates,
  ranges, step values) is shown read-only rather than rewritten.
  This needs the extra IAM permissions listed in the app.
- Lists the newest files under `results_prefix` so a run's output can be
  opened or downloaded.

Configure it with an `[aws]` block in `.streamlit/secrets.toml` (see
`.streamlit/secrets.toml.example`; the file is git-ignored) and the same
block in *Settings → Secrets* on Streamlit Cloud. Leave `sku_key` blank to
browse for the list in the app, then paste the key into secrets.

The IAM user needs `s3:GetObject`/`s3:PutObject` on the SKU key and its
`backups/` folder, `s3:ListBucket` on the bucket, and
`lambda:InvokeFunction` on the function.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Streamlit UI (upload → validate → run → live progress → results/downloads) |
| `crawler_app/jobs.py` | Background job runner + single-job registry |
| `crawler_app/runner.py` | Row loop shared by both crawlers: worker pool, cancel, per-row error isolation |
| `crawler_app/archive.py` | Splits `assets/` into ZIP parts of bounded size (JPEGs stored, CSV deflated) |
| `crawler_app/netutil.py` / `images.py` | Pooled HTTP session with timeouts + retries; image helpers |
| `fasthouse/scrape.py`, `seven/scrape.py` | Site-specific parsing |
| `fasthouse/sizechart.py` | Kiwi Sizing extraction + 2000x2000 size chart renderer |
| `crawler_app/awsio.py` / `skucheck_ui.py` | S3 SKU list + Lambda invoke, and the SKU checker UI |
| `crawler_app/schedule.py` | Reads/edits the EventBridge schedule that fires the checker |

## Downloads: one ZIP or split parts

A 1500×1500 JPEG is ~0.4 MB and a product has up to 9 of them, so a 135-row
batch produces a ~560 MB archive. `st.download_button` has to hold its payload
in RAM, which is what used to crash the app past ~150 ASINs. Archives are
written to `static/downloads/` and served straight from disk
(`server.enableStaticServing` in `.streamlit/config.toml`), so nothing is
buffered in memory.

Streamlit's static route refuses files above 200 MB, so `crawler_app/bigfiles.py`
raises that cap at startup and the batch is delivered as **one ZIP** by default
(*Download as one ZIP file* in the sidebar). Untick it to split the archive into
parts of at most 190 MB instead - useful on a flaky connection. If the cap
cannot be raised (a future Streamlit version moving the constant) or static
serving is off, the app falls back to split parts automatically.

## Running locally

```bash
pip install -r requirements.txt
streamlit run main.py
```
