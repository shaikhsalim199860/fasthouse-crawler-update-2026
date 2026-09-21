import io
import logging

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# (connect timeout, read timeout) in seconds. A hung CDN request used to
# block the whole crawl forever because no timeout was set at all. The
# image CDN occasionally stalls a request for minutes while a retry on a
# fresh connection succeeds in under a second, so the read timeout is kept
# short and the retry policy below does the recovering.
DEFAULT_TIMEOUT = (10, 30)


def make_session(pool_size: int = 8) -> requests.Session:
    """A pooled session with automatic retries on transient failures.

    Connection pooling avoids a fresh TLS handshake for every one of the
    ~7 requests made per ASIN; the retry policy covers 429/5xx and
    connection/read errors for both page and image downloads.
    """
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    return session


def fetch_bytes(session: requests.Session, url: str, timeout=DEFAULT_TIMEOUT) -> bytes:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def fetch_image(session: requests.Session, url: str, timeout=DEFAULT_TIMEOUT) -> Image.Image:
    """Download an image fully into memory and decode it.

    `Image.open(resp.raw)` (the old approach) breaks on gzip-encoded
    responses and keeps the socket open while PIL lazily decodes; reading
    the bytes first and calling `.load()` releases the connection back to
    the pool immediately.
    """
    data = fetch_bytes(session, url, timeout)
    img = Image.open(io.BytesIO(data))
    img.load()
    return img
