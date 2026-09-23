"""Let Streamlit serve one large archive instead of several parts.

Streamlit's app-static route refuses any file above
`MAX_APP_STATIC_FILE_SIZE` (200 MB by default), which is why big image
batches are otherwise split into parts. The constant is read per request,
so raising it lets the server stream a single multi-hundred-MB ZIP
straight from disk - it is never buffered in memory, unlike
`st.download_button`.

The constant lives in different modules depending on the Streamlit
version, and is imported by value into the routing module, so every known
location is patched. If a future version moves it, nothing is patched,
`raise_static_file_limit` returns 0 and the caller keeps splitting.
"""
import importlib
import logging

log = logging.getLogger(__name__)

ATTRIBUTE = "MAX_APP_STATIC_FILE_SIZE"

# Newest layout first. The routes module does `from ... import
# MAX_APP_STATIC_FILE_SIZE`, so it holds its own copy that must be set too.
CANDIDATE_MODULES = (
    "streamlit.web.server.starlette.starlette_server_config",
    "streamlit.web.server.starlette.starlette_routes",
    "streamlit.web.server.app_static_file_handler",
    "streamlit.web.server.server_util",
    "streamlit.web.server.server",
)


def raise_static_file_limit(limit_bytes: int) -> int:
    """Raise the static-file size limit wherever it is defined.

    Returns the number of modules patched; 0 means the limit could not be
    found and archives should still be split.
    """
    patched = 0
    for name in CANDIDATE_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - module absent in this version
            continue
        if not hasattr(module, ATTRIBUTE):
            continue
        try:
            if getattr(module, ATTRIBUTE) < limit_bytes:
                setattr(module, ATTRIBUTE, limit_bytes)
            patched += 1
        except Exception as e:  # noqa: BLE001 - read-only / unexpected type
            log.warning("Could not raise %s in %s: %s", ATTRIBUTE, name, e)

    if patched:
        log.info(
            "Static file limit raised to %.0f MB in %d module(s)",
            limit_bytes / (1024 * 1024), patched,
        )
    else:
        log.warning("Could not raise %s - large archives will stay split", ATTRIBUTE)
    return patched
