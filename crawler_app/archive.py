import logging
import zipfile
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

MB = 1024 * 1024

# Streamlit's static file server refuses files above 200 MB unless the
# limit is raised (see crawler_app.bigfiles), and holding a
# multi-hundred-MB zip in RAM for st.download_button is what used to crash
# the app. `max_part_bytes=None` produces one archive of any size, which is
# safe only when it is served from disk.
DEFAULT_PART_BYTES = 150 * MB
MAX_PART_BYTES = 190 * MB


def build_zip_parts(
    src_dir: Path,
    out_dir: Path,
    base_name: str,
    max_part_bytes: Optional[int] = DEFAULT_PART_BYTES,
) -> List[Path]:
    """Zip every file under `src_dir` into one or more archives of at most
    ~`max_part_bytes` each (a single file larger than the limit still gets
    its own part); `None` means no limit, i.e. exactly one archive. JPEGs
    are stored uncompressed (deflate gains nothing and costs CPU); CSV/text
    files are deflated. Returns the part paths in order.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = float("inf") if max_part_bytes is None else max(1 * MB, min(int(max_part_bytes), MAX_PART_BYTES))

    files = sorted(p for p in src_dir.rglob("*") if p.is_file())
    # Small companion files (the image CSV) go first so they land in part 1.
    files.sort(key=lambda p: (p.suffix.lower() not in {".csv", ".xlsx", ".txt"}, p.name))

    # First pass: decide the grouping so the part count is known up front.
    groups: List[List[Path]] = []
    current: List[Path] = []
    current_size = 0
    for f in files:
        size = f.stat().st_size
        if current and current_size + size > limit:
            groups.append(current)
            current, current_size = [], 0
        current.append(f)
        current_size += size
    if current:
        groups.append(current)
    if not groups:
        groups = [[]]

    total = len(groups)
    parts: List[Path] = []
    for i, group in enumerate(groups, start=1):
        name = f"{base_name}.zip" if total == 1 else f"{base_name}.part{i:02d}of{total:02d}.zip"
        path = out_dir / name
        with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
            for f in group:
                compress = zipfile.ZIP_DEFLATED if f.suffix.lower() in {".csv", ".xlsx", ".txt"} else zipfile.ZIP_STORED
                zf.write(f, arcname=str(f.relative_to(src_dir)), compress_type=compress)
        parts.append(path)
        log.info("Wrote %s (%d files, %.1f MB)", path.name, len(group), path.stat().st_size / MB)

    return parts


def remove_matching(out_dir: Path, base_name: str) -> None:
    """Delete previous archives for `base_name` so old parts never mix
    with a new run's parts."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    for p in out_dir.glob(f"{base_name}*.zip"):
        try:
            p.unlink()
        except OSError as e:
            log.warning("Could not delete %s: %s", p, e)
