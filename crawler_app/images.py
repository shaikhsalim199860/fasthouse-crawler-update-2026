import logging
import os
from typing import Tuple

from PIL import Image

log = logging.getLogger(__name__)

# Amazon accepts 1000px+ for zoom and recommends 1600px on the longest
# side. Fasthouse's masters are 1200px, so anything above that is
# interpolation either way - landing on the recommended number costs
# nothing real and keeps listing audits quiet.
SQUARE_SIZE = 1600


def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def ensure_min_size(img: Image.Image, min_size: Tuple[int, int] = (550, 550)) -> Image.Image:
    """Upscale tiny images (any side <= 500px) to the minimum accepted size."""
    if any(sz <= 500 for sz in img.size):
        return img.resize(min_size)
    return img


def center_square(img: Image.Image, size: int = SQUARE_SIZE) -> Image.Image:
    """Center-crop to a square and resize to `size` x `size` in one pass."""
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def save_jpeg(img: Image.Image, path: str, quality: int = 95, subsampling: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_rgb(img).save(path, "JPEG", quality=quality, subsampling=subsampling)


def is_background_white(img: Image.Image, sample: int = 96) -> bool:
    """True when the most common colour is pure white.

    The previous implementation called `getcolors()` on every pixel of a
    1500x1500 image, allocating a tuple per unique colour (hundreds of
    thousands per photo). Sampling the image down with NEAREST keeps the
    real pixel values (no blending), so pure white stays (255, 255, 255)
    while the check costs ~9k pixels instead of ~2.25M.
    """
    small = to_rgb(img).resize((sample, sample), Image.NEAREST)
    colors = small.getcolors(sample * sample) or []
    if not colors:
        return False
    return max(colors)[1] == (255, 255, 255)
