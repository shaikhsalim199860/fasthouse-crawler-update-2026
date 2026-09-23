"""Fasthouse size charts.

Fasthouse renders its size charts client-side with the Kiwi Sizing Shopify
app, so they never appear in the page HTML. The product page does embed a
`KiwiSizing.data = {...}` block though, and Kiwi's public endpoint
`/kiwiSizing/api/getSizingChart` answers with the chart as structured data
(heading, measurement diagram, "How to Measure" text, the table in inches,
footer notes). This module fetches that data and renders it to a
2000x2000 PNG with Pillow - no browser required.
"""
import html as html_lib
import io
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from crawler_app.netutil import fetch_bytes, fetch_image

log = logging.getLogger(__name__)

KIWI_API = "https://app.kiwisizing.com/kiwiSizing/api/getSizingChart"
KIWI_KEYS = ["collections", "tags", "product", "vendor", "type", "title"]
_JS_STRING = r'"((?:[^"\\]|\\.)*)"'

UNIT_CHOICES = ("Inches", "Centimetres", "Both")
CANVAS = 2000
MARGIN = 90
FONT_PATH = Path(__file__).resolve().parent.parent / "crawler_app" / "fonts" / "Inter-Variable.ttf"


# ---------------------------------------------------------------- data model

@dataclass
class ChartTable:
    rows: List[List[dict]]           # Kiwi cells: {type, value, unitType}
    footer: str = ""


@dataclass
class ChartNote:
    """A line of chart prose that is not a measurement step, e.g. the red
    "All measurements are garment measurements" warning. The inline colour
    and bold from the source are kept so the note reads as it does on the
    site."""
    text: str
    color: Optional[Tuple[int, int, int]] = None
    bold: bool = False


@dataclass
class SizeChart:
    id: int
    name: str
    updated_at: str
    heading: str = ""
    diagram_url: str = ""
    how_to_measure: List[Tuple[str, str]] = field(default_factory=list)   # (label, instruction)
    intro_notes: List["ChartNote"] = field(default_factory=list)   # before the table
    notes: List["ChartNote"] = field(default_factory=list)         # after the table
    tables: List[ChartTable] = field(default_factory=list)
    decimals: int = 1

    @property
    def cache_key(self) -> str:
        return f"{self.id}:{self.updated_at}"

    @property
    def sizes(self) -> List[str]:
        out = []
        for t in self.tables:
            for row in t.rows[1:]:
                if row and row[0].get("type") == "header":
                    out.append(str(row[0].get("value", "")).strip())
        return out

    @property
    def measurements(self) -> List[str]:
        out = []
        for t in self.tables:
            if t.rows:
                out += [str(c.get("value", "")).strip() for c in t.rows[0][1:] if str(c.get("value", "")).strip()]
        return out

    @property
    def how_to_measure_sentence(self) -> str:
        """'Step 1: Chest - ...; Step 2: Sleeve - ...' in one sentence."""
        if not self.how_to_measure:
            return ""
        parts = []
        for i, (label, text) in enumerate(self.how_to_measure, start=1):
            body = f"{label} - {text}" if label and text else (label or text)
            parts.append(f"Step {i}: {body.rstrip('.')}")
        return "; ".join(parts) + "."


# ---------------------------------------------------------------- extraction

def parse_kiwi_data(page_html: str) -> Optional[Dict[str, str]]:
    """Pull the `KiwiSizing.data` object out of the page (it is a JS object
    literal, not JSON, so it is read key by key)."""
    start = page_html.find("KiwiSizing.data = {")
    if start < 0:
        return None
    block = page_html[start:start + 4000]
    data: Dict[str, str] = {}
    for key in KIWI_KEYS:
        m = re.search(r"\b" + key + r":\s*" + _JS_STRING, block)
        if m:
            data[key] = json.loads('"' + m.group(1) + '"')
    m = re.search(r'KiwiSizing\.shop\s*=\s*"([^"]+)"', page_html)
    if m:
        data["shop"] = m.group(1)
    if "product" not in data or "shop" not in data:
        return None
    return data


_COLOR_RE = re.compile(r"color:\s*rgb\((\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\)", re.I)
_LINE_SPLIT_RE = re.compile(r"<br\s*/?>|</p>|</h[1-6]>|</li>|</div>", re.I)


def _html_to_rich_lines(fragment: str) -> List[Tuple[str, Optional[Tuple[int, int, int]], bool]]:
    """Split a Kiwi rich-text block into (text, colour, bold) lines.

    Kiwi wraps each line in its own <span style="color: rgb(...)">, so the
    emphasis the site shows (such as the red garment-measurements warning)
    can be carried through to the rendered chart.
    """
    lines = []
    for chunk in _LINE_SPLIT_RE.split(fragment or ""):
        text = re.sub(r"<[^>]+>", "", chunk)
        text = html_lib.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        match = _COLOR_RE.search(chunk)
        color = tuple(min(255, int(g)) for g in match.groups()) if match else None
        lines.append((text, color, "<strong" in chunk.lower() or "<b>" in chunk.lower()))
    return lines


def _html_to_lines(fragment: str) -> List[str]:
    return [text for text, _, _ in _html_to_rich_lines(fragment)]


def _parse_how_to_measure(rich_lines) -> Tuple[List[Tuple[str, str]], List[ChartNote]]:
    """Split a "How to Measure" block into numbered steps and plain notes.

    The site writes each step as "Label: instruction"; anything else in the
    block is prose, such as "All measurements are garment measurements, NOT
    body measurements." - a warning, not a step, so it must not be numbered.
    """
    steps: List[Tuple[str, str]] = []
    notes: List[ChartNote] = []
    for text, color, bold in rich_lines:
        if re.match(r"^\s*how\s+to\s+measure\s*:?\s*$", text, re.I):
            continue
        text = re.sub(r"^\s*how\s+to\s+measure\s*:\s*", "", text, flags=re.I).strip()
        if not text:
            continue
        m = re.match(r"^([A-Za-z][A-Za-z /&()-]{0,40}?)\s*:\s*(.+)$", text)
        if m and len(m.group(1).split()) <= 3:
            steps.append((m.group(1).strip(), m.group(2).strip()))
        else:
            notes.append(ChartNote(text=text, color=color, bold=bold))
    return steps, notes


def parse_charts(payload: dict) -> List[SizeChart]:
    charts: List[SizeChart] = []
    settings = payload.get("settings") or {}
    decimals = int(settings.get("conversionSignificantDecimalPoints", 1) or 1)
    for sizing in payload.get("sizings") or []:
        if sizing.get("isEnabled") is False:
            continue
        chart = SizeChart(
            id=int(sizing.get("id", 0)),
            name=str(sizing.get("name", "")).strip(),
            updated_at=str(sizing.get("updatedAt", "")),
            decimals=decimals,
        )
        tables = sizing.get("tables") or {}

        def add_notes(items: List[ChartNote]) -> None:
            # Notes keep their position relative to the table, as on the site.
            (chart.notes if chart.tables else chart.intro_notes).extend(items)

        for block in (sizing.get("layout") or {}).get("data") or []:
            btype = block.get("type")
            if btype == 0:                              # rich text
                rich = _html_to_rich_lines(block.get("value") or "")
                if not rich:
                    continue
                joined = " ".join(text for text, _, _ in rich)
                if re.search(r"how\s+to\s+measure", joined, re.I):
                    steps, extra = _parse_how_to_measure(rich)
                    chart.how_to_measure += steps
                    add_notes(extra)
                elif not chart.heading and not chart.tables:
                    chart.heading = rich[0][0]
                    add_notes([ChartNote(t, c, b) for t, c, b in rich[1:]])
                else:
                    add_notes([ChartNote(t, c, b) for t, c, b in rich])
            elif btype == 6:                            # image
                url = ((block.get("data") or {}).get("url") or "").strip()
                if url and not chart.diagram_url:
                    chart.diagram_url = url
            elif btype == 1:                            # table reference
                table = tables.get(block.get("value"))
                if table and table.get("data") and str(table.get("hide", "")).lower() != "true":
                    footer = table.get("footer") or ""
                    chart.tables.append(ChartTable(rows=table["data"], footer=str(footer)))
        if not chart.heading:
            chart.heading = chart.name
        charts.append(chart)
    return charts


def _fingerprint(chart: SizeChart) -> tuple:
    """Identity of a chart by what it actually shows, ignoring its name."""
    tables = tuple(
        tuple(
            tuple((str(cell.get("value", "")), str(cell.get("unitType", ""))) for cell in row)
            for row in table.rows
        )
        for table in chart.tables
    )
    return (
        chart.heading.strip().lower(),
        chart.diagram_url,
        tuple(chart.how_to_measure),
        tables,
        tuple(n.text for n in chart.intro_notes),
        tuple(n.text for n in chart.notes),
    )


def dedupe_charts(charts: List[SizeChart]) -> List[SizeChart]:
    """Drop charts that render identically.

    Kiwi can return the same chart more than once when a shop has left a
    duplicate definition in place - Fasthouse has a
    "Gloves - SpeedStyle - Adult clone" matched by the same product tag as
    the original. Without this, a product would get the same size chart
    twice and the gallery would be pushed down a slot for nothing. Charts
    that genuinely differ (bikini top vs bottom) are all kept; among
    identical ones the lowest id wins, which is the original definition
    rather than a later copy.
    """
    winners: Dict[tuple, SizeChart] = {}
    order: List[tuple] = []
    for chart in charts:
        key = _fingerprint(chart)
        if key not in winners:
            winners[key] = chart
            order.append(key)
            continue
        kept = winners[key]
        loser, keeper = (kept, chart) if chart.id < kept.id else (chart, kept)
        winners[key] = keeper
        log.info("Dropping duplicate size chart %r (id %s); keeping %r (id %s)",
                 loser.name, loser.id, keeper.name, keeper.id)
    return [winners[key] for key in order]


def fetch_size_charts(session: requests.Session, kiwi_data: Dict[str, str]) -> List[SizeChart]:
    raw = fetch_bytes(session, KIWI_API + "?" + requests.compat.urlencode(kiwi_data))
    return dedupe_charts(parse_charts(json.loads(raw)))


# ---------------------------------------------------------------- units

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def convert_value(value: str, unit_type: str, target: str, decimals: int) -> str:
    """Convert every number in a cell ("38", "38-40", "25.75") in->cm or
    cm->in. Non-measurement cells are returned untouched."""
    value = str(value)
    if unit_type not in ("in", "cm") or unit_type == target:
        return value
    factor = 2.54 if target == "cm" else 1 / 2.54

    def repl(m):
        num = float(m.group(0)) * factor
        text = f"{num:.{decimals}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    return _NUMBER.sub(repl, value)


def table_to_strings(table: ChartTable, target: str, decimals: int) -> List[List[str]]:
    out = []
    for row in table.rows:
        out.append([convert_value(c.get("value", ""), c.get("unitType", "string"), target, decimals) for c in row])
    return out


# ---------------------------------------------------------------- rendering

class _Fonts:
    _lock = threading.Lock()
    _cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}

    @classmethod
    def get(cls, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        with cls._lock:
            font = cls._cache.get(key)
            if font is None:
                try:
                    font = ImageFont.truetype(str(FONT_PATH), size)
                    try:
                        font.set_variation_by_name(b"Bold" if bold else b"Regular")
                    except Exception:  # noqa: BLE001 - older FreeType without name tables
                        axes = font.get_variation_axes()
                        font.set_variation_by_axes([a.get("default", 0) for a in axes[:-1]] + [700 if bold else 400])
                except Exception:  # noqa: BLE001 - bundled font missing/unsupported
                    font = ImageFont.load_default(size=size)
                cls._cache[key] = font
            return font


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, first_width: Optional[int] = None) -> List[str]:
    """Greedy word wrap; `first_width` lets the first line be narrower
    (used when a bold label is drawn in front of it)."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        limit = first_width if (first_width is not None and not lines) else max_width
        trial = f"{current} {word}".strip()
        if _text_width(draw, trial, font) <= limit or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_font(draw: ImageDraw.ImageDraw, text: str, size: int, bold: bool, max_width: int, min_size: int = 16):
    """Largest font at or below `size` whose rendering of `text` fits."""
    while size > min_size and _text_width(draw, text, _Fonts.get(size, bold)) > max_width:
        size -= 2
    return _Fonts.get(size, bold)


class ChartRenderer:
    """Lays the chart out on a 2000x2000 white canvas.

    The layout is computed at a scale factor; if the content does not fit,
    the scale is reduced and everything (fonts, image, table) shrinks
    together so long charts stay legible instead of being cropped.
    """

    def __init__(self, chart: SizeChart, units: str, diagram: Optional[Image.Image]):
        self.chart = chart
        self.units = units
        self.diagram = diagram

    def render(self, size: int = CANVAS) -> Image.Image:
        for scale in (1.0, 0.9, 0.8, 0.7, 0.62, 0.55, 0.48, 0.42, 0.36, 0.3):
            img, fits, bottom = self._render_at(scale, size)
            if fits:
                break
        # Centre short charts vertically instead of leaving the bottom empty.
        slack = size - MARGIN - bottom
        if fits and slack > 0:
            shifted = Image.new("RGB", (size, size), "white")
            shifted.paste(img.crop((0, 0, size, bottom + MARGIN)), (0, slack // 2))
            return shifted
        return img

    # Every dimension below is expressed for scale 1.0 and multiplied.
    def _render_at(self, scale: float, size: int) -> Tuple[Image.Image, bool, int]:
        s = lambda v: max(1, int(round(v * scale)))  # noqa: E731
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        content_w = size - 2 * MARGIN
        y = MARGIN
        fits = True

        # Heading
        font_h = _Fonts.get(s(72), bold=True)
        for line in _wrap(draw, self.chart.heading.upper(), font_h, content_w):
            w = _text_width(draw, line, font_h)
            draw.text(((size - w) // 2, y), line, font=font_h, fill="black")
            y += s(86)
        y += s(20)

        # Diagram
        if self.diagram is not None:
            max_w, max_h = int(content_w * 0.55 * scale), s(560)
            dw, dh = self.diagram.size
            ratio = min(max_w / dw, max_h / dh)
            dsz = (max(1, int(dw * ratio)), max(1, int(dh * ratio)))
            resized = self.diagram.resize(dsz, Image.LANCZOS)
            img.paste(resized, ((size - dsz[0]) // 2, y))
            y += dsz[1] + s(40)

        # How to measure - one sentence
        sentence = self.chart.how_to_measure_sentence
        if sentence:
            font_b = _Fonts.get(s(34), bold=True)
            font_t = _Fonts.get(s(34))
            label = "How to Measure: "
            label_w = _text_width(draw, label, font_b)
            draw.text((MARGIN, y), label, font=font_b, fill="black")
            lines = _wrap(draw, sentence, font_t, content_w, first_width=content_w - label_w)
            for i, line in enumerate(lines):
                draw.text((MARGIN + label_w if i == 0 else MARGIN, y), line, font=font_t, fill="black")
                y += s(46)
            y += s(30)

        # Notes that belong above the table (the red garment-measurements
        # warning), left aligned and in their original colour.
        for note in self.chart.intro_notes:
            font_i = _Fonts.get(s(30), bold=note.bold)
            for line in _wrap(draw, note.text, font_i, content_w):
                draw.text((MARGIN, y), line, font=font_i, fill=note.color or (40, 40, 40))
                y += s(40)
            y += s(14)

        # Tables (one per unit system requested)
        targets = {"Inches": ["in"], "Centimetres": ["cm"], "Both": ["in", "cm"]}[self.units]
        for table in self.chart.tables:
            for target in targets:
                y = self._draw_table(draw, table, target, y, size, scale)
                if y > size - MARGIN:
                    fits = False
                y += s(40)

        # Notes / footer
        for note in self.chart.notes:
            font_n = _Fonts.get(s(28), bold=False)
            for line in _wrap(draw, note.text, font_n, content_w):
                w = _text_width(draw, line, font_n)
                draw.text(((size - w) // 2, y), line, font=font_n, fill=note.color or (60, 60, 60))
                y += s(38)
        if y > size - MARGIN // 2:
            fits = False
        return img, fits, y

    def _draw_table(self, draw, table: ChartTable, target: str, y: int, size: int, scale: float) -> int:
        s = lambda v: max(1, int(round(v * scale)))  # noqa: E731
        rows = table_to_strings(table, target, self.chart.decimals)
        if not rows:
            return y
        content_w = size - 2 * MARGIN
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]

        font_head = _Fonts.get(s(36), bold=True)
        font_cell = _Fonts.get(s(36))
        unit_label = "INCHES" if target == "in" else "CENTIMETRES"
        font_u = _Fonts.get(s(30), bold=True)
        draw.text((MARGIN, y), unit_label, font=font_u, fill=(90, 90, 90))
        y += s(44)

        row_h = s(64)
        pad = s(16)
        # Column widths follow the widest text in each column (with a floor)
        # so long headers such as "WAIST (Extended)" are not cut off.
        natural = []
        for c_i in range(n_cols):
            widest = max(
                _text_width(draw, str(r[c_i]), font_head if (r_i == 0 or c_i == 0) else font_cell)
                for r_i, r in enumerate(rows)
            )
            natural.append(max(widest + 2 * pad, s(120)))
        # Blend equal shares with content-proportional shares so a long
        # header widens its column without swallowing the table.
        total = sum(natural)
        equal = content_w / n_cols
        col_ws = [max(s(80), int(0.5 * equal + 0.5 * w * content_w / total)) for w in natural]
        col_x = [MARGIN]
        for w in col_ws[:-1]:
            col_x.append(col_x[-1] + w)
        table_right = col_x[-1] + col_ws[-1]

        for r_i, row in enumerate(rows):
            is_header = r_i == 0
            fill = (34, 39, 45) if is_header else ((245, 245, 245) if r_i % 2 == 0 else "white")
            draw.rectangle([MARGIN, y, table_right, y + row_h], fill=fill, outline=(200, 200, 200))
            for c_i, cell in enumerate(row):
                bold = is_header or c_i == 0
                color = "white" if is_header else "black"
                text = str(cell)
                font = _fit_font(draw, text, s(36), bold, col_ws[c_i] - pad, min_size=max(12, s(18)))
                w = _text_width(draw, text, font)
                cx = col_x[c_i] + (col_ws[c_i] - w) // 2
                draw.text((cx, y + (row_h - font.size) // 2 - s(4)), text, font=font, fill=color)
                draw.line([col_x[c_i], y, col_x[c_i], y + row_h], fill=(200, 200, 200))
            y += row_h
        if table.footer:
            font_f = _Fonts.get(s(26))
            for line in _wrap(draw, _html_to_lines(table.footer) and " ".join(_html_to_lines(table.footer)) or "", font_f, content_w):
                draw.text((MARGIN, y + s(8)), line, font=font_f, fill=(80, 80, 80))
                y += s(34)
        return y


class SizeChartCache:
    """Charts are shared by many products (every Grindhouse men's jersey
    maps to one Kiwi chart), so each (chart, units) is rendered once and
    the PNG bytes are reused for every ASIN."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._png: Dict[str, bytes] = {}
        self._diagrams: Dict[str, Optional[Image.Image]] = {}

    def _diagram(self, session: requests.Session, url: str) -> Optional[Image.Image]:
        if not url:
            return None
        with self._lock:
            if url in self._diagrams:
                return self._diagrams[url]
        try:
            img = fetch_image(session, url)
            if img.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", img.size, "white")
                bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
        except Exception as e:  # noqa: BLE001 - chart still renders without the diagram
            log.warning("Diagram download failed for %s: %s", url, e)
            img = None
        with self._lock:
            self._diagrams[url] = img
        return img

    def png(self, session: requests.Session, chart: SizeChart, units: str) -> bytes:
        key = f"{chart.cache_key}:{units}"
        with self._lock:
            cached = self._png.get(key)
        if cached is not None:
            return cached
        diagram = self._diagram(session, chart.diagram_url)
        image = ChartRenderer(chart, units, diagram).render()
        buf = io.BytesIO()
        image.save(buf, "PNG", optimize=True)
        data = buf.getvalue()
        with self._lock:
            self._png[key] = data
        return data
