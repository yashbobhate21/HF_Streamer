# Run:      python -m streamlit run stream.py
# Install:  pip install "streamlit>=1.50" datasets huggingface_hub pillow
# Optional: add `--theme.base dark` to the run command for a fully dark Streamlit chrome.
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import base64
import html
import inspect
import io
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import streamlit as st
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)
from PIL import Image

Image.MAX_IMAGE_PIXELS = 400_000_000  # allow large scans; still guards against decompression bombs

MAX_RECORDS = 500  # per split; records are kept in session memory so UI reruns never re-stream
CORE = ("image_name", "image", "text")
REPO_RE = re.compile(r"^[\w.-]{1,96}(/[\w.-]{1,96})?$")
SPLIT_ORDER = {"train": 0, "validation": 1, "val": 1, "dev": 1, "test": 2}
WEB_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
MAX_BYTES, MAX_SIDE = 6_000_000, 1600  # bigger or non-web images are downscaled for display
THUMB = (360, 160)  # table-cell thumbnails: small, so the table stays light even with hundreds of rows
CELL_CHARS = 600  # text shown in a table cell is cut here (the cell itself clamps to 4 lines)
WORKERS = min(8, (os.cpu_count() or 4) * 2)
EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
NULL = "<span class='nul'>null</span>"

CSS = """<style>
:root{color-scheme:dark}
.stApp,[data-testid=stHeader]{background:#0e1218}
[data-testid=stSidebar]{background:#131922;border-right:1px solid #222b38}
.stApp,.stApp p,.stApp label,.stApp h1,.stApp h2,.stApp h3{color:#e7e5df}
.stApp input{background:#1a212c!important;color:#e7e5df!important}
[data-testid=stForm]{border-color:#222b38}
[data-testid=stVerticalBlockBorderWrapper]{background:#141a23;border-color:#252f3d!important;border-radius:12px;
  transition:border-color .15s}
[data-testid=stVerticalBlockBorderWrapper]:hover{border-color:#d9b26f88!important}
[data-testid^=stBaseButton-primary]{background:#d9b26f;border:0}
[data-testid^=stBaseButton-primary] p{color:#14181f!important;font-weight:600}
[data-testid^=stBaseButton-secondary]{background:#1a212c;border:1px solid #2a3544}
.stProgress>div>div>div>div{background-color:#d9b26f}
.hero h1{font:600 2rem/1.2 Georgia,"Iowan Old Style",serif;margin:0 0 .25rem}
.hero p{color:#9aa4b2!important;margin:0 0 1.25rem;max-width:60ch}
.name{font-weight:600;font-size:.95rem;color:#d9b26f;margin-bottom:.5rem;word-break:break-all}
.txt{margin:.65rem 0;padding:.65rem .8rem;background:#0e1218;border-radius:8px;max-height:9.5rem;
  overflow:auto;white-space:pre-wrap;font:.95rem/1.6 Georgia,"Iowan Old Style",serif;color:#cfd3da}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.25rem 0 1rem}
.stApp .chip{background:#141a23;border:1px solid #2a3544;border-radius:999px;padding:.25rem .8rem;
  font-size:.82rem;color:#c3cad4}
.chip b{color:#d9b26f;margin-right:.4rem;font-weight:600}
.ds-panel{margin:1.1rem 0 1.35rem;padding:1.1rem 1.2rem 1rem;border:1px solid #3a3429;
  border-radius:16px;background:linear-gradient(135deg,#191d24 0%,#141a23 58%,#211d17 100%);
  box-shadow:0 10px 28px #0003}
.ds-heading{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem}
.ds-kicker{font:600 .66rem ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.16em;color:#d9b26f}
.ds-title{font:600 1.35rem Georgia,"Iowan Old Style",serif;color:#f0e8d9}
.ds-status{margin-left:auto;border-radius:999px;padding:.28rem .65rem;font-size:.7rem;letter-spacing:.04em}
.ds-status.yes{color:#a8d5b2;background:#183126;border:1px solid #315b43}
.ds-status.no{color:#e8c27f;background:#352b1b;border:1px solid #6a512b}
.ds-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem}
.ds-metric{min-width:0;padding:.8rem .9rem;border:1px solid #2b3441;border-radius:11px;
  background:#10161f;position:relative;overflow:hidden}
.ds-metric:before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:#b58e52}
.ds-label{font:600 .65rem/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.07em;color:#929eae}
.ds-value{font:600 1.55rem/1.2 Georgia,"Iowan Old Style",serif;color:#f0d39e;margin-top:.35rem}
.ds-note{margin-top:.85rem;color:#8e9aaa;font-size:.76rem}
.ds-error{margin-top:.35rem;color:#e8c27f;font-size:.76rem;overflow-wrap:anywhere}
@media(max-width:700px){.ds-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.ds-value{font-size:1.3rem}}


/* ---- Hugging Face style dataset table ---- */
[data-testid=stMarkdownContainer],[data-testid=stElementContainer]{min-width:0;max-width:100%}
.stApp .tbl-wrap{overflow:auto;max-height:72vh;max-width:100%;border:1px solid #252f3d;border-radius:12px;
  background:#141a23;scrollbar-color:#34415a #141a23;overscroll-behavior:contain}
.stApp table.hf{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;margin:0;
  color:#cfd3da;font-size:.85rem;border:0}
.stApp table.hf th,.stApp table.hf td{padding:.5rem .8rem;border:0;border-bottom:1px solid #1f2733;
  border-right:1px solid #1f2733;text-align:left;vertical-align:middle;min-width:150px;max-width:400px;
  background:#141a23;color:#cfd3da}
.stApp table.hf th{position:sticky;top:0;z-index:2;background:#171e29;border-bottom:1px solid #2a3544;
  white-space:nowrap;font-weight:400}
.stApp table.hf th.idx,.stApp table.hf td.idx{position:sticky;left:0;z-index:1;min-width:56px;width:56px;
  max-width:72px;text-align:right;color:#6b7686;font:.78rem ui-monospace,SFMono-Regular,Menlo,monospace}
.stApp table.hf th.idx{z-index:3;background:#171e29}
.stApp table.hf tbody tr:hover td{background:#19212c}
.stApp table.hf.img td{height:96px}
.stApp table.hf td.im{min-width:200px}
.stApp table.hf img{display:block;max-height:80px;max-width:340px;width:auto;height:auto;border-radius:4px;
  background:#fff}
.stApp .cn{font-weight:600;color:#e7e5df;overflow:hidden;text-overflow:ellipsis}
.stApp .ct{font:.7rem ui-monospace,SFMono-Regular,Menlo,monospace;color:#d9b26f;margin-top:.1rem}
.stApp .cell{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:4;overflow:hidden;
  white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45}
.stApp .cell.code{font:.76rem/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;color:#b9c2cf}
.stApp .nul{color:#6b7686;font-style:italic}
.stApp .num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.stApp table.hf tr.ld td{color:#9aa4b2;text-align:left;animation:hfpulse 1.2s ease-in-out infinite}
@keyframes hfpulse{50%{opacity:.4}}
</style>"""


class UserError(Exception):
    """Error whose message is safe and meant to be shown to the user."""


def explain(e: Exception, token: str) -> str:
    """Turn any exception into a friendly message that can never contain the token."""
    if isinstance(e, UserError):
        return str(e)
    if isinstance(e, HFValidationError):
        return "Invalid repository name. Use the form `owner/dataset-name`."
    if isinstance(e, GatedRepoError):
        return "This repository is gated. Request access on Hugging Face, then try again."
    if isinstance(e, RepositoryNotFoundError):
        return "Repository not found, or the token is invalid or has no access. Check the name and that the token has read permission."
    if isinstance(e, HfHubHTTPError):
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 401:
            return "The access token is invalid or expired."
        if code == 403:
            return "The token does not have permission to read this repository."
        if code == 429:
            return "Hugging Face is rate limiting requests. Wait a moment and try again."
    if isinstance(e, FileNotFoundError):
        return "No readable data files were found in this repository."
    name = type(e).__name__
    if isinstance(e, OSError) or any(k in name for k in ("Connect", "Timeout", "Network")):
        return "Network problem while contacting Hugging Face. Check your connection and try again."
    return f"Unexpected error ({name}): {str(e).replace(token, '***')[:300]}"


def connect(token: str, repo: str) -> list[str]:
    """One API call validates token and repo, and returns the Parquet files."""
    info = HfApi(token=token).repo_info(repo, repo_type="dataset")
    files = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".parquet")]
    if not files:
        raise UserError("No Parquet files were found in this repository.")
    return files



def dataset_counts(token: str, repo: str, parquet_files: list[str]) -> dict:
    """Read full split row counts from Hugging Face metadata without streaming all rows."""
    result = {
        "train": None, "validation": None, "test": None, "other": None,
        "images": None, "transcriptions": None, "paired": None,
        "shards": len(parquet_files), "failed_shards": None,
        "complete": False, "error": None,
    }
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(repo, token=token)
        split_info = builder.info.splits or {}
        if not split_info:
            raise ValueError("Hugging Face did not provide split row-count metadata.")

        counts = {}
        for split_name, info in split_info.items():
            number = getattr(info, "num_examples", None)
            if number is None or int(number) < 0:
                raise ValueError(f"Row count is unavailable for split '{split_name}'.")
            counts[str(split_name).lower()] = int(number)

        result["train"] = sum(v for k, v in counts.items() if k == "train")
        result["validation"] = sum(v for k, v in counts.items() if k in ("validation", "val", "dev"))
        result["test"] = sum(v for k, v in counts.items() if k == "test")
        result["other"] = sum(
            v for k, v in counts.items()
            if k not in ("train", "validation", "val", "dev", "test")
        )
        total = sum(counts.values())
        features = builder.info.features or {}
        has_image, has_text = "image" in features, "text" in features
        result["images"] = total if has_image else 0
        result["transcriptions"] = total if has_text else 0
        result["paired"] = total if has_image and has_text else 0
        result["failed_shards"] = 0
        result["complete"] = True
    except Exception as e:
        result["error"] = str(e)[:220]
    return result


def dataset_counts_html(c: dict) -> str:
    """Decorative full-dataset summary, separate from the limited streamed sample."""
    def value(v):
        return "—" if v is None else f"{v:,}" if isinstance(v, int) else str(v)

    metrics = [
        ("TRAIN SAMPLES", c.get("train")),
        ("VALIDATION SAMPLES", c.get("validation")),
        ("TEST SAMPLES", c.get("test")),
        ("OTHER SPLIT SAMPLES", c.get("other")),
        ("TOTAL IMAGE RECORDS", c.get("images")),
        ("TOTAL TRANSCRIPTION RECORDS", c.get("transcriptions")),
        ("TOTAL PAIRED SAMPLES", c.get("paired")),
        ("PARQUET SHARDS FOUND", c.get("shards")),
        ("PARQUET SHARDS FAILED", c.get("failed_shards")),
    ]
    cards = "".join(
        f"<div class='ds-metric'><div class='ds-label'>{html.escape(label)}</div>"
        f"<div class='ds-value'>{value(number)}</div></div>"
        for label, number in metrics
    )
    status = (
        "<span class='ds-status yes'>✓ COUNT COMPLETE</span>"
        if c.get("complete") else "<span class='ds-status no'>! COUNT INCOMPLETE</span>"
    )
    note = ("<div class='ds-note'>Full dataset totals from Hugging Face split metadata · "
            "independent of the sample limit selected for streaming.</div>")
    if c.get("error"):
        note += f"<div class='ds-error'>Full counts unavailable: {esc(c['error'])}</div>"
    return (
        "<section class='ds-panel'><div class='ds-heading'>"
        "<span class='ds-kicker'>DATA OVERVIEW</span>"
        "<span class='ds-title'>Dataset Counts</span>"
        f"{status}</div><div class='ds-grid'>{cards}</div>{note}</section>"
    )


def type_label(f) -> str:
    """Short dtype label for a `datasets` feature, like the type shown under a column name in the HF viewer."""
    if isinstance(f, dict):
        return "dict"
    if isinstance(f, list):
        return "list"
    n = type(f).__name__
    if n == "Value":
        return str(getattr(f, "dtype", "value"))
    if n == "ClassLabel":
        return "class_label"
    if n in ("List", "LargeList", "Sequence"):
        return "list"
    return n.lower()


def open_streams(
    token: str, repo: str, n: int, shuffle: bool = True, seed: int = 42, buffer_size: int = 10_000
) -> tuple[dict, dict]:
    """Create bounded-memory streams and optionally shuffle before taking n rows."""
    from datasets import Image as HFImage, load_dataset  # imported late so the page renders instantly

    ds = load_dataset(repo, streaming=True, token=token)
    if not isinstance(ds, dict):
        ds = {"default": ds}
    if not ds:
        raise UserError("The dataset has no splits.")
    out, schemas = {}, {}
    ordered_splits = sorted(ds, key=lambda s: (SPLIT_ORDER.get(s, 3), s))
    for split_index, name in enumerate(ordered_splits):
        d = ds[name]
        feats = d.features or {}
        for col, f in feats.items():
            if type(f).__name__ == "Image":  # keep image payloads undecoded while streaming
                d = d.cast_column(col, HFImage(decode=False))
        if shuffle:
            # IterableDataset.shuffle uses a bounded buffer; it does not download the full split.
            # A larger buffer improves mixing but uses more memory.
            d = d.shuffle(seed=seed + split_index, buffer_size=buffer_size)
        out[name] = d.take(n)
        schemas[name] = {col: type_label(f) for col, f in feats.items()}
    return out, schemas


def to_bytes(v) -> bytes | None:
    """Raw bytes of an image cell (bytes / {'bytes','path'} dict / PIL image)."""
    if isinstance(v, dict):
        v = v.get("bytes") or v.get("path")
        if isinstance(v, str) and os.path.exists(v):
            with open(v, "rb") as f:
                v = f.read()
    if hasattr(v, "save"):  # already a PIL image
        buf = io.BytesIO()
        v.save(buf, "PNG")
        v = buf.getvalue()
    if not v:
        return None
    return bytes(v)


def prepare_image(v) -> bytes | None:
    """Return display-ready image bytes. Raises on corrupted data."""
    raw = to_bytes(v)
    if not raw:
        return None
    img = Image.open(io.BytesIO(raw))
    if img.format in WEB_FORMATS and len(raw) <= MAX_BYTES:
        return raw  # pass original bytes through directly to avoid file pointer corruption
    img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # fast downscaled JPEG decode (no-op for other formats)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()


def thumb_uri(raw: bytes) -> str:
    """Small JPEG data-URI used inside table cells."""
    im = Image.open(io.BytesIO(raw))
    im.draft("RGB", (THUMB[0] * 2, THUMB[1] * 2))
    im.thumbnail(THUMB)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def mime_of(b: bytes) -> str:
    if b[:2] == b"\xff\xd8":
        return "image/jpeg"
    if b[:4] == b"\x89PNG":
        return "image/png"
    return "image/webp" if b[8:12] == b"WEBP" else "image/gif"


def brief(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v):,} bytes>"
    if isinstance(v, dict) and "bytes" in v:
        return "<binary>"
    return str(v)[:200]


def esc(s) -> str:
    """HTML-escape and keep the markup on ONE line (blank lines would end a markdown HTML block)."""
    return html.escape(str(s)).replace("\r", "").replace("\n", "&#10;").replace("$", "&#36;")


def is_image_value(v) -> bool:
    return hasattr(v, "save") or (isinstance(v, dict) and "bytes" in v and "path" in v)


def fmt_cell(v) -> tuple[str, str, str]:
    """(dtype label, cell html, searchable text) for a non-image value."""
    if v is None:
        return "null", NULL, ""
    if isinstance(v, bool):
        return "bool", f"<span class='num'>{str(v).lower()}</span>", str(v)
    if isinstance(v, int):
        return "int64", f"<span class='num'>{v}</span>", str(v)
    if isinstance(v, float):
        return "float64", f"<span class='num'>{v!r}</span>", str(v)
    if isinstance(v, str):
        shown = v if len(v) <= CELL_CHARS else v[:CELL_CHARS] + "…"
        return "string", f"<div class='cell' dir='auto'>{esc(shown)}</div>", v[:2000]
    if isinstance(v, (bytes, bytearray)):
        return "binary", f"<span class='nul'>{esc(brief(v))}</span>", ""
    try:
        s = json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        s = str(v)
    kind = "list" if isinstance(v, (list, tuple)) else "dict" if isinstance(v, dict) else type(v).__name__.lower()
    return kind, f"<div class='cell code'>{esc(s[:CELL_CHARS])}</div>", s[:2000]


def build_record(row: dict, i: int, image_cols: frozenset) -> dict:
    """Runs in a worker thread: must not call any st.* function."""
    img, err, bad = None, None, False
    if "image" not in row:
        err = "This record has no image column."
    else:
        try:
            img = prepare_image(row["image"])
            err = None if img else "The image field is empty."
        except Exception:
            err = "The image is corrupted or in an unsupported format."
    name, text = row.get("image_name"), row.get("text")
    name = str(name) if name not in (None, "") else f"record_{i + 1}"

    cells, kinds, hay = {}, {}, [name]
    for k, v in row.items():
        if k in image_cols or is_image_value(v):
            kinds[k] = "image"
            if v is None:
                cells[k] = NULL
                continue
            try:
                raw = img if (k == "image" and img) else to_bytes(v)
                if not raw:
                    raise ValueError("empty image")
                cells[k] = f"<img src='{thumb_uri(raw)}' loading='lazy' alt=''>"
            except Exception:
                bad = True
                cells[k] = "<span class='nul'>invalid image</span>"
        else:
            kinds[k], cells[k], t = fmt_cell(v)
            if t:
                hay.append(t)

    mime = mime_of(img) if img else None
    return {
        "i": i,
        "name": name,
        "file": os.path.splitext(name)[0] + EXT[mime] if mime else name,
        "text": "" if text is None else str(text),
        "image": img,
        "mime": mime,
        "err": err,
        "bad": bad,
        "extra": {k: brief(v) for k, v in row.items() if k not in CORE},
        "cells": cells,
        "kinds": kinds,
        "hay": " ".join(hay).lower(),
        "tr": None,  # row html, built once on first render and reused by every later redraw
    }


# ----------------------------------------------------------------------------- table rendering
def row_html(r: dict, cols: list[str]) -> str:
    tr = r["tr"]
    if tr is None:
        tds = "".join(
            f"<td{' class=im' if r['kinds'].get(c) == 'image' else ''}>{r['cells'].get(c, NULL)}</td>" for c in cols
        )
        tr = r["tr"] = f"<tr><td class='idx'>{r['i']}</td>{tds}</tr>"
    return tr


def col_types(recs: list[dict], cols: list[str], schema: dict) -> dict:
    """Dtype per column: from the dataset features when known, otherwise inferred from the first non-null value."""
    return {
        c: schema.get(c)
        or next((r["kinds"][c] for r in recs if r["kinds"].get(c) not in (None, "null")), "null")
        for c in cols
    }


def table_html(recs: list[dict], cols: list[str], types: dict, loading: bool = False) -> str:
    """One scrollable table: sticky header (name + type), sticky row-number column, one column per dataset column."""
    th = "".join(f"<th><div class='cn'>{esc(c)}</div><div class='ct'>{esc(types.get(c, ''))}</div></th>" for c in cols)
    body = "".join(row_html(r, cols) for r in recs)
    foot = f"<tr class='ld'><td colspan='{len(cols) + 1}'>Streaming more rows…</td></tr>" if loading else ""
    img = " img" if "image" in types.values() else ""
    return (
        f"<div class='tbl-wrap'><table class='hf{img}'><thead><tr><th class='idx'></th>{th}</tr></thead>"
        f"<tbody>{body}{foot}</tbody></table></div>"
    )


def paint(s: dict, schema: dict, loading: bool):
    """Redraw one split's table into its placeholder (main thread only)."""
    recs = s["records"]
    if not recs:
        return
    if s["colnames"] is None:
        s["colnames"] = list(schema) or list(recs[0]["cells"])
    s["types"] = col_types(recs, s["colnames"], schema)
    s["ph"].markdown(table_html(recs, s["colnames"], s["types"], loading), unsafe_allow_html=True)
    s["dirty"], s["loading"], s["last"] = False, loading, time.time()


# ----------------------------------------------------------------------------- cards (unchanged layout)
def show_image_compatible(image_bytes: bytes):
    """Display an image across older and newer Streamlit versions."""
    parameters = inspect.signature(st.image).parameters
    if "use_container_width" in parameters:
        st.image(image_bytes, use_container_width=True)
    elif "use_column_width" in parameters:
        st.image(image_bytes, use_column_width=True)
    else:
        st.image(image_bytes)


def card(r: dict, key: str, show_text: bool):
    with st.container(border=True):
        st.markdown(f"<div class='name'>{html.escape(r['name'])}</div>", unsafe_allow_html=True)
        if r["image"]:
            show_image_compatible(r["image"])
        else:
            st.warning(r["err"])
        if show_text:
            body = html.escape(r["text"]) if r["text"] else "No text for this record."
            st.markdown(f"<div class='txt' dir='auto'>{body}</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            if r["image"]:
                st.download_button(
                    "Save image", r["image"], file_name=r["file"], mime=r["mime"],
                    key=f"dl_{key}", use_container_width=True,
                )
        with right:
            with st.popover("Details", use_container_width=True):
                st.code(r["text"] or "(no text)", language=None, wrap_lines=True)
                if r["extra"]:
                    st.json(r["extra"])


def chips_html(m: dict) -> str:
    multi = len(m["splits"]) > 1
    items = [
        ("Repository", m["repo"]),
        ("Parquet files", m["files"]),
        ("Requested", f"{m['n']} per split" if multi else m["n"]),
        ("Splits", ", ".join(m["splits"])),
    ]
    items.append(("Shuffle", "On" if m.get("shuffle") else "Off"))
    if m.get("shuffle"):
        items.append(("Seed", m.get("seed")))
        items.append(("Buffer", f"{m.get('buffer_size'):,}"))
    if m.get("columns"):
        items.append(("Columns", f"{len(m['columns'])}: " + ", ".join(m["columns"])[:120]))
    if "streamed" in m:
        items += [("Streamed", m["streamed"]), ("Time", f"{m['elapsed']:.1f}s")]
        if m["issues"]:
            items.append(("Image issues", m["issues"]))
    chips = "".join(f"<span class='chip'><b>{html.escape(k)}</b>{html.escape(str(v))}</span>" for k, v in items)
    return f"<div class='chips'>{chips}</div>"


def produce(s: dict, stream, pool):
    """Background thread: pulls rows from the stream and hands images to the worker pool."""
    try:
        for i, row in enumerate(stream):
            s["cols"] = s["cols"] or list(row)
            s["futs"].append(pool.submit(build_record, row, i, s["img_cols"]))
    except Exception as e:
        s["err"] = e
    finally:
        s["done"] = True


def run(token: str, repo: str, n: int, shuffle: bool, seed: int, buffer_size: int):
    t0 = time.time()
    try:
        with st.spinner("Connecting to Hugging Face..."):
            files = connect(token, repo)
            counts = dataset_counts(token, repo, files)
            streams, schemas = open_streams(token, repo, n, shuffle, seed, buffer_size)
    except Exception as e:
        st.error(explain(e, token))
        return

    m = {
        "key": (repo, n, shuffle, seed, buffer_size), "repo": repo, "files": len(files),
        "n": n, "splits": list(streams), "shuffle": shuffle, "seed": seed,
        "buffer_size": buffer_size, "counts": counts,
    }
    head, notes = st.empty(), st.container()
    head.markdown(chips_html(m), unsafe_allow_html=True)
    st.markdown(dataset_counts_html(counts), unsafe_allow_html=True)
    bar = st.progress(0.0, text="Waiting for the first record...")
    tabs = st.tabs(list(streams))

    # Every split streams concurrently; images are processed in parallel; the table grows as rows are ready.
    pool = ThreadPoolExecutor(WORKERS)
    S = {}
    for (name, stream), tab in zip(streams.items(), tabs):
        schema = schemas[name]
        S[name] = {
            "futs": deque(), "done": False, "err": None, "cols": None, "records": [],
            "ph": tab.empty(), "dirty": False, "loading": False, "last": 0.0,
            "colnames": None, "types": {},
            "img_cols": frozenset(c for c, t in schema.items() if t == "image"),
        }
        Thread(target=produce, args=(S[name], stream, pool), daemon=True).start()

    total = n * len(S)
    while True:
        progressed, finished = False, True
        for name, s in S.items():
            done = s["done"]  # read first: done=True with no pending futures means truly finished
            while s["futs"] and s["futs"][0].done():
                s["records"].append(s["futs"].popleft().result())
                s["dirty"] = progressed = True
            fin = done and not s["futs"]
            finished &= fin
            # Redraw is throttled (the table is re-sent whole), but always runs once a split is complete.
            if (s["dirty"] or (fin and s["loading"])) and (
                fin or time.time() - s["last"] >= 0.25 + 0.004 * len(s["records"])
            ):
                paint(s, schemas[name], loading=not fin)
            if s["cols"] and not m.get("columns"):
                m["columns"] = s["cols"]
                head.markdown(chips_html(m), unsafe_allow_html=True)
                missing = [c for c in CORE if c not in s["cols"]]
                if missing:
                    notes.info("Cards layout expects these columns, which are missing: " + ", ".join(f"`{c}`" for c in missing))
        got = sum(len(s["records"]) for s in S.values())
        bar.progress(min(got / total, 1.0), text=f"Streaming {got} of {total} records")
        if finished:
            break
        if not progressed:
            time.sleep(0.02)
    pool.shutdown(wait=False)
    bar.empty()

    for name, s in S.items():
        if s["err"]:
            notes.error(f"Split `{name}` stopped after {len(s['records'])} records. {explain(s['err'], token)}")
        elif not s["records"]:
            notes.warning(f"Split `{name}` is empty.")
    data = {name: s["records"] for name, s in S.items()}
    if not any(data.values()):
        return
    m.update(
        streamed=sum(map(len, data.values())),
        elapsed=time.time() - t0,
        issues=sum(r["bad"] for recs in data.values() for r in recs),
        data=data,
        layout={name: (s["colnames"] or [], s["types"]) for name, s in S.items()},
    )
    st.session_state["run"] = m  # the token is never stored
    st.rerun()  # redraw from memory with search enabled


def view(cols: int, show_text: bool, layout: str):
    m = st.session_state["run"]
    st.markdown(dataset_counts_html(m.get("counts", {})), unsafe_allow_html=True)
    st.markdown(chips_html(m), unsafe_allow_html=True)
    q = st.text_input(
        "Search", key="q", placeholder="Search any column...", label_visibility="collapsed"
    ).strip().lower()
    tabs = st.tabs([f"{name} ({len(recs)})" for name, recs in m["data"].items()])
    for tab, (name, recs) in zip(tabs, m["data"].items()):
        with tab:
            shown = [(i, r) for i, r in enumerate(recs) if not q or q in r["hay"]]
            if q:
                st.caption(f"{len(shown)} of {len(recs)} records match")
            if not shown:
                st.info("No records to show." if not q else "No records match your search.")
            elif layout == "Table":
                ccols, types = m["layout"][name]
                st.markdown(table_html([r for _, r in shown], ccols, types), unsafe_allow_html=True)
            else:
                for start in range(0, len(shown), cols):
                    for col, (i, r) in zip(st.columns(cols), shown[start:start + cols]):
                        with col:
                            card(r, f"{name}_{i}", show_text)


def main():
    st.set_page_config(page_title="Dataset streamer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'><h1>Private dataset streamer</h1>"
        "<p>Stream a limited number of records from every split of a private Hugging Face Parquet dataset "
        "without downloading the whole thing.</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Connection")
        with st.form("connection"):
            token = st.text_input("Hugging Face access token", type="password", placeholder="hf_...")
            repo = st.text_input("Repository name", placeholder="owner/dataset-name")
            n = st.number_input(
                "How many records do you want to stream?", min_value=1, max_value=MAX_RECORDS, value=30, step=10,
                help="Applied to each split (train, validation, test).",
            )
            shuffle = st.checkbox("Shuffle before random sampling", value=True)
            seed = st.number_input(
                "Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1,
                help="Use the same seed and settings to reproduce the same sample.",
            )
            buffer_size = st.number_input(
                "Shuffle buffer size", min_value=100, max_value=1_000_000, value=10_000, step=1000,
                help="Larger buffers improve random mixing but use more memory.",
            )
            go = st.form_submit_button("Stream records", type="primary", use_container_width=True)
        st.subheader("View")
        layout = st.radio("Layout", ["Table", "Cards"], horizontal=True,
                          help="Table shows every column like the Hugging Face viewer. Cards adds image downloads and details.")
        cols = st.slider("Grid columns (Cards)", 1, 5, 3)
        show_text = st.toggle("Show text on cards", value=True)
        st.caption("Change view options after streaming finishes.")
        if st.button("Clear results", use_container_width=True):
            st.session_state.pop("run", None)

    if go:
        token, repo = (token or "").strip(), (repo or "").strip()
        if not token or not repo:
            st.error("Enter both an access token and a repository name.")
        elif not REPO_RE.fullmatch(repo):
            st.error("Invalid repository name. Use the form `owner/dataset-name`.")
        elif st.session_state.get("run", {}).get("key") != (repo, int(n), bool(shuffle), int(seed), int(buffer_size)):
            st.session_state.pop("run", None)
            run(token, repo, int(n), bool(shuffle), int(seed), int(buffer_size))

    if "run" in st.session_state:
        view(cols, show_text, layout)
    elif not go:
        st.info("Enter your token, the repository name, and a record count in the sidebar to begin.")


main()