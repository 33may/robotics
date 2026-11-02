"""LimX documentation HTML → clean Markdown extractor.

Pulls Next.js SSR pages from limx.cn doc viewer, isolates the
md-editor-preview body, strips chrome, extracts inline base64 images
to a shared images/ folder (dedup by content hash, human-readable
names), runs pandoc → GFM, and post-processes to:

  * inject language tags onto fenced code blocks (pandoc-gfm drops them)
  * convert ATX `# 1.2.3 Title` lines into proper depth (`##`, `###`)
  * strip residual md-editor chrome (`复制代码`, stray HTML wrappers, '图片' image alt)
  * rewrite image references to local relative paths.

Usage
-----
    python extract.py                       # full pipeline, all three docs
    python extract.py --no-fetch            # reuse cached raw HTML
    python extract.py --doc <id>            # single doc

Dependencies
------------
    pip install beautifulsoup4 pyyaml requests
    pandoc on PATH (>=2.x)

Outputs go to ../sources/ relative to this script.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import gzip
import zlib
from urllib.request import Request, urlopen

import yaml
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(__file__).resolve().parent / "raw"
SOURCES = ROOT / "sources"
IMAGES = SOURCES / "images"
META = SOURCES / "_meta"

DOCS = [
    {
        "id": "831851699013554176",
        "doc_id": "quick-start",
        "slug": "quick_start",
        "filename": "LimX_EDU_Quick_Start_Guide.md",
        "title": "LimX EDU Quick Start Guide",
    },
    {
        "id": "823924477418147840",
        "doc_id": "user-manual",
        "slug": "user_manual",
        "filename": "Oli_EDU_User_Manual.md",
        "title": "Oli EDU User Manual",
    },
    {
        "id": "823930550015365120",
        "doc_id": "sdk-guide",
        "slug": "sdk_guide",
        "filename": "Oli_EDU_SDK_Development_Guide.md",
        "title": "Oli EDU SDK Development Guide",
    },
]

BASE_URL = "https://limx.cn/en/documents/{id}"

# Section labels we strip wholesale (chrome / icons / copy buttons)
STRIP_CLASSES = (
    "md-editor-code-flag",
    "md-editor-code-action",
    "md-editor-collapse-tips",
    "md-editor-code-lang",
    "md-editor-copy-button",
    "md-editor-catalog-tips",
)


def fetch(url: str) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urlopen(req, timeout=60) as r:
        raw = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip":
        raw = gzip.decompress(raw)
    elif enc == "deflate":
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_body(soup: BeautifulSoup) -> Tag:
    """The actual article body is the md-editor-preview div."""
    body = soup.find("div", class_="md-editor-preview")
    if body is None:
        # fallback: first h1 with data-line=0
        h1 = soup.find("h1", attrs={"data-line": "0"})
        if h1 is None:
            raise RuntimeError("could not locate document body")
        body = h1.find_parent()
    if not isinstance(body, Tag):
        raise RuntimeError("document body is not an HTML tag")
    return body


def heading_level(text: str) -> int | None:
    """Given a heading text starting with N or N.N(.N)…, return target ATX depth.

    1            -> 1
    1.2          -> 2
    1.2.3        -> 3
    1.2.3.4      -> 4
    Otherwise None (don't override).
    """
    m = re.match(r"^(\d+(?:\.\d+)*)(?:\s|\.|:)", text)
    if not m:
        return None
    return min(6, len(m.group(1).split(".")))


_HTML_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S | re.I)


def _cell_to_gfm(cell: Tag) -> str:
    """Render a <td>/<th> as a single-line GFM table cell.

    - <br> -> '<br>' (GFM supports inline HTML in cells)
    - newlines -> single space
    - escape pipes
    """
    # replace <br> with marker first
    for br in cell.find_all(["br"]):
        br.replace_with("§§BR§§")
    text = cell.get_text(" ", strip=True)
    text = text.replace("§§BR§§", "<br>")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    # bold markers preserved from <strong> would have been lost — re-inject:
    # (we keep it simple: don't preserve inline formatting, only structure)
    return text


def _html_table_to_gfm(html: str) -> str:
    sub = BeautifulSoup(html, "html.parser")
    table = sub.find("table")
    if table is None:
        return html
    rows = table.find_all("tr")
    if not rows:
        return ""
    # split into header + body
    header_cells = rows[0].find_all(["th", "td"])
    header = [_cell_to_gfm(c) for c in header_cells]
    ncols = len(header) or 1
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * ncols) + "|")
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        vals = [_cell_to_gfm(c) for c in cells]
        # pad short rows
        while len(vals) < ncols:
            vals.append("")
        out.append("| " + " | ".join(vals[:ncols]) + " |")
    return "\n".join(out)


def _html_tables_to_gfm(md: str) -> str:
    return _HTML_TABLE_RE.sub(lambda m: _html_table_to_gfm(m.group(0)), md)


def extract_doc(doc: dict, fetch_fresh: bool) -> dict:
    raw_path = RAW / f"{doc['id']}.html"
    if fetch_fresh or not raw_path.exists():
        url = BASE_URL.format(id=doc["id"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(fetch(url))
    raw_bytes = raw_path.read_bytes()
    raw_html = raw_bytes.decode("utf-8", errors="replace")

    # ---- raw counters for fidelity report --------------------------------
    n_lang_classes = {
        lang: len(re.findall(rf'class="language-{lang}[^"]*"', raw_html))
        for lang in ["bash", "cpp", "html", "json", "python", "shell", "c", "javascript", "typescript", "yaml"]
    }
    n_lang_total = sum(n_lang_classes.values())
    n_tables_html = raw_html.count("<table")
    n_pre_html = raw_html.count("<pre")

    soup = BeautifulSoup(raw_html, "html.parser")
    body = find_body(soup)

    # strip chrome
    for cls in STRIP_CLASSES:
        for el in body.select(f".{cls}"):
            el.decompose()

    # any element whose visible text is just '复制代码' -> remove
    for el in list(body.find_all(string=lambda s: isinstance(s, NavigableString) and s.strip() == "复制代码")):
        # remove parent if it only holds this string
        parent = el.parent
        el.extract()
        if parent and not parent.get_text(strip=True):
            parent.decompose()

    # ---- images: base64 -> file ------------------------------------------
    images_count = 0
    for img in body.find_all("img"):
        src_attr = img.get("src", "")
        src = " ".join(src_attr) if isinstance(src_attr, list) else str(src_attr)
        blob: bytes | None = None
        ext_raw: str = ""
        m = re.match(r"data:image/([\w+]+);base64,(.+)", src, re.S)
        if m:
            ext_raw = m.group(1)
            try:
                blob = base64.b64decode(m.group(2))
            except Exception:
                img.decompose()
                continue
        elif src.startswith("http://") or src.startswith("https://"):
            # remote URL — fetch
            try:
                blob = fetch(src)
            except Exception as exc:
                print(f"  [warn] failed to fetch remote image {src}: {exc}", file=sys.stderr)
                continue
            ext_raw = src.rsplit(".", 1)[-1].lower().split("?")[0]
            if ext_raw not in {"jpg", "jpeg", "png", "webp", "gif", "svg"}:
                ext_raw = "png"
        else:
            if src.startswith("data:"):
                img.decompose()
            continue
        if blob is None:
            continue
        # drop tiny svg chrome (chevrons etc)
        if ext_raw.startswith("svg") and len(blob) < 1500:
            img.decompose()
            continue
        ext = "svg" if ext_raw.startswith("svg") else ext_raw
        if ext == "jpeg":
            ext = "jpg"
        h = hashlib.sha1(blob).hexdigest()[:12]
        fname = f"{doc['slug']}_{images_count + 1:02d}_{h}.{ext}"
        # dedupe across docs: if we already saved this hash, reuse name
        existing = list(IMAGES.glob(f"*_{h}.{ext}"))
        if existing:
            fname = existing[0].name
        else:
            IMAGES.mkdir(parents=True, exist_ok=True)
            (IMAGES / fname).write_bytes(blob)
        img["src"] = f"images/{fname}"
        # clean alt text — replace Chinese '图片' placeholder
        alt_attr = img.get("alt", "")
        alt_text = " ".join(alt_attr) if isinstance(alt_attr, list) else str(alt_attr)
        alt = alt_text.replace("图片", "").strip()
        img["alt"] = alt or "figure"
        images_count += 1

    # ---- capture language per <pre> in document order --------------------
    pre_langs: list[str] = []
    for pre in body.find_all("pre"):
        code = pre.find("code")
        lang = ""
        if code is not None:
            class_attr = code.get("class")
            classes = class_attr if isinstance(class_attr, list) else [str(class_attr)]
            for cls in classes:
                cls_text = str(cls)
                if cls_text.startswith("language-") and cls_text != "language-":
                    lang = cls_text.removeprefix("language-")
                    break
        pre_langs.append(lang)

    # write clean HTML and run pandoc
    body_html_path = RAW / f"{doc['id']}_body.html"
    body_html_path.write_text(str(body), encoding="utf-8")

    md_path = SOURCES / doc["filename"]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            "-f", "html",
            "-t", "gfm",
            "--wrap=preserve",
            str(body_html_path),
            "-o", str(md_path),
        ],
        check=True,
    )

    md = md_path.read_text(encoding="utf-8")

    # ---- post-process the markdown ---------------------------------------
    lines = md.splitlines()
    out: list[str] = []
    fence_idx = 0
    in_fence = False
    for line in lines:
        stripped = line.strip()

        # inject language on opening fence
        if not in_fence and stripped == "```":
            lang = pre_langs[fence_idx] if fence_idx < len(pre_langs) else ""
            fence_idx += 1
            in_fence = True
            out.append("```" + lang)
            continue
        if in_fence and stripped == "```":
            in_fence = False
            out.append("```")
            continue
        if in_fence:
            out.append(line)
            continue

        # already-fenced lines like ```python from pandoc — pass through (rare)
        if stripped.startswith("```") and not in_fence and stripped != "```":
            in_fence = True
            out.append(line)
            fence_idx += 1
            continue

        # heading depth normalization
        m = re.match(r"^(#+)\s+(.*?)\s*$", line)
        if m:
            text = m.group(2)
            depth = heading_level(text)
            if depth is not None:
                out.append("#" * depth + " " + text)
                continue
        # strip residual '复制代码' that may have slipped through
        if stripped == "复制代码":
            continue
        # strip empty image alt placeholder lines
        if stripped == "![图片]()" or stripped == "![](图片)":
            continue
        # filter trailing '图片' artifacts that pandoc sometimes emits as plain text
        if stripped == "图片":
            continue

        out.append(line)

    cleaned = "\n".join(out)
    # convert any remaining HTML <table> blocks to GFM tables
    cleaned = _html_tables_to_gfm(cleaned)
    # collapse 3+ blank lines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned.rstrip()) + "\n"
    md_path.write_text(cleaned, encoding="utf-8")

    # ---- fidelity counters -----------------------------------------------
    md_text = cleaned
    # count fenced openings per language
    fence_lang_counts: dict[str, int] = {}
    fence_open = re.findall(r"^```([\w+-]*)\s*$", md_text, flags=re.M)
    for f in fence_open:
        # fence_open captures all fences (open+close). Open fences come at even index pairs
        # but easier: count language tag occurrences when nonempty
        pass
    # robust pairing: walk lines
    fence_state = False
    for ln in md_text.splitlines():
        s = ln.strip()
        if s.startswith("```") and not fence_state:
            tag = s.removeprefix("```").strip()
            fence_lang_counts[tag or "_none_"] = fence_lang_counts.get(tag or "_none_", 0) + 1
            fence_state = True
        elif s == "```" and fence_state:
            fence_state = False

    # GFM table header count: lines starting with | that are followed by '|---'
    md_lines = md_text.splitlines()
    table_count = 0
    for i, ln in enumerate(md_lines[:-1]):
        if ln.lstrip().startswith("|") and re.match(r"^\s*\|[\s:|-]+\|\s*$", md_lines[i + 1]):
            table_count += 1

    has_copy_marker = "复制代码" in md_text
    has_tupian = re.search(r"(?<![A-Za-z])图片(?![A-Za-z])", md_text) is not None

    return {
        "id": doc["id"],
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "source_url": BASE_URL.format(id=doc["id"]),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_sha256": sha256(raw_bytes),
        "clean_sha256": sha256(md_path.read_bytes()),
        "filename": doc["filename"],
        "counts": {
            "html_pre": n_pre_html,
            "html_tables": n_tables_html,
            "html_language_classes_total": n_lang_total,
            "html_language_classes": n_lang_classes,
            "md_fences_by_lang": fence_lang_counts,
            "md_fences_total": sum(fence_lang_counts.values()),
            "md_tables": table_count,
            "images_extracted": images_count,
        },
        "fidelity": {
            "copy_marker_absent": not has_copy_marker,
            "tupian_placeholder_absent": not has_tupian,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="reuse cached raw HTML")
    ap.add_argument("--check", action="store_true", help="fetch/verify only; do not modify sources")
    ap.add_argument("--doc", help="only process this doc id")
    args = ap.parse_args()

    docs = [d for d in DOCS if not args.doc or d["id"] == args.doc or d["doc_id"] == args.doc]

    if args.check:
        manifest_path = META / "manifest.yaml"
        if not manifest_path.exists():
            raise SystemExit("manifest missing; run extractor before --check")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        previous = {entry["doc_id"]: entry for entry in manifest.get("docs", [])}
        drifted = False
        for d in docs:
            raw = fetch(BASE_URL.format(id=d["id"]))
            current_raw = sha256(raw)
            expected = previous.get(d["doc_id"])
            if expected is None:
                print(f"[check] {d['doc_id']}: missing manifest entry", file=sys.stderr)
                drifted = True
                continue
            if expected.get("raw_sha256") != current_raw:
                print(
                    f"[check] {d['doc_id']}: raw_sha256 drift previous={expected.get('raw_sha256')} current={current_raw}",
                    file=sys.stderr,
                )
                drifted = True
            md_path = SOURCES / d["filename"]
            if not md_path.exists():
                print(f"[check] {d['doc_id']}: missing source file {md_path}", file=sys.stderr)
                drifted = True
                continue
            current_clean = sha256(md_path.read_bytes())
            if expected.get("clean_sha256") != current_clean:
                print(
                    f"[check] {d['doc_id']}: clean_sha256 mismatch previous={expected.get('clean_sha256')} current={current_clean}",
                    file=sys.stderr,
                )
                drifted = True
        raise SystemExit(1 if drifted else 0)

    SOURCES.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    results = []
    for d in docs:
        print(f"[extract] {d['id']} {d['title']}", file=sys.stderr)
        r = extract_doc(d, fetch_fresh=not args.no_fetch)
        results.append(r)

    (META / "manifest.yaml").write_text(
        yaml.safe_dump({"docs": results}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
