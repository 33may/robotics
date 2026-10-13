"""Convert a two-column text PDF with no HTML version to plain markdown.

usage: python3 -I pdf_to_md.py <src.pdf> <out.md> "<title>" "<source url>"

Reads text blocks with PyMuPDF and orders them column by column: blocks that span
both columns split the page into bands, and inside a band the left column is read
before the right one. Figures, tables and formulas are not recovered.
"""
import re
import sys

import pymupdf as fitz

src, out, title, url = sys.argv[1:5]

HEAD = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{2,70})$")
ROMAN = re.compile(r"^([IVX]{1,4})\.\s+([A-Z][A-Z0-9 \-:,&]{3,70})$")
LETTER = re.compile(r"^([A-H])\.\s+([A-Z][^\n]{2,70})$")


def heading(line):
    words = len(line.split())
    h = HEAD.match(line)
    if h and words <= 10 and not line.endswith((".", ",")) and not h.group(2).startswith("Courtesy"):
        return "#" * (2 + h.group(1).count(".")) + f" {h.group(1)} {h.group(2)}"
    r = ROMAN.match(line)
    if r:
        return f"## {r.group(1)}. {r.group(2).title()}"
    le = LETTER.match(line)
    if le and 3 <= words <= 10 and not line.endswith((".", ":")) and ". " not in le.group(2):
        return f"### {le.group(1)}. {le.group(2)}"
    return None


def ordered_blocks(page):
    width = page.rect.width
    mid = width / 2
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    blocks.sort(key=lambda b: (b[1], b[0]))
    bands, left, right = [], [], []
    for b in blocks:
        x0, x1 = b[0], b[2]
        full = (x1 - x0) > 0.6 * width or (x0 < mid - 20 and x1 > mid + 20)
        if full:
            bands.extend(left + right)
            bands.append(b)
            left, right = [], []
        elif (x0 + x1) / 2 < mid:
            left.append(b)
        else:
            right.append(b)
    bands.extend(left + right)
    return bands


doc = fitz.open(src)
body = []
for n, page in enumerate(doc, start=1):
    paras = []
    for b in ordered_blocks(page):
        text = b[4]
        for lig, plain in (("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb00", "ff"), ("\ufb03", "ffi"), ("\ufb04", "ffl")):
            text = text.replace(lig, plain)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        while lines:
            h = heading(lines[0])
            if not h:
                break
            paras.append(h)
            lines = lines[1:]
        if lines:
            paras.append(" ".join(lines))
    body.append(f"<!-- page {n} -->\n\n" + "\n\n".join(paras))

note = (
    f"# {title}\n\n> Source: {url}. Text extracted from the PDF with PyMuPDF on 2026-10-06, "
    f"read column by column. Figures, tables and formulas are not recovered, "
    f"see `../pdf/{src.split('/')[-1]}`.\n\n"
)
open(out, "w", encoding="utf-8").write(note + "\n\n".join(body) + "\n")
print(f"{out}: {len(doc)} pages, {sum(len(b) for b in body)} chars")
