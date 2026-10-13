"""Light AsciiDoc to markdown pass for the 3D Tiles spec (headings, code, images, links).

usage: python3 -I adoc_to_md.py <src.adoc> <out.md> "<source url>"
"""
import re
import sys

src, out, url = sys.argv[1:4]
lines = open(src, encoding="utf-8").read().split("\n")
res, in_code = [], False
for ln in lines:
    if re.match(r"^(ifdef|ifndef|endif|include|:[\w-]+:)", ln) or re.match(r"^\[#[\w-]+\]$", ln) or ln.startswith("//"):
        continue
    if re.match(r"^\[source", ln) or re.match(r"^\[%unnumbered\]", ln) or re.match(r"^\[\.\w", ln):
        continue
    if ln.strip() in ("----", "...."):
        res.append("```")
        in_code = not in_code
        continue
    if in_code:
        res.append(ln)
        continue
    h = re.match(r"^(=+)\s+(.*)$", ln)
    if h:
        res.append("#" * len(h.group(1)) + " " + h.group(2))
        continue
    ln = re.sub(r"image::?([^\[\s]+)\[([^\]]*)\]", lambda m: f"![{m.group(2)}](https://raw.githubusercontent.com/CesiumGS/3d-tiles/main/specification/{m.group(1)})", ln)
    ln = re.sub(r"<<([^,>]+),([^>]+)>>", r"\2", ln)
    ln = re.sub(r"<<([^>]+)>>", r"\1", ln)
    ln = re.sub(r"xref:[^\[\s]+\[([^\]]*)\]", r"\1", ln)
    ln = re.sub(r"link:([^\[\s]+)\[([^\]]*)\]", r"[\2](\1)", ln)
    ln = re.sub(r"(https?://[^\[\s]+)\[([^\]]+)\]", r"[\2](\1)", ln)
    ln = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"**\1**", ln)
    ln = re.sub(r"^\.([A-Z][^\n]*)$", r"*\1*", ln)
    res.append(ln)
md = re.sub(r"\n{3,}", "\n\n", "\n".join(res))
note = (f"> Source: {url}. Core spec of 3D Tiles (OGC community standard), converted from AsciiDoc "
        f"with a light regex pass on 2026-10-06. Tables keep their AsciiDoc form.\n\n")
open(out, "w", encoding="utf-8").write(note + md.strip() + "\n")
print(out, len(md), "chars")
