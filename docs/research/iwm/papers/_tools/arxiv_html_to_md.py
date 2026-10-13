"""Convert one arXiv HTML page (LaTeXML) to markdown.

usage: python3 -I arxiv_html_to_md.py <src.html> <out.md> <arxiv_id> <pdf_name> <lua_filter>

Math is taken from the authors' LaTeX (the alttext attribute), not re-derived from MathML.
"""
import html
import re
import subprocess
import sys

src, out, arxiv_id, pdf_name, lua = sys.argv[1:6]
page = open(src, encoding="utf-8", errors="ignore").read()

m = re.search(r"<article\b.*?</article>", page, re.S)
if not m:
    sys.exit(f"no <article> in {src}")
art = m.group(0)

ver = re.search(r"arXiv:%s(v\d+)" % re.escape(arxiv_id), page)
version = ver.group(1) if ver else ""
base = f"https://arxiv.org/html/{arxiv_id}{version}/"

# 1. math -> tokens carrying the original LaTeX
maths = []


def math_token(mm):
    tag = mm.group(0)
    head = tag[: tag.find(">") + 1]
    a = re.search(r'alttext="([^"]*)"', head)
    if not a:
        return tag  # leave it to pandoc
    maths.append([html.unescape(a.group(1)).strip(), 'display="block"' in head])
    return f"MATHTOKEN{len(maths) - 1:05d}END"


art = re.sub(r"<math\b.*?</math>", math_token, art, flags=re.S)


# 2. equation tables -> one paragraph with the tokens and the equation number
def eq_table(mm):
    block = mm.group(0)
    toks = re.findall(r"MATHTOKEN(\d{5})END", block)
    if not toks:
        return block
    for t in toks:
        maths[int(t)][1] = True
    tag = re.search(r'ltx_tag_equation[^>]*>\s*(\([^)<]*\))', block)
    num = f" EQNUM{tag.group(1)}" if tag else ""
    return "<p>" + " ".join(f"MATHTOKEN{t}END" for t in toks) + num + "</p>"


art = re.sub(
    r'<table\b[^>]*class="[^"]*ltx_eqn_table[^"]*"[^>]*>.*?</table>', eq_table, art, flags=re.S
)

# 3. drop clutter
art = re.sub(r"<button\b.*?</button>", "", art, flags=re.S)
art = re.sub(r"<svg\b.*?</svg>", "<p>[inline vector figure, see the PDF]</p>", art, flags=re.S)
art = re.sub(r"<annotation\b.*?</annotation>", "", art, flags=re.S)


# 4. absolute image links
def abs_src(mm):
    url = mm.group(2)
    if re.match(r"^(https?:|data:|/)", url):
        return mm.group(0)
    return f'{mm.group(1)}{base}{url}"'


art = re.sub(r'(<img\b[^>]*?\bsrc=")([^"]+)"', abs_src, art)

md = subprocess.run(
    ["pandoc", "-f", "html", "-t", "gfm", "--wrap=none", "--lua-filter", lua],
    input=art.encode("utf-8"),
    capture_output=True,
    check=True,
).stdout.decode("utf-8")


# 5. put the LaTeX back
def restore(mm):
    tex, display = maths[int(mm.group(1))]
    tex = re.sub(r"\s+", " ", tex)
    return f"\n\n$$\n{tex}\n$$\n\n" if display else f"${tex}$"


md = re.sub(r"MATHTOKEN(\d{5})END", restore, md)
md = re.sub(r"\s*EQNUM(\([^)]*\))", r"\n\n\1\n\n", md)
md = re.sub(r"[ \t]+\n", "\n", md)
md = re.sub(r"\n{3,}", "\n\n", md)

note = (
    f"> Source: https://arxiv.org/abs/{arxiv_id} ({version or 'latest'}). "
    f"Converted from the arXiv HTML with pandoc on 2026-10-06. "
    f"Math is the authors' LaTeX. Figures link to arxiv.org. "
    f"PDF: `../pdf/{pdf_name}`. Tables may be imperfect, check the PDF before quoting a number.\n\n"
)
open(out, "w", encoding="utf-8").write(note + md.strip() + "\n")
print(f"{out}: {len(md)} chars, {len(maths)} formulas, base {base}")
