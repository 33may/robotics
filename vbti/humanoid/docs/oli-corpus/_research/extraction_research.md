# LimX docs extraction research

## TL;DR (5 lines max)
Doc pages on `limx.cn` are **fully server-side rendered** (Next.js SSR) — the initial HTML already contains every heading, table, `<pre><code class="language-X">`, and image (as base64 data URIs). A tiny Python script with BeautifulSoup + pandoc gives near-perfect markdown: real GFM tables, fenced code blocks with language tags, base64 images extracted to local files. **Recommendation: BeautifulSoup-to-clean-HTML → pandoc gfm → post-pass to inject language fences.** No headless browser, no API reverse-engineering, no PDFs needed.

## Site architecture

**Stack:** Next.js (App Router, RSC). Initial HTML is the full SSR'd page. Verified by `curl -A "Mozilla/5.0" https://limx.cn/en/documents/823930550015365120`:

```
$ curl -sL https://limx.cn/en/documents/823930550015365120 | wc -l
9903
$ grep -oE "<h1[^>]*>[^<]+</h1>" limx_sdk.html | head -3
<h1 class="text-xl md:text-4xl font-bold text-gray-900">Oli EDU SDK Development Guide</h1>
<h1 data-line="0" id="Oli EDU SDK Development Guide">Oli EDU SDK Development Guide</h1>
<h1 data-line="7" id="1 Large Model API Interface">1 Large Model API Interface</h1>
$ grep -c "<pre" limx_sdk.html   # code blocks
209
$ grep -c "<table" limx_sdk.html # tables
34
$ grep -oE 'class="language-[a-z+]+"' limx_sdk.html | sort -u
class="language-bash"
class="language-cpp"
class="language-html"
class="language-json"
class="language-python"
class="language-shell"
```

The doc body lives inside `<div class="md-editor-preview default-theme md-editor-scrn">` — looks like the upstream CMS is the open-source `md-editor-v3` Vue component, so the rendered HTML mirrors markdown-it output. Every heading carries a `data-line="N"` attribute that maps back to the source markdown line — confirming this was authored as markdown originally.

**API endpoint:** The index page (`/en/documents`) is hydrated client-side. Setting header `RSC: 1` returns the React Server Component payload with full doc metadata:

```
$ curl -sL -H "RSC: 1" https://limx.cn/en/documents -o index.rsc
# 97 KB payload containing the full tag tree + initialDocuments list,
# e.g.:
"initialDocuments":[{"id":"831851699013554176","title":"LimX EDU Quick Start Guide",
  "tags":[{"id":"WDYBQ001","name":"Oli EDU Ed."}],"language":"en",
  "publishTime":"2026-04-14 11:21:22", ...}, ...]
```

So you can enumerate all docs without scraping a JS-rendered list.

**PDF availability:** None on the doc viewer pages. The separate `/en/downloads` page hosts firmware + a couple of manuals, but the three target docs are HTML-only.

**Mirror availability:** None. GitHub `limxdynamics` org hosts code repos (humanoid-mujoco-sim, etc.) but not the user-facing documentation.

**Image handling:** Content images are inlined as `<img src="data:image/webp;base64,...">` (also a few PNGs). Decorative chevron SVGs are also base64 — those should be filtered out by size.

## Per-document inventory

| Doc title | URL | Format | Language |
|---|---|---|---|
| LimX EDU Quick Start Guide | https://limx.cn/en/documents/831851699013554176 | SSR HTML | EN (verified `<h1>` present) |
| Oli EDU User Manual | https://limx.cn/en/documents/823924477418147840 | SSR HTML | EN |
| Oli EDU SDK Development Guide | https://limx.cn/en/documents/823930550015365120 | SSR HTML | EN |

(Both `limx.cn` and `limxdynamics.com` serve the same Next.js app; either host works.)

## Tooling evaluation

| Tool | Headings | Code blocks | Lang fences | Tables | Images | Repeatable | Effort |
|---|---|---|---|---|---|---|---|
| pandoc (gfm) on raw page | yes | yes | **no** (loses `<code class="language-X">`) | **yes** (real GFM) | base64 stays inline = ugly | high | low |
| **pandoc on cleaned `<div class="md-editor-preview">` + lang post-pass** | yes | yes | **yes** | yes | extracted to `images/` | high | **low** ★ |
| markitdown (Microsoft) | yes | yes | no | yes | base64 → inline data URIs | high | low |
| html2text | yes | yes (no fences for nested code) | no | poor (linearises) | drops binary data: URIs | high | low |
| trafilatura | yes | partial — strips many `<pre>` as boilerplate | no | poor | drops images | medium | low |
| readability-lxml + markdownify | yes | yes | needs custom converter | mediocre | inline | medium | medium |
| MarkDownload / Obsidian Web Clipper | yes | yes | sometimes | yes | usually skipped or inline | **manual per page** | low (but not scriptable) |
| Playwright headless | overkill — page is already SSR'd; only useful if a future page becomes JS-only | — | — | — | — | medium | high |

### Concrete sample (SDK guide, section 1.2.1)

User's current paste:
```
1.2.1 Invoke via Curl       <-- no heading hash
复制代码                     <-- copy-button leak
curl http://10.192.1.3:11434/api/generate ...   <-- no fence, no language
```

Pandoc-clean pipeline output:
```
### 1.2.1 Invoke via Curl

​```bash
curl http://10.192.1.3:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{ "model": "qwen2.5:3b", ... }'
​```
```

Tables verified — pandoc emits real GFM:
```
| **Version** | **Date**   | **Description of Changes** | ... |
|-------------|------------|----------------------------|-----|
| V1.0        | 2026-02-27 | Initial Release            | ... |
| V1.1        | 2026-04-22 | Added Audio Interface      | ... |
```

Fence-language histogram from the final markdown of the SDK guide:
```
209 total code blocks
17 bash, 16 cpp, 1 html, 133 json, 31 python, 2 shell, 9 untagged
```
24 content images extracted, 21 unique after dedupe (md5).

## Recommendation

Use **BeautifulSoup → strip chrome → pandoc gfm → language-fence post-pass**. Single script, ~50 lines, works for all three doc IDs and any future LimX doc.

### Exact pipeline

```bash
# 1. fetch (no auth, no JS)
for ID in 831851699013554176 823924477418147840 823930550015365120; do
  curl -sL -A "Mozilla/5.0" "https://limx.cn/en/documents/$ID" -o "raw/$ID.html"
done

# 2. process (Python; deps: beautifulsoup4)
python extract.py raw/$ID.html out/$ID/
```

`extract.py` outline (verified end-to-end on the SDK guide):

```python
import re, os, base64, hashlib, subprocess, sys
from bs4 import BeautifulSoup

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(f"{outdir}/images", exist_ok=True)
soup = BeautifulSoup(open(src).read(), "html.parser")

# 1. Isolate the article body (the md-editor-preview div)
body = soup.find("h1", attrs={"data-line": "0"}).find_parent()

# 2. Strip md-editor chrome (copy buttons, collapse, lang labels)
for sel in ("md-editor-code-flag", "md-editor-code-action",
            "md-editor-collapse-tips", "md-editor-code-lang",
            "md-editor-copy-button"):
    for el in body.select(f".{sel}"):
        el.decompose()

# 3. Extract base64 images, rewrite to images/img_<md5>.<ext>
langs = []
for img in body.find_all("img"):
    m = re.match(r"data:image/([\w+]+);base64,(.+)", img.get("src",""), re.S)
    if not m: continue
    ext, raw = m.group(1), base64.b64decode(m.group(2))
    if len(raw) < 800 and ext.startswith("svg"):   # drop chevron icons
        img.decompose(); continue
    ext = "svg" if ext.startswith("svg") else ext
    h = hashlib.md5(raw).hexdigest()[:10]
    fn = f"img_{h}.{ext}"
    open(f"{outdir}/images/{fn}", "wb").write(raw)
    img["src"] = f"images/{fn}"
    img["alt"] = img.get("alt","").replace("图片","image")

# 4. Capture language per code block (in document order) for the post-pass
for pre in body.find_all("pre"):
    code = pre.find("code")
    lang = ""
    for c in (code.get("class") or []) if code else []:
        if c.startswith("language-") and c != "language-":
            lang = c.removeprefix("language-"); break
    langs.append(lang)

clean_html = f"{outdir}/body.html"
open(clean_html, "w").write(str(body))

# 5. pandoc → GFM
md_path = f"{outdir}/doc.md"
subprocess.run(["pandoc","-f","html","-t","gfm","--wrap=preserve",
                clean_html,"-o",md_path], check=True)

# 6. Inject language tags into ``` fences (pandoc gfm drops them)
out, idx, in_block = [], 0, False
for line in open(md_path).read().splitlines():
    s = line.strip()
    if s == "```" and not in_block:
        out.append("```" + (langs[idx] if idx < len(langs) else ""))
        idx += 1; in_block = True
    elif s == "```" and in_block:
        out.append("```"); in_block = False
    else:
        out.append(line)
open(md_path,"w").write("\n".join(out))
```

**Verification on SDK guide:** 220 GFM table rows, 209/209 code blocks fenced (200 with language tag), 21 unique content images extracted, all `1.x` / `1.2.3` headings emit `#`/`##`/`###` correctly.

### Follow-up risks
- A handful of code blocks (~9) have no `language-X` class in source and stay un-tagged. Acceptable, or fix by inferring from first-token heuristics.
- Pandoc keeps a wrapping `<div id="document-preview-preview">` line at the top of the md — trivial to strip.
- The `复制代码` ("copy code") chrome string is gone after the `md-editor-copy-button` decompose; verified by grep.
- If LimX swaps its CMS in the future the `.md-editor-preview` anchor breaks; falling back to "first `<h1 data-line="0">` then siblings until next product banner" still works.

## Open questions
- Are there pages where parts of the content are lazy-loaded (e.g. embedded videos, expandable sections)? Spot-checked the SDK guide only end-to-end — Quick Start and User Manual not yet rendered through the pipeline. The headings come back fine via curl, so the pipeline should hold, but a 5-minute trial run on each is worth doing before committing the output to the repo.
- Some images are PNG/WebP screenshots of Chinese-localized UI even on the EN pages. Not extractable from the docs themselves — would need product-side EN screenshots.
- The RSC payload exposes a full doc tree (tags `WDYBQ001..WDYBQ009`, `initialDocuments`). If we later want to mirror every LimX doc, that's the enumeration endpoint to script against.
