---
name: reference-vbti-visuals
description: VBTI visuals figure-factory (skill vbti-visuals) — location, how to render, and how to restore if the engine vanishes
metadata:
  type: reference
---

The `vbti-visuals` skill (HTML+CSS+Playwright figure factory, warm editorial brand) lives at `vbti/oficial/docs/visuals/` inside the Obsidian vault `/home/may33/Documents/vbti/`.

- **Render:** compose `decks/<name>.html` linking `../theme.css`; `node render.mjs decks/<name>.html` (or `make render FILE=...`) → `out/<name>-NN.png`. Flags: `--pdf`, `--selector "<css>"` crop, `--scale n`. `theme.css` is the authoritative design system — extend it, never fork per-deck. Self-validate by reading the output PNG.
- **Non-obvious gotchas:** the vault is **NOT a git repo**, and `node_modules` + engine files are **not committed** — they can and did go missing (2026-07-06). If `theme.css`/`render.mjs`/`Makefile`/`templates/`/`decks/` are absent, restore from the backup tarball **`~/Downloads/vbti-visuals.tar.gz`** (paths prefixed `vbti/oficial/docs/visuals/`), then `npm install` in the visuals dir (pulls chromium headless shell).
- Added a `.timeline` component (horizontal spine + accent phase nodes + parallel lane) for the Albert humanoid deck. See [[feedback_show_dont_tell]].
