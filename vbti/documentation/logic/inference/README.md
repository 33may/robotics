# `logic/inference`

## Purpose

`logic/inference` deploys policies on the real robot and records structured physical evaluation evidence.

## Files

| File | Purpose |
|---|---|
| `run_real_inference.py` | Interactive real-robot policy inference and legacy eval wrapper. |
| `eval_engine.py` | Protocol-based real-robot evaluation engine. |
| `eval_helpers.py` | List/inspect/play evaluation sessions. |
| `eval_render.py` | Render heatmaps/grids from sessions. |
| `async_chunk_runner.py` | Asynchronous action chunk runner. |
| `prompt_gui.py` | Prompt UI helper. |
| `voice_prompt.py` | Voice prompt helper. |
| `protocols/` | Protocol JSONs, render/edit tools, generators. |

## Docs

- `real_inference.md` - live policy deployment.
- `evaluation_engine.md` - protocol trial execution.
- `protocols.md` - protocol schemas/tools.
- `session_analysis.md` - helpers, heatmaps, evidence interpretation.

## Critical Rule

Protocol evaluation sessions are the source of quantitative real-robot claims. Ad-hoc demos are useful for debugging but not for comparing checkpoints.
