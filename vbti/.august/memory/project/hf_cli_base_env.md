---
name: Use huggingface_hub from conda envs, not upgraded base
description: HF Hub work uses each conda env's own huggingface_hub (~0.35); do not upgrade base to 1.x — it breaks transformers
type: project
originSessionId: 6764f38e-c482-44ba-ad27-1fce66892784
---
Do HF Hub operations (`hf` CLI, `huggingface_hub` API) from within the conda envs (`lerobot`, `groot`) — they carry their own `huggingface_hub` (~0.35.x) — rather than relying on a globally-upgraded base CLI.

`huggingface_hub` 1.x was tried in miniconda base on 2026-05-18 to get a modern `hf` CLI (`repo delete`, etc.), but `transformers` hard-requires `huggingface_hub<1.0` and stopped importing. Reverted base to `huggingface_hub==0.35.3`.

**Why:** `transformers`/`lerobot` pin `huggingface_hub<1.0`; a global 1.x upgrade breaks them. The user wants HF tooling scoped to the envs, not a special base install.
**How to apply:** don't `pip install -U huggingface_hub` in base. The 0.35.x `hf` CLI lacks `repo delete` / `datasets` — for those, use the Python `huggingface_hub` API (e.g. `delete_repo`, `HfApi`), which works fine on 0.35.x.
