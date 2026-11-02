#!/usr/bin/env bash
# Extra build steps AFTER the conda solve (compile vendor/, apply patches, download nothing fat).
# Runs inside `bench-<name>` via `locbench env create`. Keep idempotent — reruns must be safe.
set -euo pipefail

# --- example (delete when real) ---
# cd "$(dirname "$0")/vendor/<upstream>"
# cmake -B build && cmake --build build -j
