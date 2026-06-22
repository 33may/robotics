#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://127.0.0.1:8080/workspace/1}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/screenshots"
OUT="$OUT_DIR/structurizr-$(date +%Y%m%d-%H%M%S).png"

mkdir -p "$OUT_DIR"

google-chrome \
  --headless=new \
  --no-sandbox \
  --disable-gpu \
  --window-size=2560,1440 \
  --virtual-time-budget=5000 \
  --screenshot="$OUT" \
  "$URL" >/dev/null 2>&1

printf '%s\n' "$OUT"
