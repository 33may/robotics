#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

chmod -R a+rwX "$SCRIPT_DIR"

docker rm -f vbti-structurizr >/dev/null 2>&1 || true
docker run -d --rm \
  --name vbti-structurizr \
  -p 8080:8080 \
  -v "$SCRIPT_DIR:/usr/local/structurizr:Z" \
  docker.io/structurizr/structurizr local

printf 'Structurizr is running at http://127.0.0.1:8080/workspace/1\n'
