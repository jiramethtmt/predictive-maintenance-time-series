#!/usr/bin/env bash
# Serves the fitted predictor from inside the training image, which already carries autogluon.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8800}"
docker rm -f scania-serve >/dev/null 2>&1 || true
docker run -d --name scania-serve \
  -p "${PORT}:${PORT}" \
  -v "${ROOT}/src:/app/src" \
  -v "${ROOT}/serve:/app/serve" \
  -v "${ROOT}/data:/app/data" \
  -v "${ROOT}/models:/app/models" \
  -w /app \
  --entrypoint python \
  scania-train:latest \
  serve/api.py "${PORT}"
echo "scania-serve started on port ${PORT}"