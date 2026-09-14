#!/usr/bin/env bash
# Ships the repo and the parquet cache to macmini-2, builds the training image there and runs it.
# Must be invoked from WSL: only WSL holds the macmini ssh key.
set -euo pipefail

HOST="${HOST:-macmini-2}"
REMOTE_DIR="${REMOTE_DIR:-~/predictive-maintenance-time-series}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$HOST" "mkdir -p $REMOTE_DIR/data/raw $REMOTE_DIR/data/cache"

rsync -az --delete \
  --exclude '.venv' --exclude '.git' --exclude 'data' --exclude 'models' \
  --exclude '__pycache__' --exclude 'web/data.json' --exclude 'web/index.html' \
  "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"

# The raw readout CSVs are 1.6 GB and are only ever read once to build the parquet cache, so the
# cache travels instead; only the small label and specification files are needed verbatim.
rsync -az --info=progress2 \
  "$LOCAL_DIR/data/cache/" "$HOST:$REMOTE_DIR/data/cache/"
rsync -az \
  --include '*_tte.csv' --include '*_labels.csv' --include '*_specifications.csv' \
  --exclude '*' \
  "$LOCAL_DIR/data/raw/" "$HOST:$REMOTE_DIR/data/raw/"

ssh "$HOST" "cd $REMOTE_DIR && docker build -f docker/Dockerfile -t scania-train ."
ssh "$HOST" "cd $REMOTE_DIR && docker run --rm \
  -v \$PWD/data:/app/data \
  -v \$PWD/models:/app/models \
  --cpus 8 --memory 9g \
  scania-train ${*:---time-limit 1800 --presets best_quality --bag-folds 8}"
