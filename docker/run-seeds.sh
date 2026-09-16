#!/usr/bin/env bash
# Repeats the containerised run across seeds so the reported cost carries a mean and a spread
# rather than a single draw. Run this on the host that owns the image (macmini-2).
set -euo pipefail

ROOT="${ROOT:-$HOME/predictive-maintenance-time-series}"
SEEDS="${SEEDS:-0 1 2}"
LOG="${LOG:-$HOME/ag-seeds.log}"
shift_args=("$@")

: > "$LOG"
for seed in $SEEDS; do
  echo "=== seed $seed ===" | tee -a "$LOG"
  rm -rf "$ROOT/models/autogluon"
  docker run --rm \
    -v "$ROOT/data:/app/data" \
    -v "$ROOT/models:/app/models" \
    -v "$ROOT/results:/app/results" \
    --cpus 8 --memory 11g \
    scania-train --seed "$seed" "${shift_args[@]}" 2>&1 | tee -a "$LOG" | grep -E "^pool rows|miss scale|^test:|^wrote"
done

echo "=== summary ===" | tee -a "$LOG"
grep -E "^test:" "$LOG" | tee -a "$LOG"
