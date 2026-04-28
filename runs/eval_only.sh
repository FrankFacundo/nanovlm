#!/usr/bin/env bash
# Run the eval suite against an existing checkpoint.

set -euo pipefail

CHECKPOINT="${1:-}"
OUT="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/runs/eval_only"
DATA="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/data"
DEV="${NANOVLM_DEVICE:-cuda}"
DTYPE="${NANOVLM_DTYPE:-bfloat16}"

INCLUDE_HARD=""
if [[ "${2:-}" == "--include-hard" ]]; then
    INCLUDE_HARD="--include-hard"
fi

if [[ -n "$CHECKPOINT" ]]; then
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
else
    CHECKPOINT_ARG=""
fi

python -m scripts.eval_bench --data-root "$DATA" --out-dir "$OUT" \
    --device-type "$DEV" --dtype "$DTYPE" --limit 200 \
    $CHECKPOINT_ARG $INCLUDE_HARD

echo "DONE: $OUT"
