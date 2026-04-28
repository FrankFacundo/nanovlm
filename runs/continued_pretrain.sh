#!/usr/bin/env bash
# Most realistic single-workstation path: load Qwen3.5-0.8B, continue
# pretraining on a small slice, then SFT + DPO + light RLVR + eval.

set -euo pipefail

OUT="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/runs/continued"
DATA="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/data"
DEV="${NANOVLM_DEVICE:-cuda}"
DTYPE="${NANOVLM_DTYPE:-bfloat16}"
mkdir -p "$OUT" "$DATA"

echo "==> downloads (cap 30 GB; weights included)"
python -m scripts.download_data --max-download-gb 30 --out-dir "$DATA" --download-weights \
    --include text-pretrain math-pretrain code-pretrain sft-text reasoning preference-text rlvr eval

echo "==> continued pretrain"
python -m scripts.pretrain --stage S1 --init checkpoint --data-root "$DATA" \
    --steps 500 --seq-len 2048 --total-batch-tokens 32768 \
    --device-type "$DEV" --dtype "$DTYPE" --out-dir "$OUT/pretrain"

echo "==> SFT"
python -m scripts.sft --init checkpoint --data-root "$DATA" \
    --steps 1000 --seq-len 2048 --device-type "$DEV" --dtype "$DTYPE" --out-dir "$OUT/sft"

echo "==> DPO"
python -m scripts.preference --init checkpoint --algo dpo \
    --data "$DATA/preference-text/ultrafeedback_clean/"*.parquet \
    --steps 400 --device-type "$DEV" --dtype "$DTYPE" --out-dir "$OUT/dpo"

echo "==> RLVR with DAPO"
python -m scripts.rlvr --init checkpoint --algo dapo \
    --data "$DATA/rlvr/rlvr_mixed/"*.parquet \
    --steps 200 --group-size 4 --max-new-tokens 256 \
    --device-type "$DEV" --dtype "$DTYPE" --out-dir "$OUT/rlvr"

echo "==> eval (simple suite)"
python -m scripts.eval_bench --data-root "$DATA" \
    --out-dir "$OUT/eval" --device-type "$DEV" --dtype "$DTYPE" --limit 100

echo "DONE: $OUT"
