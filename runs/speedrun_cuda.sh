#!/usr/bin/env bash
# Full pipeline on CUDA (single-node, multi-GPU via torchrun).
# Realistic continued-pretrain + post-train arc on Qwen3.5-0.8B init.
# End-to-end ~3-6 hours on 8x A100 / H100; longer with --from-scratch.

set -euo pipefail

NPROC="${NPROC:-8}"
OUT="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/runs/speedrun_cuda"
DATA="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/data"
mkdir -p "$OUT" "$DATA"

echo "==> download data + Qwen weights (cap 100 GB)"
python -m scripts.download_data --max-download-gb 100 --out-dir "$DATA" --download-weights

TORCHRUN="torchrun --standalone --nproc_per_node=${NPROC}"

echo "==> pretrain S1 (general)"
$TORCHRUN -m scripts.pretrain --stage S1 --init checkpoint --data-root "$DATA" \
    --steps 2000 --seq-len 4096 --batch-size 1 --total-batch-tokens 1048576 \
    --device-type cuda --dtype bfloat16 --compile --out-dir "$OUT/pretrain_S1"

echo "==> pretrain S2 (STEM + code)"
$TORCHRUN -m scripts.pretrain --stage S2 --init checkpoint --data-root "$DATA" \
    --steps 1000 --seq-len 4096 --batch-size 1 --total-batch-tokens 524288 \
    --device-type cuda --dtype bfloat16 --compile --out-dir "$OUT/pretrain_S2"

echo "==> midtrain (Dolmino-style annealing)"
$TORCHRUN -m scripts.midtrain --init checkpoint --data-root "$DATA" \
    --steps 500 --seq-len 4096 --batch-size 1 --total-batch-tokens 524288 \
    --device-type cuda --dtype bfloat16 --compile --out-dir "$OUT/midtrain"

echo "==> vision alignment"
$TORCHRUN -m scripts.vision_align --init checkpoint --data-root "$DATA" \
    --steps 1000 --device-type cuda --dtype bfloat16 --out-dir "$OUT/vision_align"

echo "==> SFT (general, text + multimodal)"
$TORCHRUN -m scripts.sft --init checkpoint --data-root "$DATA" \
    --steps 2000 --seq-len 2048 --device-type cuda --dtype bfloat16 \
    --thinking-ratio 0.25 --out-dir "$OUT/sft"

echo "==> reasoning SFT (OpenThoughts3 / OpenR1-Math)"
$TORCHRUN -m scripts.reasoning_sft --steps 1000 --device-type cuda --dtype bfloat16 \
    --out-dir "$OUT/reasoning_sft"

echo "==> MPO (multimodal preference)"
$TORCHRUN -m scripts.preference --init checkpoint --algo mpo \
    --data "$DATA/preference-text/ultrafeedback_clean/"*.parquet \
    --steps 800 --device-type cuda --dtype bfloat16 --out-dir "$OUT/mpo"

echo "==> RLVR with DAPO"
$TORCHRUN -m scripts.rlvr --init checkpoint --algo dapo \
    --data "$DATA/rlvr/rlvr_mixed/"*.parquet "$DATA/rlvr/dapo_math_17k/"*.parquet \
    --steps 500 --group-size 8 --max-new-tokens 512 \
    --device-type cuda --dtype bfloat16 --out-dir "$OUT/rlvr"

echo "==> agentic RL"
$TORCHRUN -m scripts.agent_rl --init checkpoint \
    --data "$DATA/rlvr/dapo_math_17k/"*.parquet \
    --steps 200 --group-size 4 --max-turns 6 \
    --device-type cuda --dtype bfloat16 --out-dir "$OUT/agent_rl"

echo "==> eval suite (simple + hard)"
python -m scripts.eval_bench --include-hard --data-root "$DATA" \
    --out-dir "$OUT/eval" --device-type cuda --dtype bfloat16 --limit 200

echo "==> aggregate reports"
python -m scripts.report --out-dir "$OUT" --title "speedrun_cuda final" || true

echo "DONE: $OUT"
