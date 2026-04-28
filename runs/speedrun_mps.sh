#!/usr/bin/env bash
# End-to-end smoke pipeline on Apple Silicon MPS using --tiny everywhere.
# Runs every script in sequence to verify the full stack works locally.
# Total runtime: ~10-30 minutes on an M3 Max.

set -euo pipefail

OUT="${NANOVLM_BASE_DIR:-$HOME/.cache/nanovlm}/runs/speedrun_mps"
DEV="${NANOVLM_DEVICE:-mps}"
mkdir -p "$OUT"

DATA="$OUT/data"
mkdir -p "$DATA"
cat > "$DATA/train.jsonl" <<'EOF'
{"text": "Vision-language models combine text and image patch embeddings."}
{"text": "Muon orthogonalizes 2-D weight matrices via Newton-Schulz iteration."}
{"text": "Reinforcement learning with verifiable rewards uses programmatic checkers."}
EOF
cat > "$DATA/sft.jsonl" <<'EOF'
{"prompt": "What is 2+2?", "response": "4"}
{"prompt": "Capital of France?", "response": "Paris"}
{"messages": [{"role": "user", "content": "Say hi."}, {"role": "assistant", "content": "Hi."}]}
EOF
cat > "$DATA/pref.jsonl" <<'EOF'
{"prompt": "Capital of France?", "chosen": "Paris", "rejected": "Berlin"}
{"prompt": "2+2?", "chosen": "4", "rejected": "5"}
EOF
cat > "$DATA/rlvr.jsonl" <<'EOF'
{"question": "Answer with the exact number: 2+2", "answer": "4"}
{"question": "Answer with the exact number: 6*7", "answer": "42"}
EOF

echo "==> pretrain"
python -m scripts.pretrain --tiny --text-only --init scratch --data "$DATA/train.jsonl" --steps 5 --seq-len 64 --batch-size 1 --total-batch-tokens 64 --device-type "$DEV" --out-dir "$OUT/pretrain" --log-every 1 --save-every 0

echo "==> sft"
python -m scripts.sft --tiny --text-only --init scratch --data "$DATA/sft.jsonl" --steps 5 --seq-len 64 --device-type "$DEV" --out-dir "$OUT/sft" --log-every 1 --save-every 0

echo "==> preference (DPO)"
python -m scripts.preference --tiny --text-only --init scratch --data "$DATA/pref.jsonl" --steps 5 --max-prompt-len 16 --max-response-len 8 --device-type "$DEV" --out-dir "$OUT/preference" --log-every 1 --save-every 0 --algo dpo

echo "==> rlvr (DAPO)"
python -m scripts.rlvr --tiny --text-only --init scratch --data "$DATA/rlvr.jsonl" --steps 3 --group-size 4 --max-new-tokens 8 --device-type "$DEV" --out-dir "$OUT/rlvr" --log-every 1 --save-every 0 --algo dapo

echo "==> agent_rl"
python -m scripts.agent_rl --tiny --text-only --init scratch --data "$DATA/rlvr.jsonl" --steps 2 --group-size 2 --max-turns 2 --max-new-tokens-per-turn 16 --device-type "$DEV" --out-dir "$OUT/agent_rl" --log-every 1 --save-every 0 --no-web

echo "==> reports"
python -m scripts.report --out-dir "$OUT/pretrain" --title "speedrun pretrain" || true
python -m scripts.report --out-dir "$OUT/sft" --title "speedrun sft" || true

echo "DONE: $OUT"
