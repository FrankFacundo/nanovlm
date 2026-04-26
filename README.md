# NanoVLM Qwen3.5 Training

Pure-PyTorch training scripts for the local `Qwen3.5-0.8B` VLM port. The runtime training path does **not** use `transformers`, `datasets`, `huggingface_hub`, `trl`, `peft`, or `accelerate`.

The scripts are designed for two modes:

- Debug and iteration on CPU/MPS with `--tiny` or very small batches.
- CUDA/DDP runs with the full Qwen3.5 config and streaming packed loaders.

Default model assets are expected at:

```bash
/Users/frankfacundo/Models/Qwen/Qwen3.5-0.8B
```

Override with `--model-path /path/to/Qwen3.5-0.8B`.

## Setup

From the repo root:

```bash
python -m pip install -r requirements.txt
```

Optional HF parity tests are skipped by default because they require `transformers`. To run them explicitly:

```bash
NANOVLM_RUN_HF_PARITY=1 python -m pytest inference/qwen3_5-0_8B/tests -q
```

## Quick Smoke Tests

Use these first after any code change:

```bash
python -m pytest -q

python -m scripts.pretrain \
  --tiny --text-only --steps 2 --seq-len 16 --batch-size 1 \
  --total-batch-tokens 16 --device-type cpu

python -m scripts.sft \
  --tiny --text-only --steps 2 --seq-len 32 --batch-size 1 \
  --device-type cpu

python -m scripts.preference \
  --tiny --text-only --steps 2 --seq-len 32 --batch-size 1 \
  --device-type cpu

python -m scripts.rlvr \
  --tiny --text-only --steps 2 --group-size 2 \
  --max-prompt-len 16 --max-new-tokens 4 --device-type cpu

python -m scripts.agent_rl --steps 2
```

## Data

The downloader builds a permissive-source download plan and enforces a global byte cap.

Dry run:

```bash
python -m scripts.download_data \
  --dry-run \
  --max-download-gb 100 \
  --out-dir ~/.cache/nanovlm/data
```

Download:

```bash
python -m scripts.download_data \
  --max-download-gb 100 \
  --out-dir ~/.cache/nanovlm/data
```

Default policy excludes noncommercial and OpenAI-synthetic sources. To opt in:

```bash
python -m scripts.download_data --allow-noncommercial
python -m scripts.download_data --allow-openai-synthetic
```

Training scripts consume JSONL paths via `--data`. For pretraining, `--data` is optional: if omitted, the script auto-creates and uses:

```bash
~/.cache/nanovlm/data/train.jsonl
```

Pass your real files with `--data file1.jsonl file2.jsonl` when you are ready.

Minimal pretraining JSONL:

```jsonl
{"text":"A vision-language model can answer questions about images and text."}
{"text":"Muon is used for matrix weights and AdamW for embeddings and norms."}
```

Minimal SFT JSONL:

```jsonl
{"prompt":"What is 2+2?","response":"4"}
{"messages":[{"role":"user","content":"Say hi."},{"role":"assistant","content":"Hi."}]}
```

Minimal preference JSONL:

```jsonl
{"prompt":"Answer with the exact number: 3+5","chosen":"8","rejected":"9"}
```

Minimal RLVR JSONL:

```jsonl
{"question":"Answer with the exact number: 6*7","answer":"42"}
{"prompt":"Write a Python function add(a,b).","tests":"from main import add\nassert add(2,3)==5"}
```

## Pretraining

Default initialization is from scratch using the local Qwen3.5 config/tokenizer.

Tiny CPU/MPS debug:

```bash
python -m scripts.pretrain \
  --tiny --text-only \
  --steps 20 \
  --seq-len 128 \
  --batch-size 1 \
  --total-batch-tokens 512 \
  --device-type mps \
  --out-dir ~/.cache/nanovlm/runs/pretrain_debug
```

Full 0.8B text-side debug on MPS:

```bash
python -m scripts.pretrain \
  --text-only \
  --init scratch \
  --stage S1 \
  --steps 100 \
  --seq-len 512 \
  --batch-size 1 \
  --total-batch-tokens 512 \
  --device-type mps \
  --out-dir ~/.cache/nanovlm/runs/qwen35_s1_mps
```

Full model from checkpoint instead of scratch:

```bash
python -m scripts.pretrain \
  --init checkpoint \
  --steps 100 \
  --seq-len 512 \
  --batch-size 1 \
  --total-batch-tokens 512 \
  --device-type mps
```

CUDA DDP:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.pretrain \
  --init scratch \
  --stage S1 \
  --steps 10000 \
  --seq-len 2048 \
  --batch-size 1 \
  --total-batch-tokens 262144 \
  --device-type cuda \
  --dtype bfloat16 \
  --compile \
  --out-dir ~/.cache/nanovlm/runs/qwen35_s1_cuda
```

Stages are labels for the run recipe:

- `S1`: general text/multimodal foundation mix.
- `S2`: STEM, code, document/OCR-heavy mix.
- `S3`: long-context mix.

The script does not infer the mixture from the stage yet; pass the intended JSONL files with `--data`.

## SFT

Assistant-token-only supervised fine-tuning:

```bash
python -m scripts.sft \
  --init checkpoint \
  --data path/to/sft.jsonl \
  --steps 1000 \
  --seq-len 1024 \
  --batch-size 1 \
  --device-type mps \
  --out-dir ~/.cache/nanovlm/runs/qwen35_sft_mps
```

The loader supports:

- `{"prompt": "...", "response": "..."}`
- `{"question": "...", "answer": "..."}`
- `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`

Use `--thinking-ratio` to mix `<think>...</think>` style examples with empty non-thinking blocks.

## Preference Training

DPO/MPO-style preference optimization:

```bash
python -m scripts.preference \
  --init checkpoint \
  --data path/to/preferences.jsonl \
  --steps 1000 \
  --seq-len 1024 \
  --batch-size 1 \
  --device-type mps \
  --out-dir ~/.cache/nanovlm/runs/qwen35_pref_mps
```

Expected records:

```jsonl
{"prompt":"...","chosen":"better answer","rejected":"worse answer"}
```

The implementation keeps a frozen reference copy in memory. On MPS, start with `--tiny` or small `--seq-len` before a full 0.8B run.

## RLVR

GRPO/DAPO-style RL with programmatic verifiers:

```bash
python -m scripts.rlvr \
  --init checkpoint \
  --data path/to/rlvr.jsonl \
  --steps 500 \
  --group-size 4 \
  --max-prompt-len 256 \
  --max-new-tokens 128 \
  --temperature 1.0 \
  --device-type mps \
  --out-dir ~/.cache/nanovlm/runs/qwen35_rlvr_mps
```

Supported verifier fields:

- `answer`: numeric or exact-match reward.
- `constraints`: instruction-following constraints such as `must_contain`, `must_not_contain`, `max_words`.
- `tests`: Python unit-test reward.

## Agentic Sandbox

The current agentic script is a deterministic local sandbox smoke harness. It logs tool trajectories and validates replayable tool execution.

```bash
python -m scripts.agent_rl \
  --steps 10 \
  --out-dir ~/.cache/nanovlm/runs/agent_rl
```

It is ready for model-generated tool-call rollouts, but the default policy is intentionally synthetic.

## Eval And Reports

Evaluate loss and sample:

```bash
python -m scripts.eval \
  --init checkpoint \
  --text-only \
  --data path/to/val.jsonl \
  --eval-steps 10 \
  --seq-len 512 \
  --prompt "Describe what a VLM does in one sentence." \
  --device-type mps
```

Regenerate a report from JSONL metrics:

```bash
python -m scripts.report --out-dir ~/.cache/nanovlm/runs/qwen35_s1_mps
```

Each training run writes:

- `*.jsonl` metrics.
- `report.html` with inline SVG graphs.
- `checkpoints/model_*.pt`, `meta_*.json`, and optimizer state.

## Practical Notes

- MPS defaults to `float32` for stability. You can try `--dtype float16` after smoke tests pass.
- CUDA should use `--dtype bfloat16` when supported.
- `--tiny` keeps the real Qwen tokenizer vocabulary but uses a small model shape for fast debugging.
- `--text-only` skips the vision tower and is the fastest way to debug optimization.
- Full multimodal training requires batches with `pixel_values`, `image_grid_thw`, and `mm_token_type_ids`; the model API supports them, while the generic JSONL loaders are text-first.
- Old inference files remain under `inference/qwen3_5-0_8B`; new training imports should use `nanovlm.models.qwen3_5`.
