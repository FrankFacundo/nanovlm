# NanoVLM

Pure-PyTorch pretraining + post-training pipeline for Qwen3.5-0.8B (text + vision).
No `transformers`, `datasets`, `huggingface_hub`, `trl`, `peft`, or `accelerate` in any code path.

Inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat) but multimodal and SOTA-aligned: the model recipe follows Qwen3 / Qwen3.5 / DeepSeek-V4 / Kimi K2.6, the data mixtures follow the 2026 open-data report (FineWeb-Edu / DCLM / Nemotron-CC / Stack-Edu / FineMath / OBELICS / PixMo / Cambrian / Tulu-3 / OpenThoughts3 / DAPO-Math / MMPR), and the training mixture spans **pretrain → mid-train → vision align → SFT → reasoning SFT → DPO/MPO → RLVR (GRPO/DAPO) → agentic RL → benchmarks**.

## Layout

```
nanovlm/
├── nanovlm/
│   ├── models/qwen3_5/        # the model: pure PyTorch, no transformers dep
│   │   ├── model.py decoder.py attention.py linear_attention.py
│   │   ├── vision.py rotary.py layers.py cache.py
│   │   ├── tokenizer.py image_processor.py weights.py config.py
│   │   └── chat_template.py    # chat + tool-call + thinking grammar
│   ├── train/
│   │   ├── common.py checkpoint.py schedule.py optim.py engine.py rollout.py
│   │   ├── losses.py            # CE, DPO, MPO, DAPO, GRPO, KL
│   │   ├── verifiers.py         # numeric, exact-match, MATH-eq, IFEval, pytest
│   │   ├── download.py          # HF parquet shards via urllib (no HF lib)
│   │   ├── report.py plots.py   # JSONL + W&B + HTML/MD report cards w/ inline SVG
│   │   ├── data/                # streaming, packing, mixture, multimodal, preference, rlvr, chat
│   │   └── tools/               # python sandbox, filesystem, web_search, browser, image_ops
│   └── eval/                    # Task / Runner / 16 benchmark tasks (12 simple + 4 hard)
├── scripts/                    # CLI entry points (one per training stage)
├── configs/                    # YAML mixtures: pretrain S1/S2/S3, midtrain, vision_align,
│                               # sft general / reasoning, preference text / multimodal,
│                               # rlvr math+IF, agent_rl
├── runs/                       # speedrun_mps.sh, speedrun_cuda.sh, continued_pretrain.sh,
│                               # eval_only.sh
└── tests/                      # pytest -q (CPU only; no network needed)
```

## Setup

```bash
pip install -r requirements.txt

# (Recommended) symlink or place Qwen3.5-0.8B weights at ~/Models/Qwen/Qwen3.5-0.8B/
# OR auto-download via the dataset downloader (see below).
```

Hardware:
- **CUDA**: bf16 + DDP via `torchrun --standalone --nproc_per_node=N`. Optional `--compile`.
- **MPS** (Apple Silicon): single-process, fp32 default. M3 Max with 128 GB unified memory
  fits full-parameter SFT/DPO/RLVR for the 0.8 B model with small batches.
  See [notes.md](notes.md) for time estimates.
- **CPU**: smoke testing only.

## Sanity check

```bash
python -m pytest -q                     # 65 tests, ~5 s on CPU, no network
bash runs/speedrun_mps.sh               # full pipeline on tiny model + 3-row fixtures (~10-30 min)
```

## Download datasets (with a 100 GB cap)

The downloader resolves HF parquet shard URLs via the public REST API and
streams each shard with `urllib`. Permissive licenses only by default
(`ODC-BY`, `CC-BY-4.0`, `Apache-2.0`, `MIT`). Add `--allow-noncommercial` or
`--allow-openai-synthetic` to opt in.

```bash
# Plan only
python -m scripts.download_data --dry-run --max-download-gb 100

# Full download + Qwen3.5-0.8B safetensors
python -m scripts.download_data --max-download-gb 100 --download-weights

# Just the eval suites
python -m scripts.download_data --max-download-gb 1 --include eval
```

Bucket allocation (proportional under the global cap):

| Bucket | Datasets |
|---|---|
| text-pretrain (~45 GB) | FineWeb-Edu (sample-10BT), DCLM-Baseline, FineWeb-2-eng, peS2o |
| math-pretrain (~12 GB) | FineMath-4+, MegaMath, Nemotron-CC-Math |
| code-pretrain (~6 GB)  | Stack-Edu, Magicoder OSS-Instruct |
| reasoning (~5 GB)      | OpenThoughts3-1.2M, OpenR1-Math-220k, s1K-1.1 |
| sft-text (~8 GB)       | Tulu-3-SFT, SmolTalk, OASST1, Aya, OpenMathInstruct-2, OpenCodeReasoning |
| preference-text (~0.5) | UltraFeedback (cleaned), HelpSteer3 |
| rlvr (~0.1)            | RLVR-GSM-MATH-IF-Mixed-Constraints, DAPO-Math-17k |
| vlm-pretrain (~14 GB)  | OBELICS, PixMo-Cap, Recap-DataComp, PixelProse |
| vlm-sft (~9 GB)        | Cambrian-10M, the-cauldron, Docmatix, PixMo-{Docs,Charts,Clocks}, LLaVA-OneVision |
| vlm-pref (~0.5)        | MMPR-v1.2, RLAIF-V |
| eval (<1)              | MMLU, GSM8K, MATH-500, HumanEval, IFEval, ARC, MMMU, ChartQA, DocVQA, AI2D, V*-bench, DeepSearchQA, HLE, SWE-Multilingual |

## Training pipeline

### Pretrain (stage S1 / S2 / S3)

```bash
# Tiny smoke on MPS
python -m scripts.pretrain --tiny --text-only --init scratch --stage S1 \
    --steps 100 --device-type mps --out-dir ~/.cache/nanovlm/runs/pretrain

# Full continued-pretrain on CUDA
torchrun --standalone --nproc_per_node=8 -m scripts.pretrain --stage S1 \
    --init checkpoint --steps 2000 --seq-len 4096 --total-batch-tokens 1048576 \
    --device-type cuda --dtype bfloat16 --compile
```

Stages:
- **S1** general: FineWeb-Edu / DCLM / FineWeb-2-eng / Stack-Edu / peS2o.
- **S2** STEM + code: FineMath / MegaMath / Nemotron-CC-Math / Stack-Edu / peS2o.
- **S3** long-context: peS2o full-doc + FineWeb-Edu + Stack-Edu (`seq_len=16384`).

### Mid-training (Dolmino-style annealing)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.midtrain --steps 500 \
    --device-type cuda --dtype bfloat16 --compile
```

### Vision alignment

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.vision_align --steps 1000 \
    --device-type cuda --dtype bfloat16
```

### SFT (text + multimodal)

```bash
python -m scripts.sft --init checkpoint --steps 1000 --seq-len 2048 \
    --thinking-ratio 0.25 --device-type cuda --dtype bfloat16
```

### Reasoning SFT (OpenThoughts3 / OpenR1-Math)

```bash
python -m scripts.reasoning_sft --steps 1000 --device-type cuda --dtype bfloat16
```

### Preference: DPO or InternVL-style MPO

```bash
python -m scripts.preference --algo dpo --data $DATA/preference-text/ultrafeedback_clean/*.parquet
python -m scripts.preference --algo mpo --quality-weight 0.05 --sft-weight 0.1 \
    --data $DATA/vlm-pref/mmpr_v12/*.parquet
```

### RLVR with full DAPO (decoupled clip + token-level loss + dynamic sampling)

```bash
python -m scripts.rlvr --algo dapo --eps-low 0.2 --eps-high 0.28 \
    --group-size 8 --max-new-tokens 512 \
    --data $DATA/rlvr/rlvr_mixed/*.parquet $DATA/rlvr/dapo_math_17k/*.parquet
```

### Agentic RL (real model rollouts with tool use)

```bash
python -m scripts.agent_rl --steps 200 --group-size 4 --max-turns 6 \
    --data $DATA/rlvr/dapo_math_17k/*.parquet
```

Tools available: `python` (sandboxed subprocess), `filesystem` (chrooted),
`web_search` (DuckDuckGo HTML by default; honors `TAVILY_API_KEY`/`SERPAPI_KEY`),
`browser` (urllib + readability), `image_ops` (PIL crop / zoom / grid / info).

## Evaluation

```bash
# Simple suite (no tools)
python -m scripts.eval_bench --tasks mmlu,arc_challenge,gsm8k,math,humaneval,ifeval,mmmu,chartqa,docvqa,ai2d,vstar_bench

# Plus the four hard tool-using benchmarks
python -m scripts.eval_bench --include-hard
```

Tasks and metrics:

| Task | Type | Metric |
|---|---|---|
| MMLU, ARC-{Easy,Challenge}, MMMU, AI2D, V*-Bench | loglikelihood MCQ | accuracy |
| GSM8K | generative | numeric_reward |
| MATH-500 | generative | math-equivalence (sympy if available) |
| HumanEval | generative | sandboxed pytest pass@1 |
| IFEval | generative | strict-constraint compliance |
| ChartQA | generative (multimodal) | relaxed-EM (5% tolerance) |
| DocVQA | generative (multimodal) | ANLS |
| **DeepSearchQA** | tool: web_search + browser | token F1 |
| **HLE-with-tools** | tool: web_search + browser + python | exact_match |
| **SWE-Multilingual** | tool: filesystem + python | patched + ran_tests |
| **V*-with-python** | tool: image_ops + python | accuracy |

The HLE dataset is gated; set `HF_TOKEN` in the environment to access it.
SWE-Multilingual reports `patched` + `ran_tests` for in-process scoring; for
official pass@1 numbers run the produced trajectories through SWE-Bench's
official harness.

## Reports

Every training script writes:
- `*.jsonl` per-step metrics
- `report.html` with inline SVG plots
- `report.md` nanochat-style report card (system info, git commit, run config,
  metric tables, embedded plots)

Standard charts auto-generated: `train_loss`, `val_loss`, `val_bpb`, `lr`,
`grad_norm`, `tokens_per_sec`, `mfu`, `dpo_margin`, `dpo_acc`, `policy_loss`,
`approx_kl`, `clip_frac`, `reward`, `pass_at_1`. The eval bench emits a bar
chart of all benchmark scores.

W&B is opt-in: pass `--wandb --wandb-project foo` (lazy import; not required).

## Speedruns

- **`runs/speedrun_mps.sh`** — full pipeline on tiny model in ~10–30 min on M3 Max.
- **`runs/speedrun_cuda.sh`** — full real pipeline on 8x H100 (~3–6 h continued-pretrain).
- **`runs/continued_pretrain.sh`** — single-workstation realistic path: download (30 GB) → continued pretrain → SFT → DPO → light RLVR → eval.
- **`runs/eval_only.sh`** — bench an existing checkpoint (`bash runs/eval_only.sh path/to/model_*.safetensors --include-hard`).

## Inference (chat)

```bash
# REPL
python -m scripts.chat_cli --enable-thinking

# Tiny stdlib HTTP UI
python -m scripts.chat_web --port 8088
```

The legacy `inference/qwen3_5-0_8B/qwen3_5_torch.py` script and parity tests
remain in place via a 3-line compat shim that re-exports the model from
`nanovlm.models.qwen3_5`.

## What's intentionally out of scope

- FlashAttention-3 kernels (model uses SDPA; FA3 is a Hopper follow-up).
- FP8 quantization (H100+ only).
- Custom RustBPE tokenizer training (we reuse the Qwen2 tokenizer).
- Multi-node distribution beyond `torchrun --standalone`.
- A React frontend.
- Fully-from-scratch 1 B-parameter pretraining as the default path (the
  pipeline supports it; speedruns assume `--init checkpoint`).

## References

- Llama 3 — https://arxiv.org/abs/2407.21783
- Tulu 3 — https://arxiv.org/abs/2411.15124
- OLMo 2 — https://arxiv.org/abs/2501.00656
- DCLM — https://arxiv.org/abs/2406.11794
- FineWeb — https://arxiv.org/abs/2406.17557
- Nemotron-CC — https://arxiv.org/abs/2412.02595
- InternVL 3 — https://arxiv.org/abs/2504.10479
- Molmo / PixMo — https://arxiv.org/abs/2409.17146
- DAPO — https://arxiv.org/abs/2503.14476
- OpenThoughts3 — https://arxiv.org/abs/2506.04178
