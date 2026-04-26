# NanoVLM Training Time Notes

## Machine Used For This Estimate

Command used:

```bash
system_profiler SPHardwareDataType SPDisplaysDataType
sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
python - <<'PY'
import platform, torch
print(platform.mac_ver()[0])
print(platform.machine())
print(torch.__version__)
print(torch.backends.mps.is_available())
PY
```

Detected system:

- MacBook Pro `Mac15,8`
- Apple M3 Max
- 16 CPU cores: 12 performance, 4 efficiency
- 40-core Apple GPU
- 128 GB unified memory
- macOS `26.3.1`
- Python `3.13.11`
- PyTorch `2.10.0`
- CUDA: unavailable
- MPS: available

Private identifiers from `system_profiler` are intentionally omitted.

## Important Assumptions

These estimates are for the current repo implementation on MPS, using full-parameter training. They are not CUDA-cluster estimates.

The full Qwen3.5 local config is about:

- `1,006,672,704` trainable parameters in text-only mode.
- FP32 model weights alone are about 4 GB.
- Full training also needs gradients, optimizer state, activations, temporary tensors, and sometimes a reference model.

The 128 GB MacBook should fit small full-parameter text-only runs, but the bottleneck is compute speed, not RAM.

For rough planning I assume:

- Full 0.8B text-only MPS training at `seq_len=512`, `batch_size=1`: about `15-50 tokens/sec`.
- Preference training is about `2.5-3.5x` slower than SFT because it evaluates chosen, rejected, and reference responses.
- Current RLVR generation is debug-grade and does not use an optimized KV-cache rollout service, so it can be much slower than normal supervised training.
- Full multimodal image training is slower and more memory-heavy than text-only, depending on image resolution and number of image tokens.

## Your Example Command

Command:

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

Tokens trained:

```text
100 steps * 512 tokens = 51,200 tokens
```

Estimated wall time:

```text
17-60 minutes, plus model initialization/checkpoint overhead
```

This is a debug run. It is enough to test loss curves, checkpointing, and code changes. It is not enough to train a useful base model from scratch.

## Estimated Times By Training Part

| Part | Typical Debug Size On This Mac | Estimated Time | Practical Meaning |
|---|---:|---:|---|
| Pretrain S1/S2/S3, text-only | `100 steps * 512 tokens = 51k tokens` | `17-60 min` | Code/debug run only |
| Pretrain S1/S2/S3, text-only | `10k steps * 512 tokens = 5.1M tokens` | `1-4 days` | Small experiment, still not a strong base model |
| Pretrain from scratch, meaningful 0.8B run | `~10B-20B tokens` | `6-35 years` | Not realistic on one M3 Max |
| SFT | `10k examples * 512 tokens = 5.1M tokens` | `1-4 days` | Useful small behavior experiment |
| SFT | `100k examples * 512 tokens = 51M tokens` | `12-40 days` | Large local run; possible but slow |
| Preference DPO/MPO | `10k prompt pairs * 512 tokens` | `3-12 days` | Slower because of reference/chosen/rejected passes |
| Preference DPO/MPO | `100k prompt pairs * 512 tokens` | `1-4 months` | Not ideal on MPS |
| RLVR | `1k prompts, group_size=4, 128 generated tokens` | `days to weeks` | Current rollout path is for debugging |
| Agentic training | Current synthetic sandbox script | `seconds to minutes` | No model rollout yet |
| Agentic training with model rollouts | `100-1k tasks, 10-100 tool steps each` | `weeks to months` | Needs optimized generation, caching, and resumable rollouts |

## Why Pretraining From Scratch Is Not Practical Here

A rough compute-optimal 0.8B pretraining target is at least:

```text
10-20 tokens per parameter
~10B-20B tokens for a ~1B parameter model
```

At `15-50 tokens/sec`:

```text
10B tokens / 50 tok/s = ~6.3 years
20B tokens / 15 tok/s = ~42 years
```

So this Mac is good for:

- testing the full training stack,
- debugging MPS correctness,
- tiny pretraining ablations,
- small SFT runs,
- verifying preference/RL code,
- building datasets and reports.

It is not good for full from-scratch 0.8B pretraining.

## Recommended Local Workflow

Start with tiny tests:

```bash
python -m scripts.pretrain \
  --tiny --text-only \
  --steps 20 \
  --seq-len 128 \
  --batch-size 1 \
  --total-batch-tokens 512 \
  --device-type mps
```

Then test the full text model for a short run:

```bash
python -m scripts.pretrain \
  --text-only \
  --init scratch \
  --stage S1 \
  --steps 100 \
  --seq-len 512 \
  --batch-size 1 \
  --total-batch-tokens 512 \
  --device-type mps
```

For useful local work, prefer checkpoint initialization plus SFT:

```bash
python -m scripts.sft \
  --init checkpoint \
  --text-only \
  --steps 1000 \
  --seq-len 512 \
  --batch-size 1 \
  --device-type mps
```

For serious pretraining, move to CUDA:

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
  --compile
```

## Best Use Of This MacBook

Use the M3 Max machine as a high-memory development box:

- validate loaders and masks,
- test Qwen architecture changes,
- test optimizer stability,
- run small SFT experiments,
- debug preference/RL losses,
- generate reports,
- prepare datasets for a future CUDA run.

For training quality, the best path is:

1. Use `--init checkpoint`.
2. Run small SFT and preference experiments locally.
3. Keep from-scratch pretraining on this Mac limited to smoke tests.
4. Use CUDA for any run above a few million tokens.
