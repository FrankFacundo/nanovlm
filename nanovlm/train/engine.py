"""Small autoregressive rollout engine for training and evaluation."""

from __future__ import annotations

import torch


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        keep = probs.cumsum(dim=-1) - probs <= top_p
        keep[:, 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        return sorted_idx.gather(-1, sampled)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int = 20,
    top_p: float = 1.0,
) -> torch.Tensor:
    model.eval()
    out = input_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=out)["logits"][:, -1, :]
        nxt = sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        out = torch.cat([out, nxt], dim=1)
        if eos_token_id is not None and bool((nxt == eos_token_id).all()):
            break
    return out
