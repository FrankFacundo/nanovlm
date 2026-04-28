"""Autoregressive generation engine: sampling + KV-cache batched generate."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovlm.models.qwen3_5 import HybridCache


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample one token per row from ``logits`` of shape ``[B, V]``. Returns ``[B, 1]``."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        keep = (cumulative - probs) <= top_p
        keep[:, 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        return sorted_idx.gather(-1, sampled)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


@dataclass
class GenerateOutput:
    sequences: torch.Tensor
    new_token_logprobs: torch.Tensor | None = None
    eos_mask: torch.Tensor | None = None


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    return_logprobs: bool = False,
    use_cache: bool = True,
    extra_inputs: dict | None = None,
) -> GenerateOutput:
    """Greedy/top-k/top-p sampling. Supports KV-cache for the text-only path.

    For multimodal (Qwen3_5ForConditionalGeneration), pass ``extra_inputs``
    with ``pixel_values``/``image_grid_thw``/``mm_token_type_ids`` and they
    will be forwarded on the first step only.
    """
    model.eval()
    device = input_ids.device
    bsz = input_ids.size(0)
    eos_set = set()
    if eos_token_id is not None:
        eos_set = set([eos_token_id]) if isinstance(eos_token_id, int) else set(eos_token_id)

    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    past = None
    logprobs: list[torch.Tensor] = []

    cur_input = input_ids
    extra = extra_inputs.copy() if extra_inputs else {}
    out_seq = input_ids

    for step in range(max_new_tokens):
        if use_cache and past is not None:
            forward_input = cur_input[:, -1:]
            forward_extra = {}
        else:
            forward_input = cur_input
            forward_extra = extra
            extra = {}

        out = model(
            input_ids=forward_input,
            past_key_values=past,
            use_cache=use_cache,
            **forward_extra,
        )
        past = out["past_key_values"] if use_cache else None
        next_logits = out["logits"][:, -1, :]
        next_token = sample_next(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)

        if return_logprobs:
            lp = torch.log_softmax(next_logits.float(), dim=-1).gather(-1, next_token).squeeze(-1)
            logprobs.append(lp)

        if eos_set:
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, next(iter(eos_set))),
                next_token,
            )
            for eid in eos_set:
                finished = finished | (next_token.squeeze(-1) == eid)

        out_seq = torch.cat([out_seq, next_token], dim=-1)
        cur_input = out_seq if not use_cache else next_token

        if eos_set and bool(finished.all().item()):
            break

    return GenerateOutput(
        sequences=out_seq,
        new_token_logprobs=torch.stack(logprobs, dim=-1) if logprobs else None,
        eos_mask=finished if eos_set else None,
    )
