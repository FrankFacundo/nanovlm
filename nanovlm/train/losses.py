"""Loss functions for SFT, preference (DPO/MPO), and RL (GRPO/DAPO).

References:
  - DPO: Rafailov et al. 2023, https://arxiv.org/abs/2305.18290
  - MPO (InternVL): Wang et al. 2024, https://arxiv.org/abs/2411.10442
  - GRPO: Shao et al. 2024 (DeepSeek-Math), https://arxiv.org/abs/2402.03300
  - DAPO: Yu et al. 2025, https://arxiv.org/abs/2503.14476
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------- Cross-entropy (SFT / pretrain) ----------------------------------

def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token-level cross entropy on ``logits`` of shape ``[B, T, V]``.

    ``labels`` of shape ``[B, T]``; positions where ``labels == -100`` or
    ``loss_mask == 0`` are ignored. Output is the mean over unmasked tokens.
    """
    labels = labels.to(device=logits.device, dtype=torch.long).clone()
    labels[labels < 0] = -100
    if loss_mask is not None:
        labels = labels.masked_fill(loss_mask.to(logits.device) == 0, -100)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )


def shift_for_causal(input_ids: torch.Tensor, labels: torch.Tensor | None = None):
    """Standard next-token shift: predict ``input_ids[:, 1:]`` from ``logits[:, :-1]``."""
    if labels is None:
        labels = input_ids
    return labels[:, 1:].contiguous()


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum of per-token log-probs of ``labels`` under ``logits``. Returns ``[B]``.

    Positions where ``labels < 0`` or ``loss_mask == 0`` contribute zero.
    """
    labels = labels.to(device=logits.device, dtype=torch.long).clone()
    valid = labels >= 0
    if loss_mask is not None:
        valid = valid & (loss_mask.to(logits.device).bool())
    safe = labels.masked_fill(~valid, 0)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * valid).sum(dim=-1)


def per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of ``labels`` under ``logits``. Returns ``[B, T]``.

    No mask; caller applies it. ``labels < 0`` positions are clamped to 0 to
    avoid OOB and should be masked downstream.
    """
    safe = labels.clamp_min(0)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return logp


# ---------- DPO / MPO ------------------------------------------------------

def dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_margin = chosen_logp - rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()
    metrics = {
        "dpo_margin": float(policy_margin.detach().mean().cpu()),
        "dpo_ref_margin": float(ref_margin.detach().mean().cpu()),
        "dpo_acc": float((policy_margin > ref_margin).float().mean().cpu()),
    }
    return loss, metrics


def mpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    chosen_sft_loss: torch.Tensor,
    *,
    beta: float = 0.1,
    quality_weight: float = 0.05,
    sft_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InternVL Mixed Preference Optimization: DPO + Bradley-Terry quality + SFT-on-chosen.

    The quality term encourages the policy's absolute log-prob on the chosen
    response to rise (a Bradley-Terry positive-pair likelihood) and on the
    rejected to fall, independent of the reference. The SFT term keeps the
    policy from drifting away from the chosen answer's surface form.
    """
    dpo, dpo_metrics = dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=beta)
    quality = -F.logsigmoid(chosen_logp - rejected_logp).mean()
    sft = chosen_sft_loss
    loss = dpo + quality_weight * quality + sft_weight * sft
    return loss, {
        **dpo_metrics,
        "mpo_dpo": float(dpo.detach().cpu()),
        "mpo_quality": float(quality.detach().cpu()),
        "mpo_sft": float(sft.detach().cpu()),
    }


# ---------- GRPO / DAPO ----------------------------------------------------

def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Within-group reward standardization (GRPO/DAPO) on ``rewards`` of shape ``[N]``.

    ``N`` must be a multiple of ``group_size``. Returns ``[N]`` advantages.
    """
    n = rewards.numel()
    assert n % group_size == 0, f"rewards size {n} not divisible by group_size {group_size}"
    grouped = rewards.view(-1, group_size).float()
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, keepdim=True).clamp_min(eps)
    advantages = ((grouped - mean) / std).view(-1)
    return advantages


def grpo_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard PPO-style clipped surrogate (single ε)."""
    return _clipped_surrogate(new_logp, old_logp, advantages, mask, eps_low=eps, eps_high=eps)


def dapo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DAPO (Yu et al. 2025): decoupled clip + token-level normalization.

    ``eps_high > eps_low`` widens the upper clip on positive-advantage tokens
    to encourage exploration; ``eps_low`` keeps the negative-advantage clip
    tight to limit destructive updates. Loss is summed over tokens and divided
    by the total unmasked-token count (token-level), not by sample count
    (sample-level), preventing long sequences from being down-weighted.
    """
    return _clipped_surrogate(new_logp, old_logp, advantages, mask, eps_low=eps_low, eps_high=eps_high)


def _clipped_surrogate(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps_low: float,
    eps_high: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    new_logp = new_logp.float()
    old_logp = old_logp.float().detach()
    mask = mask.float()
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1).expand_as(new_logp)
    advantages = advantages.float().detach()

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    pos = advantages > 0
    upper = 1.0 + eps_high
    lower = 1.0 - eps_low
    clipped_ratio = torch.where(pos, ratio.clamp(max=upper), ratio.clamp(min=lower))
    clipped = clipped_ratio * advantages

    loss_terms = -torch.minimum(unclipped, clipped) * mask
    denom = mask.sum().clamp_min(1.0)
    loss = loss_terms.sum() / denom

    clip_mask = ((ratio > upper) & pos) | ((ratio < lower) & ~pos)
    clip_frac = float(((clip_mask.float() * mask).sum() / denom).detach().cpu())
    approx_kl = float((((ratio - 1.0) - log_ratio) * mask).sum().detach().cpu() / float(denom.detach().cpu()))

    return loss, {
        "policy_loss": float(loss.detach().cpu()),
        "clip_frac": clip_frac,
        "approx_kl": approx_kl,
        "ratio_mean": float((ratio * mask).sum().detach().cpu() / float(denom.detach().cpu())),
    }


def grpo_kl_penalty(
    new_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Schulman approx KL: ``exp(r) - 1 - r`` where ``r = ref_logp - new_logp``.

    Always non-negative; matches the form used in DeepSeek-Math GRPO.
    """
    diff = ref_logp.float().detach() - new_logp.float()
    kl = torch.exp(diff) - 1.0 - diff
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom
