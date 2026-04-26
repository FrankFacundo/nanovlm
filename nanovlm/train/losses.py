"""Loss functions for SFT, preference optimization, and RLVR."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long).clone()
    labels[labels == -1] = -100
    if loss_mask is not None:
        labels = labels.masked_fill(loss_mask.to(logits.device) == 0, -100)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long).clone()
    valid = labels >= 0
    if loss_mask is not None:
        valid = valid & (loss_mask.to(logits.device).bool())
    safe_labels = labels.masked_fill(~valid, 0)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (logp * valid).sum(dim=-1)


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
    return loss, {
        "dpo_margin": float(policy_margin.detach().mean().cpu()),
        "dpo_ref_margin": float(ref_margin.detach().mean().cpu()),
        "dpo_acc": float((logits.detach() > 0).float().mean().cpu()),
    }


def mpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    chosen_ce: torch.Tensor,
    beta: float = 0.1,
    quality_weight: float = 0.05,
    sft_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    dpo, metrics = dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=beta)
    quality = F.softplus(-(chosen_logp - rejected_logp)).mean()
    loss = dpo + quality_weight * quality + sft_weight * chosen_ce
    metrics.update({
        "mpo_quality": float(quality.detach().cpu()),
        "mpo_sft": float(chosen_ce.detach().cpu()),
    })
    return loss, metrics


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6, zscore: bool = False) -> torch.Tensor:
    if rewards.numel() % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    adv = grouped - grouped.mean(dim=1, keepdim=True)
    if zscore:
        adv = adv / grouped.std(dim=1, keepdim=True).clamp_min(eps)
    return adv.reshape_as(rewards)


def grpo_policy_loss(
    token_logp: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    ref_token_logp: torch.Tensor | None = None,
    kl_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = token_mask.to(token_logp.device).float()
    denom = mask.sum().clamp_min(1.0)
    pg = -((token_logp * advantages.to(token_logp.device).unsqueeze(-1)) * mask).sum() / denom
    kl = torch.zeros((), device=token_logp.device)
    if ref_token_logp is not None and kl_coef > 0:
        kl = ((token_logp - ref_token_logp.to(token_logp.device)) * mask).sum() / denom
    loss = pg + kl_coef * kl
    return loss, {"rl_pg": float(pg.detach().cpu()), "rl_kl": float(kl.detach().cpu())}
