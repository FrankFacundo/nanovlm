import math

import pytest
import torch

from nanovlm.train.losses import (
    dapo_loss,
    dpo_loss,
    group_advantages,
    grpo_kl_penalty,
    masked_ce_loss,
    mpo_loss,
    sequence_logprobs,
)


def test_masked_ce_loss_ignores_negative_labels():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[0, 1, -100, 2], [3, -100, 4, 0]])
    loss = masked_ce_loss(logits, labels)
    # 6 valid tokens; verify loss matches torch.nn.functional.cross_entropy on valid only
    valid = labels >= 0
    expected = torch.nn.functional.cross_entropy(
        logits[valid].float(),
        labels[valid].long(),
    )
    assert torch.allclose(loss, expected, atol=1e-5)


def test_sequence_logprobs_matches_manual_sum():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 4)
    labels = torch.tensor([[1, 2, 0]])
    expected = torch.log_softmax(logits.float(), dim=-1)
    expected = expected[0, 0, 1] + expected[0, 1, 2] + expected[0, 2, 0]
    out = sequence_logprobs(logits, labels)
    assert torch.allclose(out.squeeze(), expected, atol=1e-5)


def test_dpo_loss_sign_and_acc():
    chosen = torch.tensor([1.0, 1.5])
    rejected = torch.tensor([0.0, 0.5])
    ref_chosen = torch.tensor([0.5, 1.0])
    ref_rejected = torch.tensor([0.5, 1.0])
    loss, m = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=1.0)
    assert loss.item() < math.log(2)  # less than uniform
    assert m["dpo_acc"] == 1.0


def test_mpo_loss_components_decompose():
    chosen = torch.tensor([1.0])
    rejected = torch.tensor([0.0])
    ref_c = torch.tensor([0.5])
    ref_r = torch.tensor([0.5])
    sft = torch.tensor(0.7)
    loss, m = mpo_loss(chosen, rejected, ref_c, ref_r, sft, beta=1.0, quality_weight=0.5, sft_weight=0.5)
    assert "mpo_dpo" in m and "mpo_quality" in m and "mpo_sft" in m
    expected = m["mpo_dpo"] + 0.5 * m["mpo_quality"] + 0.5 * m["mpo_sft"]
    assert math.isclose(loss.item(), expected, rel_tol=1e-4)


def test_group_advantages_zero_mean_within_group():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    adv = group_advantages(rewards, group_size=4)
    grouped = adv.view(-1, 4)
    assert torch.allclose(grouped.mean(-1), torch.zeros(2), atol=1e-5)


def test_dapo_decoupled_clip_widens_upper_for_positive_advantage():
    # Construct a single token where ratio is exactly above eps_high but below
    # what classic ε=0.2 would clip; DAPO with eps_high=0.28 should NOT clip.
    new_lp = torch.tensor([[math.log(1.25)]], requires_grad=True)
    old_lp = torch.tensor([[0.0]])
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    _, m_dapo = dapo_loss(new_lp, old_lp, adv, mask, eps_low=0.2, eps_high=0.28)
    # 1.25 < 1.28 → not clipped
    assert m_dapo["clip_frac"] == 0.0
    # Compare to GRPO ε=0.2: 1.25 > 1.20 → clipped
    from nanovlm.train.losses import grpo_policy_loss
    _, m_grpo = grpo_policy_loss(new_lp, old_lp, adv, mask, eps=0.2)
    assert m_grpo["clip_frac"] == 1.0


def test_dapo_token_level_normalization_handles_long_sequences():
    new_lp = torch.zeros(1, 100, requires_grad=True)
    old_lp = torch.zeros(1, 100)
    adv = torch.ones(1)
    mask = torch.ones(1, 100)
    loss, _ = dapo_loss(new_lp, old_lp, adv, mask)
    # all ratios = 1.0 → loss = -mean(adv) = -1.0
    assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)


def test_grpo_kl_is_nonneg():
    new = torch.randn(2, 4)
    ref = torch.randn(2, 4)
    mask = torch.ones(2, 4)
    kl = grpo_kl_penalty(new, ref, mask)
    assert kl.item() >= 0
