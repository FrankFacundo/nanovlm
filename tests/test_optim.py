import torch

from nanovlm.train.optim import (
    Muon,
    OptimConfig,
    _newton_schulz5,
    build_optimizer,
    set_lr_and_wd,
)


def test_newton_schulz_bounds_singular_values():
    """Newton-Schulz with Frobenius scaling pulls SVs into a tight band around 1.

    We check the band [0.5, 1.5] — Random gaussian SVs span much wider, so this
    confirms orthogonalization is happening even though the iteration does not
    drive SVs to exactly 1 in 5 steps under conservative scaling.
    """
    torch.manual_seed(0)
    g = torch.randn(64, 16)
    x = _newton_schulz5(g, steps=5)
    sv = torch.linalg.svdvals(x.float())
    assert sv.min().item() > 0.5
    assert sv.max().item() < 1.5


def test_muon_step_changes_weights():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    p.grad = torch.randn(8, 8)
    opt = Muon([p], lr=0.01)
    before = p.detach().clone()
    opt.step()
    assert not torch.allclose(p.detach(), before)


def test_combined_optimizer_splits_groups():
    model = torch.nn.Sequential(
        torch.nn.Embedding(50, 16),
        torch.nn.Linear(16, 32),
        torch.nn.LayerNorm(32),
        torch.nn.Linear(32, 5),
    )
    cfg = OptimConfig(matrix_lr=0.02, adam_lr=2e-4)
    opt = build_optimizer(model, cfg)
    kinds = [g["kind"] for g in opt.param_groups]
    assert kinds == ["muon", "adamw"]
    matrix = opt.param_groups[0]["params"]
    other = opt.param_groups[1]["params"]
    # Embedding goes to AdamW, Linear weights go to Muon
    assert all(p.dim() == 2 for p in matrix)
    # 2 Linear weights → matrix; embedding + 2 biases + 2 LayerNorm params → other
    assert len(matrix) == 2


def test_set_lr_and_wd_scales_each_group_independently():
    model = torch.nn.Linear(4, 4)
    cfg = OptimConfig(matrix_lr=0.02, adam_lr=2e-4)
    opt = build_optimizer(model, cfg)
    set_lr_and_wd(opt, 0.5, wd=0.05)
    assert abs(opt.param_groups[0]["lr"] - 0.01) < 1e-9
    assert abs(opt.param_groups[1]["lr"] - 1e-4) < 1e-9
    for g in opt.param_groups:
        assert g["weight_decay"] == 0.05
