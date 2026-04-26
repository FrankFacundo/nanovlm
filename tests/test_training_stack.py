from __future__ import annotations

import ast
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from nanovlm.train.agent import LocalSandbox
from nanovlm.train.data import PackedTextLoader, SFTLoader, make_pair_batch
from nanovlm.train.download import DEFAULT_SOURCES, build_download_plan
from nanovlm.train.losses import dpo_loss, group_advantages, masked_ce_loss
from nanovlm.train.model_factory import init_qwen_weights, tiny_config
from nanovlm.train.optim import OptimConfig, build_optimizer


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [(ord(c) % 90) + 1 for c in text]

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(chr((int(i) - 1) % 90 + 32) for i in ids if int(i) > 0)


def test_masked_ce_ignores_negative_labels():
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, -1], [3, 4, 5, -100]])
    loss = masked_ce_loss(logits, labels)
    assert torch.isfinite(loss)


def test_dpo_and_group_advantages():
    chosen = torch.tensor([2.0, 1.0])
    rejected = torch.tensor([0.0, 0.5])
    ref_chosen = torch.tensor([1.0, 1.0])
    ref_rejected = torch.tensor([0.0, 0.0])
    loss, metrics = dpo_loss(chosen, rejected, ref_chosen, ref_rejected)
    assert loss.item() > 0
    assert metrics["dpo_acc"] >= 0.5
    adv = group_advantages(torch.tensor([1.0, 0.0, 3.0, 1.0]), group_size=2)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 1.0, -1.0]))


def test_packed_and_sft_loader_shapes():
    tok = TinyTokenizer()
    batch = next(PackedTextLoader(tok, None, batch_size=2, seq_len=8))
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    sft = next(SFTLoader(tok, None, batch_size=2, seq_len=16, thinking_ratio=0.0))
    assert sft["labels"].shape == (2, 16)
    assert (sft["labels"] == -100).any()


def test_pair_batch_shapes():
    tok = TinyTokenizer()
    batch = make_pair_batch(tok, [{"prompt": "q", "chosen": "a", "rejected": "b"}], seq_len=12)
    assert batch["chosen_input_ids"].shape == (1, 12)
    assert batch["rejected_labels"].shape == (1, 12)


def test_tiny_model_loss_and_optimizer_grouping():
    cfg = tiny_config(vocab_size=512)
    model = Qwen3_5ForConditionalGeneration(cfg)
    init_qwen_weights(model)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    out = model(input_ids=x, labels=y)
    assert torch.isfinite(out["loss"])
    opt = build_optimizer(model, OptimConfig())
    kinds = {g.get("kind") for g in opt.param_groups}
    assert "muon" in kinds
    assert "adamw" in kinds


def test_download_budget_plan_is_capped():
    plan = build_download_plan(DEFAULT_SOURCES, max_bytes=1024)
    assert sum(item["budget_bytes"] for item in plan) <= 1024
    assert all(not item["allow_noncommercial"] for item in plan)


def test_sandbox_logs_and_runs_python(tmp_path):
    sandbox = LocalSandbox(root=tmp_path / "box", log_path=tmp_path / "traj.jsonl")
    result = sandbox.run_python("print(40 + 2)")
    sandbox.close()
    assert result.returncode == 0
    assert "42" in result.stdout
    assert (tmp_path / "traj.jsonl").exists()


def test_runtime_has_no_forbidden_training_imports():
    forbidden = {"transformers", "datasets", "huggingface_hub", "trl", "peft", "accelerate"}
    roots = [Path("nanovlm"), Path("scripts")]
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = {node.module.split(".")[0]}
                else:
                    continue
                assert not (names & forbidden), f"{path} imports {names & forbidden}"
