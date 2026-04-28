"""Build a Qwen3.5 model + tokenizer for training scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import (
    DEFAULT_MODEL_PATH,
    Qwen2Tokenizer,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    load_qwen3_5_weights,
)


def tiny_text_config(vocab_size: int = 248320) -> Qwen3_5TextConfig:
    """Tiny text-only config for fast CPU/MPS smoke tests."""
    return Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )


def tiny_config(vocab_size: int = 248320) -> Qwen3_5Config:
    """Tiny multimodal config for smoke tests."""
    text = tiny_text_config(vocab_size)
    vision = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=text.hidden_size,
        patch_size=16,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    return Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True)


def load_tokenizer(model_path: str | Path = DEFAULT_MODEL_PATH) -> Qwen2Tokenizer:
    return Qwen2Tokenizer.from_pretrained(str(model_path))


def build_model(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    init: str = "scratch",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    tiny: bool = False,
    text_only: bool = False,
) -> torch.nn.Module:
    """Construct a Qwen3.5 model.

    init=scratch     : random weights from config (or tiny if --tiny).
    init=checkpoint  : load safetensors from ``model_path``.
    text_only=True   : returns ``Qwen3_5ForCausalLM`` (decoder-only).
    text_only=False  : returns ``Qwen3_5ForConditionalGeneration`` (VLM).
    """
    model_path = Path(model_path).expanduser()

    if tiny:
        if text_only:
            cfg = tiny_text_config()
            model = Qwen3_5ForCausalLM(cfg)
        else:
            cfg = tiny_config()
            model = Qwen3_5ForConditionalGeneration(cfg)
    else:
        cfg = Qwen3_5Config.from_pretrained(str(model_path))
        if text_only:
            model = Qwen3_5ForCausalLM(cfg.text_config)
        else:
            model = Qwen3_5ForConditionalGeneration(cfg)

    if init == "checkpoint":
        if tiny:
            raise ValueError("init=checkpoint is incompatible with tiny=True")
        load_qwen3_5_weights(model, model_path)
    elif init != "scratch":
        raise ValueError(f"unknown init: {init!r}")

    model = model.to(device=device, dtype=dtype)
    return model
