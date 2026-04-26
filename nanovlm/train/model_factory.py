"""Model construction for training scripts."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

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


def tiny_config(vocab_size: int = 248320) -> Qwen3_5Config:
    text = Qwen3_5TextConfig(
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
    vision = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=text.hidden_size,
        num_position_embeddings=16,
    )
    return Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=248056 if vocab_size > 248057 else min(vocab_size - 3, 200),
        video_token_id=248057 if vocab_size > 248057 else min(vocab_size - 2, 201),
        vision_start_token_id=248053 if vocab_size > 248057 else min(vocab_size - 5, 198),
        vision_end_token_id=248054 if vocab_size > 248057 else min(vocab_size - 4, 199),
        tie_word_embeddings=True,
    )


def init_qwen_weights(model: nn.Module, std: float = 0.02) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv3d)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm,)):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.__class__.__name__ == "RMSNorm":
            nn.init.zeros_(module.weight)
        elif module.__class__.__name__ == "RMSNormGated":
            nn.init.ones_(module.weight)
    for name, param in model.named_parameters():
        if name.endswith("dt_bias"):
            nn.init.ones_(param)
        elif name.endswith("A_log"):
            with torch.no_grad():
                param.copy_(torch.empty_like(param).uniform_(0.1, 16.0).log())
        elif param.ndim == 1 and "norm" not in name.lower() and not name.endswith("bias"):
            nn.init.zeros_(param)


def load_tokenizer(model_path: str | Path = DEFAULT_MODEL_PATH):
    return Qwen2Tokenizer.from_pretrained(model_path)


def build_model(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    init: str = "scratch",
    device: torch.device,
    dtype: torch.dtype,
    tiny: bool = False,
    text_only: bool = False,
) -> nn.Module:
    if tiny:
        cfg = tiny_config()
    else:
        cfg = Qwen3_5Config.from_pretrained(model_path)

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg.text_config) if text_only else Qwen3_5ForConditionalGeneration(cfg)
    model.to_empty(device=device)
    init_qwen_weights(model, std=cfg.text_config.initializer_range)

    if init == "checkpoint":
        if tiny:
            raise ValueError("--init checkpoint is not supported with --tiny")
        load_qwen3_5_weights(model, model_path, text_only=text_only, strict=False, dtype=dtype)
    elif init != "scratch":
        raise ValueError("--init must be scratch or checkpoint")

    model.to(device=device, dtype=dtype)
    return model
