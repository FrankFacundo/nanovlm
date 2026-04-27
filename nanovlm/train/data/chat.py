"""Loader for chat-format records ``{messages: [...]}`` with assistant-only mask."""

from __future__ import annotations

from typing import Iterable

import torch

from nanovlm.models.qwen3_5.chat_template import render_chat_for_training


def build_chat_record(rec: dict) -> dict:
    """Normalize a record into ``{messages: [...]}``.

    Accepts:
      - ``{"prompt": ..., "response": ...}``
      - ``{"question": ..., "answer": ...}``
      - ``{"messages": [...]}``
    """
    if "messages" in rec:
        return {"messages": rec["messages"]}
    user = rec.get("prompt") or rec.get("question") or ""
    asst = rec.get("response") or rec.get("answer") or ""
    return {
        "messages": [
            {"role": "user", "content": str(user)},
            {"role": "assistant", "content": str(asst)},
        ]
    }


class ChatLoader:
    """Yield one packed chat sample per ``__next__`` (``batch_size=1``).

    For multi-batch packing, wrap with ``BestFitPacker`` with pre-tokenized
    records (``{"input_ids": ..., "loss_mask": ...}``).
    """

    def __init__(
        self,
        records: Iterable[dict],
        tokenizer,
        *,
        seq_len: int,
        thinking_ratio: float = 0.0,
    ):
        self.records = iter(records)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.thinking_ratio = thinking_ratio

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        rec = next(self.records)
        chat = build_chat_record(rec)
        out = render_chat_for_training(
            chat["messages"],
            self.tokenizer,
            seq_len=self.seq_len,
            enable_thinking=True,
        )
        ids = torch.tensor([out["input_ids"]], dtype=torch.long)
        mask = torch.tensor([out["loss_mask"]], dtype=torch.long)
        labels = ids.clone()
        labels[mask == 0] = -100
        attn = torch.ones_like(ids)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "loss_mask": mask,
        }
