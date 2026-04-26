"""Streaming JSONL/parquet-ish data helpers and packed batch loaders.

The loaders are intentionally simple and lazy: records are read one at a time,
tokenized on demand, and packed into fixed-size tensors. They do not depend on
Hugging Face libraries.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch


DEFAULT_PRETRAIN_RECORDS = [
    {"text": "Vision-language models combine text tokens with image patch embeddings."},
    {"text": "Muon optimizes matrix weights while AdamW handles embeddings and normalization."},
    {"prompt": "What is 2+2?", "response": "4"},
    {"question": "Return the final answer only: 7 * 6", "answer": "42"},
]


def ensure_default_pretrain_jsonl(path: str | Path) -> Path:
    """Create a tiny default JSONL training file if it does not exist."""
    path = Path(path).expanduser()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in DEFAULT_PRETRAIN_RECORDS:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def validate_data_paths(paths: Iterable[str | Path], *, default_hint: str | Path | None = None) -> list[str]:
    resolved = []
    missing = []
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            missing.append(str(path))
        else:
            resolved.append(str(p))
    if missing:
        hint = f" Omit --data to use the default starter file at {default_hint}." if default_hint else ""
        raise FileNotFoundError(f"training data file(s) not found: {', '.join(missing)}.{hint}")
    return resolved


def iter_jsonl(paths: Iterable[str | Path]) -> Iterator[dict]:
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def record_text(record: dict) -> str:
    for key in ("text", "content", "document", "caption", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    if "prompt" in record and "response" in record:
        return f"{record['prompt']}\n{record['response']}"
    if "question" in record and "answer" in record:
        return f"{record['question']}\n{record['answer']}"
    if "messages" in record:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in record["messages"])
    return json.dumps(record, ensure_ascii=False)


def iter_synthetic_records() -> Iterator[dict]:
    while True:
        for ex in DEFAULT_PRETRAIN_RECORDS:
            yield ex


def cycle_records(paths: list[str | Path] | None, *, rank: int = 0, world_size: int = 1) -> Iterator[dict]:
    if not paths:
        yield from iter_synthetic_records()
        return
    while True:
        for i, rec in enumerate(iter_jsonl(paths)):
            if i % world_size == rank:
                yield rec


def eos_id(tokenizer) -> int:
    value = getattr(tokenizer, "eos_token_id", None)
    return int(value if value is not None else 0)


def encode_record(tokenizer, record: dict) -> list[int]:
    return tokenizer.encode(record_text(record), add_special_tokens=False) + [eos_id(tokenizer)]


@dataclass
class LoaderState:
    docs_seen: int = 0
    tokens_seen: int = 0


class PackedTextLoader:
    """Pack streaming records into dense next-token LM batches."""

    def __init__(
        self,
        tokenizer,
        paths: list[str | Path] | None,
        batch_size: int,
        seq_len: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | None = None,
    ):
        self.tokenizer = tokenizer
        self.records = cycle_records(paths, rank=rank, world_size=world_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.state = LoaderState()
        self.buffer: list[int] = []

    def _fill(self, n: int) -> None:
        while len(self.buffer) < n:
            toks = encode_record(self.tokenizer, next(self.records))
            self.buffer.extend(toks)
            self.state.docs_seen += 1
            self.state.tokens_seen += len(toks)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        need = self.batch_size * (self.seq_len + 1)
        self._fill(need)
        chunk = self.buffer[:need]
        del self.buffer[:need]
        x = torch.tensor(chunk, dtype=torch.long).view(self.batch_size, self.seq_len + 1)
        batch = {
            "input_ids": x[:, :-1].contiguous(),
            "labels": x[:, 1:].contiguous(),
            "attention_mask": torch.ones((self.batch_size, self.seq_len), dtype=torch.long),
            "loss_mask": torch.ones((self.batch_size, self.seq_len), dtype=torch.long),
        }
        if self.device is not None:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                else:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def render_chat_tokens(tokenizer, messages: list[dict], *, train_assistant_only: bool = True) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _flatten_content(msg.get("content", ""))
        prefix = f"<|im_start|>{role}\n"
        suffix = "<|im_end|>\n"
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        suffix_ids = tokenizer.encode(suffix)
        ids.extend(prefix_ids)
        mask.extend([0] * len(prefix_ids))
        ids.extend(content_ids)
        mask.extend([1 if (role == "assistant" or not train_assistant_only) else 0] * len(content_ids))
        ids.extend(suffix_ids)
        mask.extend([1 if (role == "assistant" and train_assistant_only) else 0] * len(suffix_ids))
    return ids, mask


def record_messages(record: dict, *, thinking: bool = False, non_thinking: bool = False) -> list[dict]:
    if "messages" in record:
        return record["messages"]
    prompt = record.get("prompt") or record.get("question") or record.get("instruction") or ""
    response = record.get("response") or record.get("answer") or record.get("output") or ""
    if thinking and "<think>" not in response:
        response = f"<think>\n{record.get('rationale', '')}\n</think>\n\n{response}"
    if non_thinking and not response.startswith("<think>"):
        response = f"<think></think>\n\n{response}"
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


class SFTLoader:
    """One conversation per row with assistant-token-only labels."""

    def __init__(
        self,
        tokenizer,
        paths: list[str | Path] | None,
        batch_size: int,
        seq_len: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | None = None,
        thinking_ratio: float = 0.25,
        seed: int = 1234,
    ):
        self.tokenizer = tokenizer
        self.records = cycle_records(paths, rank=rank, world_size=world_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = random.Random(seed + rank)
        self.thinking_ratio = thinking_ratio

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        rows, labels, masks = [], [], []
        pad = eos_id(self.tokenizer)
        for _ in range(self.batch_size):
            rec = next(self.records)
            thinking = self.rng.random() < self.thinking_ratio
            ids, loss_mask = render_chat_tokens(
                self.tokenizer,
                record_messages(rec, thinking=thinking, non_thinking=not thinking),
                train_assistant_only=True,
            )
            ids = ids[: self.seq_len + 1]
            loss_mask = loss_mask[: self.seq_len + 1]
            if len(ids) < self.seq_len + 1:
                pad_n = self.seq_len + 1 - len(ids)
                ids.extend([pad] * pad_n)
                loss_mask.extend([0] * pad_n)
            row = ids[:-1]
            target = ids[1:]
            target_mask = loss_mask[1:]
            rows.append(row)
            labels.append([t if m else -100 for t, m in zip(target, target_mask)])
            masks.append([1 if tok != pad else 0 for tok in row])
        batch = {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "loss_mask": (torch.tensor(labels, dtype=torch.long) >= 0).long(),
        }
        if self.device is not None:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch


def pair_records(paths: list[str | Path] | None) -> Iterator[dict]:
    if not paths:
        while True:
            yield {
                "prompt": "Answer with the exact number: 3+5",
                "chosen": "8",
                "rejected": "9",
            }
    while True:
        for rec in iter_jsonl(paths):
            if "chosen" in rec and "rejected" in rec:
                yield rec


def make_pair_batch(tokenizer, records: list[dict], seq_len: int) -> dict[str, torch.Tensor]:
    rows = {"chosen": [], "rejected": []}
    labels = {"chosen": [], "rejected": []}
    pad = eos_id(tokenizer)
    for rec in records:
        prompt = rec.get("prompt", "")
        for side in ("chosen", "rejected"):
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rec[side]}]
            ids, mask = render_chat_tokens(tokenizer, messages, train_assistant_only=True)
            ids = ids[: seq_len + 1]
            mask = mask[: seq_len + 1]
            if len(ids) < seq_len + 1:
                ids += [pad] * (seq_len + 1 - len(ids))
                mask += [0] * (seq_len + 1 - len(mask))
            rows[side].append(ids[:-1])
            labels[side].append([t if m else -100 for t, m in zip(ids[1:], mask[1:])])
    return {
        "chosen_input_ids": torch.tensor(rows["chosen"], dtype=torch.long),
        "chosen_labels": torch.tensor(labels["chosen"], dtype=torch.long),
        "rejected_input_ids": torch.tensor(rows["rejected"], dtype=torch.long),
        "rejected_labels": torch.tensor(labels["rejected"], dtype=torch.long),
    }


class PreferenceLoader:
    def __init__(self, tokenizer, paths: list[str | Path] | None, batch_size: int, seq_len: int):
        self.tokenizer = tokenizer
        self.records = pair_records(paths)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        return make_pair_batch(self.tokenizer, [next(self.records) for _ in range(self.batch_size)], self.seq_len)
