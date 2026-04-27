"""BOS-aligned best-fit token packing.

Pattern from nanochat: every packed row begins with the BOS/EOS-as-separator
token; documents are appended greedily until the next won't fit; the last
document is cropped to fill exactly the configured sequence length. The
returned ``loss_mask`` excludes the leading BOS so that the next-token
shift target is well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch


@dataclass
class PackedBatch:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    loss_mask: torch.LongTensor
    docs_in_batch: int = 0
    tokens_in_batch: int = 0


class BestFitPacker:
    """Pack tokenized records into fixed-size sequences.

    ``record_iter`` yields dicts. The packer uses ``record["text"]`` for
    pretraining (encoded with ``tokenizer.encode``) or ``record["input_ids"]``
    if pre-tokenized. For SFT, pass ``assistant_only=True`` and records with
    ``loss_mask`` aligned to ``input_ids``.
    """

    def __init__(
        self,
        record_iter: Iterable,
        tokenizer,
        *,
        seq_len: int,
        batch_size: int,
        sep_token_id: int | None = None,
        assistant_only: bool = False,
    ):
        self.records = iter(record_iter)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sep_token_id = sep_token_id if sep_token_id is not None else _default_sep(tokenizer)
        self.assistant_only = assistant_only
        self._buffer_ids: list[int] = []
        self._buffer_mask: list[int] = []
        self._docs_in_buffer = 0

    def __iter__(self):
        return self

    def __next__(self) -> PackedBatch:
        rows_ids = []
        rows_mask = []
        docs = 0
        while len(rows_ids) < self.batch_size:
            self._fill_one_row()
            rows_ids.append(self._row_ids)
            rows_mask.append(self._row_mask)
            docs += self._row_docs

        ids = torch.tensor(rows_ids, dtype=torch.long)
        mask = torch.tensor(rows_mask, dtype=torch.long)
        labels = ids.clone()
        # Cause masked positions to be ignored by losses.masked_ce_loss
        labels[mask == 0] = -100
        return PackedBatch(input_ids=ids, labels=labels, loss_mask=mask, docs_in_batch=docs, tokens_in_batch=int(ids.numel()))

    def _fill_one_row(self) -> None:
        row_ids: list[int] = [self.sep_token_id]
        row_mask: list[int] = [0]
        docs = 0
        while len(row_ids) < self.seq_len:
            if not self._buffer_ids:
                rec = next(self.records)
                ids, msk = self._encode(rec)
                if not ids:
                    continue
                self._buffer_ids = ids + [self.sep_token_id]
                self._buffer_mask = msk + [1]
                self._docs_in_buffer = 1
            need = self.seq_len - len(row_ids)
            chunk_ids = self._buffer_ids[:need]
            chunk_mask = self._buffer_mask[:need]
            row_ids.extend(chunk_ids)
            row_mask.extend(chunk_mask)
            self._buffer_ids = self._buffer_ids[need:]
            self._buffer_mask = self._buffer_mask[need:]
            if not self._buffer_ids:
                docs += self._docs_in_buffer
                self._docs_in_buffer = 0
        self._row_ids = row_ids[: self.seq_len]
        self._row_mask = row_mask[: self.seq_len]
        self._row_docs = docs

    def _encode(self, rec: dict) -> tuple[list[int], list[int]]:
        if "input_ids" in rec:
            ids = list(rec["input_ids"])
            if self.assistant_only and "loss_mask" in rec:
                mask = list(rec["loss_mask"])
            else:
                mask = [1] * len(ids)
            return ids, mask
        text = rec.get("text") or rec.get("content") or ""
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text)
        ids = self.tokenizer.encode(str(text))
        return ids, [1] * len(ids)


def _default_sep(tokenizer) -> int:
    eid = getattr(tokenizer, "eos_token_id", None)
    if eid is not None:
        return int(eid)
    return 0


def pack_records(
    records: Iterable[dict],
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    sep_token_id: int | None = None,
) -> Iterator[PackedBatch]:
    return BestFitPacker(
        records,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        sep_token_id=sep_token_id,
    )
