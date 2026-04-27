"""Loader for preference (DPO/MPO) records: ``(prompt, chosen, rejected)``."""

from __future__ import annotations

from typing import Iterable, Iterator

import torch

from nanovlm.models.qwen3_5.chat_template import render_chat_for_training


class PreferenceLoader:
    """Yield batches of size 1 with chosen/rejected token streams.

    Returned dict::

        {
          "prompt_ids":   [1, T_p]  long
          "chosen_ids":   [1, T_c]  long  (prompt + chosen)
          "rejected_ids": [1, T_r]  long  (prompt + rejected)
          "chosen_mask":  [1, T_c]  long  (1 on chosen response tokens)
          "rejected_mask":[1, T_r]  long  (1 on rejected response tokens)
        }
    """

    def __init__(
        self,
        records: Iterable[dict],
        tokenizer,
        *,
        max_prompt_len: int = 1024,
        max_response_len: int = 1024,
    ):
        self.records = iter(records)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        rec = next(self.records)
        prompt = rec.get("prompt") or rec.get("question") or ""
        chosen = rec.get("chosen", "")
        rejected = rec.get("rejected", "")

        prompt_ids = self.tokenizer.encode(str(prompt))[: self.max_prompt_len]
        chosen_ids, chosen_mask = self._concat_with_mask(prompt_ids, str(chosen))
        rejected_ids, rejected_mask = self._concat_with_mask(prompt_ids, str(rejected))

        return {
            "prompt_ids": torch.tensor([prompt_ids], dtype=torch.long),
            "chosen_ids": torch.tensor([chosen_ids], dtype=torch.long),
            "rejected_ids": torch.tensor([rejected_ids], dtype=torch.long),
            "chosen_mask": torch.tensor([chosen_mask], dtype=torch.long),
            "rejected_mask": torch.tensor([rejected_mask], dtype=torch.long),
        }

    def _concat_with_mask(self, prompt_ids: list[int], response: str) -> tuple[list[int], list[int]]:
        resp_ids = self.tokenizer.encode(response)[: self.max_response_len]
        ids = prompt_ids + resp_ids
        mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
        return ids, mask
