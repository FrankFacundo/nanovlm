"""Multimodal batch builder: lazy image loading + image-text packed batches.

Records are dicts of the form::

    {
      "messages": [...],         # chat-format (uses chat_template)
      "images": ["path/to/img1.jpg", ...],
    }

or simple captioning::

    {"image": "path/to/img.jpg", "text": "a cat sitting on a mat"}

Output batch has the keys expected by ``Qwen3_5ForConditionalGeneration``:
``input_ids``, ``attention_mask``, ``labels``, ``loss_mask``, ``pixel_values``,
``image_grid_thw``, ``mm_token_type_ids`` (1 where image patch tokens go).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import torch

from nanovlm.models.qwen3_5 import Qwen2VLImageProcessor, render_chat
from nanovlm.models.qwen3_5.chat_template import (
    IMAGE_PAD,
    VISION_END,
    VISION_START,
    render_chat_for_training,
)


class MultimodalLoader:
    """Yield single-row multimodal training batches.

    On each ``__next__``: pulls a record, renders chat into text + assistant
    mask, loads listed images via PIL, runs the image processor to produce
    ``pixel_values``/``image_grid_thw``, expands ``<|image_pad|>`` placeholders
    in the token ids to N image-token slots (one per merged patch), and builds
    ``mm_token_type_ids`` (1 on image-token positions, 0 elsewhere).
    """

    def __init__(
        self,
        records: Iterable[dict],
        tokenizer,
        *,
        seq_len: int,
        image_root: str | Path | None = None,
        image_processor: Qwen2VLImageProcessor | None = None,
        spatial_merge_size: int = 2,
    ):
        self.records = iter(records)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.image_root = Path(image_root).expanduser() if image_root else Path(".")
        self.image_processor = image_processor or Qwen2VLImageProcessor()
        self.spatial_merge_size = spatial_merge_size
        self.image_pad_id = self._lookup_id(IMAGE_PAD)

    def _lookup_id(self, token: str) -> int:
        added = getattr(self.tokenizer, "_added", {})
        if token in added:
            return int(added[token])
        ids = self.tokenizer.encode(token)
        if len(ids) == 1:
            return ids[0]
        raise ValueError(f"tokenizer does not contain a single id for {token!r}")

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        from PIL import Image

        rec = next(self.records)
        messages = self._normalize(rec)
        image_paths = self._extract_image_paths(rec, messages)

        rendered = render_chat_for_training(
            messages,
            self.tokenizer,
            seq_len=None,
        )
        ids: list[int] = rendered["input_ids"]
        mask: list[int] = rendered["loss_mask"]

        images = []
        grid_thw_rows = []
        if image_paths:
            for p in image_paths:
                full = (self.image_root / p) if not Path(p).is_absolute() else Path(p)
                images.append(Image.open(full).convert("RGB"))
            proc = self.image_processor(images=images, return_tensors="pt")
            pixel_values = proc["pixel_values"]
            image_grid_thw = proc["image_grid_thw"]
            ids, mask, mm_type = self._expand_image_pads(ids, mask, image_grid_thw)
            grid_thw_rows = image_grid_thw
        else:
            pixel_values = torch.empty(0)
            image_grid_thw = torch.empty(0, 3, dtype=torch.long)
            mm_type = [0] * len(ids)

        ids = ids[: self.seq_len]
        mask = mask[: self.seq_len]
        mm_type = mm_type[: self.seq_len]
        # Right-pad to seq_len with EOS / mask 0
        pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        pad_n = self.seq_len - len(ids)
        ids.extend([pad_id] * pad_n)
        mask.extend([0] * pad_n)
        mm_type.extend([0] * pad_n)

        ids_t = torch.tensor([ids], dtype=torch.long)
        mask_t = torch.tensor([mask], dtype=torch.long)
        mm_t = torch.tensor([mm_type], dtype=torch.int32)
        attn = torch.ones_like(ids_t)
        labels = ids_t.clone()
        labels[mask_t == 0] = -100

        out = {
            "input_ids": ids_t,
            "attention_mask": attn,
            "labels": labels,
            "loss_mask": mask_t,
            "mm_token_type_ids": mm_t,
        }
        if image_paths:
            out["pixel_values"] = pixel_values
            out["image_grid_thw"] = image_grid_thw
        return out

    def _normalize(self, rec: dict) -> list[dict]:
        if "messages" in rec:
            return rec["messages"]
        text = rec.get("text") or rec.get("caption") or ""
        return [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]},
            {"role": "assistant", "content": str(text)},
        ]

    def _extract_image_paths(self, rec: dict, messages: list[dict]) -> list[str]:
        if "images" in rec:
            return list(rec["images"])
        if "image" in rec:
            return [rec["image"]]
        # Pull from messages: any item with type=image carrying a path
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        if item.get("path"):
                            out.append(item["path"])
        return out

    def _expand_image_pads(
        self,
        ids: list[int],
        mask: list[int],
        image_grid_thw: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """Replace each ``<|image_pad|>`` token with N copies, where N is the
        number of vision tokens that the spatial-merge stage emits for the
        corresponding image. Builds ``mm_token_type_ids`` in lockstep.
        """
        merge = self.spatial_merge_size
        n_img = image_grid_thw.size(0)
        per_image = []
        for i in range(n_img):
            t, h, w = [int(v) for v in image_grid_thw[i].tolist()]
            per_image.append(t * (h // merge) * (w // merge))
        out_ids: list[int] = []
        out_mask: list[int] = []
        out_mm: list[int] = []
        ptr = 0
        for tok, mk in zip(ids, mask):
            if tok == self.image_pad_id and ptr < n_img:
                n = per_image[ptr]
                ptr += 1
                out_ids.extend([self.image_pad_id] * n)
                out_mask.extend([0] * n)  # never train on image-token positions
                out_mm.extend([1] * n)
            else:
                out_ids.append(tok)
                out_mask.append(mk)
                out_mm.append(0)
        return out_ids, out_mask, out_mm
