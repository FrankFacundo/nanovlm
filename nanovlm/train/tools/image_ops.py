"""PIL-backed image operations for V*-with-python and visual reasoning agents."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path


def image_op(op: str, **kwargs) -> dict:
    """Dispatch on ``op``:

      - ``crop``: ``{path, bbox: [x0, y0, x1, y1]}`` → cropped image saved to /tmp.
      - ``zoom``: ``{path, bbox: [...], factor: float}`` → resized crop.
      - ``grid``: ``{path, rows, cols}`` → image with grid overlay (debug aid).
      - ``info``: ``{path}`` → ``{width, height, mode}``.
    """
    from PIL import Image, ImageDraw

    op = op.lower()
    src = Path(str(kwargs.pop("path"))).expanduser()
    img = Image.open(src).convert("RGB")
    if op == "info":
        return {"width": img.width, "height": img.height, "mode": img.mode}

    out_dir = Path(tempfile.gettempdir()) / "nanovlm_image_ops"
    out_dir.mkdir(parents=True, exist_ok=True)

    if op == "crop":
        bbox = _bbox(kwargs.get("bbox"), img.size)
        crop = img.crop(bbox)
        return _save(crop, out_dir)
    if op == "zoom":
        bbox = _bbox(kwargs.get("bbox"), img.size)
        factor = float(kwargs.get("factor", 2.0))
        crop = img.crop(bbox)
        new = (max(1, int(crop.width * factor)), max(1, int(crop.height * factor)))
        return _save(crop.resize(new, Image.LANCZOS), out_dir)
    if op == "grid":
        rows = int(kwargs.get("rows", 3))
        cols = int(kwargs.get("cols", 3))
        canvas = img.copy()
        d = ImageDraw.Draw(canvas)
        for r in range(1, rows):
            y = int(r * img.height / rows)
            d.line([(0, y), (img.width, y)], fill=(255, 0, 0), width=2)
        for c in range(1, cols):
            x = int(c * img.width / cols)
            d.line([(x, 0), (x, img.height)], fill=(255, 0, 0), width=2)
        return _save(canvas, out_dir)
    return {"ok": False, "error": f"unknown image_op {op!r}"}


def _bbox(value, size):
    if not value or len(value) != 4:
        return (0, 0, size[0], size[1])
    x0, y0, x1, y1 = [int(v) for v in value]
    x0, x1 = sorted((max(0, x0), min(size[0], x1)))
    y0, y1 = sorted((max(0, y0), min(size[1], y1)))
    if x1 <= x0:
        x1 = min(size[0], x0 + 1)
    if y1 <= y0:
        y1 = min(size[1], y0 + 1)
    return (x0, y0, x1, y1)


def _save(img, out_dir):
    name = f"img_{uuid.uuid4().hex[:12]}.png"
    path = out_dir / name
    img.save(path)
    return {"ok": True, "path": str(path), "width": img.width, "height": img.height}
