"""Tiny pure-stdlib SVG plotters: line and bar charts. No matplotlib hard dep."""

from __future__ import annotations

import html
import math
from typing import Sequence


def line_plot(
    series: dict[str, Sequence[tuple[float, float]]],
    *,
    width: int = 720,
    height: int = 280,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    margin: tuple[int, int, int, int] = (40, 60, 40, 60),  # top right bottom left
) -> str:
    """Render multiple ``(x, y)`` series as an SVG <svg> string.

    ``series`` maps a label to a sequence of ``(x, y)``. Returns a single
    self-contained SVG string (no external CSS/scripts).
    """
    pts_all = [pt for s in series.values() for pt in s]
    if not pts_all:
        return _empty_svg(width, height, title)
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all if math.isfinite(p[1])]
    if not ys:
        return _empty_svg(width, height, title)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    top, right, bottom, left = margin
    plot_w = width - left - right
    plot_h = height - top - bottom

    def _x(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def _y(y):
        return top + (1 - (y - y_min) / (y_max - y_min)) * plot_h

    palette = ["#2f6fdb", "#dd6633", "#33aa55", "#aa44aa", "#cc8800", "#666666"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white" stroke="#ddd"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="13" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#888"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>',
        f'<text x="{left}" y="{top + plot_h + 14}" font-size="10" fill="#444">{x_min:.4g}</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 14}" text-anchor="end" font-size="10" fill="#444">{x_max:.4g}</text>',
        f'<text x="{left - 4}" y="{top + plot_h}" text-anchor="end" font-size="10" fill="#444">{y_min:.4g}</text>',
        f'<text x="{left - 4}" y="{top + 8}" text-anchor="end" font-size="10" fill="#444">{y_max:.4g}</text>',
    ]
    if xlabel:
        parts.append(f'<text x="{left + plot_w/2}" y="{height-8}" text-anchor="middle" font-size="11">{html.escape(xlabel)}</text>')
    if ylabel:
        parts.append(f'<text transform="rotate(-90,{left-32},{top+plot_h/2})" x="{left-32}" y="{top+plot_h/2}" text-anchor="middle" font-size="11">{html.escape(ylabel)}</text>')

    legend_y = top + 4
    for i, (label, pts) in enumerate(series.items()):
        color = palette[i % len(palette)]
        path = ""
        in_path = False
        for x, y in pts:
            if not math.isfinite(y):
                in_path = False
                continue
            if not in_path:
                path += f"M{_x(x):.2f} {_y(y):.2f}"
                in_path = True
            else:
                path += f"L{_x(x):.2f} {_y(y):.2f}"
        if path:
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        parts.append(f'<rect x="{left + plot_w + 8}" y="{legend_y}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{left + plot_w + 22}" y="{legend_y + 9}" font-size="11">{html.escape(label)}</text>')
        legend_y += 14

    parts.append("</svg>")
    return "".join(parts)


def bar_plot(
    bars: dict[str, float],
    *,
    width: int = 720,
    height: int = 320,
    title: str = "",
    ylabel: str = "score",
) -> str:
    """Vertical bar chart for benchmark summaries."""
    if not bars:
        return _empty_svg(width, height, title)
    items = list(bars.items())
    n = len(items)
    top, right, bottom, left = 40, 30, 80, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for v in bars.values() if math.isfinite(v)]
    y_max = max(values) if values else 1.0
    y_max = y_max if y_max > 0 else 1.0
    bar_w = max(8.0, plot_w / max(1, n) * 0.7)
    spacing = plot_w / max(1, n)
    palette = ["#2f6fdb", "#dd6633", "#33aa55", "#aa44aa", "#cc8800", "#666666"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white" stroke="#ddd"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="13" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#888"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>',
        f'<text x="{left - 4}" y="{top + 8}" text-anchor="end" font-size="10" fill="#444">{y_max:.4g}</text>',
        f'<text x="{left - 4}" y="{top + plot_h}" text-anchor="end" font-size="10" fill="#444">0</text>',
        f'<text transform="rotate(-90,{left-40},{top+plot_h/2})" x="{left-40}" y="{top+plot_h/2}" text-anchor="middle" font-size="11">{html.escape(ylabel)}</text>',
    ]
    for i, (label, val) in enumerate(items):
        if not math.isfinite(val):
            continue
        h = (val / y_max) * plot_h
        x = left + i * spacing + (spacing - bar_w) / 2
        y = top + plot_h - h
        color = palette[i % len(palette)]
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.2f}" y="{y - 3:.2f}" text-anchor="middle" font-size="10" fill="#222">{val:.3g}</text>')
        parts.append(f'<text x="{x + bar_w/2:.2f}" y="{top + plot_h + 14}" text-anchor="middle" font-size="10" transform="rotate(35,{x + bar_w/2:.2f},{top + plot_h + 14})">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _empty_svg(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">'
        f'<rect width="{width}" height="{height}" fill="white" stroke="#ddd"/>'
        f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="12" fill="#888">'
        f'{html.escape(title)} (no data)</text></svg>'
    )
