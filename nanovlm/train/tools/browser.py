"""HTTP fetch + readability-style text extraction (no third-party deps)."""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1 nanovlm/0.1"

_TAG_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_BLOCK_RE = re.compile(r"</?(p|div|li|h\d|br|tr|td)[^>]*>", re.IGNORECASE)
_OTHER_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def browse(url: str, *, max_chars: int = 8000, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read(1 << 20)
    except urllib.error.URLError as e:
        return {"ok": False, "error": str(e)}
    body = data.decode("utf-8", errors="ignore")
    text = extract_text(body)
    return {
        "ok": True,
        "url": url,
        "content_type": ct,
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def extract_text(html_doc: str) -> str:
    s = _TAG_RE.sub("", html_doc)
    s = _BLOCK_RE.sub("\n", s)
    s = _OTHER_TAG_RE.sub("", s)
    s = html.unescape(s)
    s = _WHITESPACE_RE.sub(" ", s)
    s = _NEWLINES_RE.sub("\n\n", s)
    return s.strip()
