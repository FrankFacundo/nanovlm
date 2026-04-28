"""Web search tool: DuckDuckGo HTML by default; Tavily/SerpAPI via env vars.

Args::

    {"query": str, "k": int = 5}

Returns::

    [{"title": str, "url": str, "snippet": str}, ...]
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1 nanovlm/0.1"


def web_search(query: str, k: int = 5) -> list[dict]:
    if os.environ.get("TAVILY_API_KEY"):
        return _tavily(query, k)
    if os.environ.get("SERPAPI_KEY"):
        return _serpapi(query, k)
    return _duckduckgo_html(query, k)


def _http(url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _duckduckgo_html(query: str, k: int) -> list[dict]:
    url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    try:
        body = _http(url).decode("utf-8", errors="ignore")
    except Exception as e:
        return [{"title": "error", "url": "", "snippet": str(e)}]
    out = []
    for m in _DDG_RESULT_RE.finditer(body):
        out.append({
            "title": _strip_tags(m.group(2)),
            "url": _resolve_redirect(m.group(1)),
            "snippet": _strip_tags(m.group(3)),
        })
        if len(out) >= k:
            break
    return out


def _tavily(query: str, k: int) -> list[dict]:
    payload = json.dumps({"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": k}).encode("utf-8")
    body = _http("https://api.tavily.com/search", data=payload, headers={"Content-Type": "application/json"})
    data = json.loads(body.decode("utf-8"))
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in data.get("results", [])]


def _serpapi(query: str, k: int) -> list[dict]:
    url = (
        "https://serpapi.com/search.json?q=" + urllib.parse.quote(query) +
        "&num=" + str(k) + "&api_key=" + urllib.parse.quote(os.environ["SERPAPI_KEY"])
    )
    body = _http(url).decode("utf-8")
    data = json.loads(body)
    out = []
    for r in data.get("organic_results", [])[:k]:
        out.append({"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")})
    return out


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _resolve_redirect(href: str) -> str:
    # DDG HTML rewrites links via /l/?uddg=<encoded>
    if href.startswith("/l/?"):
        q = urllib.parse.parse_qs(href.split("?", 1)[1])
        return q.get("uddg", [href])[0]
    if href.startswith("//"):
        return "https:" + href
    return href
