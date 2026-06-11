"""Web-search fallback (default ON) — DuckDuckGo for links + Jina for full text.

Free, no API key, no self-hosting:
  1. DuckDuckGo (via `ddgs`) -> result URLs, restricted to INCLUDE_DOMAINS
  2. Jina Reader -> clean full-page text per URL (no key)
  3. results join the pipeline and go through reranking

Fires only when RAG retrieval is weak (top score < threshold), so for questions
already covered by the local corpus it never runs. DuckDuckGo is an unofficial
source and can rate-limit; failures degrade gracefully (returns []), and the
backend is swappable (Brave / Tavily) by editing only `_search_links`.
"""
from urllib.parse import urlparse

import requests
import config

READER = "https://r.jina.ai/"
JINA_HEADERS = {"X-Return-Format": "text"}


def _allowed(url):
    """Enforce INCLUDE_DOMAINS on the ACTUAL url. The `site:` search filter is only
    a hint; this is the real containment boundary against fetching off-domain pages
    (SSRF / prompt-injection from arbitrary sites)."""
    if not config.INCLUDE_DOMAINS:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in config.INCLUDE_DOMAINS)


def _jina_fetch(url):
    try:
        r = requests.get(READER + url, headers=JINA_HEADERS, timeout=60)
        r.raise_for_status()
        return r.text[:12000]
    except Exception:
        return ""


def _search_links(query, k):
    """Return [(url, snippet), ...] from DuckDuckGo. Swap for Brave/Tavily later if needed."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=k)
    except Exception:
        return []
    out = []
    for r in results:
        url = r.get("href") or r.get("url") or r.get("link")
        if url:
            out.append((url, r.get("body") or r.get("snippet") or ""))
    return out


def web_search(query, k=3):
    if not config.WEB_SEARCH:
        return []
    site = " ".join(f"site:{d}" for d in config.INCLUDE_DOMAINS)
    links = _search_links(f"{query} {site}".strip(), k)
    hits = []
    for url, snippet in links:
        if not _allowed(url):  # enforce the domain allow-list before fetching
            continue
        text = _jina_fetch(url) or snippet
        if text and len(text.strip()) >= 80:  # skip empty/too-thin pages (would embed noise)
            hits.append({"text": text, "source": url, "score": 0.0})
    return hits
