"""SearxNG search and result ranking for Library.

query_variations()  -- deterministic keyword + prose variants for a question
search()            -- run queries against SearxNG, return raw result pool
rank_results()      -- deduplicate, score, and pick top-N from the pool

Configuration (env vars, all optional):
  LIBRARY_SEARXNG_URL  -- base URL of the local SearxNG instance.
                          Default: http://127.0.0.1:8888 (matches the
                          2gpu workstation stack's user-scope SearxNG
                          unit). Must be a SearxNG instance with the
                          JSON output format enabled.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any


SEARXNG_URL = os.environ.get("LIBRARY_SEARXNG_URL") or "http://127.0.0.1:8888"
SEARCH_TIMEOUT_SEC = 15
USER_AGENT = "library-mcp/0.1 (local research helper)"

DOC_DOMAIN_BOOST = (
    "proxmox.com", "immich.app", "debian.org", "ubuntu.com",
    "kernel.org", "github.com", "docker.com", "traefik.io",
    "grafana.com", "prometheus.io",
)

_STOPWORDS = frozenset((
    "what", "which", "where", "when", "why", "how", "does", "do",
    "is", "are", "was", "were", "the", "this", "that", "with",
    "for", "from", "into", "about", "can", "should", "would",
    "and", "but", "not", "you", "your", "there", "their",
    "support", "supports",
))


def query_variations(question: str) -> list[str]:
    q = question.strip()
    words = re.findall(r"[\w-]+", q)
    keywords = [w for w in words if w.lower() not in _STOPWORDS and len(w) >= 3]
    variations: list[str] = []
    if len(keywords) >= 2:
        variations.append(" ".join(keywords[:8]))
    variations.append(q)
    seen: set[str] = set()
    return [v for v in variations if not (v in seen or seen.add(v))]


def search(question: str) -> list[dict[str, Any]]:
    """Run all query variations and return the pooled result list."""
    all_results: list[dict[str, Any]] = []
    for q in query_variations(question):
        url = f"{SEARXNG_URL}/search?{urllib.parse.urlencode({'q': q, 'format': 'json'})}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_SEC) as r:
                data = json.loads(r.read())
            for i, result in enumerate(data.get("results", [])[:10]):
                result["_query_position"] = i
                all_results.append(result)
        except Exception:
            pass
    return all_results


def rank_results(results: list[dict[str, Any]], question: str, max_n: int) -> list[dict[str, Any]]:
    q_tokens = {w.lower() for w in re.findall(r"\w+", question) if len(w) >= 4}
    seen_urls: set[str] = set()
    scored: list[tuple[float, dict[str, Any]]] = []

    for r in results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        host = urllib.parse.urlparse(url).netloc.lower()
        title = (r.get("title") or "").lower()
        score = float(r.get("_query_position", 99))

        if any(dom in host for dom in DOC_DOMAIN_BOOST):
            score -= 5.0

        t_tokens = {w for w in re.findall(r"\w+", title) if len(w) >= 4}
        overlap = len(q_tokens & t_tokens)
        score -= 2.0 * overlap
        if overlap == 0 and not any(dom in host for dom in DOC_DOMAIN_BOOST):
            score += 20.0

        scored.append((score, r))

    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:max_n]]
