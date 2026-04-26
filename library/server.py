#!/usr/bin/env python3
"""Library MCP server.

Three tools:

  research(question, max_sources, return_chunks, force_refresh)
    → summary layer (default) or chunk layer (return_chunks=True)
    Web acquisition: SearxNG search → fetch → chunk → embed → rank → summarize

  read_file(path, query, return_chunks)
    → summary layer (default) or chunk layer (return_chunks=True)
    Local acquisition: read file → chunk → embed → rank → summarize

  get_skill(name)
    → skill layer: full skill file contents, no pipeline

Escalation protocol (primary carries round count):
  - Call with return_chunks=False first (summary layer, can_escalate=True)
  - Call again with return_chunks=True if summary is insufficient (same round)
  - A new/refined query starts a new round (max 3 rounds per topic)
  - After 3 rounds on a topic without sufficient result: fetch directly
  - A new topic always starts at round 1
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .cache import Cache, CachedEntry, make_file_id, make_web_id
from .chunkers import chunk as chunk_file
from .embedder import EmbedderError, cosine_similarity, embed_batch, embed_one
from .fetcher import FetchError, fetch_and_extract
from .searcher import search, rank_results
from .summarizer import summarize


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MAX_SOURCES = 5
DEFAULT_TOP_K = 5
SKILLS_DIR = Path(__file__).parent / "skills"
_EMBED_BATCH = 16


# ── Logging ──────────────────────────────────────────────────────────────────

def _log(event: str, **fields: Any) -> None:
    line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}
    print(json.dumps(line), file=sys.stderr, flush=True)


# ── Shared pipeline helpers ───────────────────────────────────────────────────

_cache = Cache()


def _embed_chunks(contents: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(contents), _EMBED_BATCH):
        batch = contents[i: i + _EMBED_BATCH]
        vectors.extend(embed_batch(batch))
    return vectors


def _error(msg: str) -> dict:
    return {"layer": "error", "error": msg, "can_escalate": False}


# ── MCP tools ─────────────────────────────────────────────────────────────────

mcp = FastMCP("library")


@mcp.tool()
def research(
    question: str,
    max_sources: int = DEFAULT_MAX_SOURCES,
    return_chunks: bool = False,
    force_refresh: bool = False,
) -> dict:
    """Answer a research question from the web.

    Returns a summary by default. If the summary is insufficient, call again
    with return_chunks=True to get the ranked raw chunks (same round — does
    not count toward the 3-round limit).

    Escalation protocol:
      Round 1: research(question)                       → summary
               research(question, return_chunks=True)   → chunks (same round)
      Round 2: research(refined_question)               → summary
      Round 3: research(further_refined_question)       → summary
      After 3 rounds without sufficient result: fetch directly (last resort).
      A new topic always resets to round 1.

    force_refresh=True bypasses the cache and re-fetches all sources. Use
    only when instructed by the user — the user is a more reliable judge of
    staleness than the primary model, which never sees the raw source.

    Args:
        question: Research question in plain English.
        max_sources: Number of sources to fetch and consult (default 5, max 8).
        return_chunks: If True, return ranked chunks instead of summary.
        force_refresh: If True, bypass cache and re-fetch all sources.

    Returns:
        Summary layer: {"layer": "summary", "query": ..., "summary": ...,
                        "sources": [...], "confidence": ..., "notes": ..., "can_escalate": true}
        Chunk layer:   {"layer": "chunks", "query": ..., "results": [...], "can_escalate": false}
        Error layer:   {"layer": "error", "error": ..., "can_escalate": false}
    """
    max_sources = max(1, min(int(max_sources), 8))
    t0 = time.time()
    _log("research_start", question=question, max_sources=max_sources,
         return_chunks=return_chunks, force_refresh=force_refresh)

    all_results = search(question)
    if not all_results:
        return _error("SearxNG returned no results for all query variations")

    picked = rank_results(all_results, question, max_sources)
    _log("picked", count=len(picked), urls=[p.get("url") for p in picked])

    all_chunks: list = []
    all_embeddings: list = []
    source_map: list[dict] = []

    for r in picked:
        url = r.get("url", "")
        if not url:
            continue

        cached = None if force_refresh else _cache.get_by_label(url)

        if cached is not None:
            _log("web_cache_hit", url=url)
            entry = cached
        else:
            try:
                title, text = fetch_and_extract(url)
            except FetchError as e:
                _log("fetch_error", url=url, error=str(e))
                continue

            _, chunks = chunk_file(url + ".md", text)
            try:
                embeddings = _embed_chunks([c.content for c in chunks])
            except EmbedderError as e:
                _log("embed_error", url=url, error=str(e))
                continue

            fetch_time = time.time()
            entry_id = make_web_id(url, fetch_time)
            entry = CachedEntry(
                entry_id=entry_id,
                source="web",
                label=url,
                chunks=chunks,
                embeddings=embeddings,
                fetch_time=fetch_time,
            )
            _cache.put(entry)
            _log("web_cached", url=url, entry_id=entry_id, n_chunks=len(chunks))

        all_chunks.extend(entry.chunks)
        all_embeddings.extend(entry.embeddings)
        source_map.extend([{"url_or_path": url, "title": r.get("title", url)}] * len(entry.chunks))

    if not all_chunks:
        return _error("no sources could be fetched or embedded")

    try:
        query_vec = embed_one(question)
    except EmbedderError as e:
        return _error(f"embed server unreachable for query ranking: {e}")
    scored = [(cosine_similarity(query_vec, all_embeddings[i]), i)
              for i in range(len(all_chunks))]
    scored.sort(reverse=True)
    top = scored[:max(1, min(DEFAULT_TOP_K, len(scored)))]

    if return_chunks:
        results = []
        for score, i in top:
            c = all_chunks[i]
            results.append({
                "chunk_id": c.chunk_id,
                "score": round(score, 4),
                "content": c.content,
                "metadata": {**c.metadata, "url_or_path": source_map[i]["url_or_path"]},
            })
        _log("research_chunks_returned", n=len(results), elapsed_sec=round(time.time() - t0, 2))
        return {"layer": "chunks", "query": question, "results": results, "can_escalate": False}

    top_chunks_for_summary = [
        {
            "content": all_chunks[i].content,
            "url_or_path": source_map[i]["url_or_path"],
            "title": source_map[i]["title"],
        }
        for _, i in top
    ]
    result = summarize(question, top_chunks_for_summary)
    _log("research_done", confidence=result.confidence, elapsed_sec=round(time.time() - t0, 2))
    return result.to_dict(question)


@mcp.tool()
def read_file(
    path: str,
    query: str,
    return_chunks: bool = False,
) -> dict:
    """Answer a question about a local file's content.

    Returns a summary by default. If the summary is insufficient, call again
    with return_chunks=True to get ranked verbatim chunks (same round).

    Prefer this over the built-in read tool when understanding something
    specific in a large file — it protects primary context by returning only
    what is relevant. Use read when you need the file verbatim.

    Escalation: same 3-round protocol as research(). A new query on the same
    file is a new round; return_chunks=True on the same query is not.

    Args:
        path: Absolute or workspace-relative path.
        query: The question or topic to match chunks against.
        return_chunks: If True, return ranked chunks instead of summary.

    Returns:
        Summary layer: {"layer": "summary", "query": ..., "summary": ...,
                        "sources": [...], "confidence": ..., "notes": ..., "can_escalate": true}
        Chunk layer:   {"layer": "chunks", "query": ..., "results": [...], "can_escalate": false}
        Error layer:   {"layer": "error", "error": ..., "can_escalate": false}
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return _error(f"file not found: {abs_path}")
    if not os.path.isfile(abs_path):
        return _error(f"not a regular file: {abs_path}")

    cached = _cache.lookup_file(abs_path)
    if cached is not None:
        _log("file_cache_hit", path=abs_path)
        entry = cached
    else:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            return _error(f"cannot read file: {e}")
        strategy, chunks = chunk_file(abs_path, content)
        try:
            embeddings = _embed_chunks([c.content for c in chunks])
        except EmbedderError as e:
            return _error(f"embed server unreachable: {e}")
        mtime = os.path.getmtime(abs_path)
        entry_id = make_file_id(abs_path, mtime)
        entry = CachedEntry(
            entry_id=entry_id,
            source="file",
            label=abs_path,
            chunks=chunks,
            embeddings=embeddings,
            fetch_time=mtime,
        )
        _cache.put(entry)
        _log("file_cached", path=abs_path, entry_id=entry_id, n_chunks=len(chunks))

    try:
        query_vec = embed_one(query)
    except EmbedderError as e:
        return _error(f"embed server unreachable for query ranking: {e}")
    scored = [(cosine_similarity(query_vec, entry.embeddings[i]), i)
              for i in range(len(entry.chunks))]
    scored.sort(reverse=True)
    top = scored[:max(1, min(DEFAULT_TOP_K, len(scored)))]

    if return_chunks:
        results = []
        for score, i in top:
            c = entry.chunks[i]
            results.append({
                "chunk_id": c.chunk_id,
                "score": round(score, 4),
                "content": c.content,
                "metadata": c.metadata,
            })
        return {"layer": "chunks", "query": query, "results": results, "can_escalate": False}

    top_chunks = [
        {"content": entry.chunks[i].content, "url_or_path": abs_path,
         "title": os.path.basename(abs_path)}
        for _, i in top
    ]
    result = summarize(query, top_chunks)
    return result.to_dict(query)


@mcp.tool()
def get_skill(name: str) -> dict:
    """Retrieve a skill's full instruction set by name.

    Skills are on-demand instruction sets stored in the Library's skills/
    directory. They are returned verbatim — no chunking, no embedding, no
    summarization. The primary model applies the skill's instructions directly.

    Available skills: annyvoice

    Args:
        name: Skill name without extension (e.g., "annyvoice").

    Returns:
        {"layer": "skill", "name": ..., "content": ...}
        or {"layer": "error", "error": ..., "can_escalate": false}
    """
    skill_path = SKILLS_DIR / f"{name}.md"
    if not skill_path.exists():
        available = [p.stem for p in SKILLS_DIR.glob("*.md")]
        return _error(f"skill '{name}' not found. Available: {available}")
    content = skill_path.read_text(encoding="utf-8")
    _log("skill_returned", name=name, chars=len(content))
    return {"layer": "skill", "name": name, "content": content}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _log("library_start")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
