#!/usr/bin/env python3
"""Library MCP server.

Three tools:

  research(question, max_sources, return_chunks, force_refresh)
    -> summary layer (default) or chunk layer (return_chunks=True)
    Web acquisition: SearxNG search -> fetch -> chunk -> embed -> rank -> summarize

  read_file(path, query, return_chunks)
    -> summary layer (default) or chunk layer (return_chunks=True)
    Local acquisition: read file -> chunk -> embed -> rank -> summarize

  get_skill(name)
    -> skill layer: full skill file contents, no pipeline

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
from .chunkers import chunk as chunk_file, chunk_document
from .converters import (
    ConversionError,
    OUTPUT_FORMATS as DOCLING_OUTPUT_FORMATS,
    convert_to_disk,
    convert_to_markdown,
    is_supported as is_binary_doc,
)
from .exporters import EXPORT_FORMATS, ExportError, export_to_disk
from .embedder import EmbedderError, cosine_similarity, embed_batch, embed_one
from .fetcher import FetchError, fetch_and_extract
from .opencode_state import get_active_session_state
from .searcher import search, rank_results
from .summarizer import summarize


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MAX_SOURCES = 5
DEFAULT_TOP_K = 5
SKILLS_DIR = Path(__file__).parent / "skills"
_EMBED_BATCH = 16


def _resolve_skill_dirs() -> list[Path]:
    """Return the ordered list of directories searched by get_skill().

    User-configured directories from WS_SKILLS_DIRS (colon-separated path
    list, like PATH) come first, in declaration order. The bundled
    SKILLS_DIR is always appended last so default skills remain
    available even when WS_SKILLS_DIRS is unset.

    Nonexistent directories are dropped silently with a stderr log line
    so a typo in user.env doesn't break the server. Path semantics:
    relative paths are resolved against the server's cwd at startup;
    absolute paths are preferred for stability across opencode sessions.
    """
    raw = os.environ.get("WS_SKILLS_DIRS", "")
    user_dirs: list[Path] = []
    seen: set[Path] = set()
    for part in raw.split(":"):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser().resolve()
        if p in seen:
            continue
        seen.add(p)
        if not p.is_dir():
            print(
                json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "skill_dir_missing",
                    "path": str(p),
                }),
                file=sys.stderr,
                flush=True,
            )
            continue
        user_dirs.append(p)
    # Bundled dir last so user dirs shadow on name collision.
    if SKILLS_DIR not in seen:
        user_dirs.append(SKILLS_DIR)
    return user_dirs


SKILL_DIRS: list[Path] = _resolve_skill_dirs()


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
    with return_chunks=True to get the ranked raw chunks (same round -- does
    not count toward the 3-round limit).

    Escalation protocol:
      Round 1: research(question)                       -> summary
               research(question, return_chunks=True)   -> chunks (same round)
      Round 2: research(refined_question)               -> summary
      Round 3: research(further_refined_question)       -> summary
      After 3 rounds without sufficient result: fetch directly (last resort).
      A new topic always resets to round 1.

    force_refresh=True bypasses the cache and re-fetches all sources. Use
    only when instructed by the user -- the user is a more reliable judge of
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
    specific in a large file -- it protects primary context by returning only
    what is relevant. Use read when you need the file verbatim.

    Supported formats:
      - Text: .md .txt .rst .py .js .ts .go .rs .json .yaml .toml ... (chunked directly)
      - Binary documents: .pdf .docx .pptx .xlsx .epub .html .htm
        and images .png .jpg .jpeg .tiff (converted to markdown via docling-serve
        sidecar, then chunked). Conversion needs the docling-serve daemon
        running on :5001; if down, returns a structured error.

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
        if is_binary_doc(abs_path):
            try:
                content = convert_to_markdown(abs_path)
            except ConversionError as e:
                return _error(f"document conversion failed: {e}")
            chunks = chunk_document(content)
            strategy = "document"
            _log("converted", path=abs_path, n_chunks=len(chunks), md_chars=len(content))
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
        _log("file_cached", path=abs_path, entry_id=entry_id, n_chunks=len(chunks), strategy=strategy)

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


def _list_available_skills() -> list[str]:
    """Return the deduped list of skill names visible across SKILL_DIRS.

    Earlier directories shadow later ones on name collision (PATH-style),
    so the returned name appears once even if multiple directories
    contain a skill of that stem.
    """
    seen: set[str] = set()
    names: list[str] = []
    for d in SKILL_DIRS:
        try:
            for p in d.glob("*.md"):
                if p.stem in seen:
                    continue
                seen.add(p.stem)
                names.append(p.stem)
        except OSError:
            continue
    names.sort()
    return names


@mcp.tool()
def get_skill(name: str) -> dict:
    """Retrieve a skill's full instruction set by name.

    Skills are on-demand instruction sets returned verbatim - no
    chunking, no embedding, no summarization. The primary model applies
    the skill's instructions directly.

    The Library searches a list of directories on each call. By default
    only the Library's bundled skills/ directory is searched. Users can
    add their own directories via the WS_SKILLS_DIRS environment
    variable (colon-separated path list, like PATH) -- typically set
    in ~/.config/workstation/user.env and propagated by
    opencode-session.sh. User directories take precedence over the
    bundled directory on name collision (first match wins). Available
    skills depend on what .md files exist across the searched
    directories at call time.

    Call with a name you expect to exist or with a placeholder; on miss
    the response lists what is available across all searched directories.

    Args:
        name: Skill name without extension (e.g., "voice-rewrite").

    Returns:
        {"layer": "skill", "name": ..., "content": ..., "source": "..."}
        or {"layer": "error", "error": ..., "can_escalate": false}
    """
    for d in SKILL_DIRS:
        skill_path = d / f"{name}.md"
        if skill_path.is_file():
            content = skill_path.read_text(encoding="utf-8")
            _log("skill_returned", name=name, chars=len(content), source=str(d))
            return {
                "layer": "skill",
                "name": name,
                "content": content,
                "source": str(d),
            }
    available = _list_available_skills()
    return _error(f"skill '{name}' not found. Available: {available}")


@mcp.tool()
def convert(
    src_path: str,
    dest_path: str | None = None,
    output_format: str = "md",
    overwrite: bool = False,
) -> dict:
    """Convert a binary document to a text format on disk.

    Use this when the user wants the *full* converted file (e.g. "convert
    foo.docx to markdown" or "give me the markdown of this PDF"), as
    opposed to read_file which returns a query-targeted summary or chunks.

    The converted content is written to disk and NOT returned -- the
    response is metadata only, so this tool's footprint on the primary
    context is constant regardless of source size.

    Backed by the docling-serve sidecar (default :5001). Supported source
    extensions: .pdf .docx .pptx .xlsx .epub .html .htm .png .jpg .jpeg
    .tiff .tif. 50 MB cap per document.

    For the inverse direction (markdown to .docx/.pdf/.odt/etc.), use
    the `export` tool.

    Args:
        src_path: Absolute or workspace-relative path to the source document.
        dest_path: Where to write the result. If omitted, writes to
                   <src_dir>/<src_stem><ext> where <ext> matches output_format
                   (e.g. foo.docx -> foo.md).
        output_format: One of: md, json, html, text, doctags. Default "md".
        overwrite: If False (default) and dest_path exists, return an error
                   instead of clobbering. Pass True only when the user has
                   explicitly asked to replace the existing file.

    Returns:
        Success: {"layer": "converted", "src_path": ..., "dest_path": ...,
                  "output_format": ..., "bytes": N}
        Error:   {"layer": "error", "error": ..., "can_escalate": false}
    """
    if not is_binary_doc(src_path):
        return _error(
            f"src_path extension not supported by docling-serve. "
            f"For text->binary export use the `export` tool instead."
        )
    if output_format not in DOCLING_OUTPUT_FORMATS:
        return _error(
            f"unknown output_format {output_format!r}; "
            f"expected one of {sorted(DOCLING_OUTPUT_FORMATS)}"
        )
    try:
        result = convert_to_disk(
            src_path=src_path,
            dest_path=dest_path,
            output_format=output_format,
            overwrite=overwrite,
        )
    except ConversionError as e:
        return _error(f"conversion failed: {e}")
    _log("converted_to_disk",
         src_path=result["src_path"],
         dest_path=result["dest_path"],
         output_format=output_format,
         bytes=result["bytes"])
    return {"layer": "converted", **result}


@mcp.tool()
def export(
    src_path: str,
    dest_path: str | None = None,
    output_format: str = "docx",
    overwrite: bool = False,
) -> dict:
    """Export a markdown (or other text) document to a binary format on disk.

    Inverse of `convert`. Use this when the user wants to produce a .docx,
    .pdf, .odt, .html, .epub, etc. from markdown they (or you) wrote.

    The result is written to disk; the response is metadata only.

    Backed by `pandoc` (system binary; install with `apt install pandoc`,
    plus `texlive-xetex` if you want PDF output).

    Source formats pandoc accepts: .md (default), .markdown, .rst, .html,
    .htm, .tex, .org, .txt. Files with other extensions are read as
    markdown.

    Args:
        src_path: Path to the source text document.
        dest_path: Where to write the result. If omitted, writes to
                   <src_dir>/<src_stem><ext> matching output_format.
        output_format: One of: docx, odt, rtf, html, epub, pdf, latex.
                       Default "docx".
        overwrite: If False (default) and dest_path exists, return an error.

    Returns:
        Success: {"layer": "exported", "src_path": ..., "dest_path": ...,
                  "output_format": ..., "bytes": N}
        Error:   {"layer": "error", "error": ..., "can_escalate": false}
    """
    if output_format not in EXPORT_FORMATS:
        return _error(
            f"unknown output_format {output_format!r}; "
            f"expected one of {sorted(EXPORT_FORMATS)}"
        )
    try:
        result = export_to_disk(
            src_path=src_path,
            dest_path=dest_path,
            output_format=output_format,
            overwrite=overwrite,
        )
    except ExportError as e:
        return _error(f"export failed: {e}")
    _log("exported_to_disk",
         src_path=result["src_path"],
         dest_path=result["dest_path"],
         output_format=output_format,
         bytes=result["bytes"])
    return {"layer": "exported", **result}


@mcp.tool()
def context_usage(directory: str | None = None) -> dict:
    """Report current context-window usage for the active opencode session.

    Use this when the user asks "how much context have I used", before a long
    operation that would consume significant context, or when deciding
    whether to start a new session. Returns ground-truth numbers from
    opencode's own session store -- no estimation.

    Resolution: prefers the most recent session whose working directory
    matches `directory` (default: this server's cwd, which is usually the
    agent's project root). If no session is found in that directory, falls
    back to the most recently updated session anywhere on the calling user's
    account. The `directory_match` field in the response signals which path
    was taken, so the answer can be honest about scope.

    Per-user by construction: an MCP server spawned by opencode runs as the
    calling user's uid and reads ~/.local/share/opencode/opencode.db, so
    each Linux user on the workstation sees only their own sessions.

    Args:
        directory: Optional working-directory filter. Defaults to the
                   server's cwd. Pass an explicit absolute path to inspect a
                   different project's session.

    Returns:
        Context layer:
            {"layer": "context",
             "session_id": "ses_...",
             "session_title": "Dev server startup",
             "session_directory": "/path/to/project",
             "directory_match": true,                  # false on fallback
             "model": "providerID/modelID",
             "context_limit": 65536,                   # tokens; null if unknown
             "current_tokens": 33962,                  # last turn total input
             "pct_used": 0.518,                        # null if limit unknown
             "input": 213,                             # last turn fresh input
             "output": 241,                            # last turn output
             "reasoning": 0,
             "cache_read": 33508,                      # cache reuse this turn
             "cache_write": 0,
             "turn_count": 47,                         # assistant turns total
             "session_age_min": 12.4}
        Error layer:  {"layer": "error", "error": ..., "can_escalate": false}
    """
    state = get_active_session_state(directory)
    if "error" in state:
        return _error(state["error"])
    return {"layer": "context", **state}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _log("library_start")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
