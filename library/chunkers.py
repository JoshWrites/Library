"""Chunking strategies for Librarian.

Two chunkers keyed on file extension. The strategy is chosen by extension at
dispatch time, not configured per-call.

- document: markdown / prose. Splits on header boundaries (# / ## / ###),
  with section-path metadata. Target ~500 tokens but respects header boundaries
  rather than hard-splitting.
- code: source files, config, markup. Fixed 500-token windows with 50-token
  overlap. No structural awareness; borrowed from the v1 code-oriented chunker.

The safety-net future work (auto-retry with alternate strategy + quality
comparison) hooks the same dispatch function; only the post-processing wrapper
changes.

TODO (v1 rejects binary): add a binary-parser chunker (PDF, ipynb, etc.) that
runs serialized against the text embedder (never concurrent) so VRAM doesn't
grow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# ── Dispatch ─────────────────────────────────────────────────────────────────

# Strategy lookup by file extension. Unknown extensions default to "code" --
# fixed-window chunking is the safe generalist; document chunking fails
# badly on files with no section structure.
CHUNKER_BY_EXT: dict[str, str] = {
    # Document / prose
    ".md": "document",
    ".markdown": "document",
    ".txt": "document",
    ".rst": "document",
    ".org": "document",
    # Source
    ".py": "code",
    ".js": "code",
    ".mjs": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".go": "code",
    ".rs": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".cc": "code",
    ".cs": "code",
    ".java": "code",
    ".kt": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".scala": "code",
    ".lua": "code",
    ".pl": "code",
    ".pm": "code",
    ".sh": "code",
    ".bash": "code",
    ".zsh": "code",
    ".fish": "code",
    # Config / data
    ".yml": "code",
    ".yaml": "code",
    ".toml": "code",
    ".json": "code",
    ".xml": "code",
    ".ini": "code",
    ".conf": "code",
    ".cfg": "code",
    ".env": "code",
    # Markup (treat as code; headers aren't markdown-style)
    ".html": "code",
    ".htm": "code",
    ".css": "code",
    ".scss": "code",
    ".sass": "code",
    ".less": "code",
    ".sql": "code",
    ".tex": "code",
    # Dockerfiles, makefiles: extension-less but the server checks names too
}

# File basenames that imply "code" when extension is missing/generic.
CODE_BASENAMES: set[str] = {
    "dockerfile",
    "makefile",
    "gnumakefile",
    "rakefile",
    "gemfile",
    "procfile",
    "caddyfile",
    "jenkinsfile",
    "vagrantfile",
}


def choose_strategy(path: str) -> str:
    """Return 'document' or 'code' based on the file's extension/basename."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in CHUNKER_BY_EXT:
        return CHUNKER_BY_EXT[ext]
    if p.name.lower() in CODE_BASENAMES:
        return "code"
    # Unknown -> safe generalist
    return "code"


# ── Chunk dataclass ──────────────────────────────────────────────────────────


@dataclass
class Chunk:
    chunk_id: int
    content: str
    byte_range: tuple[int, int]  # [start, end) in the source file
    metadata: dict[str, str | int]  # strategy-specific


# ── Token estimation (approximate, no tokenizer dep) ─────────────────────────

# mxbai-embed-large has a 512-token context hard limit. We size chunks to never
# overflow it even for token-dense content.
#
# Empirical: a 560-char Python chunk tokenized to 560 tokens (mxbai's tokenizer
# effectively allocates ~1 token per character for code with short identifiers,
# operators, and whitespace). So for the worst case we must assume 1 char = 1
# token. That gives a hard char ceiling of 500.
#
# For prose (~4 chars/token), 500 chars is under-sized, but that's okay -- the
# retriever will just produce more, smaller chunks, each still semantically
# coherent at the section level.
#
# TARGET_CHARS is the chunk-aim; MAX_CHARS is the hard limit we never cross.
TARGET_CHARS = 400
MAX_CHARS = 500
OVERLAP_CHARS = 50


# ── Document chunker (markdown / prose) ──────────────────────────────────────


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_document(content: str) -> list[Chunk]:
    """Chunk markdown-style prose on header boundaries.

    Strategy: split on ATX headers (`# H1`, `## H2`, etc.). Track section path
    as we descend the header hierarchy. If a section body exceeds TARGET_CHARS,
    sub-split it into overlapping windows while preserving the section path.

    For files with no headers, falls back to a single-document paragraph-based
    split.
    """
    chunks: list[Chunk] = []
    chunk_id = 0

    # Collect header positions
    headers = []
    for m in _HEADER_RE.finditer(content):
        headers.append((m.start(), m.end(), len(m.group(1)), m.group(2).strip()))

    if not headers:
        # No structure -> fall back to fixed-window splitting with paragraph
        # awareness (prefer splitting at double-newline boundaries when possible)
        return _chunk_fixed_windows(content, metadata_for_code=False)

    # Build sections: [(start, end, depth, title)]
    sections: list[tuple[int, int, int, str]] = []
    for i, (hstart, hend, depth, title) in enumerate(headers):
        section_end = headers[i + 1][0] if i + 1 < len(headers) else len(content)
        sections.append((hstart, section_end, depth, title))

    # Track the current section-path stack
    section_stack: list[tuple[int, str]] = []  # (depth, title)

    for sec_start, sec_end, depth, title in sections:
        # Pop deeper-or-equal levels so this header replaces them
        while section_stack and section_stack[-1][0] >= depth:
            section_stack.pop()
        section_stack.append((depth, title))
        section_path = " > ".join(t for _, t in section_stack)

        body = content[sec_start:sec_end]

        # If the section fits the target, one chunk. Otherwise sub-split.
        # (We cap at TARGET_CHARS, not MAX_CHARS, so we always have overlap
        # headroom. MAX_CHARS is a never-exceed safety net if someone tweaks
        # the chunker logic in the future.)
        if len(body) <= TARGET_CHARS:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=body,
                    byte_range=(sec_start, sec_end),
                    metadata={"section_path": section_path, "heading": title},
                )
            )
            chunk_id += 1
        else:
            # Sub-split oversized section with overlap
            sub_start = 0
            while sub_start < len(body):
                sub_end = min(sub_start + TARGET_CHARS, len(body))
                # Prefer splitting on a paragraph break if there's one nearby
                if sub_end < len(body):
                    window = body[sub_start:sub_end]
                    last_break = window.rfind("\n\n")
                    if last_break > TARGET_CHARS // 2:
                        sub_end = sub_start + last_break + 2
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=body[sub_start:sub_end],
                        byte_range=(sec_start + sub_start, sec_start + sub_end),
                        metadata={
                            "section_path": section_path,
                            "heading": title,
                            "window": "partial",
                        },
                    )
                )
                chunk_id += 1
                if sub_end == len(body):
                    break
                sub_start = sub_end - OVERLAP_CHARS

    return chunks


# ── Code chunker (fixed-window with overlap) ─────────────────────────────────


def chunk_code(content: str) -> list[Chunk]:
    """Chunk source files into fixed 500-token windows with 50-token overlap.

    No structural awareness. Metadata: byte_range and line_range.
    Borrowed (in spirit) from the v1 code-oriented chunker.
    """
    return _chunk_fixed_windows(content, metadata_for_code=True)


def _chunk_fixed_windows(content: str, *, metadata_for_code: bool) -> list[Chunk]:
    if not content:
        return []
    chunks: list[Chunk] = []
    chunk_id = 0
    pos = 0
    length = len(content)
    # Pre-compute line starts for line_range calculation (code case)
    line_starts: list[int] = []
    if metadata_for_code:
        line_starts.append(0)
        for i, ch in enumerate(content):
            if ch == "\n":
                line_starts.append(i + 1)

    while pos < length:
        end = min(pos + TARGET_CHARS, length)
        # Try to end on a line boundary for code readability
        if metadata_for_code and end < length:
            next_nl = content.rfind("\n", pos, end)
            if next_nl > pos + TARGET_CHARS // 2:
                end = next_nl + 1
        chunk_content = content[pos:end]
        metadata: dict[str, str | int] = {}
        if metadata_for_code:
            # Compute line range (1-based, inclusive-end)
            start_line = _line_for_offset(line_starts, pos)
            end_line = _line_for_offset(line_starts, end - 1)
            metadata["line_range"] = f"{start_line}-{end_line}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                content=chunk_content,
                byte_range=(pos, end),
                metadata=metadata,
            )
        )
        chunk_id += 1
        if end == length:
            break
        pos = end - OVERLAP_CHARS
    return chunks


def _line_for_offset(line_starts: list[int], offset: int) -> int:
    """Binary-search for the 1-based line number containing the given byte offset."""
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


# ── Main dispatch ────────────────────────────────────────────────────────────


def chunk(path: str, content: str) -> tuple[str, list[Chunk]]:
    """Dispatch on path extension and return (strategy_name, chunks)."""
    strategy = choose_strategy(path)
    if strategy == "document":
        return strategy, chunk_document(content)
    return strategy, chunk_code(content)
