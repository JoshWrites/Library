# Architecture

Library is the unification of two earlier MCP servers (`distiller` for web,
`librarian` for local files) plus a new on-demand skill mechanism plus a
binary-document conversion path. This doc explains *why* the design looks
the way it does -- the constraints that shaped each choice and the
tradeoffs we accepted.

## The problem statement

The original two-MCP setup had three weaknesses observed in real opencode
sessions:

1. **Asymmetric returns.** `distiller` returned a summary; `librarian`
   returned raw chunks. Same conceptual question ("what does this say
   about X?") got fundamentally different shapes depending on whether the
   source was online or local. The primary model had to handle both.

2. **Self-fetch context bloat.** When `distiller` results were
   insufficient, opencode reflexively reached for `webfetch` and pulled
   raw HTML into primary context. A 50 KB page got dropped whole into a
   64 KB context window. The summary had been the right shape; we just
   needed an escalation path that stayed structured.

3. **No room for non-text.** PDFs, DOCX, PPTX -- common in real research --
   couldn't go through `librarian` at all. The primary's only options
   were "open it manually" (impossible for binary) or "ask the user."

Library's design starts from those observations.

## Design principles

### Two-layer retrieval

Every retrieval call returns one of two layers:

- **Summary layer** (default): secondary-model-distilled answer with citations,
  ~1-5 KB regardless of source size.
- **Chunk layer** (`return_chunks=True`): top-ranked verbatim text chunks
  with metadata, for when the primary needs to see exactly what the source
  said.

Same call, same arguments -- only the `return_chunks` flag changes. The
primary picks the layer that fits the moment. This unifies the
distiller/librarian split: a question about a Wikipedia article gets the
same shape whether the article is online or saved locally.

The chunk layer **does not consume a research round**. Asking "show me the
sources for that summary" is the same intent as the original question,
not a new one.

### The 3-round escalation protocol

For each distinct topic in a turn:

| Round | Action |
|-------|--------|
| 1 | Default call -> summary. Optional same-call escalation to chunks. |
| 2 | Refined query -> summary. Optional chunks. |
| 3 | Further refined -> summary. Optional chunks. |
| Fallback | Direct `webfetch` or built-in `read` -- the Library has had its turn. |

A new topic resets to round 1. The Library is **stateless**; the primary
carries the round count. Documenting this explicitly in the tool docstring
and in `~/.config/opencode/AGENTS.md` is what makes the behavior actually
happen -- without it, the primary keeps calling the same tool with the same
query forever, or gives up after one bad summary and self-fetches.

### force_refresh is user-instructed

Web pages and local files are cached in DRAM after first access. The
cache is invalidated:

- **Files**: automatically when `mtime` changes (free correctness).
- **Web pages**: only when the user says "force refresh" or "the doc has
  changed" -- never automatically.

The reasoning: the primary model never sees the raw source. It cannot
judge whether a cached version is stale. The user *can* -- they have
out-of-band knowledge ("I just edited that page") that the model cannot
acquire. Putting the staleness decision in the user's hands is the only
honest design.

`research(question, force_refresh=True)` is the explicit knob. The MCP
prompt instructs the primary to set it only when the user asks.

### Graceful degradation everywhere

No tool call ever raises into the MCP protocol. Every error path returns:

```json
{ "layer": "error", "error": "...", "can_escalate": false }
```

This includes embed server unreachable, summarizer unreachable,
docling-serve unreachable, SearxNG unreachable, malformed file, network
DNS failure, anything. The `can_escalate: false` tells the primary "don't
try chunk-layer escalation; the path is broken at the infrastructure
level -- fall back to direct fetch/read."

Each sidecar can fail independently:

- **Embed server down** -> no ranking, but conversion + summarization still
  work? No -- without ranking we can't pick top chunks for the summary
  prompt either, so the whole call returns an error. (This is the
  weakest link; if it becomes a problem we add a "first-N-chunks"
  fallback.)
- **Summarizer down** -> the summarizer module catches the failure and
  returns a `low-confidence` SummaryResult containing the raw top chunks
  joined together. The primary sees `confidence: "low"` and chooses
  whether to escalate.
- **docling-serve down** -> only affects binary doc conversion;
  `read_file` on a `.txt` or `.md` works normally.
- **SearxNG down** -> only affects `research`; `read_file` and `get_skill`
  unaffected.
- **All sidecars down** -> all calls return errors; primary falls back to
  built-in tools. The MCP itself stays alive.

## The Y-shaped pipeline

Web acquisition and local acquisition share a chunk -> embed -> rank ->
summarize backbone. They differ only in the acquisition arms:

```
research(question)                read_file(path, query)
       v                                  v
  searcher.search()              looks_binary(path)?
       v                          v             v
  rank_results()           docling-serve    open() + decode
       v                          v             v
  for each URL:                   └─── content ─┘
    fetcher.fetch()                       v
       v                            chunkers.chunk()
       └──── content ────────────────────┘
                              v
                    chunkers.chunk(strategy = document | code)
                              v
                    embedder.embed_batch() ── llama-embed :11437
                              v
                    cache.put(entry)
                              v
                    cosine_similarity(query_vec, chunk_vecs)
                              v
                    top_k chunks
                              v
              ┌───────────────┴───────────────┐
              v                               v
       return_chunks=True              return_chunks=False
              v                               v
        chunk layer                  summarizer.summarize()
                                              v
                                         summary layer
```

The two arms diverge for ~150 lines (search and fetch vs. file open and
extension dispatch) and then merge. Once content is in `chunks`, the
pipeline doesn't know or care where it came from.

## Cache shape

A single `Cache` class (`library/cache.py`) holds both file and web
entries:

```python
@dataclass
class CachedEntry:
    entry_id: str          # "f_<hash>" for files, "w_<hash>" for web
    source: str            # "file" | "web"
    label: str             # absolute path or URL
    chunks: list[Chunk]
    embeddings: list[list[float]]
    fetch_time: float      # mtime for files, fetch wall-clock for web
    last_accessed: float
```

LRU eviction at 40 entries. Lookup by ID, by label, or `lookup_file(path)`
which checks mtime and auto-invalidates if the file has changed since
caching.

Why one cache instead of two: simpler, evicts both kinds equitably under
memory pressure, and a session that mixes web research with local file
mining doesn't end up with split memory budgets.

## Summarizer contract

The summarizer is the only inference component (besides embeddings).
It calls the secondary llama-server on `:11435` with a structured prompt
that asks for:

```json
{ "summary": "...", "confidence": "high|medium|low", "notes": "..." }
```

If the secondary returns invalid JSON, garbage, or anything, the
summarizer normalizes to `confidence: "low"` and packages whatever it can
salvage. **It never raises.** This is critical -- the secondary model is
the most failure-prone component (long generations, occasional drift) and
its failures must not break the MCP.

The contract returns a `SummaryResult` dataclass with a `to_dict(query)`
method that produces the wire shape:

```json
{
  "layer": "summary",
  "query": "...",
  "summary": "...",
  "sources": [{"url_or_path": "...", "title": "..."}, ...],
  "confidence": "high|medium|low",
  "notes": "...",
  "can_escalate": true
}
```

## Docling sidecar

Binary documents (PDF, DOCX, PPTX, XLSX, EPUB, HTML, images) go through
`docling-serve`, a Linux Foundation AI project that converts to clean
markdown. Library hits its HTTP API on `127.0.0.1:5001` and feeds the
resulting markdown into the existing document chunker -- no format-specific
chunking lives in Library itself.

### Why a sidecar, not in-process

- Docling pulls torch + transformers + ~5 GB of layout/table models. We
  don't want that weight in every Library process.
- Models load once at sidecar startup (~10s) and stay resident. Every
  conversion is fast (~0.5s for a 30 KB DOCX with many tables).
- Library can fail-open if docling-serve is down -- text files still work.
  In-process binding would mean a docling import crash kills the MCP.

### Why CPU-only torch

The workstation is AMD (no CUDA) and the GPUs are claimed by llama-server.
docling-serve runs on the Ryzen 5950X's 16 cores at ~1.5 pages/sec for
text-layer documents. The whole point is "don't unload the primary model
to ingest a document," so GPU contention is exactly what we're avoiding.

### Why not OCR

Default install includes RapidOCR for text extraction from scanned PDFs,
but Library's V1 doesn't exercise it. Most homelab research documents
have a text layer (docs, exported slides, tutorials). When scanned-PDF
support becomes a real need, the OCR path is already wired -- only a
config flag away.

### Why files travel via HTTP, not paths

docling-serve runs as a dedicated system user with `ProtectHome=yes`,
so it cannot read user home directories. Library reads file bytes
itself, with the session's user permissions, and POSTs them. This
keeps docling-serve isolated from user data and lets it serve every
local user without cross-home permission complexity.

## Skills

A skill is a markdown file in `library/skills/`. Calling
`library_get_skill(name)` returns the file content verbatim. No chunking,
no embedding, no summarization -- skills are *finalized instruction sets*
intended for the primary to follow as written.

Why a tool, not just bundling skills into AGENTS.md:

- Skills can be long (a single voice-rewriting pass clocks in at ~6 KB).
  Loading every skill into every session burns context.
- Skills are situational. A prose-rewriting skill applies to prose
  moments, not every turn. Pull-on-demand keeps context lean.
- New skills can land without rewriting AGENTS.md every time. Drop the
  markdown, restart the MCP, done.

The first skill was a voice-rewriting pass for the user's own prose,
and that one skill seeded this whole project: it needed to be callable
from opencode without bloating context, which led to "use an MCP,"
which led to "the existing MCPs have shape problems," which led to
Library.

## Multilingual support

Library handles non-English source documents (notably Hebrew, but also
Arabic, Russian, Chinese, and 90+ other languages the embed model
supports) without changing the primary chat model.

**Retrieval is multilingual.** `multilingual-e5-large` embeds queries
and chunks in any supported language into the same 1024-dim space.
Cross-language ranking works: an English query against a Hebrew
document returns the relevant Hebrew chunks, and vice versa.
Verification: empirical cosine-similarity tests on Hebrew/English
education-domain queries showed relevant content always ranking above
unrelated content regardless of query/passage language pair.

**Summarization is currently English-primary.** Qwen3-4B is the
secondary model. Its English summarization is solid; on non-English
sources it degrades -- sometimes producing generic summaries,
sometimes hallucinating, sometimes defaulting to English explanations
of non-English material. For non-English documents, consumers should
escalate to `return_chunks=True` early; chunks are faithful to the
source language even when summaries aren't.

**A bilingual summarizer swap is planned but not shipped.** The plan
calls for replacing Qwen3-4B with Aya Expanse 8B (one of 23
explicitly multilingual languages, Hebrew first-class trained). The
swap was paused on an open question: Aya runs ~3-4x slower than
Qwen3-4B on the secondary's Vulkan/5700XT slot, which would slow
*every* summary call (English included), not just non-English ones.
The dual-model alternative (Qwen3 for English, Aya for Hebrew) doesn't
fit on the 5700 XT's 8 GB VRAM. ROCm acceleration isn't viable on
gfx1010 (RDNA 1) hardware. The decision blocker is whether Aya's
English summarization quality is comparable to Qwen3's -- that
benchmark hasn't been run yet.

**VRAM budget update (2026-04-29):** the 5700 XT now hosts a third
resident -- `llama-coder` (Qwen2.5-Coder-3B, :11438) for Zed edit
prediction. Validated co-resident with summarize + embed under load
(see `edit-prediction-on-secondary-research.md`). Steady-state usage
is ~8.12 GB of the card's 8.57 GB; ~0.45 GB free. Any future swap that
*adds* VRAM pressure (Aya-8B, multi-model summarize) now has to
account for this third tenant.

See `local-mcp-servers/docs/superpowers/plans/2026-04-26-library-multilingual-progress.md`
for the full status, the resume path, and the cheap experiment that
unblocks the next step.

## Tradeoffs we accepted

- **One MCP for three jobs.** Could have kept three separate MCPs. We
  chose unification because (a) the failure domains are already shared
  (same machine, same sidecars), (b) the primary doesn't need to know
  which surface to call, and (c) the return-shape unification is the
  whole point.

- **DRAM-only cache, no persistence.** A session restart loses the
  cache. We chose this because cache invalidation is the wrong problem
  to solve at the same time as everything else. Pages are cheap to
  re-fetch; documents re-convert in 0.5s. If session-warm-restore
  becomes a need, it's an additive feature, not a redesign.

- **multilingual-e5-large at 512-token chunks.** Embedding model context
  cap is 512 tokens, so we chunk at 500 chars (worst-case 1 char ~= 1
  token) to never overflow. For prose this is small -- more, smaller
  chunks. We accept the granularity hit because retrieval quality at
  smaller chunk sizes is generally fine and the alternative is rewriting
  with a larger embed model. (Originally mxbai-embed-large; swapped to
  multilingual-e5-large for Hebrew + 90 other languages -- see the
  Multilingual support section below.)

- **No retry on inference failures.** If llama-summarize times out or
  returns garbage, we degrade rather than retry. Retries hide
  problems and waste time on a failure mode that almost never recovers
  in <1 second. Better to surface "low confidence, here's the raw
  chunks" than spin for 30 seconds.

- **Unbounded SSRF defense scope.** `is_safe_public_url` rejects all
  RFC1918 + loopback + multicast + reserved IPs at every redirect hop.
  This blocks legitimate use cases like fetching from a homelab service
  by IP. We accept this because the tool is a primary-model-callable
  surface and the risk of "model fetches private network resource on
  command" outweighs the convenience.

## Future directions (not in V1)

- **OCR for scanned PDFs.** Wire up the RapidOCR path; one config flag.
- **IPYNB native handling.** Convert via `nbconvert` or parse JSON cells
  directly -- preserves cell-type metadata.
- **Table-aware chunking.** Detect table boundaries in markdown and emit
  tables as whole chunks (or row-per-chunk for large tables) rather than
  letting the document chunker split mid-table.
- **Persistent cache.** SQLite or filesystem-backed entries so a session
  restart doesn't lose state. Only meaningful once we have heavy users.
- **More skills.** The voice-rewriting pass was the seed. As more
  proven instruction sets emerge, they drop in without code changes.

## Provenance

Library was built in a single subagent-driven session on 2026-04-26.
Plan: `local-mcp-servers/docs/superpowers/plans/2026-04-26-library-mcp.md`.
Module-level commits tell the build order: chunkers + embedder ported
from librarian, then cache, fetcher, searcher, summarizer, server, then
docling integration as a follow-on. 12 commits, 35 unit tests + 12
converter tests + 7 e2e tests = 54 total at first cut.
