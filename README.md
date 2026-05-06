# Library

A local MCP (Model Context Protocol) server for opencode that unifies
web research, file mining, and binary-document conversion behind a
single tool surface. Each tool returns a *summary* by default to
protect the agent's primary context budget; raw chunks are available
on demand.

Library was built for the
[2gpu-local-ai-workstation](https://github.com/JoshWrites/2gpu-local-ai-workstation)
homelab stack -- a dual-AMD-GPU box where the chat model lives on the
big card and Library fans retrieval out to small services on the
secondary card and CPU. The integration there is well-trodden, but
nothing about Library *requires* the umbrella stack: every sidecar
URL is overridable via environment variables, so any host with the
right HTTP services can drive Library directly.

## What it does

Five MCP tools, three return contracts:

| Tool                    | Purpose                                                  |
|-------------------------|----------------------------------------------------------|
| `library_research`      | Web search → fetch → chunk → embed → rank → summarize    |
| `library_read_file`     | Local file (or binary doc) → chunk → embed → rank → summarize |
| `library_convert`       | Binary document → text format on disk (docling sidecar)  |
| `library_export`        | Markdown (or other text) → binary format on disk (pandoc)|
| `library_context_usage` | Active opencode session token usage from the local DB    |

**Summary tools.** `research` and `read_file` return a focused
summary by default. If the summary is insufficient, the same call
with `return_chunks=True` returns the top-ranked verbatim chunks.
This protects primary-model context -- a 50-page PDF returns
~1-5 KB of distilled answer, not 30 KB of raw text. Measured
compaction ratios run 22-43× depending on per-source bloat (see the
umbrella's
[stack one-sheet](https://github.com/JoshWrites/2gpu-local-ai-workstation/blob/main/docs/research/2026-05-03-stack-one-sheet.md)
for the methodology).

**Disk-write tools.** `convert` and `export` write the result to
disk and return only metadata (path, byte count). The agent never
sees the converted content. Use them when the user wants the *full*
file, not a summary. Cost on primary context is constant (~150 tokens)
regardless of source size.

`convert` accepts pdf, docx, pptx, xlsx, epub, html, png, jpg, tiff;
emits md (default), json, html, text, or doctags. `export` is the
inverse -- markdown (or rst, html, tex, org, txt) to docx, odt, rtf,
html, epub, pdf, or latex.

`context_usage` reads opencode's own SQLite session store and
reports the active session's token total vs. the model's configured
context limit. Use when the user asks "how much context have I used"
-- burns ~150 tokens instead of the ~1500-token guess the agent
would otherwise produce. Per-user by construction: each user's MCP
subprocess reads their own `~/.local/share/opencode/opencode.db`,
never anyone else's.

## Architecture (one screen)

```
                      opencode primary model
                               v (MCP/stdio)
                        Library MCP server  ──────────────────────┐
                           v (HTTP localhost)                     │
    ┌──────────────────────┼──────────────────────┐              │
    v                      v                      v              v
  SearxNG :8888       embeddings :11437      summarizer :11435   docling-serve :5001
  (search)            (e.g. e5-large)        (e.g. Qwen3-4B)     (binary doc -> markdown)
```

Library is the orchestrator. All four sidecars are independent
services with their own lifecycles. If any sidecar is down, Library
returns a structured `{"layer": "error", ...}` -- the MCP protocol
never sees a crash.

For the *why* behind the design (two-layer retrieval, 3-round
escalation, force-refresh ownership), see
[docs/architecture.md](docs/architecture.md).

## Prerequisites

**Sidecar services.** All four are HTTP services Library reaches over
localhost. Library calls *clients* of these services; you install and
run them yourself.

| Service          | Default URL                  | Override env var       |
|------------------|------------------------------|------------------------|
| SearxNG          | `http://127.0.0.1:8888`      | `LIBRARY_SEARXNG_URL`  |
| Embeddings       | `http://127.0.0.1:11437`     | `LIBRARY_EMBED_URL`    |
| Summarizer       | `http://127.0.0.1:11435`     | `LIBRARY_SUMMARIZE_URL`|
| docling-serve    | `http://127.0.0.1:5001`      | `LIBRARY_DOCLING_URL`  |

The defaults match the 2gpu-local-ai-workstation port layout. To run
Library against a different deployment, set the env vars when
launching the MCP subprocess (in opencode's `mcp.<name>.command.env`
block, or in your shell environment if you're driving Library
directly).

**Model expectations.**

| Sidecar         | Default model                 | Override env var          | Notes                                                  |
|-----------------|-------------------------------|---------------------------|--------------------------------------------------------|
| Embeddings      | `multilingual-e5-large`       | `LIBRARY_EMBED_MODEL`     | Must produce 1024-dim vectors if your downstream store needs that. |
| Summarizer      | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `LIBRARY_SUMMARIZE_MODEL` | Must be chat/instruct-tuned (not a base/completion model). |

If you swap the embedding model, any vector store that already
embedded under the old model will silently reject writes from the
new one (dimension mismatch). Drop downstream collections first.

**System binaries on PATH.**

- `pandoc` -- required by `library_export`. `sudo apt install pandoc`.
- `texlive-xetex` -- required only if `library_export` should produce
  PDF. `sudo apt install texlive-xetex`. Other export targets (docx,
  odt, html, epub, rtf, latex) need only pandoc.

`library_convert` does not need pandoc; it goes through docling-serve.
`library_export` does not need docling; it goes through pandoc. The
two engines never overlap.

## Quickstart

Library uses [uv](https://docs.astral.sh/uv/) for venv management.
If you don't have `uv` yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and bootstrap:

```bash
git clone https://github.com/JoshWrites/Library.git ~/Documents/Repos/Library
cd ~/Documents/Repos/Library
uv sync
```

Verify the entry point:

```bash
uv run --project ~/Documents/Repos/Library library --help
```

Wire it into opencode (`~/.config/opencode/opencode.json`):

```jsonc
{
  "mcp": {
    "library": {
      "type": "local",
      "command": ["/usr/local/bin/uv", "run", "--project",
                  "/home/<you>/Documents/Repos/Library", "library"],
      "enabled": true,
      "timeout": 180000
    }
  }
}
```

Restart opencode to pick up the new MCP. The 2gpu workstation
umbrella renders this block automatically; if you're integrating
elsewhere, edit the JSON yourself.

To run against non-default sidecar URLs, add an `env` block:

```jsonc
"mcp": {
  "library": {
    "type": "local",
    "command": [...],
    "env": {
      "LIBRARY_SEARXNG_URL": "http://my-searx.local:8888",
      "LIBRARY_EMBED_URL": "http://gpu1:11437/v1/embeddings",
      "LIBRARY_SUMMARIZE_URL": "http://gpu1:11435/v1/chat/completions",
      "LIBRARY_DOCLING_URL": "http://docling.local:5001/v1/convert/file"
    },
    "enabled": true
  }
}
```

## Languages

**Retrieval (embedding) is multilingual** with the default
`multilingual-e5-large`. Cross-language retrieval works: an English
query against a Hebrew document ranks the relevant Hebrew chunks
correctly, and vice versa. Coverage spans English, Hebrew, Arabic,
Russian, Chinese, and 90+ other languages.

**Summarization is currently English-primary** with the default
Qwen3-4B summarizer. The model handles English well but degrades
on non-English source material -- summaries may be generic,
hallucinated, or default to English explanations of non-English
content. For non-English documents, escalate to `return_chunks=True`
early; the chunks themselves are faithful to the source language.

A bilingual summarizer swap (e.g. Aya Expanse 8B) is planned but
not yet shipped.

## Module map

```
library/
├── server.py        MCP entry point. Tool definitions, dispatch, logging.
├── cache.py         DRAM LRU cache. Files (mtime-keyed) + web pages (URL-keyed).
├── chunkers.py      Document (header-aware) and code (fixed-window) strategies.
├── converters.py    Binary doc -> markdown via docling-serve HTTP.
├── embedder.py      Cosine similarity + batch HTTP client for the embeddings sidecar.
├── exporters.py     Markdown -> binary format on disk via pandoc.
├── fetcher.py       Web fetch with SSRF defense and readability extraction.
├── opencode_state.py Reads the local opencode SQLite store for context_usage.
├── searcher.py      SearxNG queries + domain-boosted ranking.
└── summarizer.py    Sidecar-model summary with structured low-confidence fallback.
```

## Running tests

```bash
# Unit tests -- no live services needed
uv run pytest tests/test_cache.py tests/test_chunkers.py tests/test_fetcher.py \
               tests/test_searcher.py tests/test_summarizer.py

# Converter tests -- mocked HTTP plus a hermetic e2e against running docling-serve
uv run python tests/test_converters.py

# End-to-end against live embeddings + summarizer sidecars
uv run python tests/test_server_e2e.py
```

The e2e suites print `SKIP` and exit 2 if their required servers
aren't reachable -- they don't fail.

## Logs

Library logs JSONL to stderr -- one event per line:

```json
{"ts": "2026-04-26T16:42:09Z", "event": "converted", "path": "...", "n_chunks": 62, "md_chars": 21546}
{"ts": "2026-04-26T16:42:11Z", "event": "file_cached", "path": "...", "entry_id": "f_fdf7a4b64dec", "n_chunks": 62, "strategy": "document"}
```

Useful event names: `research_start`, `research_done`, `picked`,
`web_cache_hit`, `web_cached`, `file_cache_hit`, `file_cached`,
`converted`, `embed_error`, `fetch_error`. Filter with
`jq -c 'select(.event == "converted")'`.

## When something breaks

| Symptom                                       | First check                                 |
|-----------------------------------------------|---------------------------------------------|
| `embed server unreachable`                    | `systemctl status <your-embeddings-unit>`   |
| `docling-serve unreachable at ...:5001`       | `systemctl status docling-serve` (or your equivalent) |
| `SearxNG returned no results`                 | `curl "$LIBRARY_SEARXNG_URL/search?q=test&format=json"` |
| Slow first call after restart                 | Sidecars warming up -- normal, ~10s          |
| Summary always low-confidence                 | Summarizer down → graceful fallback to chunks |

Errors are always structured: `{"layer": "error", "error": "...",
"can_escalate": false}`. The primary model is told to fall back to
direct webfetch/read after 3 rounds of insufficient results.

## License

[MIT](LICENSE). Third-party components and licenses for sidecar
services and model weights are listed in [LICENSES.md](LICENSES.md).
