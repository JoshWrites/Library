# Library

A local MCP server for opencode that unifies web research, file mining, and
on-demand skill injection into one tool surface. Designed for a homelab
two-user, two-GPU workstation where the primary card stays loaded with the
chat model and all retrieval work happens on the secondary card or CPU.

## What it does

Three MCP tools, one return contract:

| Tool                    | Purpose                                       |
|-------------------------|-----------------------------------------------|
| `library_research`      | Web search → fetch → chunk → embed → rank → summarize |
| `library_read_file`     | Local file (or binary doc) → chunk → embed → rank → summarize |
| `library_get_skill`     | Verbatim skill instruction set, no pipeline   |

`research` and `read_file` return a **summary** by default. If the summary is
insufficient, the same call with `return_chunks=True` returns the top-ranked
verbatim chunks. This protects primary-model context — a 50-page PDF returns
~1-5 KB of distilled answer, not 30 KB of raw text.

`get_skill` returns the full skill content as plain text — skills are
finalized instruction sets (e.g., voice-rewriting passes) that should land
in primary context exactly as written.

## Architecture (one screen)

```
                     opencode primary model
                              ↓ (MCP/stdio)
                       Library MCP server  ──────────────────────┐
                          ↓ (HTTP localhost)                     │
   ┌──────────────────────┼──────────────────────┐              │
   ↓                      ↓                      ↓              ↓
 SearxNG :8888       llama-embed :11437     llama-summarize :11435   docling-serve :5001
 (search)            (mxbai-embed-large)    (Qwen3-4B)               (binary doc → markdown)
                        GPU0                   GPU0                   CPU only
```

Library is the orchestrator. All four sidecars are independent system
services with their own lifecycles. If any sidecar is down, Library returns
a structured `{"layer": "error", ...}` — the MCP protocol never sees a
crash.

For the *why* behind the design (two-layer retrieval, 3-round escalation,
force-refresh ownership), see [docs/architecture.md](docs/architecture.md).

## Prerequisites

| Service           | Where it lives        | Purpose                              |
|-------------------|-----------------------|--------------------------------------|
| SearxNG           | `127.0.0.1:8888`      | Privacy-respecting search aggregator |
| llama-embed       | `127.0.0.1:11437`     | mxbai-embed-large, GPU0              |
| llama-summarize   | `127.0.0.1:11435`     | Qwen3-4B, GPU0                       |
| docling-serve     | `127.0.0.1:5001`      | Binary document conversion (CPU)     |

The first three are part of the broader homelab AI stack (see
`Workstation/second-opinion`). docling-serve installs as a dedicated
system service — see [docs/architecture.md](docs/architecture.md#docling-sidecar)
for the full setup.

## Quickstart

```bash
# Clone (Anny: or follow the same steps in your home)
git clone git@github.com:JoshWrites/Library.git ~/Documents/Repos/Library
cd ~/Documents/Repos/Library

# Build the venv and install deps
uv sync

# Wire it into opencode (~/.config/opencode/opencode.json):
# {
#   "mcp": {
#     "library": {
#       "type": "local",
#       "command": ["/usr/local/bin/uv", "run", "--project",
#                   "/home/<you>/Documents/Repos/Library", "library"],
#       "enabled": true,
#       "timeout": 180000
#     }
#   }
# }

# Restart opencode to pick up the new MCP.
```

Anny's setup is identical except for the path. Both users get their own
checkout, their own venv, and use the shared system sidecars.

## Module map

```
library/
├── server.py        ← MCP entry point. Three tools, dispatch, logging.
├── cache.py         ← DRAM LRU cache. Files (mtime-keyed) + web pages (URL-keyed).
├── chunkers.py      ← Document (header-aware) and code (fixed-window) strategies.
├── converters.py    ← Binary doc → markdown via docling-serve HTTP.
├── embedder.py      ← Cosine similarity + batch HTTP client for llama-embed.
├── fetcher.py       ← Web fetch with SSRF defense and readability extraction.
├── searcher.py      ← SearxNG queries + domain-boosted ranking.
├── summarizer.py    ← Secondary-model summary with structured fallback.
└── skills/
    └── annyvoice.md ← Voice-rewriting pass for Anny's prose.
```

## Running tests

```bash
# Unit tests — no live services needed
uv run pytest tests/test_cache.py tests/test_chunkers.py tests/test_fetcher.py \
              tests/test_searcher.py tests/test_summarizer.py

# Converter tests — mocked HTTP plus a hermetic e2e against running docling-serve
uv run python tests/test_converters.py

# End-to-end against live llama-embed + llama-summarize
uv run python tests/test_server_e2e.py
```

The e2e suites print `SKIP` and exit 2 if their required servers aren't
reachable — they don't fail.

## Adding a skill

A skill is a markdown file in `library/skills/`. Drop the file in, restart
the MCP, and it's callable via `library_get_skill("filename_without_ext")`.

The first line should be a YAML frontmatter block declaring `name` and
`description`. The rest is verbatim instructions for the primary model.

## Logs

Library logs JSONL to stderr — one event per line:

```json
{"ts": "2026-04-26T16:42:09Z", "event": "converted", "path": "...", "n_chunks": 62, "md_chars": 21546}
{"ts": "2026-04-26T16:42:11Z", "event": "file_cached", "path": "...", "entry_id": "f_fdf7a4b64dec", "n_chunks": 62, "strategy": "document"}
```

Useful event names: `research_start`, `research_done`, `picked`, `web_cache_hit`,
`web_cached`, `file_cache_hit`, `file_cached`, `converted`, `skill_returned`,
`embed_error`, `fetch_error`. Filter with `jq -c 'select(.event == "converted")'`.

## When something breaks

| Symptom                                       | First check                                 |
|-----------------------------------------------|---------------------------------------------|
| `embed server unreachable`                    | `systemctl status llama-embed`              |
| `docling-serve unreachable at ...:5001`       | `systemctl status docling-serve`            |
| `SearxNG returned no results`                 | `curl http://127.0.0.1:8888/search?q=test`  |
| Slow first call after restart                 | Sidecars warming up — normal, ~10s          |
| Summary always low-confidence                 | llama-summarize down → graceful fallback    |

Errors are always structured: `{"layer": "error", "error": "...", "can_escalate": false}`.
The primary model is told to fall back to direct webfetch / read after
3 rounds of insufficient results.

## License

Private. Do not redistribute.
