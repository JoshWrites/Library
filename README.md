# Library

A local MCP server for opencode that unifies web research, file mining, and on-demand skill injection into one tool surface. It's designed for a homelab two-user, two-GPU workstation -- the primary card stays loaded with the chat model, and all retrieval work happens on the secondary card or CPU.

Here's the thing: you want one tool that handles research, file mining, and skill injection all at once. That's what this does.

## What it does

Six MCP tools, two return contracts:

| Tool                    | Purpose                                       |
|-------------------------|-----------------------------------------------|
| `library_research`      | Web search -> fetch -> chunk -> embed -> rank -> summarize |
| `library_read_file`     | Local file (or binary doc) -> chunk -> embed -> rank -> summarize |
| `library_get_skill`     | Verbatim skill instruction set, no pipeline   |
| `library_convert`       | Binary document -> text format on disk (docling sidecar) |
| `library_export`        | Markdown (or other text) -> binary format on disk (pandoc) |
| `library_context_usage` | Current opencode session token usage from the local DB |

**Summary tools.** `research` and `read_file` return a **summary** by default.
If the summary is insufficient, the same call with `return_chunks=True`
returns the top-ranked verbatim chunks. This protects primary-model
context -- a 50-page PDF returns ~1-5 KB of distilled answer, not 30 KB
of raw text.

`get_skill` is a two-step tool: by default (`load=False`) it returns
metadata only -- description, an approximate token count, and the list
of helper files alongside the skill -- so the agent can ask the user
*"may I load skill X (~N tokens) for reason Y?"* before any of the
skill's content reaches the primary context. On approval, the agent
calls again with `load=True` to get the full skill content verbatim,
ready to apply.

Skills are looked up across a list of directories: the Library's
bundled `library/skills/` plus any directories the user lists in
`WS_SKILLS_DIRS` (colon-separated, like `PATH`). User directories take
precedence over the bundled set on name collision (first match wins).
Two on-disk formats are supported per directory: flat `<dir>/<name>.md`
and the Anthropic Skills convention of `<dir>/<name>/SKILL.md` (with
optional helper files alongside). Listing a missing skill aggregates
names across all directories and both formats, deduped quietly.

**Disk-write tools.** `convert` and `export` are different: they write the
result to disk and return only metadata (path, byte count). The agent
never sees the converted content. Use them when the user wants the *full*
file, not a summary. Cost on primary context is constant (~150 tokens)
regardless of source size.

`convert` accepts the same binary formats `read_file` does (pdf, docx,
pptx, xlsx, epub, html, png, jpg, tiff) and emits md (default), json,
html, text, or doctags. `export` is the inverse -- markdown (or rst,
html, tex, org, txt) to docx, odt, rtf, html, epub, pdf, or latex.

`context_usage` reads opencode's own SQLite session store and reports the
active session's token total vs. the model's configured context limit. Use
when the user asks "how much context have I used" -- burns ~150 tokens
instead of the ~1500-token guess the agent would otherwise produce. Per-user
by construction: each user's MCP subprocess reads their own
`~/.local/share/opencode/opencode.db`, never anyone else's.

## Architecture (one screen)

```
                      opencode primary model
                               v (MCP/stdio)
                        Library MCP server  ──────────────────────┐
                           v (HTTP localhost)                     │
    ┌──────────────────────┼──────────────────────┐              │
    v                      v                      v              v
  SearxNG :8888       llama-embed :11437       llama-summarize :11435   docling-serve :5001
  (search)            (multilingual-e5-large)  (Qwen3-4B)               (binary doc -> markdown)
                         GPU1 (Vulkan)            GPU1 (Vulkan)            CPU only
```

Library is the orchestrator. All four sidecars are independent system
services with their own lifecycles. If any sidecar is down, Library returns
a structured `{"layer": "error", ...}` -- the MCP protocol never sees a
crash.

A fifth service, `llama-coder` on `:11438` (Qwen2.5-Coder-3B, GPU1 / Vulkan),
also lives on the secondary card but is **not a Library client** -- it's the
edit-prediction backend for Zed. Library doesn't call it; it shares VRAM
with llama-embed and llama-summarize. See
[docs/edit-prediction-on-secondary-research.md](docs/edit-prediction-on-secondary-research.md)
for the validation story.

For the *why* behind the design (two-layer retrieval, 3-round escalation,
force-refresh ownership), see [docs/architecture.md](docs/architecture.md).

## Prerequisites

**Sidecar services:**

| Service           | Where it lives        | Purpose                                          |
|-------------------|-----------------------|--------------------------------------------------|
| SearxNG           | `127.0.0.1:8888`      | Privacy-respecting search aggregator             |
| llama-embed       | `127.0.0.1:11437`     | multilingual-e5-large, GPU1 (Vulkan)             |
| llama-summarize   | `127.0.0.1:11435`     | Qwen3-4B, GPU1 (Vulkan)                          |
| docling-serve     | `127.0.0.1:5001`      | Binary document conversion (CPU)                 |

The first three are part of the broader homelab AI stack (see
`Workstation/second-opinion`). docling-serve installs as a dedicated
system service -- see [docs/architecture.md](docs/architecture.md#docling-sidecar)
for the full setup.

**System binaries on PATH:**

- `pandoc` -- required by `library_export` (markdown -> docx/pdf/odt/etc.).
  Install via `sudo apt install pandoc`.
- `texlive-xetex` -- only required if you want `library_export` to produce
  PDF. Install via `sudo apt install texlive-xetex`. Other export targets
  (docx, odt, html, epub, rtf, latex) need only `pandoc`.

`library_convert` does not need pandoc; it goes through the docling-serve
sidecar. `library_export` does not need docling; it goes through pandoc.
The two engines never overlap.

## Quickstart

```bash
# Clone (or follow the same steps in your home)
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

Your setup is identical to mine. Both users get their own checkout, their own venv, and use the shared system sidecars.

## Languages

**Retrieval (embedding) is multilingual.** `multilingual-e5-large` covers
English, Hebrew, Arabic, Russian, Chinese, and 90+ other languages.
Cross-language retrieval works: an English query against a Hebrew document
ranks the relevant Hebrew chunks correctly, and vice versa.

**Summarization is currently English-primary.** The summarizer model
(Qwen3-4B) handles English well but degrades on non-English source
material -- summaries may be generic, hallucinated, or default to English
explanations of non-English content. For non-English documents, escalate
to `return_chunks=True` early; the chunks themselves are faithful to the
source language.

A bilingual summarizer swap (Aya Expanse 8B) is planned but not yet
shipped. See `local-mcp-servers/docs/superpowers/plans/2026-04-26-library-multilingual-progress.md`
for status and the resume path.

## Module map

```
library/
├── server.py        <- MCP entry point. Three tools, dispatch, logging.
├── cache.py         <- DRAM LRU cache. Files (mtime-keyed) + web pages (URL-keyed).
├── chunkers.py      <- Document (header-aware) and code (fixed-window) strategies.
├── converters.py    <- Binary doc -> markdown via docling-serve HTTP.
├── embedder.py      <- Cosine similarity + batch HTTP client for llama-embed.
├── fetcher.py       <- Web fetch with SSRF defense and readability extraction.
├── searcher.py      <- SearxNG queries + domain-boosted ranking.
├── summarizer.py    <- Secondary-model summary with structured fallback.
└── skills/
    └── (drop your skill markdown files here; empty by default)
```

These modules do what they say on the tin. The server is the entry point, cache handles DRAM LRU, chunkers do document vs. code strategies, converters handle binary docs, embedder does cosine similarity, fetcher handles web fetch with SSRF defense, searcher handles SearxNG queries, and summarizer handles the secondary-model summary. The skills folder is where you drop new skills.

## Running tests

```bash
# Unit tests -- no live services needed
uv run pytest tests/test_cache.py tests/test_chunkers.py tests/test_fetcher.py \
               tests/test_searcher.py tests/test_summarizer.py

# Converter tests -- mocked HTTP plus a hermetic e2e against running docling-serve
uv run python tests/test_converters.py

# End-to-end against live llama-embed + llama-summarize
uv run python tests/test_server_e2e.py
```

The e2e suites print `SKIP` and exit 2 if their required servers aren't
reachable -- they don't fail.

## Adding a skill

A skill is a markdown file in `library/skills/`. Drop the file in, restart
the MCP, and it's callable via `library_get_skill("filename_without_ext")`.

The first line should be a YAML frontmatter block declaring `name` and
`description`. The rest is verbatim instructions for the primary model.

Here's how you do it: you create a markdown file in the skills folder, write your instructions as plain text, and restart the MCP. That's it.

## Logs

Library logs JSONL to stderr -- one event per line:

```json
{"ts": "2026-04-26T16:42:09Z", "event": "converted", "path": "...", "n_chunks": 62, "md_chars": 21546}
{"ts": "2026-04-26T16:42:11Z", "event": "file_cached", "path": "...", "entry_id": "f_fdf7a4b64dec", "n_chunks": 62, "strategy": "document"}
```

Useful event names: `research_start`, `research_done`, `picked`, `web_cache_hit`,
`web_cached`, `file_cache_hit`, `file_cached`, `converted`, `skill_returned`,
`embed_error`, `fetch_error`. Filter with `jq -c 'select(.event == "converted")'`.

These events tell you what's happening under the hood -- when research starts, when files get cached, when chunks are converted, when skills are returned, and when errors occur. Keep an eye on them.

## When something breaks

| Symptom                                       | First check                                 |
|-----------------------------------------------|---------------------------------------------|
| `embed server unreachable`                    | `systemctl status llama-embed`              |
| `docling-serve unreachable at ...:5001`       | `systemctl status docling-serve`            |
| `SearxNG returned no results`                 | `curl http://127.0.0.1:8888/search?q=test`  |
| Slow first call after restart                 | Sidecars warming up -- normal, ~10s          |
| Summary always low-confidence                 | llama-summarize down -> graceful fallback    |

Errors are always structured: `{"layer": "error", "error": "...", "can_escalate": false}`.
The primary model is told to fall back to direct webfetch / read after
3 rounds of insufficient results.

## License

MIT. See the LICENSE file in the umbrella repo
(`2gpu-local-ai-workstation`) for the full text. Library is included
there as a submodule and shares its license.