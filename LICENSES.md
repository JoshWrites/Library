# Third-Party Components and Licenses

Library's own code (everything under `library/` and `tests/`) is
MIT-licensed; see [LICENSE](LICENSE) for the full text.

The full *running stack* combines this code with several upstream
Python packages and (when integrated with the 2gpu workstation umbrella)
sidecar services and model weights. Each has its own license. This
file lists the components a working install pulls in, so downstream
users know what they can do with the whole.

If you redistribute, modify, or build commercially on this stack,
read each component's actual LICENSE -- this list is a navigation
aid, not a substitute.

---

## Python dependencies (pyproject.toml)

| Package | License | Source |
|---|---|---|
| mcp\[cli\] | MIT | https://github.com/modelcontextprotocol/python-sdk |
| readability-lxml | Apache 2.0 | https://github.com/buriy/python-readability |
| beautifulsoup4 | MIT | https://www.crummy.com/software/BeautifulSoup/ |
| lxml | BSD-3-Clause | https://lxml.de |

Optional dev dependencies:

| Package | License | Source |
|---|---|---|
| pytest | MIT | https://docs.pytest.org |

## Sidecar services Library reaches at runtime

These are HTTP services Library calls; Library itself does not
bundle or distribute them. Each runs in its own process and you
install it separately.

| Service | License | Source | Default URL |
|---|---|---|---|
| llama.cpp llama-server (embeddings + summarizer) | MIT | https://github.com/ggml-org/llama.cpp | http://127.0.0.1:11437, :11435 |
| SearxNG | AGPL-3.0 | https://github.com/searxng/searxng | http://127.0.0.1:8888 |
| docling-serve | MIT | https://github.com/DS4SD/docling-serve | http://127.0.0.1:5001 |
| pandoc (used by export tool) | GPL-2.0+ | https://pandoc.org | (CLI) |

The default URLs above are documented for the 2gpu workstation
umbrella; override via the `LIBRARY_*_URL` environment variables
(see each module's docstring).

## Model weights

Library does not bundle model weights. The defaults assume:

- `multilingual-e5-large` for embeddings (MIT,
  https://huggingface.co/intfloat/multilingual-e5-large)
- `Qwen3-4B-Instruct-2507` for summarization (Apache 2.0,
  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF)

Override via `LIBRARY_EMBED_MODEL` and `LIBRARY_SUMMARIZE_MODEL`. The
embedding model's vector dimension must match whatever vector store
consumes the output -- if you swap the model and a downstream store
already exists, drop the existing collection first.

If you swap models, check the new model's license yourself. License
shapes you might run into:

- **Apache 2.0 / MIT** -- generally permissive for commercial use.
- **CC-BY-NC-4.0** -- non-commercial only.
- **Llama license** -- Meta-specific terms; read the model card.
- **Custom research-only** -- common on small experimental GGUFs.

---

## "Can I redistribute this whole stack?"

Library's MIT license covers Library's source. Each upstream component
follows its own license when you actually run it.

In particular:

- **You CAN ship your own fork or modification of Library** under MIT
  or any compatible license, with attribution.
- **You CANNOT ship pre-built model weights** without the upstream
  model's permission.
- **SearxNG is AGPL-3.0** -- if you build a public service that uses
  it, you may need to publish your modifications.
- **pandoc is GPL-2.0+** -- using its CLI is fine; linking against
  its library would have GPL implications.

For commercial deployments, audit each component's license against
your specific use case.
