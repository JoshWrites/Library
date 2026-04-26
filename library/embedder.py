"""HTTP client for the llama-embed server on :11437.

Uses the OpenAI-compatible /v1/embeddings endpoint. Batches sent as a single
request to leverage llama-server's internal batching.

No retry here — retry policy lives at the caller level (the server.py tool).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


EMBED_URL = "http://127.0.0.1:11437/v1/embeddings"
EMBED_MODEL = "mxbai-embed-large"
REQUEST_TIMEOUT_SEC = 30

# mxbai-embed-large was trained at 512 tokens; llama-server rejects inputs
# that exceed the configured context (-c 512). We defensively cap the
# character count per input so even token-dense content (worst case 1 char
# per token) cannot overflow.
MAX_CHARS_PER_INPUT = 500


class EmbedderError(RuntimeError):
    pass


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns a list of vectors in the same order.

    Raises EmbedderError on HTTP or parsing failure. No partial results —
    either every text gets a vector, or the call raises.

    Inputs longer than MAX_CHARS_PER_INPUT are truncated defensively to
    prevent the embed server from 500-ing on over-budget text. Chunkers
    should stay under this limit; truncation here is a safety net.
    """
    if not texts:
        return []

    safe_texts = [t[:MAX_CHARS_PER_INPUT] for t in texts]
    payload = json.dumps({"model": EMBED_MODEL, "input": safe_texts}).encode("utf-8")
    req = urllib.request.Request(
        EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise EmbedderError(f"embed HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from e
    except urllib.error.URLError as e:
        raise EmbedderError(f"embed unreachable at {EMBED_URL}: {e.reason}") from e
    except TimeoutError as e:
        raise EmbedderError(f"embed timeout after {REQUEST_TIMEOUT_SEC}s") from e

    try:
        resp_json = json.loads(body)
    except json.JSONDecodeError as e:
        raise EmbedderError(f"embed returned invalid JSON: {e}") from e

    data = resp_json.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise EmbedderError(
            f"embed returned {len(data) if isinstance(data, list) else 'non-list'} "
            f"vectors for {len(texts)} inputs"
        )

    vectors: list[list[float]] = []
    for item in data:
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise EmbedderError("embed response item missing 'embedding'")
        vectors.append(emb)
    return vectors


def embed_one(text: str) -> list[float]:
    """Embed a single query. Convenience wrapper."""
    return embed_batch([text])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Used for ranking chunks against a query."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
