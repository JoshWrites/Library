"""Secondary inference for Library.

Sends a question + list of text chunks to the secondary llama-server
and returns a SummaryResult. If the server is offline, returns a
low-confidence result so the caller can offer chunk escalation instead.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass


LLAMA_URL = "http://127.0.0.1:11435/v1/chat/completions"
LLAMA_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
DISTILL_TIMEOUT_SEC = 45

_SYSTEM_PROMPT = """\
You are a research assistant. You will be given a user question and a numbered
set of text chunks. Produce a concise answer grounded only in the chunks.

Rules:
- Never make confident claims the chunks don't support. If the chunks don't
  cover the question, return confidence "low" and say so in notes.
- Answer directly. No preamble.
- Maximum ~400 tokens of summary.
- Output ONLY valid JSON — no markdown fences, no surrounding prose:

{
  "summary": "<1-3 paragraphs>",
  "sources": [{"url_or_path": "<url or path>", "title": "<title>", "used_for": "<brief phrase>"}],
  "confidence": "high" | "medium" | "low",
  "notes": "<optional one-line hedge or gap>"
}
"""


@dataclass
class SummaryResult:
    summary: str
    sources: list[dict]
    confidence: str  # "high" | "medium" | "low"
    notes: str

    def to_dict(self, query: str) -> dict:
        return {
            "layer": "summary",
            "query": query,
            "summary": self.summary,
            "sources": self.sources,
            "confidence": self.confidence,
            "notes": self.notes,
            "can_escalate": True,
        }


def summarize(question: str, chunks: list[dict]) -> SummaryResult:
    """Call secondary model with question + chunks. Returns SummaryResult.

    chunks: list of {"content": str, "url_or_path": str, "title": str}

    Gracefully degrades: if the server is unreachable, returns a low-confidence
    result so the caller can offer chunk escalation instead.
    """
    bundle_lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        label = f"{c.get('title', '')} ({c.get('url_or_path', '')})"
        bundle_lines.append(f"[{i}] {label}:")
        bundle_lines.append(c.get("content", ""))
        bundle_lines.append("")

    user_msg = (
        f"Question: {question}\n\n"
        f"Chunks:\n{''.join(bundle_lines)}\n\n"
        "Return JSON only."
    )

    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 600,
        "temperature": 0.2,
        "stream": False,
    }

    fallback_sources = [
        {"url_or_path": c.get("url_or_path", ""), "title": c.get("title", ""), "used_for": "not consulted"}
        for c in chunks
    ]

    try:
        req = urllib.request.Request(
            LLAMA_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=DISTILL_TIMEOUT_SEC) as r:
            resp = json.loads(r.read())
        raw = resp["choices"][0]["message"]["content"]
    except Exception as e:
        return SummaryResult(
            summary="secondary model offline; request chunks for direct access",
            sources=fallback_sources,
            confidence="low",
            notes=f"llama-server error: {e}",
        )

    return _parse_output(raw, fallback_sources)


def _parse_output(raw: str, fallback_sources: list[dict]) -> SummaryResult:
    cleaned = raw.strip()
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return SummaryResult("library: no parseable JSON from secondary model",
                             fallback_sources, "low", "model output contained no JSON object")
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return SummaryResult("library: JSON parse failed", fallback_sources, "low", str(e))

    confidence = parsed.get("confidence")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return SummaryResult(
        summary=parsed.get("summary", "")[:4000],
        sources=parsed.get("sources") or fallback_sources,
        confidence=confidence,
        notes=parsed.get("notes", ""),
    )
