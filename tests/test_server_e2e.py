"""End-to-end tests against live servers.

Prereqs:
  - llama-embed.service on :11437
  - llama-server (secondary) on :11435

Run: uv run python tests/test_server_e2e.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from library import cache as _cache_mod
from library import server as _server_mod


def _fresh_cache():
    _server_mod._cache = _cache_mod.Cache()


def _server_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


DOC_CONTENT = """# Homelab Network Architecture

## Physical Layout
The homelab sits behind a single WireGuard VPN endpoint.

## VLAN Segmentation
We segment into three VLANs.

### Management VLAN
Proxmox UI and SSH. Reachable only over VPN.

### DMZ VLAN
Public-facing services: web, reverse proxy.

### Trust VLAN
Internal services: NAS, git, matrix.

## Firewall Policy
Default deny. Explicit allow per destination port.
"""


def test_read_file_summary_layer():
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(DOC_CONTENT)
    try:
        _fresh_cache()
        result = _server_mod.read_file(path, "what VLAN is public-facing?")
        assert result["layer"] == "summary", f"expected summary, got: {result}"
        assert result["can_escalate"] is True
        assert "DMZ" in result["summary"] or result["confidence"] in ("medium", "low")
    finally:
        os.unlink(path)


def test_read_file_chunk_layer():
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(DOC_CONTENT)
    try:
        _fresh_cache()
        result = _server_mod.read_file(path, "firewall policy", return_chunks=True)
        assert result["layer"] == "chunks"
        assert result["can_escalate"] is False
        assert len(result["results"]) >= 1
        assert "score" in result["results"][0]
        assert "content" in result["results"][0]
    finally:
        os.unlink(path)


def test_read_file_cache_hit_on_second_call():
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(DOC_CONTENT)
    try:
        _fresh_cache()
        r1 = _server_mod.read_file(path, "firewall?")
        assert r1["layer"] == "summary"
        r2 = _server_mod.read_file(path, "DNS strategy?")
        assert r2["layer"] == "summary"
        assert "summary" in r2
    finally:
        os.unlink(path)


def test_get_skill_annyvoice():
    result = _server_mod.get_skill("annyvoice")
    assert result["layer"] == "skill"
    assert result["name"] == "annyvoice"
    assert len(result["content"]) > 100
    assert "voice" in result["content"].lower()


def test_get_skill_missing_returns_error():
    result = _server_mod.get_skill("does_not_exist")
    assert result["layer"] == "error"
    assert result["can_escalate"] is False
    assert "available" in result["error"].lower() or "not found" in result["error"].lower()


def test_read_file_missing_returns_error():
    _fresh_cache()
    result = _server_mod.read_file("/tmp/does_not_exist_xyz_library.md", "anything")
    assert result["layer"] == "error"
    assert result["can_escalate"] is False


def test_read_file_chunk_scores_are_ranked():
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(DOC_CONTENT)
    try:
        _fresh_cache()
        result = _server_mod.read_file(path, "DMZ public-facing services", return_chunks=True)
        assert result["layer"] == "chunks"
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True), "chunks should be ordered best-score-first"
    finally:
        os.unlink(path)


if __name__ == "__main__":
    missing = []
    if not _server_up("http://127.0.0.1:11437/v1/models"):
        missing.append("llama-embed on :11437")
    if not _server_up("http://127.0.0.1:11435/v1/models"):
        missing.append("llama-server (secondary) on :11435")
    if missing:
        print(f"SKIP: servers not reachable: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    import inspect
    current = sys.modules[__name__]
    tests = [(n, o) for n, o in inspect.getmembers(current)
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            t0 = time.monotonic()
            fn()
            print(f"PASS  {name:60s}  ({time.monotonic() - t0:.2f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
