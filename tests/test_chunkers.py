"""Strategy dispatch tests — no network, no embed server needed.

Verifies that the chunker chooses the right strategy based on file extension
and basename. This is the V1 "hardcoded dispatch" behavior; future auto-retry
wrappers will sit above this function and leave it unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running the test file directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from library.chunkers import choose_strategy, chunk, chunk_document, chunk_code


def test_extensions_route_to_document():
    docs = ["notes.md", "/tmp/readme.markdown", "rfc.txt", "api.rst", "life.org"]
    for path in docs:
        assert choose_strategy(path) == "document", f"{path} should be document"


def test_extensions_route_to_code():
    code = [
        "server.py", "main.go", "src/app.ts", "index.js",
        "config.yml", "settings.toml", "package.json", "Caddyfile.html",
        "db.sql", "styles.css", "shell.sh", "build.rs",
    ]
    for path in code:
        assert choose_strategy(path) == "code", f"{path} should be code"


def test_extensionless_basenames_route_to_code():
    basenames = ["Dockerfile", "Makefile", "Caddyfile", "Jenkinsfile"]
    for name in basenames:
        assert choose_strategy(f"/tmp/{name}") == "code", f"{name} should be code"


def test_unknown_extension_falls_back_to_code():
    # Truly unknown extension — defaults to code (safe generalist)
    assert choose_strategy("weird.xyzzy") == "code"


def test_document_chunker_splits_on_headers():
    md = """# Intro
Hello world.

## Section A
Alpha content paragraph.

## Section B
Beta content paragraph.
"""
    strategy, chunks = chunk("/tmp/test.md", md)
    assert strategy == "document"
    assert len(chunks) == 3, f"expected 3 header sections, got {len(chunks)}"
    headings = [c.metadata.get("heading") for c in chunks]
    assert headings == ["Intro", "Section A", "Section B"]


def test_document_chunker_tracks_section_path():
    md = """# Top
Alpha.

## Middle
Beta.

### Leaf
Gamma.
"""
    _, chunks = chunk("/tmp/t.md", md)
    paths = [c.metadata.get("section_path") for c in chunks]
    assert paths == ["Top", "Top > Middle", "Top > Middle > Leaf"]


def test_code_chunker_includes_line_range_metadata():
    code = "\n".join(f"line {i}" for i in range(1, 101))  # 100 lines
    strategy, chunks = chunk("/tmp/t.py", code)
    assert strategy == "code"
    assert all("line_range" in c.metadata for c in chunks), "every code chunk should have line_range"
    # Format sanity: "N-M" where N <= M
    for c in chunks:
        lr = c.metadata["line_range"]
        a, b = lr.split("-")
        assert int(a) <= int(b), f"line_range {lr} is not ordered"


def test_code_chunker_handles_empty_file():
    _, chunks = chunk("/tmp/empty.py", "")
    assert chunks == []


def test_document_chunker_headerless_falls_back_to_windows():
    # No headers → document chunker should still produce chunks
    text = "plain text with no headers.\n\n" * 200
    strategy, chunks = chunk("/tmp/plain.md", text)
    assert strategy == "document"
    assert len(chunks) > 0


if __name__ == "__main__":
    import inspect
    # Discover and run every test_* function in this module.
    current = sys.modules[__name__]
    tests = [(name, obj) for name, obj in inspect.getmembers(current)
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
