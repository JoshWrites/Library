"""Converter tests.

Unit tests use a mocked HTTP layer (no daemon needed). The e2e test at the
bottom hits the real docling-serve on :5001 if it's up; skipped otherwise.

Run: uv run python tests/test_converters.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from library.converters import (
    ConversionError,
    SUPPORTED_EXTS,
    convert_to_markdown,
    is_supported,
    _build_multipart,
)


def _fake_response(payload: dict) -> io.BytesIO:
    """Build a urlopen()-compatible response object from a dict payload."""
    buf = io.BytesIO(json.dumps(payload).encode())
    return buf


def test_is_supported_known_extensions():
    assert is_supported("/tmp/foo.pdf")
    assert is_supported("/tmp/foo.docx")
    assert is_supported("/tmp/foo.PPTX")  # case-insensitive
    assert is_supported("foo.epub")


def test_is_supported_rejects_text_formats():
    # Text formats go through the existing chunker path, not docling.
    assert not is_supported("/tmp/foo.md")
    assert not is_supported("/tmp/foo.py")
    assert not is_supported("/tmp/foo.txt")
    assert not is_supported("/tmp/foo.ipynb")  # not in V1 scope


def test_supported_exts_is_frozen():
    assert isinstance(SUPPORTED_EXTS, frozenset)


def test_build_multipart_well_formed():
    body, ctype = _build_multipart(
        filename="x.pdf",
        file_bytes=b"PDFDATA",
        fields={"to_formats": "md"},
    )
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    text = body.decode("latin-1")
    # Both fields present
    assert 'name="to_formats"' in text
    assert 'name="files"; filename="x.pdf"' in text
    assert "PDFDATA" in text
    # Closes with --boundary--
    assert text.rstrip("\r\n").endswith(f"--{boundary}--")


def test_convert_missing_file_raises():
    with pytest_raises(ConversionError, "cannot stat"):
        convert_to_markdown("/tmp/definitely_does_not_exist_xyz.pdf")


def test_convert_oversize_raises():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"hello")  # 5 bytes
    try:
        from library import converters as _c
        original = _c.MAX_BYTES
        _c.MAX_BYTES = 4  # smaller than file
        try:
            with pytest_raises(ConversionError, "too large"):
                convert_to_markdown(path)
        finally:
            _c.MAX_BYTES = original
    finally:
        os.unlink(path)


def test_convert_daemon_unreachable_raises():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"not really a pdf, but bytes are bytes")

    def fake_urlopen(*a, **kw):
        raise urllib.error.URLError("Connection refused")

    try:
        with patch("library.converters.urllib.request.urlopen", fake_urlopen):
            with pytest_raises(ConversionError, "unreachable"):
                convert_to_markdown(path)
    finally:
        os.unlink(path)


def test_convert_http_error_raises():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"x")

    def fake_urlopen(*a, **kw):
        raise urllib.error.HTTPError(
            url="http://x", code=500, msg="boom", hdrs={}, fp=io.BytesIO(b"server error")
        )

    try:
        with patch("library.converters.urllib.request.urlopen", fake_urlopen):
            with pytest_raises(ConversionError, "HTTP 500"):
                convert_to_markdown(path)
    finally:
        os.unlink(path)


def test_convert_status_failure_raises():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"x")

    def fake_urlopen(*a, **kw):
        return _CtxWrap(_fake_response({"status": "failure", "errors": ["boom"]}))

    try:
        with patch("library.converters.urllib.request.urlopen", fake_urlopen):
            with pytest_raises(ConversionError, "status="):
                convert_to_markdown(path)
    finally:
        os.unlink(path)


def test_convert_missing_md_content_raises():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"x")

    def fake_urlopen(*a, **kw):
        return _CtxWrap(_fake_response({"status": "success", "document": {}}))

    try:
        with patch("library.converters.urllib.request.urlopen", fake_urlopen):
            with pytest_raises(ConversionError, "missing md_content"):
                convert_to_markdown(path)
    finally:
        os.unlink(path)


def test_convert_success_returns_markdown():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"x")

    expected = "# Heading\n\nBody text."

    def fake_urlopen(*a, **kw):
        return _CtxWrap(_fake_response({
            "status": "success",
            "document": {"md_content": expected},
        }))

    try:
        with patch("library.converters.urllib.request.urlopen", fake_urlopen):
            md = convert_to_markdown(path)
        assert md == expected
    finally:
        os.unlink(path)


# ── E2E test: hits real docling-serve on :5001, skips if down ─────────────────

def test_e2e_real_docling_serve():
    """Convert a tiny synthesized DOCX through the real daemon.

    Skipped if docling-serve isn't reachable on :5001. We synthesize the
    DOCX inline so the test is hermetic -- no fixture file needed.
    """
    if not _docling_up():
        print(f"  SKIP test_e2e_real_docling_serve: docling-serve not on :5001")
        return

    # Build a minimal DOCX from raw zip bytes. We don't want a hard dep on
    # python-docx in the test suite, so we hand-roll the smallest possible
    # valid .docx file.
    docx_path = _make_minimal_docx()
    try:
        md = convert_to_markdown(docx_path)
        assert isinstance(md, str)
        assert len(md) > 0
        assert "Hello from Library" in md
    finally:
        os.unlink(docx_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _CtxWrap:
    """Wraps a BytesIO so it satisfies the with-statement contract urlopen uses."""
    def __init__(self, body: io.BytesIO):
        self._body = body
    def __enter__(self):
        return self._body
    def __exit__(self, *a):
        return False


def pytest_raises(exc_type, *substrs):
    """Tiny stand-in for pytest.raises so we don't import pytest just for one feature."""
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {exc_type.__name__}, no exception raised")
            if not issubclass(et, exc_type):
                return False
            msg = str(ev)
            for s in substrs:
                if s not in msg:
                    raise AssertionError(f"expected substr {s!r} in error, got: {msg!r}")
            return True
    return _Ctx()


def _docling_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:5001/health", timeout=2)
        return True
    except Exception:
        try:
            urllib.request.urlopen("http://127.0.0.1:5001/docs", timeout=2)
            return True
        except Exception:
            return False


def _make_minimal_docx() -> str:
    """Produce the smallest valid .docx file containing the phrase
    'Hello from Library'. .docx is just a zip of a few XML parts."""
    import zipfile
    fd, path = tempfile.mkstemp(suffix=".docx")
    os.close(fd)

    # Minimal three-part docx: [Content_Types].xml, word/_rels/document.xml.rels,
    # word/document.xml, and the package rels.
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    pkg_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Hello from Library</w:t></w:r></w:p>
  </w:body>
</w:document>"""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", pkg_rels)
        z.writestr("word/document.xml", document)
    return path


# ── Direct-run runner (matches existing test_*.py style) ──────────────────────

if __name__ == "__main__":
    import inspect
    tests = [(n, o) for n, o in inspect.getmembers(sys.modules[__name__])
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
