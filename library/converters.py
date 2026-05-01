"""Binary document conversion via docling-serve sidecar.

Posts file bytes to the local docling-serve daemon (default :5001) and
returns markdown. Callers feed the markdown into the existing document
chunker -- no format-specific chunking lives here.

Design: Library reads file bytes itself (it has the user's permissions);
docling-serve does not need filesystem access to user paths. This keeps
docling-serve isolated and lets it run as a dedicated system user.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

# Multipart formatting via stdlib only -- no extra deps.
import mimetypes
import os
import secrets


DOCLING_URL = "http://127.0.0.1:5001/v1/convert/file"
CONVERT_TIMEOUT_SEC = 120  # generous; large PDFs can take a while
MAX_BYTES = 50_000_000  # 50 MB cap per document

# Extensions docling-serve handles. Keep narrow in V1; expand as needed.
SUPPORTED_EXTS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".html", ".htm", ".epub",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
})


class ConversionError(RuntimeError):
    pass


def is_supported(path: str) -> bool:
    """True if docling-serve handles this extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def convert_to_markdown(path: str) -> str:
    """POST file bytes to docling-serve, return markdown.

    Raises ConversionError on any failure (file too large, daemon down,
    HTTP error, malformed response). Caller decides how to surface the
    error to the MCP client.
    """
    abs_path = os.path.abspath(path)
    try:
        size = os.path.getsize(abs_path)
    except OSError as e:
        raise ConversionError(f"cannot stat {abs_path}: {e}") from e
    if size > MAX_BYTES:
        raise ConversionError(
            f"file too large for conversion: {size} bytes > {MAX_BYTES}"
        )

    try:
        with open(abs_path, "rb") as f:
            file_bytes = f.read()
    except OSError as e:
        raise ConversionError(f"cannot read {abs_path}: {e}") from e

    body, content_type = _build_multipart(
        filename=os.path.basename(abs_path),
        file_bytes=file_bytes,
        fields={"to_formats": "md"},
    )
    req = urllib.request.Request(
        DOCLING_URL,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=CONVERT_TIMEOUT_SEC) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise ConversionError(
            f"docling HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
        ) from e
    except urllib.error.URLError as e:
        raise ConversionError(
            f"docling-serve unreachable at {DOCLING_URL}: {e.reason}"
        ) from e
    except TimeoutError as e:
        raise ConversionError(f"docling timeout after {CONVERT_TIMEOUT_SEC}s") from e

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConversionError(f"docling returned invalid JSON: {e}") from e

    if parsed.get("status") != "success":
        errors = parsed.get("errors") or "unknown"
        raise ConversionError(f"docling status={parsed.get('status')!r}: {errors}")

    md = (parsed.get("document") or {}).get("md_content")
    if not isinstance(md, str) or not md.strip():
        raise ConversionError("docling response missing md_content")
    return md


def _build_multipart(
    *, filename: str, file_bytes: bytes, fields: dict[str, str]
) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body. stdlib-only, no `requests` dep.

    Returns (body_bytes, content_type_header).
    """
    boundary = "----LibraryBoundary" + secrets.token_hex(8)
    crlf = b"\r\n"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode())

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="files"; filename="{filename}"'.encode()
    )
    parts.append(f"Content-Type: {mime}".encode())
    parts.append(b"")
    parts.append(file_bytes)
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")

    body = crlf.join(parts)
    return body, f"multipart/form-data; boundary={boundary}"
