"""Binary document conversion via docling-serve sidecar.

Posts file bytes to the local docling-serve daemon (default :5001) and
returns markdown. Callers feed the markdown into the existing document
chunker -- no format-specific chunking lives here.

Design: Library reads file bytes itself (it has the user's permissions);
docling-serve does not need filesystem access to user paths. This keeps
docling-serve isolated and lets it run as a dedicated system user.

Configuration (env vars, all optional):
  LIBRARY_DOCLING_URL  -- full /v1/convert/file URL. Default:
                          http://127.0.0.1:5001/v1/convert/file. Set to
                          empty string to disable docling integration
                          entirely; the convert tool will then return a
                          structured error for any binary input.
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


DOCLING_URL = os.environ.get(
    "LIBRARY_DOCLING_URL", "http://127.0.0.1:5001/v1/convert/file"
)
CONVERT_TIMEOUT_SEC = 120  # generous; large PDFs can take a while
MAX_BYTES = 50_000_000  # 50 MB cap per document

# Extensions docling-serve handles. Keep narrow in V1; expand as needed.
SUPPORTED_EXTS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".html", ".htm", ".epub",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
})

# docling-serve OutputFormat values we expose. Keys are the user-facing
# format names; values are the JSON keys docling returns the content under.
OUTPUT_FORMATS: dict[str, str] = {
    "md": "md_content",
    "json": "json_content",
    "html": "html_content",
    "text": "text_content",
    "doctags": "doctags_content",
}

# File extension to write for each output format.
FORMAT_EXTENSIONS: dict[str, str] = {
    "md": ".md",
    "json": ".json",
    "html": ".html",
    "text": ".txt",
    "doctags": ".doctags",
}


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
    return convert_to_format(path, output_format="md")


def convert_to_format(path: str, output_format: str = "md") -> str:
    """POST file bytes to docling-serve, return converted content as a string.

    Args:
        path: Path to the binary document.
        output_format: One of OUTPUT_FORMATS keys (md, json, html, text, doctags).

    Raises ConversionError on any failure.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ConversionError(
            f"unknown output_format {output_format!r}; "
            f"expected one of {sorted(OUTPUT_FORMATS)}"
        )

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
        fields={"to_formats": output_format},
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

    content_key = OUTPUT_FORMATS[output_format]
    content = (parsed.get("document") or {}).get(content_key)
    if content is None:
        raise ConversionError(f"docling response missing {content_key}")
    if not isinstance(content, str):
        # json/yaml come back as nested objects in some configs; serialize.
        content = json.dumps(content, indent=2)
    if not content.strip():
        raise ConversionError(f"docling returned empty {content_key}")
    return content


def convert_to_disk(
    src_path: str,
    dest_path: str | None = None,
    output_format: str = "md",
    overwrite: bool = False,
) -> dict:
    """Convert a binary document and write the result to disk.

    Returns metadata only -- no file content. Designed so the caller's
    primary-context footprint is the same regardless of source size.

    Args:
        src_path: Path to the binary document.
        dest_path: Output path. If None, defaults to <src_dir>/<src_stem><ext>
                   where <ext> matches output_format.
        output_format: One of OUTPUT_FORMATS keys. Default "md".
        overwrite: If False and dest exists, raise ConversionError.

    Returns:
        {"src_path", "dest_path", "output_format", "bytes"}

    Raises ConversionError on any failure.
    """
    abs_src = os.path.abspath(src_path)
    if not os.path.isfile(abs_src):
        raise ConversionError(f"src_path is not a regular file: {abs_src}")

    if output_format not in OUTPUT_FORMATS:
        raise ConversionError(
            f"unknown output_format {output_format!r}; "
            f"expected one of {sorted(OUTPUT_FORMATS)}"
        )

    if dest_path is None:
        stem = Path(abs_src).stem
        ext = FORMAT_EXTENSIONS[output_format]
        abs_dest = str(Path(abs_src).parent / f"{stem}{ext}")
    else:
        abs_dest = os.path.abspath(dest_path)

    if os.path.exists(abs_dest) and not overwrite:
        raise ConversionError(
            f"dest_path exists: {abs_dest} (pass overwrite=True to replace)"
        )
    if abs_dest == abs_src:
        raise ConversionError(f"dest_path equals src_path: {abs_src}")

    parent = os.path.dirname(abs_dest)
    if parent and not os.path.isdir(parent):
        raise ConversionError(f"dest_path parent does not exist: {parent}")

    content = convert_to_format(abs_src, output_format=output_format)
    try:
        with open(abs_dest, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        raise ConversionError(f"cannot write {abs_dest}: {e}") from e

    return {
        "src_path": abs_src,
        "dest_path": abs_dest,
        "output_format": output_format,
        "bytes": len(content.encode("utf-8")),
    }


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
