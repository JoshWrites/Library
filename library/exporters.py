"""Markdown -> binary document export via pandoc.

Inverse of converters.py. Reads markdown (or other text source formats
pandoc accepts) and produces docx/pdf/odt/html/epub/etc. on disk.
Returns metadata only -- no content reaches the caller's context.

pandoc is invoked as a subprocess; no Python deps. Add `pandoc` to the
system install (apt: `sudo apt install pandoc`; for PDF output also
install a TeX engine, e.g. `texlive-xetex`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PANDOC_TIMEOUT_SEC = 120

# Output formats we expose. Keys are user-facing; values are the pandoc
# `-t` argument and the file extension we'll write.
EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "docx":  ("docx",  ".docx"),
    "odt":   ("odt",   ".odt"),
    "rtf":   ("rtf",   ".rtf"),
    "html":  ("html",  ".html"),
    "epub":  ("epub",  ".epub"),
    "pdf":   ("pdf",   ".pdf"),
    "latex": ("latex", ".tex"),
}

# Source formats pandoc reads. Defaults to "markdown" if the source
# extension isn't in this map.
SOURCE_FORMATS: dict[str, str] = {
    ".md":       "markdown",
    ".markdown": "markdown",
    ".rst":      "rst",
    ".html":     "html",
    ".htm":      "html",
    ".tex":      "latex",
    ".org":      "org",
    ".txt":      "markdown",  # treat plain text as minimally-formatted md
}


class ExportError(RuntimeError):
    pass


def is_supported_source(path: str) -> bool:
    """True if pandoc can read this source extension."""
    return Path(path).suffix.lower() in SOURCE_FORMATS


def export_to_disk(
    src_path: str,
    dest_path: str | None = None,
    output_format: str = "docx",
    overwrite: bool = False,
) -> dict:
    """Run pandoc and write the result to disk.

    Returns metadata only -- no file content.

    Args:
        src_path: Path to the source text document (markdown by default).
        dest_path: Output path. If None, defaults to
                   <src_dir>/<src_stem><ext> where <ext> matches output_format.
        output_format: One of EXPORT_FORMATS keys. Default "docx".
        overwrite: If False and dest exists, raise ExportError.

    Returns:
        {"src_path", "dest_path", "output_format", "bytes"}

    Raises ExportError on any failure.
    """
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise ExportError(
            "pandoc not found on PATH; install with `sudo apt install pandoc`"
        )

    abs_src = os.path.abspath(src_path)
    if not os.path.isfile(abs_src):
        raise ExportError(f"src_path is not a regular file: {abs_src}")

    if output_format not in EXPORT_FORMATS:
        raise ExportError(
            f"unknown output_format {output_format!r}; "
            f"expected one of {sorted(EXPORT_FORMATS)}"
        )
    pandoc_to, ext = EXPORT_FORMATS[output_format]

    if dest_path is None:
        stem = Path(abs_src).stem
        abs_dest = str(Path(abs_src).parent / f"{stem}{ext}")
    else:
        abs_dest = os.path.abspath(dest_path)

    if os.path.exists(abs_dest) and not overwrite:
        raise ExportError(
            f"dest_path exists: {abs_dest} (pass overwrite=True to replace)"
        )
    if abs_dest == abs_src:
        raise ExportError(f"dest_path equals src_path: {abs_src}")

    parent = os.path.dirname(abs_dest)
    if parent and not os.path.isdir(parent):
        raise ExportError(f"dest_path parent does not exist: {parent}")

    src_ext = Path(abs_src).suffix.lower()
    pandoc_from = SOURCE_FORMATS.get(src_ext, "markdown")

    cmd = [
        pandoc,
        "-f", pandoc_from,
        "-t", pandoc_to,
        "-o", abs_dest,
        abs_src,
    ]
    # PDF output via pandoc requires a TeX engine; xelatex tends to handle
    # unicode best. If not installed, pandoc will return a clear error.
    if output_format == "pdf":
        cmd.insert(1, "--pdf-engine=xelatex")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PANDOC_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        raise ExportError(f"pandoc timeout after {PANDOC_TIMEOUT_SEC}s") from e
    except OSError as e:
        raise ExportError(f"failed to invoke pandoc: {e}") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:400]
        raise ExportError(
            f"pandoc exit {proc.returncode}: {stderr or '(no stderr)'}"
        )
    if not os.path.isfile(abs_dest):
        raise ExportError(f"pandoc reported success but {abs_dest} was not written")

    return {
        "src_path": abs_src,
        "dest_path": abs_dest,
        "output_format": output_format,
        "bytes": os.path.getsize(abs_dest),
    }
