"""Web fetch and text extraction for Library.

Handles URL safety validation (SSRF defense), HTTP fetch with redirect
re-checking, and readability-based text extraction. Returns plain text;
callers handle chunking and embedding.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from readability import Document
from bs4 import BeautifulSoup


FETCH_TIMEOUT_SEC = 10
USER_AGENT = "library-mcp/0.1 (local research helper)"
MAX_BYTES = 2_000_000  # 2 MB cap per page


class FetchError(RuntimeError):
    pass


def is_safe_public_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Reject non-http(s), private/loopback, link-local."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"bad scheme: {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return False, "no hostname"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        return False, f"dns failed: {e}"
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False, f"non-public address {ip_str}"
    return True, "ok"


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        ok, reason = is_safe_public_url(newurl)
        if not ok:
            raise urllib.error.URLError(f"unsafe redirect: {reason}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def fetch_and_extract(url: str) -> tuple[str, str]:
    """Fetch url and return (title, plain_text). Raises FetchError on failure."""
    ok, reason = is_safe_public_url(url)
    if not ok:
        raise FetchError(f"blocked: {reason}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with _OPENER.open(req, timeout=FETCH_TIMEOUT_SEC) as r:
            raw_bytes = r.read(MAX_BYTES)
            ctype = r.headers.get("Content-Type", "")
            charset = r.headers.get_content_charset() or "utf-8"
    except urllib.error.URLError as e:
        raise FetchError(f"fetch failed: {e}") from e

    if "html" not in ctype and "text" not in ctype:
        raise FetchError(f"non-text content-type: {ctype}")

    raw_text = raw_bytes.decode(charset, errors="replace")
    doc = Document(raw_text)
    title = doc.short_title() or url
    html = doc.summary(html_partial=True)
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text
