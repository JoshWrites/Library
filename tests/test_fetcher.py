"""Fetcher tests -- no live network needed for SSRF and rejection tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running the test file directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from library.fetcher import is_safe_public_url, FetchError


def test_http_loopback_rejected():
    ok, reason = is_safe_public_url("http://127.0.0.1/anything")
    assert not ok
    assert "non-public" in reason or "loopback" in reason.lower()


def test_https_private_range_rejected():
    ok, reason = is_safe_public_url("http://192.168.1.1/admin")
    assert not ok


def test_non_http_scheme_rejected():
    ok, reason = is_safe_public_url("file:///etc/passwd")
    assert not ok
    assert "scheme" in reason


def test_ftp_scheme_rejected():
    ok, reason = is_safe_public_url("ftp://example.com/file")
    assert not ok


def test_no_hostname_rejected():
    ok, reason = is_safe_public_url("http:///no-host")
    assert not ok


def test_public_url_accepted():
    # example.com resolves to a public IP -- safe to check without fetching
    ok, reason = is_safe_public_url("https://example.com/page")
    assert ok, f"expected ok, got: {reason}"
