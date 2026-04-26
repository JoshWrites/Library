"""Summarizer tests — validates offline contract and JSON parsing.

These tests do NOT require the secondary llama-server to be running.
They test the parse/fallback path.
"""
from __future__ import annotations
from library.summarizer import _parse_output, SummaryResult

def test_valid_json_parsed_correctly():
    raw = '{"summary": "Proxmox supports LVM, ZFS, and directory storage.", "sources": [{"url_or_path": "https://proxmox.com", "title": "Proxmox Docs", "used_for": "storage types"}], "confidence": "high", "notes": ""}'
    result = _parse_output(raw, fallback_sources=[])
    assert result.summary == "Proxmox supports LVM, ZFS, and directory storage."
    assert result.confidence == "high"
    assert len(result.sources) == 1

def test_markdown_fences_stripped():
    raw = '```json\n{"summary": "test", "sources": [], "confidence": "medium", "notes": "x"}\n```'
    result = _parse_output(raw, fallback_sources=[])
    assert result.summary == "test"

def test_invalid_json_produces_low_confidence():
    result = _parse_output("not json at all", fallback_sources=[
        {"url_or_path": "https://example.com", "title": "Example", "used_for": ""}
    ])
    assert result.confidence == "low"
    assert len(result.sources) == 1

def test_empty_response_produces_low_confidence():
    result = _parse_output("", fallback_sources=[])
    assert result.confidence == "low"

def test_invalid_confidence_value_normalised_to_low():
    raw = '{"summary": "ok", "sources": [], "confidence": "very_high", "notes": ""}'
    result = _parse_output(raw, fallback_sources=[])
    assert result.confidence == "low"

def test_summary_result_to_dict():
    r = SummaryResult(
        summary="test",
        sources=[{"url_or_path": "x", "title": "y", "used_for": "z"}],
        confidence="high",
        notes="",
    )
    d = r.to_dict("my query")
    assert d["layer"] == "summary"
    assert d["query"] == "my query"
    assert d["can_escalate"] is True
