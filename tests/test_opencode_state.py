"""opencode_state unit tests — uses fixture SQLite DBs, no live opencode needed."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from library import opencode_state


def _make_fixture_db(path: Path, sessions: list[dict], messages: list[dict]) -> None:
    """Build a minimal opencode-shaped SQLite DB.

    Only the columns we actually read are populated. Schema mirrors what
    opencode 1.14.x writes; if opencode changes shape, this fixture stays
    independent and the production code's tolerance kicks in.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE session ("
        "id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, slug TEXT, "
        "directory TEXT, title TEXT, version TEXT, share_url TEXT, "
        "summary_additions INTEGER, summary_deletions INTEGER, "
        "summary_files INTEGER, summary_diffs TEXT, revert TEXT, "
        "permission TEXT, time_created INTEGER, time_updated INTEGER, "
        "time_compacting INTEGER, time_archived INTEGER, workspace_id TEXT)"
    )
    conn.execute(
        "CREATE TABLE message ("
        "id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, "
        "time_updated INTEGER, data TEXT)"
    )
    for s in sessions:
        conn.execute(
            "INSERT INTO session (id, directory, title, time_created, time_updated) "
            "VALUES (?, ?, ?, ?, ?)",
            (s["id"], s["directory"], s["title"], s["time_created"], s["time_updated"]),
        )
    for m in messages:
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (m["id"], m["session_id"], m["time_created"], m["time_created"], json.dumps(m["data"])),
        )
    conn.commit()
    conn.close()


def _config_with_limit(path: Path, provider: str, model: str, limit: int) -> None:
    path.write_text(json.dumps({
        "provider": {provider: {"models": {model: {"limit": {"context": limit}}}}}
    }))


def test_resolves_session_by_directory(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    cfg = tmp_path / "opencode.json"
    _make_fixture_db(db, sessions=[
        {"id": "ses_old", "directory": "/proj/a", "title": "old",
         "time_created": 1000, "time_updated": 2000},
        {"id": "ses_new", "directory": "/proj/a", "title": "new",
         "time_created": 3000, "time_updated": 4000},
        {"id": "ses_other", "directory": "/proj/b", "title": "other",
         "time_created": 5000, "time_updated": 6000},
    ], messages=[
        {"id": "m1", "session_id": "ses_new", "time_created": 3500, "data": {
            "role": "assistant", "providerID": "p", "modelID": "m",
            "tokens": {"total": 1000, "input": 50, "output": 100,
                       "reasoning": 0, "cache": {"read": 850, "write": 0}},
        }},
    ])
    _config_with_limit(cfg, "p", "m", 8000)
    monkeypatch.setattr(opencode_state, "DB_PATH", db)
    monkeypatch.setattr(opencode_state, "CONFIG_PATH", cfg)

    result = opencode_state.get_active_session_state("/proj/a")
    assert "error" not in result
    assert result["session_id"] == "ses_new"  # most recent in /proj/a
    assert result["directory_match"] is True
    assert result["model"] == "p/m"
    assert result["context_limit"] == 8000
    assert result["current_tokens"] == 1000
    assert result["pct_used"] == 0.125


def test_falls_back_to_most_recent_when_no_match(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    cfg = tmp_path / "opencode.json"
    _make_fixture_db(db, sessions=[
        {"id": "ses_a", "directory": "/proj/a", "title": "A",
         "time_created": 1000, "time_updated": 9999},
    ], messages=[
        {"id": "m1", "session_id": "ses_a", "time_created": 1500, "data": {
            "role": "assistant", "providerID": "p", "modelID": "m",
            "tokens": {"total": 500, "input": 10, "output": 20,
                       "reasoning": 0, "cache": {"read": 470, "write": 0}},
        }},
    ])
    _config_with_limit(cfg, "p", "m", 4000)
    monkeypatch.setattr(opencode_state, "DB_PATH", db)
    monkeypatch.setattr(opencode_state, "CONFIG_PATH", cfg)

    result = opencode_state.get_active_session_state("/nowhere")
    assert "error" not in result
    assert result["session_id"] == "ses_a"
    assert result["directory_match"] is False


def test_skips_user_messages_finds_assistant(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    cfg = tmp_path / "opencode.json"
    # Most recent message is a user turn (no tokens). Should walk back.
    _make_fixture_db(db, sessions=[
        {"id": "ses", "directory": "/p", "title": "t",
         "time_created": 1000, "time_updated": 9999},
    ], messages=[
        {"id": "m_assistant", "session_id": "ses", "time_created": 100, "data": {
            "role": "assistant", "providerID": "p", "modelID": "m",
            "tokens": {"total": 200, "input": 5, "output": 50,
                       "reasoning": 0, "cache": {"read": 145, "write": 0}},
        }},
        {"id": "m_user_newer", "session_id": "ses", "time_created": 200, "data": {
            "role": "user",
        }},
    ])
    _config_with_limit(cfg, "p", "m", 1000)
    monkeypatch.setattr(opencode_state, "DB_PATH", db)
    monkeypatch.setattr(opencode_state, "CONFIG_PATH", cfg)

    result = opencode_state.get_active_session_state("/p")
    assert result["current_tokens"] == 200


def test_unknown_model_returns_null_pct(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    cfg = tmp_path / "opencode.json"
    cfg.write_text("{}")  # empty config, no limits
    _make_fixture_db(db, sessions=[
        {"id": "ses", "directory": "/p", "title": "t",
         "time_created": 1000, "time_updated": 9999},
    ], messages=[
        {"id": "m1", "session_id": "ses", "time_created": 100, "data": {
            "role": "assistant", "providerID": "unknown", "modelID": "model",
            "tokens": {"total": 500, "input": 1, "output": 1,
                       "reasoning": 0, "cache": {"read": 498, "write": 0}},
        }},
    ])
    monkeypatch.setattr(opencode_state, "DB_PATH", db)
    monkeypatch.setattr(opencode_state, "CONFIG_PATH", cfg)

    result = opencode_state.get_active_session_state("/p")
    assert result["context_limit"] is None
    assert result["pct_used"] is None
    assert result["current_tokens"] == 500


def test_db_missing_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(opencode_state, "DB_PATH", tmp_path / "nonexistent.db")
    result = opencode_state.get_active_session_state("/p")
    assert "error" in result
    assert "not found" in result["error"]


def test_no_sessions_returns_error(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    _make_fixture_db(db, sessions=[], messages=[])
    monkeypatch.setattr(opencode_state, "DB_PATH", db)
    monkeypatch.setattr(opencode_state, "CONFIG_PATH", tmp_path / "missing.json")
    result = opencode_state.get_active_session_state("/p")
    assert "error" in result
    assert "no opencode sessions" in result["error"]


def test_session_with_no_token_data_returns_error(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    cfg = tmp_path / "opencode.json"
    cfg.write_text("{}")
    _make_fixture_db(db, sessions=[
        {"id": "ses", "directory": "/p", "title": "t",
         "time_created": 1000, "time_updated": 9999},
    ], messages=[
        {"id": "m_user", "session_id": "ses", "time_created": 100, "data": {
            "role": "user",
        }},
    ])
    monkeypatch.setattr(opencode_state, "DB_PATH", db)
    monkeypatch.setattr(opencode_state, "CONFIG_PATH", cfg)
    result = opencode_state.get_active_session_state("/p")
    assert "error" in result
    assert "no assistant turn" in result["error"]
