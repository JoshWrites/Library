"""Read-only access to opencode's persisted session state.

opencode keeps message-level state -- including per-turn token counts -- in a
SQLite database at ``~/.local/share/opencode/opencode.db``. Each user has
their own database; an MCP server spawned by opencode inherits the user's
uid, so ``Path.home()`` resolves to the right place automatically.

This module exposes one function -- ``get_active_session_state`` -- used by
the ``context_usage`` MCP tool. It opens the DB read-only (WAL-mode
concurrent reads are safe while opencode is writing) and returns a small
dict the tool can wrap.

Schema notes (opencode 1.14.x):
  - ``session(id, directory, title, time_created, time_updated, ...)``
  - ``message(id, session_id, time_created, time_updated, data)`` where
    ``data`` is a JSON blob. Assistant messages carry
    ``{role: "assistant", tokens: {total, input, output, reasoning,
    cache: {write, read}}, modelID, providerID, ...}``.

Tolerated failures (return ``{"error": ...}`` rather than raising):
  - Database file missing (opencode never run on this user account)
  - No session in this directory and no fallback session anywhere
  - Most recent assistant turn has no token data (very fresh session)
  - Model not declared in ``opencode.json`` (return tokens but null pct)
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"

# How many recent messages to scan when looking for the latest turn with
# token data. opencode emits user messages, assistant messages, and tool
# parts; we only want assistant turns. 20 is generous -- token data lives
# on every assistant message, so usually message[0] is the answer.
_RECENT_MESSAGE_SCAN = 20


def _load_context_limits() -> dict[tuple[str, str], int]:
    """Map (providerID, modelID) -> context-window size from opencode config.

    Returns an empty dict if the config file is missing or malformed; callers
    should treat a missing limit as "unknown" rather than failing.
    """
    if not CONFIG_PATH.exists():
        return {}
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[str, str], int] = {}
    for provider_id, pdata in cfg.get("provider", {}).items():
        if not isinstance(pdata, dict):
            continue
        for model_id, mdata in pdata.get("models", {}).items():
            if not isinstance(mdata, dict):
                continue
            limit = mdata.get("limit", {}).get("context") if isinstance(mdata.get("limit"), dict) else None
            if isinstance(limit, int) and limit > 0:
                out[(provider_id, model_id)] = limit
    return out


def _parse_message_data(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def get_active_session_state(directory: str | None = None) -> dict:
    """Return token-usage state for the active opencode session.

    Resolution: prefer the most recent session whose ``directory`` exactly
    matches ``directory`` (defaults to ``os.getcwd()``). If none, fall back
    to the most recently updated session anywhere on this user account.
    The returned ``directory_match`` field signals which path was taken.

    On any error, returns ``{"error": "<reason>"}``. Callers wrap this for
    MCP transport.
    """
    if directory is None:
        directory = os.getcwd()

    if not DB_PATH.exists():
        return {"error": f"opencode database not found at {DB_PATH}"}

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.OperationalError as e:
        return {"error": f"cannot open opencode database: {e}"}

    try:
        # Exact directory match first.
        row = conn.execute(
            "SELECT id, title, time_created, time_updated, directory "
            "FROM session WHERE directory = ? "
            "ORDER BY time_updated DESC LIMIT 1",
            (directory,),
        ).fetchone()
        directory_match = row is not None

        # Fall back to most recent session anywhere.
        if row is None:
            row = conn.execute(
                "SELECT id, title, time_created, time_updated, directory "
                "FROM session ORDER BY time_updated DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return {"error": "no opencode sessions found for this user"}

        sid, title, t_created, t_updated, sess_dir = row

        # Walk recent messages newest-first looking for an assistant turn
        # with token data.
        msg_rows = conn.execute(
            "SELECT data FROM message WHERE session_id = ? "
            "ORDER BY time_created DESC LIMIT ?",
            (sid, _RECENT_MESSAGE_SCAN),
        ).fetchall()

        last_turn: dict | None = None
        for (raw,) in msg_rows:
            d = _parse_message_data(raw)
            if d is None:
                continue
            if d.get("role") != "assistant":
                continue
            tokens = d.get("tokens")
            if not isinstance(tokens, dict):
                continue
            # Skip in-flight placeholder messages: opencode creates an
            # assistant row at request time with all-zero token fields and
            # no `total` key, then updates it on completion. The placeholder
            # is the *most recent* row when a tool call is in-flight (e.g.
            # the call this tool is responding to), so picking it would
            # always report 0. The completed message has both `total` and
            # the `finish` field.
            if "total" not in tokens:
                continue
            last_turn = d
            break

        if last_turn is None:
            return {
                "error": (
                    f"no assistant turn with token data in last "
                    f"{_RECENT_MESSAGE_SCAN} messages of session {sid}"
                )
            }

        # Total assistant turn count. SQL LIKE on a JSON blob is hacky but
        # cheap and good enough -- this number is informational, not load-bearing.
        turn_count = conn.execute(
            "SELECT COUNT(*) FROM message WHERE session_id = ? "
            "AND data LIKE '%\"role\":\"assistant\"%'",
            (sid,),
        ).fetchone()[0]
    finally:
        conn.close()

    tokens = last_turn["tokens"]
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    provider_id = last_turn.get("providerID", "unknown")
    model_id = last_turn.get("modelID", "unknown")

    limits = _load_context_limits()
    ctx_limit = limits.get((provider_id, model_id))
    current_total = tokens.get("total", 0) if isinstance(tokens.get("total"), int) else 0
    pct_used = round(current_total / ctx_limit, 4) if ctx_limit and current_total else None

    # opencode timestamps are millis since epoch.
    age_ms = max(0, int(t_updated) - int(t_created))

    return {
        "session_id": sid,
        "session_title": title,
        "session_directory": sess_dir,
        "directory_match": directory_match,
        "model": f"{provider_id}/{model_id}",
        "context_limit": ctx_limit,
        "current_tokens": current_total,
        "pct_used": pct_used,
        "input": tokens.get("input", 0),
        "output": tokens.get("output", 0),
        "reasoning": tokens.get("reasoning", 0),
        "cache_read": cache.get("read", 0),
        "cache_write": cache.get("write", 0),
        "turn_count": int(turn_count or 0),
        "session_age_min": round(age_ms / 60000, 1),
    }
