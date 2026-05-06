"""Unified DRAM LRU cache for Library.

Holds CachedEntry objects for both local files and web pages.
Keyed by entry_id; secondary lookup by label (path or URL).

File entries: keyed by make_file_id(path, mtime). Stale on mtime change.
Web entries:  keyed by make_web_id(url, fetch_time). Never auto-expire;
              force_refresh bypasses cache at the tool level.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


MAX_CACHED_ENTRIES = 40


@dataclass
class CachedEntry:
    entry_id: str
    source: str          # "file" | "web"
    label: str           # absolute path or URL
    chunks: list         # list[Chunk] for files; list[str] for web
    embeddings: list     # list[list[float]], parallel to chunks
    fetch_time: float
    last_accessed: float = field(default_factory=time.time)


def make_file_id(path: str, mtime: float) -> str:
    h = hashlib.sha256(f"file|{path}|{mtime}".encode()).hexdigest()
    return f"f_{h[:12]}"


def make_web_id(url: str, fetch_time: float) -> str:
    h = hashlib.sha256(f"web|{url}|{fetch_time}".encode()).hexdigest()
    return f"w_{h[:12]}"


class Cache:
    def __init__(self, max_entries: int = MAX_CACHED_ENTRIES) -> None:
        self._entries: OrderedDict[str, CachedEntry] = OrderedDict()
        self._max = max_entries

    def get(self, entry_id: str) -> Optional[CachedEntry]:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        self._entries.move_to_end(entry_id)
        entry.last_accessed = time.time()
        return entry

    def get_by_label(self, label: str) -> Optional[CachedEntry]:
        for entry in reversed(list(self._entries.values())):
            if entry.label == label:
                self._entries.move_to_end(entry.entry_id)
                entry.last_accessed = time.time()
                return entry
        return None

    def put(self, entry: CachedEntry) -> None:
        self._entries[entry.entry_id] = entry
        self._entries.move_to_end(entry.entry_id)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def release(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None

    def lookup_file(self, path: str) -> Optional[CachedEntry]:
        # File entry IDs include mtime, so an mtime change makes the old
        # ID un-findable -- correctness is automatic. But the orphaned
        # entry stays in the cache occupying a slot until LRU eviction
        # eventually reaches it; under heavy editing of one file that
        # could push other useful entries out. On every miss we sweep
        # any other entries with this label and release them, keeping
        # the cache to one entry per (label, current-mtime).
        #
        # Note: this also implicitly assumes that the chunking strategy
        # is a function of the path's extension, so a cache hit on the
        # current (path, mtime) returns chunks produced by the current
        # extension's strategy. A file renamed to a different-strategy
        # extension while preserving mtime would return stale-strategy
        # chunks; in practice mtime updates whenever inode metadata
        # changes via rename, so this is near-impossible.
        abs_path = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            return None
        expected_id = make_file_id(abs_path, mtime)
        hit = self.get(expected_id)
        if hit is not None:
            return hit
        stale_ids = [
            entry.entry_id
            for entry in self._entries.values()
            if entry.source == "file" and entry.label == abs_path
        ]
        for sid in stale_ids:
            self._entries.pop(sid, None)
        return None

    def stats(self) -> dict:
        return {
            "cached_entries": len(self._entries),
            "max_entries": self._max,
            "entry_ids": list(self._entries.keys()),
        }
