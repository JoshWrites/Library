"""Cache unit tests -- no network, no embed server needed."""
from __future__ import annotations
import time
from library.cache import Cache, make_file_id, make_web_id, CachedEntry

def test_file_entry_roundtrip():
    c = Cache()
    entry = CachedEntry(
        entry_id="f_abc123",
        source="file",
        label="/tmp/test.md",
        chunks=["chunk1", "chunk2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        fetch_time=time.time(),
    )
    c.put(entry)
    got = c.get("f_abc123")
    assert got is not None
    assert got.label == "/tmp/test.md"
    assert got.chunks == ["chunk1", "chunk2"]

def test_web_entry_roundtrip():
    c = Cache()
    t = time.time()
    entry = CachedEntry(
        entry_id="w_xyz789",
        source="web",
        label="https://example.com/article",
        chunks=["web chunk"],
        embeddings=[[0.5, 0.6]],
        fetch_time=t,
    )
    c.put(entry)
    got = c.get_by_label("https://example.com/article")
    assert got is not None
    assert got.entry_id == "w_xyz789"

def test_lru_eviction_drops_oldest():
    c = Cache(max_entries=3)
    for i in range(4):
        c.put(CachedEntry(f"id_{i}", "file", f"/tmp/f{i}.md", [], [], time.time()))
    assert c.get("id_0") is None  # evicted
    assert c.get("id_3") is not None

def test_release_removes_entry():
    c = Cache()
    c.put(CachedEntry("id_a", "file", "/tmp/a.md", [], [], time.time()))
    assert c.release("id_a") is True
    assert c.get("id_a") is None
    assert c.release("id_a") is False

def test_lookup_file_by_path_and_mtime():
    import os, tempfile
    fd, path = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    try:
        mtime = os.path.getmtime(path)
        eid = make_file_id(path, mtime)
        c = Cache()
        c.put(CachedEntry(eid, "file", path, [], [], time.time()))
        got = c.lookup_file(path)
        assert got is not None and got.entry_id == eid
    finally:
        os.unlink(path)

def test_lookup_file_misses_after_mtime_change():
    import os, tempfile
    fd, path = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    try:
        old_mtime = os.path.getmtime(path)
        eid = make_file_id(path, old_mtime)
        c = Cache()
        c.put(CachedEntry(eid, "file", path, [], [], time.time()))
        time.sleep(1.1)
        with open(path, "w") as f:
            f.write("changed")
        got = c.lookup_file(path)
        assert got is None  # mtime changed -> miss
    finally:
        os.unlink(path)

def test_make_file_id_is_stable():
    id1 = make_file_id("/tmp/test.md", 1234567890.0)
    id2 = make_file_id("/tmp/test.md", 1234567890.0)
    assert id1 == id2
    assert id1.startswith("f_")

def test_make_web_id_is_stable():
    id1 = make_web_id("https://example.com", 9999.0)
    id2 = make_web_id("https://example.com", 9999.0)
    assert id1 == id2
    assert id1.startswith("w_")
