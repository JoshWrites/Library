"""Searcher tests -- no live SearxNG needed for unit tests."""
from __future__ import annotations
from library.searcher import query_variations, rank_results

def test_query_variations_strips_stopwords():
    variations = query_variations("what is the best way to configure proxmox storage")
    # First variation should be keyword-only
    first = variations[0]
    for stopword in ("what", "is", "the", "to"):
        assert stopword not in first.split(), f"stopword '{stopword}' in keyword query: {first}"
    assert "proxmox" in first
    assert "storage" in first

def test_query_variations_includes_original():
    q = "how does immich handle duplicates"
    variations = query_variations(q)
    assert q in variations

def test_query_variations_deduplicates():
    # Very short query -- keyword strip may equal original
    variations = query_variations("proxmox storage")
    assert len(variations) == len(set(variations))

def test_rank_deduplicates_urls():
    results = [
        {"url": "https://proxmox.com/page", "title": "Proxmox Storage", "_query_position": 0},
        {"url": "https://proxmox.com/page", "title": "Proxmox Storage", "_query_position": 1},
    ]
    ranked = rank_results(results, "proxmox storage types", max_n=5)
    assert len(ranked) == 1

def test_rank_boosts_known_domains():
    results = [
        {"url": "https://random-blog.com/proxmox", "title": "Proxmox tips", "_query_position": 0},
        {"url": "https://proxmox.com/docs/storage", "title": "Proxmox Storage", "_query_position": 2},
    ]
    ranked = rank_results(results, "proxmox storage", max_n=5)
    # proxmox.com should rank above random-blog even with higher query position
    assert ranked[0]["url"].startswith("https://proxmox.com")

def test_rank_respects_max_n():
    results = [
        {"url": f"https://site{i}.com/page", "title": f"Title {i}", "_query_position": i}
        for i in range(10)
    ]
    ranked = rank_results(results, "anything", max_n=3)
    assert len(ranked) == 3
