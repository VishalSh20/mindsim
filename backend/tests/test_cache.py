import time

import pytest

from mindsim.cache.disk_cache import DiskCache


def test_cache_set_get_roundtrip(tmp_path):
    cache = DiskCache("test_ns", ttl_seconds=60, root=tmp_path)
    cache.set({"q": "hello"}, {"result": 42})
    assert cache.get({"q": "hello"}) == {"result": 42}


def test_cache_miss_returns_none(tmp_path):
    cache = DiskCache("test_ns", ttl_seconds=60, root=tmp_path)
    assert cache.get({"q": "not_there"}) is None


def test_cache_key_canonicalized(tmp_path):
    cache = DiskCache("test_ns", ttl_seconds=60, root=tmp_path)
    # Dict ordering must not change the cache key.
    assert cache._key({"a": 1, "b": 2}) == cache._key({"b": 2, "a": 1})


def test_cache_key_differs_on_different_input(tmp_path):
    cache = DiskCache("test_ns", ttl_seconds=60, root=tmp_path)
    assert cache._key({"a": 1}) != cache._key({"a": 2})


def test_cache_ttl_expiry(tmp_path):
    # ttl=0: stored entry is considered expired the instant it is read.
    cache = DiskCache("test_ns", ttl_seconds=0, root=tmp_path)
    cache.set({"k": 1}, "v")
    assert cache.get({"k": 1}) is None


def test_cache_ttl_within_window(tmp_path):
    cache = DiskCache("test_ns", ttl_seconds=3600, root=tmp_path)
    cache.set({"k": 1}, "v")
    assert cache.get({"k": 1}) == "v"


def test_cache_clear_removes_entries(tmp_path):
    cache = DiskCache("test_ns", ttl_seconds=60, root=tmp_path)
    cache.set({"k": 1}, "v1")
    cache.set({"k": 2}, "v2")
    removed = cache.clear()
    assert removed == 2
    assert cache.get({"k": 1}) is None
    assert cache.get({"k": 2}) is None


def test_cache_namespaces_isolated(tmp_path):
    a = DiskCache("ns_a", ttl_seconds=60, root=tmp_path)
    b = DiskCache("ns_b", ttl_seconds=60, root=tmp_path)
    a.set({"k": 1}, "from_a")
    assert b.get({"k": 1}) is None
    assert a.get({"k": 1}) == "from_a"


def test_cache_rejects_negative_ttl(tmp_path):
    with pytest.raises(ValueError):
        DiskCache("ns", ttl_seconds=-1, root=tmp_path)


def test_cache_survives_corrupt_file(tmp_path):
    cache = DiskCache("ns", ttl_seconds=60, root=tmp_path)
    cache.set({"k": 1}, "v")
    # Corrupt the cache file.
    for path in cache.root.glob("*.pkl"):
        path.write_bytes(b"not a pickle")
    # Should return None, not raise.
    assert cache.get({"k": 1}) is None
