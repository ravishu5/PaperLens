"""Store invariants."""
from paperlens.graph.store import Store


def test_rate_limit_bucket_persists_across_connections(tmp_path):
    """A crash loop must not stampede arXiv, so the bucket lives in the DB."""
    db = tmp_path / "g.db"
    assert Store(db).take_token("arxiv") == 0.0
    wait = Store(db).take_token("arxiv")   # fresh Store, same file
    assert 2.0 < wait <= 3.0


def test_cache_roundtrip(tmp_path):
    s = Store(tmp_path / "g.db")
    s.cache_put("k", "arxiv", "http://x", b"payload")
    assert s.cache_get("k") == b"payload"
    assert s.cache_get("missing") is None
