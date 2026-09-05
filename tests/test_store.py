import pytest

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


OLD_TABLE = """
CREATE TABLE implementation_candidates (
  id TEXT PRIMARY KEY,
  paper_version TEXT NOT NULL,
  repo_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK (relation IN ('OFFICIAL')),
  confidence TEXT NOT NULL CHECK (confidence IN ('CONFIRMED','LIKELY','POSSIBLE','UNKNOWN')),
  coverage_score REAL,
  coverage_checked INTEGER NOT NULL DEFAULT 0,
  rank INTEGER
)"""


def _with_old_table(db):
    """A store whose candidate table predates the DECLARED_DEPENDENCY relation."""
    s = Store(db)
    s.upsert_paper("2103.00020", "CLIP")
    s.upsert_paper_version("2103.00020v1", "2103.00020", 1, source_kind="LATEX",
                           fidelity="LATEX_EXACT", source_sha256=None,
                           flattened_tex="x")
    s.execute("DROP TABLE implementation_candidates")
    s.conn.executescript(OLD_TABLE)
    s.execute("INSERT OR IGNORE INTO repos (id, owner, name) VALUES ('o/r','o','r')")
    s.execute("INSERT OR IGNORE INTO repos (id, owner, name) VALUES ('o/r2','o','r2')")
    s.execute("INSERT INTO implementation_candidates (id, paper_version, repo_id, "
              "relation, confidence) VALUES ('keep','2103.00020v1','o/r',"
              "'OFFICIAL','CONFIRMED')")
    s.commit()
    s.close()
    return s


def test_a_changed_check_constraint_is_migrated(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a changed
    CHECK was silently ignored until a write failed with IntegrityError against a
    definition nobody could see. Reopening the store must rebuild the table."""
    db = tmp_path / "g.db"
    _with_old_table(db)

    migrated = Store(db)          # reopening rebuilds the drifted table
    assert "DECLARED_DEPENDENCY" in migrated.one(
        "SELECT sql FROM sqlite_master WHERE name='implementation_candidates'")["sql"]
    migrated.execute(
        "INSERT INTO implementation_candidates (id, paper_version, repo_id, "
        "relation, confidence) VALUES ('b','2103.00020v1','o/r2',"
        "'DECLARED_DEPENDENCY','LIKELY')")
    migrated.commit()
    assert migrated.one(
        "SELECT COUNT(*) c FROM implementation_candidates")["c"] == 2


def test_migration_preserves_rows_and_unrelated_tables(tmp_path):
    db = tmp_path / "g.db"
    _with_old_table(db)
    s = Store(db)
    assert s.one("SELECT relation FROM implementation_candidates "
                 "WHERE id='keep'")["relation"] == "OFFICIAL"
    assert s.one("SELECT title FROM papers WHERE arxiv_id='2103.00020'")["title"] == "CLIP"


def test_an_unchanged_schema_is_left_alone(tmp_path):
    db = tmp_path / "g.db"
    s = Store(db)
    s.upsert_paper("2103.00020", "CLIP")
    s.commit()
    s.close()
    again = Store(db)
    assert again.one("SELECT COUNT(*) c FROM papers")["c"] == 1
