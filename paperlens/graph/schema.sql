-- PaperLens research graph. See ARCHITECTURE.md section 2.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- ─── Papers ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
  arxiv_id        TEXT PRIMARY KEY,
  title           TEXT NOT NULL,
  abstract        TEXT,
  doi             TEXT,
  s2_paper_id     TEXT,
  openalex_id     TEXT,
  published_at    TEXT,
  authors_json    TEXT,
  latest_version  TEXT
);

CREATE TABLE IF NOT EXISTS paper_versions (
  paper_version   TEXT PRIMARY KEY,
  arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
  version         INTEGER NOT NULL,
  source_kind     TEXT NOT NULL CHECK (source_kind IN ('LATEX','PDF','NONE')),
  fidelity        TEXT NOT NULL CHECK (fidelity IN ('LATEX_EXACT','PDF_DERIVED','UNAVAILABLE')),
  source_sha256   TEXT,
  parser_version  TEXT NOT NULL,
  flattened_tex   TEXT,
  ingested_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pv_arxiv ON paper_versions(arxiv_id);

-- ─── Paper artifacts ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sections (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  section_path    TEXT NOT NULL,
  level           INTEGER NOT NULL,
  title           TEXT NOT NULL,
  latex_label     TEXT,
  body            TEXT NOT NULL,
  src_file        TEXT,
  src_line_start  INTEGER,
  src_line_end    INTEGER,
  ordinal         INTEGER NOT NULL,
  UNIQUE (paper_version, section_path)
);
CREATE INDEX IF NOT EXISTS ix_sec_pv ON sections(paper_version);

CREATE TABLE IF NOT EXISTS equations (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  content_hash    TEXT NOT NULL,
  latex           TEXT NOT NULL,
  environment     TEXT NOT NULL,
  is_numbered     INTEGER NOT NULL,
  derived_number  TEXT,
  number_conf     TEXT NOT NULL CHECK (number_conf IN ('CONFIRMED','LIKELY','POSSIBLE','UNKNOWN')),
  latex_label     TEXT,
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  src_line        INTEGER,
  ordinal         INTEGER NOT NULL,
  UNIQUE (paper_version, content_hash)
);
CREATE INDEX IF NOT EXISTS ix_eq_pv ON equations(paper_version);
CREATE INDEX IF NOT EXISTS ix_eq_num ON equations(paper_version, derived_number);

CREATE TABLE IF NOT EXISTS components (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  slug            TEXT NOT NULL,
  name            TEXT NOT NULL,
  kind            TEXT NOT NULL CHECK (kind IN
                    ('architecture','module','loss','optimizer','dataset',
                     'preprocessing','training','inference','evaluation','other')),
  description     TEXT,
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  source          TEXT NOT NULL CHECK (source IN ('DETERMINISTIC','AGENT','SERVER_LLM')),
  UNIQUE (paper_version, slug)
);
CREATE INDEX IF NOT EXISTS ix_comp_pv ON components(paper_version);

CREATE TABLE IF NOT EXISTS algorithms (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  slug            TEXT NOT NULL,
  name            TEXT,
  body            TEXT,
  presentation    TEXT NOT NULL CHECK (presentation IN
                    ('latex_env','listing','figure_image','prose')),
  extractable     INTEGER NOT NULL,
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  src_line        INTEGER,
  UNIQUE (paper_version, slug)
);

CREATE TABLE IF NOT EXISTS stated_values (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  symbol          TEXT,
  value_text      TEXT NOT NULL,
  value_num       REAL,
  context         TEXT NOT NULL,
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  src_line        INTEGER
);
CREATE INDEX IF NOT EXISTS ix_sv_pv ON stated_values(paper_version);
CREATE INDEX IF NOT EXISTS ix_sv_num ON stated_values(paper_version, value_num);

CREATE TABLE IF NOT EXISTS declared_urls (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  url             TEXT NOT NULL,
  host            TEXT NOT NULL,
  owner           TEXT,
  repo            TEXT,
  context         TEXT,
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  in_abstract     INTEGER NOT NULL DEFAULT 0,
  src_line        INTEGER,
  UNIQUE (paper_version, url)
);
CREATE INDEX IF NOT EXISTS ix_url_pv ON declared_urls(paper_version);

-- Bibliography entries, resolved from the paper's own .bbl/.bib. These are the
-- outgoing half of the research lineage and need no network at all.
CREATE TABLE IF NOT EXISTS bib_entries (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  bib_key         TEXT NOT NULL,
  raw             TEXT NOT NULL,
  authors         TEXT,
  title           TEXT,
  year            INTEGER,
  arxiv_id        TEXT,
  doi             TEXT,
  cite_count      INTEGER NOT NULL DEFAULT 0,
  UNIQUE (paper_version, bib_key)
);
CREATE INDEX IF NOT EXISTS ix_bib_pv ON bib_entries(paper_version);

-- Where in the paper each reference is cited, with the sentence that cites it.
CREATE TABLE IF NOT EXISTS citation_sites (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  bib_key         TEXT NOT NULL,
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  context         TEXT NOT NULL,
  src_line        INTEGER,
  command         TEXT
);
CREATE INDEX IF NOT EXISTS ix_cs_pv ON citation_sites(paper_version, bib_key);

CREATE TABLE IF NOT EXISTS claims (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  statement       TEXT NOT NULL,
  kind            TEXT NOT NULL CHECK (kind IN
                    ('contribution','result','assumption','limitation','requirement')),
  section_id      TEXT REFERENCES sections(id) ON DELETE SET NULL,
  source          TEXT NOT NULL CHECK (source IN ('DETERMINISTIC','AGENT','SERVER_LLM'))
);

-- ─── Code side ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS repos (
  id              TEXT PRIMARY KEY,
  owner           TEXT NOT NULL,
  name            TEXT NOT NULL,
  default_branch  TEXT,
  stars           INTEGER,
  license         TEXT,
  archived        INTEGER,
  last_push_at    TEXT,
  metadata_fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS repo_snapshots (
  id              TEXT PRIMARY KEY,
  repo_id         TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  commit_sha      TEXT NOT NULL,
  indexed_at      TEXT,
  indexer         TEXT NOT NULL,
  symbol_count    INTEGER,
  file_count      INTEGER,
  clone_path      TEXT,
  UNIQUE (repo_id, commit_sha)
);

CREATE TABLE IF NOT EXISTS symbols (
  id              TEXT PRIMARY KEY,
  snapshot_id     TEXT NOT NULL REFERENCES repo_snapshots(id) ON DELETE CASCADE,
  qualified_name  TEXT NOT NULL,
  kind            TEXT,
  file_path       TEXT NOT NULL,
  line_start      INTEGER,
  line_end        INTEGER,
  signature       TEXT
);
CREATE INDEX IF NOT EXISTS ix_sym_snap ON symbols(snapshot_id);

-- ─── Implementation discovery ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS implementation_candidates (
  id                 TEXT PRIMARY KEY,
  paper_version      TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  repo_id            TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  relation           TEXT NOT NULL CHECK (relation IN
                       ('OFFICIAL','ORGANIZATION','REPRODUCTION','THIRD_PARTY','DERIVED',
                        'DECLARED_DEPENDENCY','UNRELATED')),
  confidence         TEXT NOT NULL CHECK (confidence IN
                       ('CONFIRMED','LIKELY','POSSIBLE','UNKNOWN')),
  coverage_score     REAL,
  coverage_checked   INTEGER NOT NULL DEFAULT 0,
  rank               INTEGER,
  UNIQUE (paper_version, repo_id)
);

-- ─── Correlation ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mappings (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  snapshot_id     TEXT NOT NULL REFERENCES repo_snapshots(id) ON DELETE CASCADE,
  anchor_kind     TEXT NOT NULL CHECK (anchor_kind IN
                    ('component','equation','algorithm','stated_value','section','claim')),
  anchor_id       TEXT NOT NULL,
  symbol_id       TEXT REFERENCES symbols(id) ON DELETE SET NULL,
  status          TEXT NOT NULL CHECK (status IN
                    ('MATCHED','ABSENT','DIVERGENT','AMBIGUOUS','UNKNOWN')),
  confidence      TEXT NOT NULL CHECK (confidence IN
                    ('CONFIRMED','LIKELY','POSSIBLE','UNKNOWN')),
  method          TEXT NOT NULL,
  reasoning       TEXT,
  analyzer_version TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mapping ON mappings(
  paper_version, snapshot_id, anchor_kind, anchor_id, IFNULL(symbol_id, ''));

CREATE TABLE IF NOT EXISTS differences (
  id              TEXT PRIMARY KEY,
  mapping_id      TEXT REFERENCES mappings(id) ON DELETE SET NULL,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  snapshot_id     TEXT NOT NULL REFERENCES repo_snapshots(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL CHECK (kind IN
                    ('MISSING_COMPONENT','EXTRA_COMPONENT','CHANGED_HYPERPARAMETER',
                     'DIFFERENT_LOSS','ARCHITECTURAL','TRAINING','INFERENCE',
                     'UNDOCUMENTED_PREPROCESSING','SHORTCUT','CONFIG_HIDDEN')),
  paper_states    TEXT,
  code_does       TEXT,
  difference      TEXT NOT NULL,
  severity        TEXT CHECK (severity IN ('BLOCKING','SIGNIFICANT','MINOR','COSMETIC')),
  confidence      TEXT NOT NULL CHECK (confidence IN
                    ('CONFIRMED','LIKELY','POSSIBLE','UNKNOWN'))
);

-- ─── Evidence ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evidence (
  id              TEXT PRIMARY KEY,
  kind            TEXT NOT NULL CHECK (kind IN
                    ('paper_section','paper_equation','paper_stated_value','paper_url',
                     'code_symbol','code_file','code_absence','citation_context',
                     'repo_metadata','external_url')),
  uri             TEXT NOT NULL,
  excerpt         TEXT,
  locator         TEXT,
  retrieved_at    TEXT NOT NULL,
  provenance_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_ev_uri ON evidence(uri);

CREATE TABLE IF NOT EXISTS evidence_links (
  evidence_id     TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
  subject_kind    TEXT NOT NULL,
  subject_id      TEXT NOT NULL,
  stance          TEXT NOT NULL CHECK (stance IN ('SUPPORTS','CONTRADICTS')),
  PRIMARY KEY (evidence_id, subject_kind, subject_id)
);
CREATE INDEX IF NOT EXISTS ix_evl_subject ON evidence_links(subject_kind, subject_id);

-- ─── Lineage ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lineage_edges (
  id              TEXT PRIMARY KEY,
  from_paper      TEXT NOT NULL,
  to_paper        TEXT NOT NULL,
  relation        TEXT NOT NULL CHECK (relation IN
                    ('CITES','FOUNDATIONAL','PREDECESSOR','SUCCESSOR','EXTENDS',
                     'IMPROVES','COMPETES','CONTRADICTS','REPRODUCES')),
  is_influential  INTEGER,
  intents_json    TEXT,
  contexts_json   TEXT,
  confidence      TEXT NOT NULL CHECK (confidence IN
                    ('CONFIRMED','LIKELY','POSSIBLE','UNKNOWN')),
  source          TEXT NOT NULL,
  UNIQUE (from_paper, to_paper, relation)
);
CREATE INDEX IF NOT EXISTS ix_lin_from ON lineage_edges(from_paper);
CREATE INDEX IF NOT EXISTS ix_lin_to ON lineage_edges(to_paper);

-- ─── Agent write-back, jobs, infrastructure ────────────────────────────────
CREATE TABLE IF NOT EXISTS analyses (
  id              TEXT PRIMARY KEY,
  paper_version   TEXT NOT NULL REFERENCES paper_versions(paper_version) ON DELETE CASCADE,
  schema_version  TEXT NOT NULL,
  fields_json     TEXT NOT NULL,
  author          TEXT NOT NULL,
  validated       INTEGER NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id              TEXT PRIMARY KEY,
  operation       TEXT NOT NULL,
  args_json       TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN
                    ('working','input_required','completed','failed','cancelled')),
  status_message  TEXT,
  progress_pct    INTEGER,
  result_json     TEXT,
  error_json      TEXT,
  poll_interval_ms INTEGER NOT NULL DEFAULT 1000,
  ttl_ms          INTEGER,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS http_cache (
  cache_key       TEXT PRIMARY KEY,
  source          TEXT NOT NULL,
  url             TEXT NOT NULL,
  body            BLOB NOT NULL,
  etag            TEXT,
  fetched_at      TEXT NOT NULL,
  expires_at      TEXT
);

CREATE TABLE IF NOT EXISTS rate_limit_buckets (
  source          TEXT PRIMARY KEY,
  tokens          REAL NOT NULL,
  last_refill_at  TEXT NOT NULL,
  capacity        REAL NOT NULL,
  refill_per_sec  REAL NOT NULL
);

-- ─── Full-text search ──────────────────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
  title, body, section_id UNINDEXED, paper_version UNINDEXED);

CREATE VIRTUAL TABLE IF NOT EXISTS components_fts USING fts5(
  name, description, component_id UNINDEXED, paper_version UNINDEXED);
