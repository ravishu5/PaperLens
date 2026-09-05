"""The CodeIntelligenceProvider port, exercised through the AST backend."""
import textwrap

import pytest

from paperlens.code.indexer import parse_repo
from paperlens.code.provider import SearchResult
from paperlens.code.python_ast import PythonAstProvider
from paperlens.resources import decode_symbol, encode_symbol, symbol_uri


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "model.py").write_text(textwrap.dedent('''
        import torch

        TEMPERATURE = 0.07

        class Encoder:
            """An encoder."""
            def __init__(self, width: int):
                self.width = width

            def forward(self, x):
                return x

        def build(cfg):
            return Encoder(cfg.width)
    ''').strip())
    (tmp_path / "README.md").write_text("# demo")
    p = PythonAstProvider()
    p.index("demo/repo", str(tmp_path))
    return p


def test_indexes_classes_methods_functions_and_constants(repo):
    names = {s.qualified_name for s in repo.list_symbols("demo/repo")}
    assert "pkg/model.py::Encoder#class" in names
    assert "pkg/model.py::Encoder.forward#method" in names
    assert "pkg/model.py::build#function" in names
    assert "pkg/model.py::TEMPERATURE#constant" in names


def test_symbol_ids_match_the_jcodemunch_format(repo):
    """Both backends must be interchangeable behind the port."""
    for s in repo.list_symbols("demo/repo"):
        assert "::" in s.qualified_name and "#" in s.qualified_name


def test_search_returns_citable_absence_rather_than_an_empty_list(repo):
    res = repo.search_symbols("demo/repo", "contrastive loss cross entropy")
    assert res.found is False
    assert res.absence_ref and res.absence_ref.startswith("absent:")


def test_absence_refs_are_reproducible():
    """A citable absence must be stable, or it cannot be cited."""
    a = SearchResult.absent("r", "q", "m").absence_ref
    b = SearchResult.absent("r", "q", "m").absence_ref
    assert a == b and a != SearchResult.absent("r", "other", "m").absence_ref


def test_symbol_source_returns_the_declaration(repo):
    src = repo.symbol_source("demo/repo", "pkg/model.py::Encoder.forward#method")
    assert src and "def forward" in src and "return x" in src


def test_text_search_reports_file_and_line(repo):
    hits = repo.search_text("demo/repo", "TEMPERATURE")
    assert hits and hits[0].file_path == "pkg/model.py" and hits[0].line == 3


def test_non_python_files_are_counted_but_flagged(repo):
    r = repo.index("demo/repo2", str(__import__("pathlib").Path(
        repo._repos["demo/repo"])))
    assert any("Python only" in w for w in r.warnings)


def test_unparseable_file_is_reported_not_fatal(tmp_path):
    (tmp_path / "broken.py").write_text("def (:::")
    r = PythonAstProvider().index("x/y", str(tmp_path))
    assert r.symbol_count == 0
    assert any("unparseable" in w for w in r.warnings)


def test_symbol_uri_encoding_round_trips():
    """Symbol ids contain '#' and '/', which a URI parser would otherwise treat
    as a fragment delimiter and a path separator."""
    qn = "clip/model.py::CLIP.forward#method"
    enc = encode_symbol(qn)
    assert "/" not in enc and "#" not in enc
    assert decode_symbol(enc) == qn
    assert symbol_uri("openai/CLIP", qn).startswith("paperlens://repo/openai/CLIP/symbol/")


@pytest.mark.parametrize("text,expected", [
    ("openai/CLIP", ("openai", "CLIP")),
    ("https://github.com/openai/CLIP", ("openai", "CLIP")),
    ("https://github.com/openai/CLIP.git", ("openai", "CLIP")),
])
def test_repo_identifier_parsing(text, expected):
    assert parse_repo(text) == expected


def test_bad_repo_identifier_is_rejected():
    with pytest.raises(ValueError):
        parse_repo("not-a-repo")
