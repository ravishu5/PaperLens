r"""Citation extraction and bibliography parsing (offline)."""
from pathlib import Path

from paperlens.paper.citations import (collect, extract_citation_sites, parse_bbl,
                                       parse_bib)

BBL = r"""
\bibitem[Wu et~al.(2018)Wu, Xiong, Yu, and Lin]{wu2018unsupervised}
Wu, Z., Xiong, Y., Yu, S., and Lin, D.
\newblock Unsupervised feature learning via non-parametric instance-level
  discrimination.
\newblock \emph{arXiv preprint arXiv:1805.01978}, 2018.

\bibitem[Sohn(2016)]{sohn2016improved}
Sohn, K.
\newblock Improved deep metric learning with multi-class n-pair loss objective.
\newblock In \emph{NeurIPS}, 2016.
"""


def test_bibitem_fields_are_split_on_newblock():
    """Splitting on "." instead broke on author initials: "Wu, Z., Xiong, Y."
    became two fields, so the title came out as ", Xiong, Y"."""
    e = parse_bbl(BBL)["wu2018unsupervised"]
    assert e.authors.startswith("Wu, Z., Xiong, Y.")
    assert e.title.startswith("Unsupervised feature learning")


def test_arxiv_id_is_recovered_from_the_entry():
    assert parse_bbl(BBL)["wu2018unsupervised"].arxiv_id == "1805.01978"
    assert parse_bbl(BBL)["sohn2016improved"].arxiv_id is None


def test_year_is_recovered():
    entries = parse_bbl(BBL)
    assert entries["wu2018unsupervised"].year == 2018
    assert entries["sohn2016improved"].year == 2016


def test_bibtex_entries_parse_too():
    bib = """@article{smith2020thing,
      author = {Smith, J. and Doe, A.},
      title = {A Thing},
      year = {2020},
      eprint = {2001.01234},
      doi = {10.1234/abc}
    }"""
    e = parse_bib(bib)["smith2020thing"]
    assert e.title == "A Thing" and e.year == 2020
    assert e.arxiv_id == "2001.01234" and e.doi == "10.1234/abc"


def test_all_citation_commands_are_recognised():
    tex = (r"\citep{a} \citet{b} \cite{c} \citealp{d} "
           r"\citep[see][p.~3]{e} \citep{f,g}")
    keys = {s.bib_key for s in extract_citation_sites(tex)}
    assert keys == set("abcdefg")


def test_context_is_the_sentence_containing_this_citation():
    r"""Picking the first sentence containing any \cite attached the wrong
    quotation whenever two citations sat close together."""
    tex = ("First sentence citing \\citep{alpha} something. "
           "Second sentence citing \\citep{beta} otherwise.")
    by_key = {s.bib_key: s.context for s in extract_citation_sites(tex)}
    assert "alpha" in by_key["alpha"] and "beta" not in by_key["alpha"]
    assert "beta" in by_key["beta"] and "alpha" not in by_key["beta"]


def test_citation_counts_join_to_bibliography(tmp_path):
    (tmp_path / "refs.bbl").write_text(BBL)
    tex = r"We follow \citep{wu2018unsupervised}. Also \citep{wu2018unsupervised}."
    entries, sites = collect(tex, tmp_path)
    assert entries["wu2018unsupervised"].cite_count == 2
    assert len(sites) == 2


def test_a_key_cited_but_absent_from_the_bibliography_is_kept(tmp_path):
    (tmp_path / "refs.bbl").write_text(BBL)
    entries, _ = collect(r"See \citep{ghost}.", tmp_path)
    assert "ghost" in entries and entries["ghost"].title is None


def test_missing_bibliography_is_not_fatal(tmp_path):
    entries, sites = collect(r"See \citep{x}.", tmp_path)
    assert len(sites) == 1 and entries["x"].arxiv_id is None
