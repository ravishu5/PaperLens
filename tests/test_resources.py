"""Resource resolution and its failure modes."""
import pytest

from paperlens.graph.store import Store
from paperlens.resources import ResourceNotFound, resolve


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "g.db")


def test_unknown_uri_scheme_is_rejected(store):
    with pytest.raises(ResourceNotFound, match="Unrecognised"):
        resolve(store, "paperlens://nonsense/thing")


def test_uningested_paper_says_so_rather_than_inventing(store):
    with pytest.raises(ResourceNotFound, match="has not been ingested"):
        resolve(store, "paperlens://paper/2103.00020")


def test_non_arxiv_identifier_is_rejected(store):
    with pytest.raises(ResourceNotFound, match="not an arXiv identifier"):
        resolve(store, "paperlens://paper/the-clip-paper")
