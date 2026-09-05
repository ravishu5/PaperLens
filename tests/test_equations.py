"""Equation counter simulation.

The counter is the only way to answer "which equation is Equation 3?", because
the number exists nowhere in the source. These tests pin the LaTeX rules.
"""
from paperlens.paper.equations import extract_equations


def test_numbered_and_starred_counting():
    tex = r"""
    \begin{equation} a = b \end{equation}
    \begin{equation*} c = d \end{equation*}
    \begin{equation} e = f \end{equation}
    """
    eqs = extract_equations(tex)
    assert [e.derived_number for e in eqs] == ["1", None, "2"]
    assert [e.is_numbered for e in eqs] == [True, False, True]
    assert eqs[1].number_conf == "UNKNOWN"


def test_align_numbers_every_row():
    tex = r"\begin{align} a &= b \\ c &= d \\ e &= f \end{align}"
    eqs = extract_equations(tex)
    assert [e.derived_number for e in eqs] == ["1", "2", "3"]


def test_nonumber_suppresses_a_row():
    tex = r"\begin{align} a &= b \\ c &= d \nonumber \\ e &= f \end{align}"
    eqs = extract_equations(tex)
    assert [e.derived_number for e in eqs] == ["1", None, "2"]


def test_tag_downgrades_confidence():
    """\\tag overrides the printed number, so our arabic count is a guess."""
    tex = r"\begin{equation} a = b \tag{7} \end{equation}"
    assert extract_equations(tex)[0].number_conf == "POSSIBLE"


def test_numberwithin_downgrades_confidence():
    """\\numberwithin makes numbers look like "3.1", not "5"."""
    tex = r"\numberwithin{equation}{section}" + r"\begin{equation} a=b \end{equation}"
    assert extract_equations(tex)[0].number_conf == "POSSIBLE"


def test_clean_numbering_is_likely_never_confirmed():
    """CONFIRMED would require checking against the compiled PDF, which we do not do."""
    eqs = extract_equations(r"\begin{equation} a = b \end{equation}")
    assert eqs[0].number_conf == "LIKELY"


def test_content_hash_ignores_labels_and_whitespace():
    a = extract_equations(r"\begin{equation}a = b\label{x}\end{equation}")[0]
    b = extract_equations(r"\begin{equation}   a  =  b   \end{equation}")[0]
    assert a.content_hash == b.content_hash


def test_bracket_display_math_is_unnumbered():
    eqs = extract_equations(r"\[ a = b \]")
    assert len(eqs) == 1 and eqs[0].derived_number is None


def test_paper_with_no_equations_yields_none():
    assert extract_equations(r"\section{Intro} Only $inline$ math here.") == []
