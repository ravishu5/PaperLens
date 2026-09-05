"""Section hierarchy, algorithms, stated values, declared URLs."""
from paperlens.paper.structure import (extract_algorithms, extract_declared_urls,
                                       extract_sections, extract_stated_values)


def test_section_paths_are_hierarchical():
    tex = (r"\section{One}\subsection{One A}\subsubsection{Deep}"
           r"\section{Two}\subsection{Two A}")
    assert [s.section_path for s in extract_sections(tex)] == \
        ["1", "1.1", "1.1.1", "2", "2.1"]


def test_appendix_switches_to_letters():
    tex = r"\section{Body}\appendix\section{First App}\section{Second App}\subsection{Sub}"
    assert [s.section_path for s in extract_sections(tex)] == ["1", "A", "B", "B.1"]


def test_starred_sections_are_unnumbered_and_do_not_advance_the_counter():
    tex = r"\section{One}\section*{Acknowledgements}\section{Two}"
    paths = [s.section_path for s in extract_sections(tex)]
    assert paths[0] == "1" and paths[1].startswith("*") and paths[2] == "2"


def test_paragraph_headings_are_not_numbered():
    """\\paragraph is a run-in heading with no number; numbering it produced
    nonsense paths like 3.1.0.1."""
    tex = r"\section{A}\subsection{B}\paragraph{Encoder:}"
    paths = [s.section_path for s in extract_sections(tex)]
    assert paths == ["1", "1.1", "*encoder"]


def test_section_label_is_captured_from_the_body():
    tex = r"\section{Method}\label{sec:method} Text."
    assert extract_sections(tex)[0].latex_label == "sec:method"


def test_figure_pseudocode_is_recorded_as_unextractable():
    """CLIP's core algorithm is a PDF image. Recording it with extractable=False
    keeps the limitation visible instead of silently dropping the algorithm."""
    tex = r"""\begin{figure}
    \includegraphics[width=1.0\columnwidth]{pseudocode.pdf}
    \caption{Numpy-like pseudocode for the core of an implementation of CLIP.}
    \end{figure}"""
    algos = extract_algorithms(tex)
    assert len(algos) == 1
    assert algos[0].presentation == "figure_image"
    assert algos[0].extractable is False


def test_real_algorithm_environment_is_extractable():
    tex = r"\begin{algorithm}\caption{Training loop}\STATE x \end{algorithm}"
    algos = extract_algorithms(tex)
    assert algos[0].extractable is True and algos[0].presentation == "latex_env"


def test_stated_value_initialised_to():
    tex = r"The learnable temperature parameter $\tau$ was initialized to 0.07 and clipped."
    vals = extract_stated_values(tex)
    assert any(v.symbol == "tau" and v.value_text == "0.07" for v in vals)


def test_stated_value_equality_form():
    vals = extract_stated_values(r"We use $\beta_2 = 0.98$ and $d_{ff} = 2048$.")
    got = {(v.symbol, v.value_text) for v in vals}
    assert ("beta_2", "0.98") in got


def test_prose_is_not_mistaken_for_a_named_quantity():
    """'initialized to the equivalent of 0.07' must not yield a symbol called
    'initialized to the equ'."""
    tex = r"$\tau$ was initialized to the equivalent of 0.07 from prior work."
    for v in extract_stated_values(tex):
        assert not (v.symbol and len(v.symbol.split()) > 1
                    and {"to", "the"} & set(v.symbol.lower().split()))


def test_layout_options_are_not_stated_values():
    tex = r"\includegraphics[width=1.0\columnwidth]{f.pdf}\aboverulesep=0.2ex"
    assert not any((v.symbol or "").lower() in {"width", "aboverulesep"}
                   for v in extract_stated_values(tex))


def test_github_url_is_extracted_with_owner_and_repo():
    tex = r"We release our code at \url{https://github.com/OpenAI/CLIP}."
    urls = extract_declared_urls(tex)
    assert len(urls) == 1
    assert (urls[0].owner, urls[0].repo) == ("OpenAI", "CLIP")


def test_url_inside_abstract_is_flagged():
    """A repo link in the abstract is a much stronger official-implementation
    signal than one buried in related work."""
    tex = (r"\begin{abstract} Code at \url{https://github.com/a/b}. \end{abstract}"
           r"\section{Related} See \url{https://github.com/c/d}.")
    by_repo = {u.repo: u for u in extract_declared_urls(tex)}
    assert by_repo["b"].in_abstract is True
    assert by_repo["d"].in_abstract is False


def test_latex_markup_residue_is_not_a_stated_value():
    r"""\begin{adjustbox}{width=0.85} collapses to "beginadjustboxwidth" and
    $2\times2\times4$ to "times2times2times4". Both were extracted as
    hyperparameters and then reported CONFIRMED-absent from repositories, which
    is noise dressed as a finding."""
    from paperlens.paper.structure import _plausible_identifier
    for residue in ("beginadjustboxwidth", "times2times2times4", "endtabular"):
        assert not _plausible_identifier(residue)
    for real in ("tau", "d_ff", "warmup_steps", "P", "learning rate"):
        assert _plausible_identifier(real)


def test_prose_counts_are_not_hyperparameters():
    """"the dataset consists of 30 subjects" is not a hyperparameter named
    "consists"."""
    tex = r"The training set consists of 30 subjects and contains 219 scans."
    got = {v.symbol for v in extract_stated_values(tex)}
    assert "consists" not in got and "contains" not in got
