"""Mapper heuristics that decide whether a claim can be made at all."""
from paperlens.correlate.mapper import (_GENERIC, _distinctive,
                                        _is_named_quantity, _looks_like_result,
                                        _stem, _tokens)


def test_common_values_carry_no_information():
    for v in ("1", "0.1", "0.5", "100"):
        assert not _distinctive(v)


def test_uncommon_constants_are_distinctive():
    """0.07 has one significant figure but is highly distinctive. An earlier
    sig-fig rule rejected it and lost the headline mapping."""
    for v in ("0.07", "0.98", "2048", "4000", "91.8"):
        assert _distinctive(v)


def test_reported_metrics_are_not_hyperparameters():
    assert _looks_like_result("accuracy", "reaches 91.8 accuracy on ImageNet")
    assert _looks_like_result(None, "a BLEU score of 41.8")
    assert not _looks_like_result("tau", "the temperature was initialized to 0.07")


def test_prose_phrases_are_not_named_quantities():
    assert _is_named_quantity("tau")
    assert _is_named_quantity("d_ff")
    assert not _is_named_quantity("assembled collection")
    assert not _is_named_quantity(None)


def test_hyphenated_names_yield_their_joined_form():
    """"V-Net" tokenises to "net" alone, which is useless; the code calls it
    VNet."""
    assert "vnet" in _tokens("V-Net Architecture")
    assert "uxnet" in _tokens("3D UX-Net backbone")


def test_stemming_lets_encoder_match_encode_image():
    assert _stem("encoder") in "clip/model.py::clip.encode_image#method"
    assert _stem("text") in "clip/model.py::clip.encode_text#method"


def test_stopwords_are_dropped_from_component_names():
    # "Pre-Training" also yields its de-hyphenated form, because "V-Net" would
    # otherwise survive only as the useless token "net".
    assert _tokens("Selecting an Efficient Pre-Training Method") == [
        "pre", "training", "method", "pretraining"]
    assert _tokens("Image Encoder") == ["image", "encoder"]


def test_section_heading_words_cannot_sustain_an_absence_claim():
    """A component named only "Inference" was reported ABSENT from a repository
    that plainly performs inference."""
    for word in ("inference", "training", "architecture", "results"):
        assert word in _GENERIC


def test_an_uncaptioned_listing_is_not_reported_absent(tmp_path):
    """An uncaptioned verbatim block gets a synthetic slug like "verbatim-l767".
    Searching code for that name and declaring it absent says nothing."""
    from paperlens.correlate.mapper import _map_algorithm

    row = {"extractable": 1, "name": None, "slug": "verbatim-l767"}
    status, _, why, ev, sym = _map_algorithm(None, None, "o/r", row)
    assert status == "UNKNOWN" and ev == [] and "no caption" in why
