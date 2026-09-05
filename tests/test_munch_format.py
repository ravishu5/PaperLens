"""Parser for jcodemunch's compact response encoding."""
from paperlens.code.munch_format import absence_ref, is_munch, parse

OUTLINE = '''#MUNCH/1 tool=get_file_outline enc=fo1

@1=clip/model.py

repo=openai/CLIP file=clip/model.py __tables=s:symbols:id|name|kind|signature|line|end_line

s,@1::CLIP#class,CLIP,class,"class CLIP(nn.Module)",243,372
s,@1::CLIP.__init__#method,__init__,method,"def __init__(self,
                 embed_dim: int,
                 vision_layers: int
                 )",244,297
'''


def test_detects_the_format():
    assert is_munch(OUTLINE) and not is_munch('{"a": 1}')


def test_aliases_expand_as_prefixes_not_whole_cells():
    """A row carries both `@1` and `@1::CLIP#class`; only prefix substitution
    handles the second."""
    rows = parse(OUTLINE)["tables"]["symbols"]
    assert rows[0]["id"] == "clip/model.py::CLIP#class"


def test_multiline_quoted_fields_do_not_break_rows():
    """A multi-line Python signature spans lines inside one quoted CSV field.
    Parsing row-by-row lost the line numbers entirely."""
    rows = parse(OUTLINE)["tables"]["symbols"]
    init = next(r for r in rows if r["name"] == "__init__")
    assert init["line"] == "244" and init["end_line"] == "297"
    assert "embed_dim" in init["signature"]


def test_header_scalars_are_read():
    assert parse(OUTLINE)["scalars"]["repo"] == "openai/CLIP"


def test_citable_absence_marker_is_recovered():
    empty = ('#MUNCH/1 tool=search_symbols enc=ss1\n\nresult_count=0 '
             '__json._meta.absence_evidence="{""ref"":""absent:842bf42855aa"",'
             '""citable"":true}" __tables=s:results:id|name\n')
    assert absence_ref(empty) == "absent:842bf42855aa"


def test_malformed_input_degrades_instead_of_raising():
    got = parse("#MUNCH/1 tool=x enc=y\n\ngarbage,,,\n")
    assert got["tables"] == {} and isinstance(got["scalars"], dict)
