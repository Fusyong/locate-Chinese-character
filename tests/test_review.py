import pytest

from hanloc.review import set_correction
from hanloc.types import Flow, Glyph, Line, Page


def _page() -> Page:
    quad = ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))
    line = Line("l1", quad, 0.0, text="甲乙", score=0.9)
    glyph = Glyph("a", "l1", quad, 0.0, char="甲", score=0.2)
    flow = Flow("f1", "horizontal_ltr", ["a"], "body", "甲乙")
    return Page("book", 40, 20, [line], [glyph], [flow])


def test_glyph_correction_does_not_replace_char():
    page = _page()
    set_correction(page, "glyph", "a", "丙")
    assert page.glyphs[0].char == "甲"
    assert page.glyphs[0].correction == "丙"
    assert page.lines[0].text == "甲乙"
    assert page.flows[0].text == "甲乙"


def test_empty_correction_means_cleared_character_and_none_removes_it():
    page = _page()
    set_correction(page, "glyph", "a", "  ")
    assert page.glyphs[0].char == "甲"
    assert page.glyphs[0].correction == ""
    set_correction(page, "glyph", "a", None)
    assert page.glyphs[0].correction is None
    assert page.glyphs[0].char == "甲"


def test_line_correction_does_not_replace_text():
    page = _page()
    set_correction(page, "line", "l1", "改正")
    assert page.lines[0].text == "甲乙"
    assert page.lines[0].correction == "改正"
    assert page.flows[0].text == "甲乙"


def test_unknown_item_and_overlong_correction():
    page = _page()
    with pytest.raises(KeyError):
        set_correction(page, "glyph", "missing", "甲")
    with pytest.raises(ValueError):
        set_correction(page, "glyph", "a", "字" * 33)
