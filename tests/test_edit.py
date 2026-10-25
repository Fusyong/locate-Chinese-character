import pytest

from hanloc.edit import (
    create_glyph,
    delete_glyphs,
    edit_glyphs,
    merge_glyphs,
    move_glyphs,
    split_glyph,
)
from hanloc.types import Flow, Glyph, Line, Page


def _quad(x, y, size=10):
    return ((x, y), (x + size, y), (x + size, y + size), (x, y + size))


def _page() -> Page:
    line = Line("line-1", _quad(0, 0, 40), 0.0, text="甲乙")
    glyphs = [
        Glyph("g1", "line-1", _quad(0, 0), 0.0, char="甲"),
        Glyph("g2", "line-1", _quad(0, 12), 0.0, char="乙", correction="乙正"),
    ]
    flow = Flow("flow-1", "vertical_rtl", ["g1", "g2"], "body", "甲乙")
    return Page("stele", 40, 40, [line], glyphs, [flow])


def test_delete_removes_the_glyph_and_refreshes_the_flow():
    page = _page()
    delete_glyphs(page, ["g1"])
    assert [glyph.id for glyph in page.glyphs] == ["g2"]
    assert page.flows[0].glyph_ids == ["g2"]
    assert page.flows[0].text == "乙"
    assert page.lines[0].text == "甲乙"


def test_delete_drops_an_empty_flow():
    page = _page()
    delete_glyphs(page, ["g1", "g2"])
    assert page.glyphs == []
    assert page.flows == []


def test_merge_joins_text_in_reading_order_and_keeps_one_box():
    page = _page()
    kept = merge_glyphs(page, ["g2", "g1"])
    assert kept == "g1"
    assert [glyph.id for glyph in page.glyphs] == ["g1"]
    survivor = page.glyphs[0]
    assert survivor.char == "甲乙"
    assert survivor.correction == "甲乙正"
    assert survivor.uncertain is True
    assert page.flows[0].glyph_ids == ["g1"]
    assert page.flows[0].text == "甲乙"
    assert survivor.quad[0] == (0, 0)
    assert survivor.quad[2] == (10, 22)


def test_split_vertical_flow_stacks_the_new_box_under_the_old():
    page = _page()
    page.glyphs[0].char = "明月"
    first, second = split_glyph(page, "g1")
    assert first == "g1"
    assert second != "g1"
    assert page.flows[0].glyph_ids == ["g1", second, "g2"]
    by_id = {glyph.id: glyph for glyph in page.glyphs}
    assert by_id["g1"].char == "明"
    assert by_id[second].char == "月"
    assert by_id["g1"].quad[0][1] < by_id[second].quad[0][1]
    assert page.flows[0].text == "明月乙"


def test_move_later_then_refuse_the_end():
    page = _page()
    move_glyphs(page, ["g1"], 1)
    assert page.flows[0].glyph_ids == ["g2", "g1"]
    assert page.flows[0].text == "乙甲"
    with pytest.raises(ValueError, match="一端"):
        move_glyphs(page, ["g1"], 1)


def test_create_inserts_by_vertical_position():
    page = _page()
    created = create_glyph(page, _quad(0, 6))
    assert page.flows[0].glyph_ids == ["g1", created, "g2"]
    assert page.glyphs[-1].line_id == "line-1"
    assert page.glyphs[-1].uncertain is True


def test_split_horizontal_places_the_new_box_to_the_right():
    page = _page()
    page.flows[0].reading = "horizontal_ltr"
    page.glyphs[0].char = "明月"
    first, second = split_glyph(page, "g1")
    by_id = {glyph.id: glyph for glyph in page.glyphs}
    assert by_id[first].quad[0][0] < by_id[second].quad[0][0]
    assert by_id[first].char == "明"
    assert by_id[second].char == "月"


def test_create_without_a_line_starts_one():
    page = _page()
    page.lines.clear()
    page.glyphs.clear()
    page.flows.clear()
    created = create_glyph(page, _quad(4, 4, 12))
    assert page.lines[0].id.startswith("line-")
    assert page.flows[0].glyph_ids == [created]
    assert page.flows[0].reading == "vertical_rtl"


def test_unknown_operation_and_missing_glyph():
    page = _page()
    with pytest.raises(ValueError, match="未知"):
        edit_glyphs(page, {"op": "spin"})
    with pytest.raises(KeyError):
        edit_glyphs(page, {"op": "delete", "ids": ["missing"]})


def test_merge_needs_two_distinct_glyphs():
    page = _page()
    with pytest.raises(ValueError):
        merge_glyphs(page, ["g1", "g1"])
