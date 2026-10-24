from hanloc.flow import build_flows, centerfold_line_ids, column_size_ids, segment_commentary
from hanloc.geometry import angle_deg
from hanloc.types import Glyph, Line


def _box(x0, y0, x1, y1):
    return (
        (float(x0), float(y0)),
        (float(x1), float(y0)),
        (float(x1), float(y1)),
        (float(x0), float(y1)),
    )


def _line(name, x0, y0, x1, y1, text=None):
    quad = _box(x0, y0, x1, y1)
    return Line(name, quad, angle_deg(quad), text=text, score=0.9)


def _glyph(name, line_id, x0, y0, x1, y1, char):
    quad = _box(x0, y0, x1, y1)
    return Glyph(name, line_id, quad, angle_deg(quad), char=char, score=0.8)


def test_horizontal_columns_do_not_interleave():
    lines = [
        _line("l1", 0, 0, 100, 20),
        _line("l2", 0, 30, 100, 50),
        _line("r1", 220, 0, 320, 20),
    ]
    glyphs = [
        _glyph("a", "l1", 0, 0, 20, 20, "甲"),
        _glyph("b", "l1", 20, 0, 40, 20, "乙"),
        _glyph("c", "l2", 0, 30, 20, 50, "丙"),
        _glyph("d", "r1", 220, 0, 240, 20, "丁"),
    ]
    flows = build_flows(lines, glyphs, "horizontal_ltr")
    assert [flow.glyph_ids for flow in flows] == [["a", "b", "c"], ["d"]]
    assert flows[0].text == "甲乙丙"
    assert flows[1].text == "丁"
    assert all(flow.role == "body" for flow in flows)


def test_spanning_header_stays_out_of_both_columns():
    lines = [
        _line("title", 0, 0, 320, 20),
        _line("l1", 0, 40, 100, 60),
        _line("r1", 220, 40, 320, 60),
    ]
    glyphs = [
        _glyph("t", "title", 0, 0, 40, 20, "题"),
        _glyph("a", "l1", 0, 40, 20, 60, "左"),
        _glyph("b", "r1", 220, 40, 240, 60, "右"),
    ]
    flows = build_flows(lines, glyphs, "horizontal_ltr")
    assert [flow.glyph_ids for flow in flows] == [["t"], ["a"], ["b"]]


def test_vertical_columns_are_read_right_to_left():
    lines = [
        _line("c1", 0, 0, 30, 80),
        _line("c2", 80, 0, 110, 80),
        _line("c3", 160, 10, 190, 40),
        _line("c3b", 160, 50, 190, 90),
    ]
    glyphs = [
        _glyph("a", "c1", 0, 0, 30, 30, "一"),
        _glyph("b", "c2", 80, 0, 110, 30, "二"),
        _glyph("c", "c3", 160, 10, 190, 40, "三"),
        _glyph("d", "c3b", 160, 50, 190, 90, "四"),
    ]
    flows = build_flows(lines, glyphs, "vertical_rtl")
    assert [flow.text for flow in flows] == ["三四", "二", "一"]


def test_commentary_is_right_column_then_left_and_returns_to_body():
    glyphs = [
        _glyph("a1", "c", 0, 0, 20, 20, "大"),
        _glyph("a2", "c", 0, 24, 20, 44, "字"),
        _glyph("a3", "c", 0, 48, 20, 68, "在"),
        _glyph("a4", "c", 0, 72, 20, 92, "前"),
        _glyph("s1", "c", 10, 100, 18, 110, "右"),
        _glyph("s2", "c", 10, 112, 18, 122, "下"),
        _glyph("s3", "c", 0, 100, 8, 110, "左"),
        _glyph("s4", "c", 0, 112, 8, 122, "下"),
        _glyph("c1", "c", 0, 140, 20, 160, "续"),
        _glyph("c2", "c", 0, 164, 20, 184, "后"),
    ]
    segments = segment_commentary(glyphs, ratio=0.7)
    assert [role for role, _items in segments] == ["body", "commentary", "body"]
    assert [glyph.char for glyph in segments[1][1]] == ["右", "下", "左", "下"]
    flows = build_flows([_line("c", 0, 0, 20, 190)], glyphs, "vertical_rtl", commentary_ratio=0.7)
    assert [(flow.role, flow.text) for flow in flows] == [
        ("body", "大字在前"),
        ("commentary", "右下左下"),
        ("body", "续后"),
    ]


def test_centerfold_is_the_narrow_column_at_the_gutter_or_edge():
    lines = [
        _line("body-r", 700, 40, 780, 400),
        _line("body-l", 500, 40, 580, 400),
        _line("gutter", 490, 40, 510, 400),
        _line("edge", 10, 40, 30, 400),
    ]
    assert centerfold_line_ids(lines, page_width=1000, mode="gutter") == {"gutter"}
    assert centerfold_line_ids(lines, page_width=1000, mode="left") == {"edge"}
    assert centerfold_line_ids(lines, page_width=1000, mode="none") == set()
    body = [line for line in lines if line.id not in {"gutter"}]
    glyphs = [_glyph("a", "body-r", 700, 40, 780, 80, "文")]
    flows = build_flows(body, glyphs, "vertical_rtl")
    assert flows[0].role == "body"
    assert "gutter" not in {glyph.line_id for glyph in glyphs if glyph.id in flows[0].glyph_ids}


def test_column_size_separates_a_caption_and_a_large_heading():
    lines = [
        _line("body-a", 400, 20, 480, 400),
        _line("body-b", 300, 20, 380, 400),
        _line("caption", 80, 420, 110, 470),
        _line("heading", 560, 10, 760, 200),
    ]
    narrow, wide = column_size_ids(lines, narrow_ratio=0.55, wide_ratio=1.75)
    assert narrow == {"caption"}
    assert wide == {"heading"}
    assert column_size_ids(lines[:2], narrow_ratio=0.55, wide_ratio=1.75) == (set(), set())
    bold = [
        _line("body", 100, 0, 180, 400),
        _line("bold", 300, 0, 460, 400),
    ]
    assert column_size_ids(bold, narrow_ratio=0.55, wide_ratio=1.75)[1] == set()
