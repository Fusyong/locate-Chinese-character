import json

from hanloc.types import page_from_dict, page_to_dict


def test_page_json_roundtrip():
    from hanloc.flow import build_flows
    from hanloc.geometry import angle_deg
    from hanloc.types import Glyph, Line, Page

    quad = ((0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0))
    lines = [Line("l1", quad, angle_deg(quad), text="甲乙", score=0.9)]
    glyphs = [
        Glyph(
            "a",
            "l1",
            ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
            0.0,
            char="甲",
            score=0.8,
        ),
        Glyph(
            "b",
            "l1",
            ((20.0, 0.0), (40.0, 0.0), (40.0, 20.0), (20.0, 20.0)),
            0.0,
            char="乙",
            score=0.8,
        ),
    ]
    flows = build_flows(lines, glyphs, "horizontal_ltr")
    page = Page("book", 100, 80, lines, glyphs, flows)
    restored = page_from_dict(json.loads(json.dumps(page_to_dict(page))))
    assert restored.lines[0].text == "甲乙"
    assert restored.lines[0].score == 0.9
    assert restored.glyphs[1].char == "乙"
    assert restored.flows[0].text == "甲乙"
    assert restored.glyphs[0].quad[0] == (0.0, 0.0)
    assert "correction" not in page_to_dict(page)["glyphs"][0]
    assert restored.glyphs[0].correction is None


def test_correction_roundtrip_keeps_original_text():
    from hanloc.types import Glyph, Line, Page

    quad = ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))
    page = Page(
        "book",
        20,
        20,
        [Line("l1", quad, 0.0, text="甲", score=0.4, correction="乙")],
        [Glyph("a", "l1", quad, 0.0, char="甲", score=0.4, correction="")],
        [],
    )
    data = page_to_dict(page)
    assert data["lines"][0]["text"] == "甲"
    assert data["lines"][0]["correction"] == "乙"
    assert data["glyphs"][0]["char"] == "甲"
    assert data["glyphs"][0]["correction"] == ""
    restored = page_from_dict(data)
    assert restored.lines[0].text == "甲"
    assert restored.lines[0].correction == "乙"
    assert restored.glyphs[0].char == "甲"
    assert restored.glyphs[0].correction == ""


def test_import_does_not_load_rapidocr():
    import sys

    sys.modules.pop("rapidocr", None)
    import hanloc

    assert "rapidocr" not in sys.modules
    assert callable(hanloc.locate)
    assert callable(hanloc.crop_glyph)
