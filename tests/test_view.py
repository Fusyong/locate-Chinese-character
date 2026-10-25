import io
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from hanloc.types import page_from_dict
from hanloc.view import _ORDER, _SESSIONS, _close, app, attach_page


def _png_bytes() -> bytes:
    import cv2

    image = np.full((40, 60, 3), 255, np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _sample_page() -> dict:
    return {
        "profile": "manual",
        "width": 60,
        "height": 40,
        "lines": [
            {
                "id": "line-1",
                "quad": [[5, 5], [50, 5], [50, 30], [5, 30]],
                "angle_deg": 0,
                "text": "测",
                "score": 0.5,
            }
        ],
        "glyphs": [
            {
                "id": "glyph-1",
                "line_id": "line-1",
                "quad": [[5, 5], [25, 5], [25, 30], [5, 30]],
                "angle_deg": 0,
                "char": "测",
                "score": 0.4,
                "uncertain": False,
            }
        ],
        "flows": [
            {
                "id": "flow-1",
                "reading": "horizontal_ltr",
                "glyph_ids": ["glyph-1"],
                "role": "body",
                "text": "测",
            }
        ],
    }


@pytest.fixture(autouse=True)
def _clear_sessions():
    yield
    for document in _SESSIONS.values():
        _close(document)
    _SESSIONS.clear()
    _ORDER.clear()


def _client() -> TestClient:
    return TestClient(app)


def test_index_is_the_proofreading_page():
    response = _client().get("/")
    assert response.status_code == 200
    assert "执行 OCR" in response.text
    assert "保存改订" in response.text
    text = response.text
    assert text.index("保存改订") < text.index("简转繁") < text.index("清除")
    assert "canvas" in response.text
    assert 'value="book">现代本' in response.text
    assert 'value="woodblock">刻本' in response.text
    assert 'value="manuscript">写本' in response.text
    assert 'value="stele">碑帖' in response.text
    assert 'value="modelbook">墨迹' in response.text
    assert 'id="ruler-x"' in response.text
    assert 'id="char-px"' in response.text
    assert "字高（行列窄边）" in response.text
    assert 'id="measure"' in response.text
    assert "二次识别书法单字" in response.text
    checkbox = response.text.split('id="calligraphy"', 1)[1].split(">", 1)[0]
    assert "checked" not in checkbox


def test_images_open_in_filename_order():
    response = _client().post(
        "/api/session",
        files=[
            ("files", ("page-10.png", io.BytesIO(_png_bytes()), "image/png")),
            ("files", ("page-2.png", io.BytesIO(_png_bytes()), "image/png")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pages"] == 2
    assert body["labels"] == ["page-2.png", "page-10.png"]
    image = _client().get(f"/api/session/{body['session_id']}/pages/0/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_correction_keeps_original_recognition():
    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert opened.status_code == 200
    session_id = opened.json()["session_id"]
    attach_page(session_id, 0, page_from_dict(_sample_page()))

    saved = client.patch(
        f"/api/session/{session_id}/pages/0/correction",
        json={"kind": "glyph", "id": "glyph-1", "correction": "正"},
    )
    assert saved.status_code == 200
    assert saved.json()["char"] == "测"
    assert saved.json()["correction"] == "正"

    loaded = client.get(f"/api/session/{session_id}/pages/0")
    glyph = loaded.json()["page"]["glyphs"][0]
    assert glyph["char"] == "测"
    assert glyph["correction"] == "正"
    assert loaded.json()["page"]["flows"][0]["text"] == "测"

    cleared = client.patch(
        f"/api/session/{session_id}/pages/0/correction",
        json={"kind": "glyph", "id": "glyph-1", "correction": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["char"] == "测"
    assert cleared.json()["correction"] is None
    again = client.get(f"/api/session/{session_id}/pages/0").json()["page"]["glyphs"][0]
    assert again["char"] == "测"
    assert "correction" not in again

    rejected = client.patch(
        f"/api/session/{session_id}/pages/0/correction",
        json={"kind": "line", "id": "line-1", "correction": "改"},
    )
    assert rejected.status_code == 400
    assert "单字" in rejected.json()["detail"]
    kept = client.get(f"/api/session/{session_id}/pages/0").json()["page"]
    assert kept["lines"][0]["text"] == "测"
    assert "correction" not in kept["lines"][0]

    crop = client.get(
        "/api/crop",
        params={"session_id": session_id, "page_index": 0, "glyph_id": "glyph-1"},
    )
    assert crop.status_code == 200
    assert crop.headers["content-type"] == "image/png"


def test_edit_and_replace_round_trip():
    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    session_id = opened.json()["session_id"]
    missing = client.post(
        f"/api/session/{session_id}/pages/0/edit",
        json={"op": "delete", "ids": ["glyph-1"]},
    )
    assert missing.status_code == 409

    attach_page(session_id, 0, page_from_dict(_sample_page()))
    deleted = client.post(
        f"/api/session/{session_id}/pages/0/edit",
        json={"op": "delete", "ids": ["glyph-1"]},
    )
    assert deleted.status_code == 200
    assert deleted.json()["ids"] == []
    assert deleted.json()["page"]["glyphs"] == []

    restored = client.put(f"/api/session/{session_id}/pages/0", json=_sample_page())
    assert restored.status_code == 200
    assert restored.json()["page"]["glyphs"][0]["char"] == "测"

    unknown = client.post(
        f"/api/session/{session_id}/pages/0/edit",
        json={"op": "split", "id": "missing"},
    )
    assert unknown.status_code == 404


def test_unknown_upload_is_rejected():
    response = _client().post(
        "/api/session",
        files={"files": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_pdf_opens_page_by_page():
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(72, 72)
    pdf.new_page(72, 72)
    raw = io.BytesIO()
    pdf.save(raw)
    response = _client().post(
        "/api/session",
        data={"dpi": "72"},
        files={"files": ("book.pdf", io.BytesIO(raw.getvalue()), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pages"] == 2
    assert body["labels"][0].endswith("第1页")
    image = _client().get(f"/api/session/{body['session_id']}/pages/1/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_page_json_used_by_download_contains_both_texts():
    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    session_id = opened.json()["session_id"]
    attach_page(session_id, 0, page_from_dict(_sample_page()))
    client.patch(
        f"/api/session/{session_id}/pages/0/correction",
        json={"kind": "glyph", "id": "glyph-1", "correction": "正"},
    )
    payload = client.get(f"/api/session/{session_id}/pages/0").json()["page"]
    encoded = json.dumps(payload, ensure_ascii=False)
    assert '"char": "测"' in encoded
    assert '"correction": "正"' in encoded


def test_ocr_skips_calligraphy_unless_asked(monkeypatch):
    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    session_id = opened.json()["session_id"]

    def fail(image, glyphs):
        raise AssertionError("未勾选时不应二次识别")

    monkeypatch.setattr("hanloc.view.locate", lambda image, profile: page_from_dict(_sample_page()))
    monkeypatch.setattr("hanloc.view.reread_glyphs", fail)
    response = client.post(
        f"/api/session/{session_id}/pages/0/ocr",
        data={"profile": "book"},
    )
    assert response.status_code == 200
    glyph = response.json()["page"]["glyphs"][0]
    assert glyph["char"] == "测"
    assert "candidates" not in glyph


def test_ocr_adds_calligraphy_candidates_without_replacing_char(monkeypatch):
    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    session_id = opened.json()["session_id"]

    def fake(image, glyphs):
        assert image.shape[0] > 0
        for glyph in glyphs:
            assert glyph.char == "测"
            glyph.candidates = (("书", 0.81), ("画", 0.05))

    monkeypatch.setattr("hanloc.view.locate", lambda image, profile: page_from_dict(_sample_page()))
    monkeypatch.setattr("hanloc.view.reread_glyphs", fake)
    response = client.post(
        f"/api/session/{session_id}/pages/0/ocr",
        data={"profile": "book", "calligraphy": "1"},
    )
    assert response.status_code == 200
    glyph = response.json()["page"]["glyphs"][0]
    assert glyph["char"] == "测"
    assert glyph["candidates"] == [
        {"char": "书", "score": 0.81},
        {"char": "画", "score": 0.05},
    ]


def test_calligraphy_failure_keeps_the_previous_page(monkeypatch):
    from hanloc.types import HanlocError

    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    session_id = opened.json()["session_id"]
    attach_page(session_id, 0, page_from_dict(_sample_page()))

    def boom(image, glyphs):
        raise HanlocError('未安装书法识别依赖。请执行：pip install -e ".[calligraphy]"')

    monkeypatch.setattr("hanloc.view.locate", lambda image, profile: page_from_dict(_sample_page()))
    monkeypatch.setattr("hanloc.view.reread_glyphs", boom)
    response = client.post(
        f"/api/session/{session_id}/pages/0/ocr",
        data={"profile": "book", "calligraphy": "1"},
    )
    assert response.status_code == 400
    assert "calligraphy" in response.json()["detail"]
    glyph = client.get(f"/api/session/{session_id}/pages/0").json()["page"]["glyphs"][0]
    assert glyph["char"] == "测"
    assert "candidates" not in glyph


def test_traditional_converts_the_field():
    pytest.importorskip("opencc")
    response = _client().post("/api/traditional", json={"text": "书法"})
    assert response.status_code == 200
    assert response.json()["text"] == "書法"


def test_traditional_rejects_a_missing_field():
    response = _client().post("/api/traditional", json={})
    assert response.status_code == 400


def test_traditional_reports_a_missing_converter(monkeypatch):
    def missing():
        raise ImportError("no opencc")

    monkeypatch.setattr("hanloc.traditional._CONVERTER", None)
    monkeypatch.setattr("hanloc.traditional._load", missing)
    response = _client().post("/api/traditional", json={"text": "书"})
    assert response.status_code == 400
    assert "OpenCC" in response.json()["detail"]


def test_char_height_must_be_a_pixel_count():
    client = _client()
    opened = client.post(
        "/api/session",
        files={"files": ("page.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    session_id = opened.json()["session_id"]
    response = client.post(
        f"/api/session/{session_id}/pages/0/ocr",
        data={"profile": "book", "char_px": "很大"},
    )
    assert response.status_code == 400
    assert "像素" in response.json()["detail"]
