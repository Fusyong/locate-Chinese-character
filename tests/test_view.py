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

    blank = client.patch(
        f"/api/session/{session_id}/pages/0/correction",
        json={"kind": "line", "id": "line-1", "correction": ""},
    )
    assert blank.status_code == 200
    assert blank.json()["text"] == "测"
    assert blank.json()["correction"] == ""

    crop = client.get(
        "/api/crop",
        params={"session_id": session_id, "page_index": 0, "glyph_id": "glyph-1"},
    )
    assert crop.status_code == 200
    assert crop.headers["content-type"] == "image/png"


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
