"""本机 OCR 核对页。只监听 127.0.0.1。"""

from __future__ import annotations

import re
import uuid
import webbrowser
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from hanloc.crop import crop_glyph
from hanloc.edit import edit_glyphs
from hanloc.measure import typical_ink_width
from hanloc.pipeline import locate
from hanloc.prep import prepare_page
from hanloc.profiles import PROFILES
from hanloc.reread import reread_glyphs
from hanloc.review import set_correction
from hanloc.scale import expected_char_px, long_side_for_char
from hanloc.traditional import to_traditional
from hanloc.types import HanlocError, Page, page_from_dict, page_to_dict

app = FastAPI(title="hanloc")
_STATIC = Path(__file__).parent / "static"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".jp2"}
_MAX_SESSIONS = 3

_SESSIONS: dict[str, _Document] = {}
_ORDER: list[str] = []


@dataclass
class _Slot:
    label: str
    image: np.ndarray | None = None
    pdf: Any = None
    pdf_index: int = 0
    page: Page | None = None
    expected_px: int | None = None
    measured_px: int | None = None
    measured_count: int = 0


@dataclass
class _Document:
    name: str
    dpi: int
    slots: list[_Slot]
    rendered: list[int] = field(default_factory=list)
    buffers: list[bytes] = field(default_factory=list)


def _measure(slot: _Slot) -> dict[str, int | None] | None:
    if slot.expected_px is None and slot.measured_px is None:
        return None
    return {
        "expected_px": slot.expected_px,
        "measured_px": slot.measured_px,
        "count": slot.measured_count,
    }


def _sort_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


def _decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法读取图片")
    return image


def _open_pdf(data: bytes) -> Any:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise HTTPException(
            status_code=400,
            detail='读取 PDF 需要 pypdfium2，请执行 pip install -e ".[view]"',
        ) from exc
    try:
        return pdfium.PdfDocument(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法打开 PDF：{exc}") from exc


def _render_pdf_page(pdf: Any, index: int, dpi: int) -> np.ndarray:
    page = pdf[index]
    bitmap = page.render(scale=dpi / 72)
    try:
        rendered = np.array(bitmap.to_pil()).copy()
    finally:
        bitmap.close()
        page.close()
    if rendered.ndim == 2:
        return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)
    if rendered.shape[2] == 4:
        return cv2.cvtColor(rendered, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)


def _close(document: _Document | None) -> None:
    if document is None:
        return
    closed: set[int] = set()
    for slot in document.slots:
        pdf = slot.pdf
        if pdf is not None and id(pdf) not in closed:
            closed.add(id(pdf))
            pdf.close()


def _remember(document: _Document) -> str:
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = document
    _ORDER.append(session_id)
    while len(_ORDER) > _MAX_SESSIONS:
        _close(_SESSIONS.pop(_ORDER.pop(0), None))
    return session_id


def _require(session_id: str) -> _Document:
    document = _SESSIONS.get(session_id)
    if document is None:
        raise HTTPException(status_code=404, detail="会话不存在，请重新打开文件")
    return document


def _slot(document: _Document, index: int) -> _Slot:
    if index < 0 or index >= len(document.slots):
        raise HTTPException(status_code=404, detail="没有这一页")
    return document.slots[index]


def _image(document: _Document, index: int) -> np.ndarray:
    slot = _slot(document, index)
    if slot.image is None:
        if slot.pdf is None:
            raise HTTPException(status_code=500, detail="这一页没有图像")
        slot.image = _render_pdf_page(slot.pdf, slot.pdf_index, document.dpi)
        document.rendered.append(index)
        while len(document.rendered) > 4:
            old = document.rendered.pop(0)
            if old != index and document.slots[old].pdf is not None:
                document.slots[old].image = None
    return slot.image


def _summary(session_id: str, document: _Document) -> dict[str, object]:
    return {
        "session_id": session_id,
        "name": document.name,
        "dpi": document.dpi,
        "pages": len(document.slots),
        "labels": [slot.label for slot in document.slots],
        "ready": [index for index, slot in enumerate(document.slots) if slot.page is not None],
    }


def attach_page(session_id: str, index: int, page: Page) -> None:
    """把已有结果放进某一页，供测试或重新载入。不改图像，也不改识别字段以外的状态。"""
    document = _require(session_id)
    _slot(document, index).page = page


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.post("/api/session")
async def create_session(
    files: list[UploadFile] = File(...),
    dpi: int = Form(300),
) -> dict[str, object]:
    if dpi < 72 or dpi > 600:
        raise HTTPException(status_code=400, detail="分辨率需在 72 到 600 之间")
    if not files:
        raise HTTPException(status_code=400, detail="请选择图片或 PDF")

    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        name = Path(upload.filename or "page").name
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"文件是空的：{name}")
        uploads.append((name, data))
    uploads.sort(key=lambda item: _sort_key(item[0]))

    document = _Document(name=uploads[0][0], dpi=dpi, slots=[])
    try:
        for name, data in uploads:
            suffix = Path(name).suffix.lower()
            if suffix == ".pdf":
                pdf = _open_pdf(data)
                count = len(pdf)
                if count < 1:
                    pdf.close()
                    raise HTTPException(status_code=400, detail=f"PDF 没有页面：{name}")
                document.buffers.append(data)
                for page_index in range(count):
                    document.slots.append(
                        _Slot(label=f"{name} 第{page_index + 1}页", pdf=pdf, pdf_index=page_index)
                    )
                continue
            if suffix not in _IMAGE_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"请上传图片或 PDF：{name}")
            document.slots.append(_Slot(label=name, image=_decode_image(data)))
    except HTTPException:
        _close(document)
        raise
    if not document.slots:
        raise HTTPException(status_code=400, detail="没有可打开的页面")
    return _summary(_remember(document), document)


@app.get("/api/session/{session_id}")
def read_session(session_id: str) -> dict[str, object]:
    return _summary(session_id, _require(session_id))


@app.get("/api/session/{session_id}/pages/{index}")
def read_page(session_id: str, index: int) -> dict[str, object]:
    document = _require(session_id)
    slot = _slot(document, index)
    image = _image(document, index)
    height, width = image.shape[:2]
    return {
        "index": index,
        "label": slot.label,
        "width": width,
        "height": height,
        "page": None if slot.page is None else page_to_dict(slot.page),
        "measure": _measure(slot),
    }


@app.get("/api/session/{session_id}/pages/{index}/image")
def read_image(session_id: str, index: int) -> Response:
    image = _image(_require(session_id), index)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="无法输出页面图像")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


def _asked(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _max_side(char_px: str, page_long: int) -> int | None:
    text = char_px.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="字高需为像素数") from exc
    if not 8 <= value <= 4000:
        raise HTTPException(status_code=400, detail="字高需在 8 到 4000 像素之间")
    return long_side_for_char(page_long, value)


@app.post("/api/session/{session_id}/pages/{index}/ocr")
def ocr_page(
    session_id: str,
    index: int,
    profile: str = Form("book"),
    centerfold: str = Form(""),
    char_px: str = Form(""),
    calligraphy: str = Form(""),
) -> dict[str, object]:
    if profile not in PROFILES:
        raise HTTPException(status_code=400, detail=f"未知版式：{profile}")
    document = _require(session_id)
    slot = _slot(document, index)
    chosen = PROFILES[profile]
    if centerfold in {"none", "gutter", "left", "right"}:
        chosen = replace(chosen, centerfold=centerfold)  # type: ignore[arg-type]
    image = _image(document, index)
    side = _max_side(char_px, max(int(image.shape[0]), int(image.shape[1])))
    if side is not None:
        chosen = replace(chosen, max_side_len=side)
    try:
        page = locate(image, chosen)
        if _asked(calligraphy):
            reread_glyphs(prepare_page(image, chosen), page.glyphs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    slot.page = page
    page_long = max(int(image.shape[0]), int(image.shape[1]))
    if char_px.strip():
        slot.expected_px = int(round(float(char_px)))
    else:
        slot.expected_px = expected_char_px(page_long, PROFILES[profile].max_side_len)
    found = typical_ink_width(image, slot.page.glyphs)
    if found is None:
        slot.measured_px = None
        slot.measured_count = 0
    else:
        slot.measured_px, slot.measured_count = found
    return {"page": page_to_dict(slot.page), "measure": _measure(slot)}


@app.post("/api/traditional")
def traditional(body: dict[str, Any] = Body(...)) -> dict[str, str]:
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="需要 text")
    if len(text) > 32:
        raise HTTPException(status_code=400, detail="改订超过 32 字")
    try:
        converted = to_traditional(text)
    except HanlocError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"text": converted}


@app.patch("/api/session/{session_id}/pages/{index}/correction")
def correct_page(
    session_id: str,
    index: int,
    body: dict[str, Any] = Body(...),
) -> dict[str, object]:
    document = _require(session_id)
    slot = _slot(document, index)
    if slot.page is None:
        raise HTTPException(status_code=409, detail="这一页还没有识别")
    kind = body.get("kind")
    item_id = body.get("id")
    if kind != "glyph" or not isinstance(item_id, str) or "correction" not in body:
        raise HTTPException(status_code=400, detail="只能改订单字，需要 id 和 correction")
    raw = body.get("correction")
    if raw is not None and not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="correction 需为字符串或 null")
    try:
        item = set_correction(slot.page, kind, item_id, raw)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有这一项") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "kind": "glyph",
        "id": item.id,
        "char": item.char,
        "score": item.score,
        "correction": item.correction,
    }


@app.post("/api/session/{session_id}/pages/{index}/edit")
def edit_page(session_id: str, index: int, body: dict[str, Any] = Body(...)) -> dict[str, object]:
    document = _require(session_id)
    slot = _slot(document, index)
    if slot.page is None:
        raise HTTPException(status_code=409, detail="这一页还没有识别")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="需要操作说明")
    try:
        selected = edit_glyphs(slot.page, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有这个字") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"page": page_to_dict(slot.page), "ids": selected}


@app.put("/api/session/{session_id}/pages/{index}")
def replace_page(
    session_id: str,
    index: int,
    body: dict[str, Any] = Body(...),
) -> dict[str, object]:
    document = _require(session_id)
    slot = _slot(document, index)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="需要页面数据")
    try:
        slot.page = page_from_dict(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="页面数据无法读入") from exc
    return {"page": page_to_dict(slot.page)}


@app.get("/api/crop")
def crop(session_id: str, glyph_id: str, page_index: int = 0, margin: float = 0.15) -> Response:
    document = _require(session_id)
    slot = _slot(document, page_index)
    if slot.page is None:
        raise HTTPException(status_code=404, detail="这一页还没有识别")
    glyph = next((item for item in slot.page.glyphs if item.id == glyph_id), None)
    if glyph is None:
        raise HTTPException(status_code=404, detail="没有这个字")
    patch = crop_glyph(_image(document, page_index), glyph, margin=margin)
    ok, encoded = cv2.imencode(".png", patch)
    if not ok:
        raise HTTPException(status_code=500, detail="无法裁切")
    return Response(content=encoded.tobytes(), media_type="image/png")


def main() -> None:
    import threading
    import time

    import uvicorn

    url = "http://127.0.0.1:8765"

    def _open() -> None:
        time.sleep(0.6)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
