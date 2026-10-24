"""核对时写入改订。识别原文保持不动。"""

from __future__ import annotations

from hanloc.types import Glyph, Line, Page

_GLYPH_LIMIT = 32
_LINE_LIMIT = 4000


def set_correction(
    page: Page,
    kind: str,
    item_id: str,
    correction: str | None,
) -> Glyph | Line:
    """保存或清除一项改订。

    ``correction`` 为 None 时清除改订。空字符串表示核对者改为空。
    不修改 ``Glyph.char``、``Line.text`` 和 ``Flow.text``。
    """
    if kind not in {"glyph", "line"}:
        raise ValueError("只能改订单字或行")
    cleaned = None if correction is None else str(correction).strip()
    limit = _GLYPH_LIMIT if kind == "glyph" else _LINE_LIMIT
    if cleaned is not None and len(cleaned) > limit:
        raise ValueError(f"改订超过 {limit} 字")

    if kind == "glyph":
        glyph = next((item for item in page.glyphs if item.id == item_id), None)
        if glyph is None:
            raise KeyError(item_id)
        glyph.correction = cleaned
        return glyph

    line = next((item for item in page.lines if item.id == item_id), None)
    if line is None:
        raise KeyError(item_id)
    line.correction = cleaned
    return line
