"""核对时写入单字改订。识别原文保持不动。"""

from __future__ import annotations

from hanloc.types import Glyph, Page

_GLYPH_LIMIT = 32


def set_correction(
    page: Page,
    kind: str,
    item_id: str,
    correction: str | None,
) -> Glyph:
    """保存或清除一个字的改订。

    ``correction`` 为 None 时清除改订。空字符串表示核对者改为空。
    不修改 ``Glyph.char`` 和 ``Flow.text``。
    """
    if kind != "glyph":
        raise ValueError("只能改订单字")
    cleaned = None if correction is None else str(correction).strip()
    if cleaned is not None and len(cleaned) > _GLYPH_LIMIT:
        raise ValueError(f"改订超过 {_GLYPH_LIMIT} 字")

    glyph = next((item for item in page.glyphs if item.id == item_id), None)
    if glyph is None:
        raise KeyError(item_id)
    glyph.correction = cleaned
    return glyph
