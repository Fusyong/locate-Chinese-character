"""页面结果的数据结构，以及 JSON 往返。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hanloc.geometry import Quad, angle_deg, as_quad

ReadingOrder = Literal["horizontal_ltr", "vertical_rtl"]
Role = Literal["body", "commentary", "title", "marginalia", "centerfold"]
Centerfold = Literal["none", "gutter", "left", "right"]
SplitMode = Literal["ctc", "ink"]


class HanlocError(RuntimeError):
    """定位过程中可向调用方说明的错误。"""


@dataclass(frozen=True)
class Profile:
    """一种页面的检测与排序参数。"""

    name: str
    reading: ReadingOrder
    rotate90: bool
    use_cls: bool
    max_side_len: int
    unclip_ratio: float
    split: SplitMode
    centerfold: Centerfold = "none"
    commentary_ratio: float = 0.7
    mark_commentary: bool = False


@dataclass
class Line:
    """一行或一列。"""

    id: str
    quad: Quad
    angle_deg: float
    text: str | None = None
    score: float | None = None
    # 核对时的改订。None 表示未改；空字符串表示改为空。不替换 text。
    correction: str | None = None


@dataclass
class Glyph:
    """一个字的外框。score 是字级识别置信度。"""

    id: str
    line_id: str
    quad: Quad
    angle_deg: float
    char: str | None = None
    score: float | None = None
    uncertain: bool = False
    # 核对时的改订。None 表示未改；空字符串表示改为空。不替换 char。
    correction: str | None = None


@dataclass
class Flow:
    """一条语流。text 由其中的字按顺序拼成。"""

    id: str
    reading: ReadingOrder
    glyph_ids: list[str]
    role: Role
    text: str


@dataclass
class Page:
    profile: str
    width: int
    height: int
    lines: list[Line]
    glyphs: list[Glyph]
    flows: list[Flow]


def _quad_to_json(quad: Quad) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in quad]


def _maybe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _correction_json(value: str | None) -> dict[str, str]:
    if value is None:
        return {}
    return {"correction": value}


def _correction_from_json(item: dict[str, Any]) -> str | None:
    if "correction" not in item or item["correction"] is None:
        return None
    return str(item["correction"]).strip()


def page_to_dict(page: Page) -> dict[str, Any]:
    return {
        "profile": page.profile,
        "width": page.width,
        "height": page.height,
        "lines": [
            {
                "id": line.id,
                "quad": _quad_to_json(line.quad),
                "angle_deg": float(line.angle_deg),
                "text": line.text,
                "score": _maybe_float(line.score),
                **_correction_json(line.correction),
            }
            for line in page.lines
        ],
        "glyphs": [
            {
                "id": glyph.id,
                "line_id": glyph.line_id,
                "quad": _quad_to_json(glyph.quad),
                "angle_deg": float(glyph.angle_deg),
                "char": glyph.char,
                "score": _maybe_float(glyph.score),
                "uncertain": glyph.uncertain,
                **_correction_json(glyph.correction),
            }
            for glyph in page.glyphs
        ],
        "flows": [
            {
                "id": flow.id,
                "reading": flow.reading,
                "glyph_ids": list(flow.glyph_ids),
                "role": flow.role,
                "text": flow.text,
            }
            for flow in page.flows
        ],
    }


def page_from_dict(data: dict[str, Any]) -> Page:
    lines = [
        Line(
            id=str(item["id"]),
            quad=as_quad(item["quad"]),
            angle_deg=float(item.get("angle_deg", angle_deg(as_quad(item["quad"])))),
            text=item.get("text"),
            score=None if item.get("score") is None else float(item["score"]),
            correction=_correction_from_json(item),
        )
        for item in data.get("lines", [])
    ]
    glyphs = [
        Glyph(
            id=str(item["id"]),
            line_id=str(item["line_id"]),
            quad=as_quad(item["quad"]),
            angle_deg=float(item.get("angle_deg", angle_deg(as_quad(item["quad"])))),
            char=item.get("char"),
            score=None if item.get("score") is None else float(item["score"]),
            uncertain=bool(item.get("uncertain", False)),
            correction=_correction_from_json(item),
        )
        for item in data.get("glyphs", [])
    ]
    flows = [
        Flow(
            id=str(item["id"]),
            reading=item["reading"],
            glyph_ids=[str(gid) for gid in item["glyph_ids"]],
            role=item["role"],
            text=str(item.get("text") or ""),
        )
        for item in data.get("flows", [])
    ]
    return Page(
        profile=str(data.get("profile") or ""),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        lines=lines,
        glyphs=glyphs,
        flows=flows,
    )
