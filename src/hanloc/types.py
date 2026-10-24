"""页面结果的数据结构，以及 JSON 往返。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hanloc.geometry import Quad, angle_deg, as_quad

ReadingOrder = Literal["horizontal_ltr", "vertical_rtl"]
Role = Literal["body", "commentary", "title", "marginalia", "centerfold"]
Centerfold = Literal["none", "gutter", "left", "right"]
SplitMode = Literal["ctc", "ink"]
Polarity = Literal["auto", "light", "dark"]
LocateMode = Literal["lines", "columns"]


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
    # dark：墨色深于纸，不反相。light：拓片浅字，强制反相。auto：看中心区域再决定。
    polarity: Polarity = "dark"
    suppress_red: bool = False
    ocr_version: str = "PP-OCRv6"
    det_model: str = "small"
    use_dilation: bool = True
    box_thresh: float = 0.5
    # 为真时按识别出的字数对齐墨迹；碑帖识别常对不上，改为只按墨迹切。
    align_chars: bool = True
    mark_marginalia: bool = False
    marginal_ratio: float = 0.55
    mark_title: bool = False
    title_ratio: float = 1.75
    # lines：先检出文本行再切字。columns：按墨迹分列，再在列内切字。
    locate_mode: LocateMode = "lines"


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
    """一个字。quad 是墨迹单字框，cell 是识别回推的字身格。"""

    id: str
    line_id: str
    quad: Quad
    angle_deg: float
    char: str | None = None
    score: float | None = None
    uncertain: bool = False
    # 核对时的改订。None 表示未改；空字符串表示改为空。不替换 char。
    correction: str | None = None
    # 识别回推的字身格。没有这项时为 None。
    cell: Quad | None = None


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


def _cell_json(quad: Quad | None) -> dict[str, list[list[float]]]:
    if quad is None:
        return {}
    return {"cell": _quad_to_json(quad)}


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
                **_cell_json(glyph.cell),
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
            cell=None if not item.get("cell") else as_quad(item["cell"]),
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
