"""改单字框：删除、合并、拆分、调序、新建。不改识别流水线。"""

from __future__ import annotations

from hanloc.geometry import (
    Quad,
    angle_deg,
    as_quad,
    center,
    left,
    lerp,
    quad_height,
    quad_width,
    top,
)
from hanloc.profiles import PROFILES
from hanloc.types import Flow, Glyph, Line, Page, ReadingOrder

_MIN_SIDE = 2.0


def edit_glyphs(page: Page, body: dict) -> list[str]:
    """按操作改这一页。返回操作之后应选中的字号。"""
    op = body.get("op")
    if op == "delete":
        delete_glyphs(page, _ids(body))
        return []
    if op == "merge":
        return [merge_glyphs(page, _ids(body))]
    if op == "split":
        return list(split_glyph(page, _one_id(body)))
    if op == "move":
        move_glyphs(page, _ids(body), _direction(body))
        return _ids(body)
    if op == "create":
        return [create_glyph(page, body.get("quad"))]
    raise ValueError("未知的字框操作")


def delete_glyphs(page: Page, ids: list[str]) -> None:
    wanted = _known(page, ids)
    page.glyphs = [glyph for glyph in page.glyphs if glyph.id not in wanted]
    _refresh(page)


def merge_glyphs(page: Page, ids: list[str]) -> str:
    if len(set(ids)) < 2:
        raise ValueError("合并至少要两个字")
    ordered = _in_reading_order(page, _known(page, ids))
    survivor = ordered[0]
    chars = [glyph.char or "" for glyph in ordered]
    shown = [
        glyph.correction if glyph.correction is not None else (glyph.char or "")
        for glyph in ordered
    ]
    survivor.quad = _bounds([glyph.quad for glyph in ordered])
    survivor.angle_deg = angle_deg(survivor.quad)
    survivor.char = "".join(chars) or None
    if any(glyph.correction is not None for glyph in ordered):
        survivor.correction = "".join(shown)
    else:
        survivor.correction = None
    survivor.score = None
    survivor.uncertain = True
    survivor.candidates = ()
    survivor.cell = None
    drop = {glyph.id for glyph in ordered[1:]}
    page.glyphs = [glyph for glyph in page.glyphs if glyph.id not in drop]
    _refresh(page)
    return survivor.id


def split_glyph(page: Page, glyph_id: str) -> tuple[str, str]:
    glyph = _glyph(page, glyph_id)
    vertical = _reading_of(page, glyph.id) == "vertical_rtl"
    first, second = _split_quad(glyph.quad, vertical)
    shown = glyph.correction if glyph.correction is not None else (glyph.char or "")
    head, tail = (shown[0], shown[1:]) if len(shown) >= 2 else (shown, "")
    glyph.quad = first
    glyph.angle_deg = angle_deg(first)
    glyph.char = head or None
    glyph.correction = None
    glyph.score = None
    glyph.uncertain = True
    glyph.candidates = ()
    glyph.cell = None
    created = _blank(page, glyph.line_id, second, tail or None)
    page.glyphs.append(created)
    for flow in page.flows:
        if glyph.id in flow.glyph_ids:
            flow.glyph_ids.insert(flow.glyph_ids.index(glyph.id) + 1, created.id)
            break
    else:
        page.flows.append(_flow(page, [glyph.id, created.id]))
    _refresh(page)
    return glyph.id, created.id


def move_glyphs(page: Page, ids: list[str], direction: int) -> None:
    if direction not in (-1, 1):
        raise ValueError("方向只能前移或后移")
    moving = _known(page, ids)
    placed = {glyph_id for flow in page.flows for glyph_id in flow.glyph_ids}
    if any(glyph_id not in placed for glyph_id in moving):
        raise ValueError("有的字不在语流里")
    changed = False
    for flow in page.flows:
        if _shift(flow.glyph_ids, moving, direction):
            changed = True
    if not changed:
        raise ValueError("这些字已经在语流的一端")
    _refresh(page)


def create_glyph(page: Page, raw_quad: object) -> str:
    quad = as_quad(raw_quad)  # type: ignore[arg-type]
    if quad_width(quad) < _MIN_SIDE or quad_height(quad) < _MIN_SIDE:
        raise ValueError("字框太小")
    line = _nearest_line(page, quad)
    if line is None:
        line = Line(
            id=_fresh({item.id for item in page.lines}, "line-"),
            quad=quad,
            angle_deg=angle_deg(quad),
        )
        page.lines.append(line)
    created = _blank(page, line.id, quad, None)
    page.glyphs.append(created)
    flow = _flow_for_line(page, line.id)
    if flow is None:
        flow = _flow(page, [])
        page.flows.append(flow)
    _insert(flow, page, created)
    _refresh(page)
    return created.id


def _ids(body: dict) -> list[str]:
    raw = body.get("ids")
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item for item in raw)
    ):
        raise ValueError("需要字的编号")
    return list(dict.fromkeys(raw))


def _one_id(body: dict) -> str:
    raw = body.get("id")
    if not isinstance(raw, str) or not raw:
        raise ValueError("需要一个字的编号")
    return raw


def _direction(body: dict) -> int:
    raw = body.get("direction")
    if raw not in (-1, 1):
        raise ValueError("方向只能前移或后移")
    return int(raw)


def _glyph(page: Page, glyph_id: str) -> Glyph:
    found = next((glyph for glyph in page.glyphs if glyph.id == glyph_id), None)
    if found is None:
        raise KeyError(glyph_id)
    return found


def _known(page: Page, ids: list[str]) -> set[str]:
    if not ids:
        raise ValueError("需要字的编号")
    present = {glyph.id for glyph in page.glyphs}
    missing = [glyph_id for glyph_id in ids if glyph_id not in present]
    if missing:
        raise KeyError(missing[0])
    return set(ids)


def _in_reading_order(page: Page, ids: set[str]) -> list[Glyph]:
    by_id = {glyph.id: glyph for glyph in page.glyphs}
    ordered: list[Glyph] = []
    seen: set[str] = set()
    for flow in page.flows:
        for glyph_id in flow.glyph_ids:
            if glyph_id in ids and glyph_id not in seen:
                ordered.append(by_id[glyph_id])
                seen.add(glyph_id)
    for glyph in page.glyphs:
        if glyph.id in ids and glyph.id not in seen:
            ordered.append(glyph)
    return ordered


def _reading_of(page: Page, glyph_id: str) -> ReadingOrder:
    for flow in page.flows:
        if glyph_id in flow.glyph_ids:
            return flow.reading
    return _page_reading(page)


def _page_reading(page: Page) -> ReadingOrder:
    if page.flows:
        return page.flows[0].reading
    profile = PROFILES.get(page.profile)
    if profile is not None:
        return profile.reading
    return "horizontal_ltr"


def _split_quad(quad: Quad, vertical: bool) -> tuple[Quad, Quad]:
    top_left, top_right, bottom_right, bottom_left = quad
    if vertical:
        mid_left = lerp(top_left, bottom_left, 0.5)
        mid_right = lerp(top_right, bottom_right, 0.5)
        upper = (top_left, top_right, mid_right, mid_left)
        lower = (mid_left, mid_right, bottom_right, bottom_left)
        return upper, lower
    mid_top = lerp(top_left, top_right, 0.5)
    mid_bottom = lerp(bottom_left, bottom_right, 0.5)
    west = (top_left, mid_top, mid_bottom, bottom_left)
    east = (mid_top, top_right, bottom_right, mid_bottom)
    return west, east


def _bounds(quads: list[Quad]) -> Quad:
    xs = [point[0] for quad in quads for point in quad]
    ys = [point[1] for quad in quads for point in quad]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _blank(page: Page, line_id: str, quad: Quad, char: str | None) -> Glyph:
    return Glyph(
        id=_fresh({glyph.id for glyph in page.glyphs}, "glyph-"),
        line_id=line_id,
        quad=quad,
        angle_deg=angle_deg(quad),
        char=char,
        uncertain=True,
    )


def _flow(page: Page, glyph_ids: list[str]) -> Flow:
    return Flow(
        id=_fresh({flow.id for flow in page.flows}, "flow-"),
        reading=_page_reading(page),
        glyph_ids=list(glyph_ids),
        role="body",
        text="",
    )


def _nearest_line(page: Page, quad: Quad) -> Line | None:
    if not page.lines:
        return None
    cx, cy = center(quad)

    def distance(line: Line) -> float:
        lx, ly = center(line.quad)
        return (lx - cx) ** 2 + (ly - cy) ** 2

    return min(page.lines, key=distance)


def _flow_for_line(page: Page, line_id: str) -> Flow | None:
    on_line = {glyph.id for glyph in page.glyphs if glyph.line_id == line_id}
    for flow in page.flows:
        if any(glyph_id in on_line for glyph_id in flow.glyph_ids):
            return flow
    return None


def _insert(flow: Flow, page: Page, glyph: Glyph) -> None:
    by_id = {item.id: item for item in page.glyphs}
    vertical = flow.reading == "vertical_rtl"
    mark = top(glyph.quad) if vertical else left(glyph.quad)
    index = 0
    for glyph_id in flow.glyph_ids:
        other = by_id.get(glyph_id)
        if other is None:
            index += 1
            continue
        key = top(other.quad) if vertical else left(other.quad)
        if key <= mark:
            index += 1
    flow.glyph_ids.insert(index, glyph.id)


def _shift(seq: list[str], moving: set[str], direction: int) -> bool:
    indexes = [index for index, glyph_id in enumerate(seq) if glyph_id in moving]
    if direction > 0:
        indexes.reverse()
    changed = False
    for index in indexes:
        target = index + direction
        if target < 0 or target >= len(seq) or seq[target] in moving:
            continue
        seq[index], seq[target] = seq[target], seq[index]
        changed = True
    return changed


def _refresh(page: Page) -> None:
    by_id = {glyph.id: glyph for glyph in page.glyphs}
    kept: list[Flow] = []
    for flow in page.flows:
        flow.glyph_ids = [glyph_id for glyph_id in flow.glyph_ids if glyph_id in by_id]
        flow.text = "".join(by_id[glyph_id].char or "" for glyph_id in flow.glyph_ids)
        if flow.glyph_ids:
            kept.append(flow)
    page.flows = kept


def _fresh(existing: set[str], prefix: str) -> str:
    number = 1
    while f"{prefix}{number}" in existing:
        number += 1
    return f"{prefix}{number}"
