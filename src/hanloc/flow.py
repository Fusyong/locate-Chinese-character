"""只根据框排语流，不读取像素。"""

from __future__ import annotations

import statistics

from hanloc.geometry import Quad, center, left, quad_height, quad_width, top, x_interval
from hanloc.types import Flow, Glyph, Line, ReadingOrder, Role

# 横向重叠低于较窄框的这个比例，就分成两栏。
_COLUMN_OVERLAP = 0.35
# 比中位列宽宽出这么多的行，当成跨栏的页眉或页脚，不并进某一栏。
_SPAN_RATIO = 1.6
# 竖排时，列中心的横向间隙大于列宽的这个倍数，就另起一列。
_VERTICAL_GAP = 0.75


def _overlap_fraction(interval: tuple[float, float], quad: Quad) -> float:
    a0, a1 = interval
    b0, b1 = x_interval(quad)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    narrow = min(a1 - a0, b1 - b0)
    if narrow <= 1e-6:
        return 0.0
    return overlap / narrow


def _flow_text(glyphs: list[Glyph]) -> str:
    return "".join(glyph.char or "" for glyph in glyphs)


def _glyphs_of(line: Line, by_line: dict[str, list[Glyph]], reading: ReadingOrder) -> list[Glyph]:
    glyphs = by_line.get(line.id, [])
    if reading == "horizontal_ltr":
        return sorted(glyphs, key=lambda glyph: left(glyph.quad))
    return sorted(glyphs, key=lambda glyph: top(glyph.quad))


def _order_double_column(block: list[Glyph]) -> list[Glyph]:
    """双行小字先右行后左行；只有一行时按从上到下。"""
    if len(block) < 2:
        return list(block)
    xs = [center(glyph.quad)[0] for glyph in block]
    spread = max(xs) - min(xs)
    typical = statistics.median(quad_width(glyph.quad) for glyph in block)
    if typical <= 1e-6 or spread < 0.8 * typical:
        return sorted(block, key=lambda glyph: top(glyph.quad))
    mid = (min(xs) + max(xs)) / 2.0
    right = sorted(
        (glyph for glyph in block if center(glyph.quad)[0] >= mid),
        key=lambda glyph: top(glyph.quad),
    )
    left_side = sorted(
        (glyph for glyph in block if center(glyph.quad)[0] < mid),
        key=lambda glyph: top(glyph.quad),
    )
    return right + left_side


def segment_commentary(glyphs: list[Glyph], ratio: float) -> list[tuple[Role, list[Glyph]]]:
    """一列里明显偏小的字收成夹注，并嵌回它们所在的纵向位置。"""
    ordered = sorted(glyphs, key=lambda glyph: center(glyph.quad)[1])
    if not ordered:
        return []
    heights = [quad_height(glyph.quad) for glyph in ordered]
    median_h = statistics.median(heights)
    if median_h <= 1e-6:
        return [("body", ordered)]
    small = [quad_height(glyph.quad) < ratio * median_h for glyph in ordered]
    segments: list[tuple[Role, list[Glyph]]] = []
    index = 0
    while index < len(ordered):
        flag = small[index]
        block: list[Glyph] = []
        while index < len(ordered) and small[index] == flag:
            block.append(ordered[index])
            index += 1
        if flag:
            segments.append(("commentary", _order_double_column(block)))
        else:
            segments.append(("body", block))
    return segments


def _thickness(quad: Quad) -> float:
    """字号近似框的短边：竖列是列宽，横行是行高。"""
    return min(quad_width(quad), quad_height(quad))


def _span(quad: Quad) -> float:
    return max(quad_width(quad), quad_height(quad))


def column_size_ids(
    lines: list[Line],
    narrow_ratio: float = 0.0,
    wide_ratio: float = 0.0,
) -> tuple[set[str], set[str]]:
    """按短边把明显偏小的列收成旁注，明显偏大的列收成标题。比例为 0 时不判这一类。"""
    if len(lines) < 2:
        return set(), set()
    sizes = [_thickness(line.quad) for line in lines]
    lengths = [_span(line.quad) for line in lines]
    median = statistics.median(sizes)
    median_length = statistics.median(lengths)
    if median <= 1e-6:
        return set(), set()
    narrow = {
        line.id
        for line, size in zip(lines, sizes, strict=True)
        if narrow_ratio > 0 and size < narrow_ratio * median
    }
    wide = {
        line.id
        for line, size, length in zip(lines, sizes, lengths, strict=True)
        if wide_ratio > 0
        and size > wide_ratio * median
        and length < 0.75 * median_length
        and line.id not in narrow
    }
    return narrow, wide


def centerfold_line_ids(
    lines: list[Line],
    page_width: int,
    mode: str,
    narrow_ratio: float = 0.5,
    edge_ratio: float = 0.12,
) -> set[str]:
    """版心是靠近中缝或页面左/右缘的窄列。宽列即使在页面中间也不算版心。"""
    if mode == "none" or not lines or page_width <= 0:
        return set()
    median_w = statistics.median(quad_width(line.quad) for line in lines)
    chosen: set[str] = set()
    for line in lines:
        width = quad_width(line.quad)
        if median_w > 0 and width >= narrow_ratio * median_w:
            continue
        cx = center(line.quad)[0]
        if mode == "gutter" and abs(cx - page_width / 2.0) <= edge_ratio * page_width:
            chosen.add(line.id)
        elif mode == "left" and cx <= edge_ratio * page_width:
            chosen.add(line.id)
        elif mode == "right" and cx >= (1.0 - edge_ratio) * page_width:
            chosen.add(line.id)
    return chosen


def _cluster_horizontal(lines: list[Line]) -> tuple[list[Line], list[list[Line]]]:
    if not lines:
        return [], []
    median_w = statistics.median(quad_width(line.quad) for line in lines) or 1.0
    spanning: list[Line] = []
    normal: list[Line] = []
    for line in lines:
        if quad_width(line.quad) > _SPAN_RATIO * median_w:
            spanning.append(line)
        else:
            normal.append(line)
    columns: list[tuple[tuple[float, float], list[Line]]] = []
    for line in sorted(normal, key=lambda item: center(item.quad)[0]):
        placed = False
        for index, (interval, members) in enumerate(columns):
            if _overlap_fraction(interval, line.quad) >= _COLUMN_OVERLAP:
                b0, b1 = x_interval(line.quad)
                merged = (min(interval[0], b0), max(interval[1], b1))
                columns[index] = (merged, members + [line])
                placed = True
                break
        if not placed:
            columns.append((x_interval(line.quad), [line]))
    columns.sort(key=lambda item: item[0][0])
    return spanning, [members for _interval, members in columns]


def _cluster_vertical(lines: list[Line]) -> list[list[Line]]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: center(line.quad)[0])
    median_w = statistics.median(quad_width(line.quad) for line in ordered) or 1.0
    groups: list[list[Line]] = [[ordered[0]]]
    for line in ordered[1:]:
        gap = center(line.quad)[0] - center(groups[-1][-1].quad)[0]
        if gap > _VERTICAL_GAP * median_w:
            groups.append([line])
        else:
            groups[-1].append(line)
    groups.sort(key=lambda group: -sum(center(line.quad)[0] for line in group) / len(group))
    return groups


def _make_flow(
    reading: ReadingOrder,
    role: Role,
    glyphs: list[Glyph],
    index: int,
) -> Flow:
    return Flow(
        id=f"flow-{index}",
        reading=reading,
        glyph_ids=[glyph.id for glyph in glyphs],
        role=role,
        text=_flow_text(glyphs),
    )


def build_flows(
    lines: list[Line],
    glyphs: list[Glyph],
    reading: ReadingOrder,
    *,
    role: Role = "body",
    commentary_ratio: float | None = None,
    id_start: int = 1,
) -> list[Flow]:
    """按版式把框排成若干条语流。跨栏的横排行单独成条，避免双栏串栏。"""
    by_line: dict[str, list[Glyph]] = {}
    for glyph in glyphs:
        by_line.setdefault(glyph.line_id, []).append(glyph)

    pieces: list[tuple[float, float, Role, list[Glyph]]] = []
    if reading == "horizontal_ltr":
        spanning, columns = _cluster_horizontal(lines)
        for line in spanning:
            glyphs_here = _glyphs_of(line, by_line, reading)
            if glyphs_here:
                pieces.append((top(line.quad), left(line.quad), role, glyphs_here))
        for column in columns:
            ordered_lines = sorted(column, key=lambda line: top(line.quad))
            column_glyphs: list[Glyph] = []
            for line in ordered_lines:
                column_glyphs.extend(_glyphs_of(line, by_line, reading))
            if column_glyphs:
                pieces.append(
                    (top(ordered_lines[0].quad), left(ordered_lines[0].quad), role, column_glyphs)
                )
        pieces.sort(key=lambda piece: (piece[0], piece[1]))
    else:
        for group in _cluster_vertical(lines):
            ordered_lines = sorted(group, key=lambda line: top(line.quad))
            column_glyphs = []
            for line in ordered_lines:
                column_glyphs.extend(_glyphs_of(line, by_line, reading))
            if not column_glyphs:
                continue
            if commentary_ratio is not None and role == "body":
                for segment_role, segment in segment_commentary(column_glyphs, commentary_ratio):
                    pieces.append(
                    (
                        top(ordered_lines[0].quad),
                        left(ordered_lines[0].quad),
                        segment_role,
                        segment,
                    )
                )
            else:
                pieces.append(
                    (
                        top(ordered_lines[0].quad),
                        left(ordered_lines[0].quad),
                        role,
                        column_glyphs,
                    )
                )

    flows: list[Flow] = []
    for _top, _left, piece_role, piece_glyphs in pieces:
        if not piece_glyphs:
            continue
        flows.append(_make_flow(reading, piece_role, piece_glyphs, id_start + len(flows)))
    return flows
