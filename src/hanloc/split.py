"""在已经检出的行或列里面切出单字。"""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass, replace

import cv2
import numpy as np

from hanloc.crop import map_through, warp_quad
from hanloc.geometry import Quad, angle_deg, lerp, quad_height, quad_width
from hanloc.types import Glyph, Line

# 比同列字宽的中位数更宽或更窄时，标成不确定。
_WIDE = 1.8
_NARROW = 0.45


@dataclass(frozen=True)
class WordBox:
    char: str
    score: float | None
    quad: Quad


class _Ids:
    def __init__(self, start: int = 0) -> None:
        self._n = start

    def next(self) -> str:
        self._n += 1
        return f"glyph-{self._n}"


def _as_quad(quad: Quad | None) -> Quad | None:
    if quad is None:
        return None
    return tuple((float(x), float(y)) for x, y in quad)  # type: ignore[return-value]


def _glyph(
    ids: _Ids,
    line: Line,
    quad: Quad,
    char: str | None,
    score: float | None,
    uncertain: bool = False,
    cell: Quad | None = None,
) -> Glyph:
    quad = _as_quad(quad)
    return Glyph(
        id=ids.next(),
        line_id=line.id,
        quad=quad,  # type: ignore[arg-type]
        angle_deg=angle_deg(quad),  # type: ignore[arg-type]
        char=char,
        score=score,
        uncertain=uncertain,
        cell=_as_quad(cell),
    )


def glyphs_from_words(line: Line, words: list[WordBox], ids: _Ids) -> list[Glyph]:
    """墨迹切不开时的退路：用识别回推的字框，否则按字数等分。"""
    if words:
        return [
            _glyph(ids, line, word.quad, word.char, word.score, cell=word.quad)
            for word in words
        ]
    text = line.text or ""
    if text:
        return _grid(line, text, ids)
    return [_glyph(ids, line, line.quad, None, line.score)]


def _grid(line: Line, text: str, ids: _Ids) -> list[Glyph]:
    count = len(text)
    quad = line.quad
    tall = quad_height(quad) >= quad_width(quad)
    glyphs: list[Glyph] = []
    for index, char in enumerate(text):
        start = index / count
        end = (index + 1) / count
        if tall:
            piece = (
                lerp(quad[0], quad[3], start),
                lerp(quad[1], quad[2], start),
                lerp(quad[1], quad[2], end),
                lerp(quad[0], quad[3], end),
            )
        else:
            piece = (
                lerp(quad[0], quad[1], start),
                lerp(quad[0], quad[1], end),
                lerp(quad[3], quad[2], end),
                lerp(quad[3], quad[2], start),
            )
        glyphs.append(_glyph(ids, line, piece, char, line.score))
    return glyphs


def assign_recognition(glyphs: list[Glyph], words: list[WordBox]) -> list[Glyph]:
    """字数一致才把识别结果抄到墨迹框上，避免对不齐时填错字。"""
    if len(glyphs) != len(words):
        return list(glyphs)
    return [
        replace(glyph, char=word.char, score=word.score)
        for glyph, word in zip(glyphs, words, strict=True)
    ]


def _slices(ink: np.ndarray, cross: int) -> list[tuple[int, int]]:
    active = ink > max(1, int(0.02 * cross))
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(active):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start >= 2:
                spans.append((start, index))
            start = None
    if start is not None and len(active) - start >= 2:
        spans.append((start, len(active)))
    return spans


def _merge_close_fragments(
    spans: list[tuple[int, int]], cross: int
) -> list[tuple[int, int]]:
    """间隙明显小于字距、宽度又不到字身的碎块，并回相邻块。标点的间隙接近字距，留着。"""
    spans = list(spans)
    skipped: set[int] = set()
    guard = 0
    while guard <= len(spans):
        guard += 1
        lengths = [end - start for start, end in spans]
        body = [length for length in lengths if length >= 0.55 * cross]
        median = statistics.median(body or lengths) or 1.0
        gaps = [spans[index + 1][0] - spans[index][1] for index in range(len(spans) - 1)]
        typical = statistics.median(gaps) if gaps else 0.0
        close = max(2.0, 0.45 * typical)
        short_at = next(
            (
                index
                for index, length in enumerate(lengths)
                if length < 0.45 * median and index not in skipped
            ),
            None,
        )
        if short_at is None:
            break
        neighbors: list[tuple[float, int]] = []
        if short_at > 0:
            gap = spans[short_at][0] - spans[short_at - 1][1]
            if gap <= close:
                neighbors.append((gap, short_at - 1))
        if short_at + 1 < len(spans):
            gap = spans[short_at + 1][0] - spans[short_at][1]
            if gap <= close:
                neighbors.append((gap, short_at + 1))
        if not neighbors:
            skipped.add(short_at)
            continue
        other = min(neighbors)[1]
        left, right = sorted((short_at, other))
        merged = (spans[left][0], spans[right][1])
        spans = [*spans[:left], merged, *spans[right + 1 :]]
        skipped.clear()
    return spans


def ink_split(
    image: np.ndarray,
    line: Line,
    ids: _Ids,
    merge_fragments: bool = False,
) -> list[Glyph]:
    """把列拉直后按墨迹间隙切字。过宽或过窄的块标为不确定。"""
    upright, inverse = warp_quad(image, line.quad, margin=0.0)
    gray = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY) if upright.ndim == 3 else upright
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape[:2]
    tall = height >= width
    if tall:
        ink = (binary > 0).sum(axis=1)
        cross = width
    else:
        ink = (binary > 0).sum(axis=0)
        cross = height
    spans = _slices(ink, cross)
    if merge_fragments:
        spans = _merge_close_fragments(spans, cross)
    if not spans:
        return [_glyph(ids, line, line.quad, None, line.score, uncertain=True)]
    lengths = [end - start for start, end in spans]
    median = statistics.median(lengths) or 1
    glyphs: list[Glyph] = []
    for start, end in spans:
        if tall:
            corners = [(0, start), (width, start), (width, end), (0, end)]
        else:
            corners = [(start, 0), (end, 0), (end, height), (start, height)]
        quad = tuple(map_through(corners, inverse))
        uncertain = (end - start) > _WIDE * median or (end - start) < _NARROW * median
        glyphs.append(_glyph(ids, line, quad, None, None, uncertain=uncertain))  # type: ignore[arg-type]
    return glyphs


def segment_line(
    image: np.ndarray,
    line: Line,
    words: list[WordBox],
    ids: _Ids,
) -> list[Glyph]:
    """按墨迹间隙先切开，再和识别出的字逐一比对。

    识别结果提供文字、置信度和一个可能偏移的位置。汉字宽度接近字高，
    标点更窄。块数比字数多就合并最像一个字的相邻块；比字数少就按宽度
    拆开最像两个字的块。对不上的位置标成不确定。
    """
    tokens = _tokens(line, words)
    if not tokens:
        return [_glyph(ids, line, line.quad, None, line.score, uncertain=True)]
    opened = _open_line(image, line)
    if opened is None:
        return glyphs_from_words(line, words, ids)
    proj, binary, inverse, tall, forward = opened
    _attach_priors(tokens, words, forward, tall, len(proj))
    cross = binary.shape[1] if tall else binary.shape[0]
    spans = _slices(proj, cross)
    pieces = _align(spans, tokens, _body_em(spans, cross), len(proj))
    if not pieces:
        count = len(tokens)
        priors = [token.prior for token in tokens]
        if all(prior is not None for prior in priors):
            prior_list = [float(prior) for prior in priors]
        else:
            prior_list = None
        forced = _fit_intervals(proj, count, prior_list)
        if len(forced) != count:
            return glyphs_from_words(line, words, ids)
        pieces = [
            (start, end, token.char, token.score, True, token.cell)
            for (start, end), token in zip(forced, tokens, strict=True)
        ]
    return _emit(pieces, proj, binary, inverse, tall, line, ids)


@dataclass
class _Token:
    char: str | None
    score: float | None
    weight: float
    prior: float | None = None
    cell: Quad | None = None


def _is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def _char_weight(char: str | None) -> float:
    """汉字接近一个字身，标点更窄，用来判断该把宽度分给谁。"""
    if not char or char.isspace():
        return 0.2
    if len(char) != 1:
        return 1.0
    if _is_han(char):
        return 1.0
    if unicodedata.category(char).startswith("P"):
        return 0.35
    return 0.55


def _tokens(line: Line, words: list[WordBox]) -> list[_Token]:
    if words:
        return [
            _Token(word.char, word.score, _char_weight(word.char), cell=word.quad)
            for word in words
        ]
    text = line.text or ""
    return [_Token(char, line.score, _char_weight(char)) for char in text]


def _attach_priors(
    tokens: list[_Token],
    words: list[WordBox],
    forward: np.ndarray | None,
    tall: bool,
    length: int,
) -> None:
    priors = _priors(words, forward, tall, length)
    if priors is None or len(priors) != len(tokens):
        return
    for token, prior in zip(tokens, priors, strict=True):
        token.prior = prior


def _body_em(spans: list[tuple[int, int]], cross: int) -> float:
    """从接近字高的墨迹块估计字身宽度。标点通常进不了这个范围。"""
    widths = [end - start for start, end in spans]
    body = [width for width in widths if 0.6 * cross <= width <= 1.35 * cross]
    if len(body) >= 2 or (len(body) == 1 and len(widths) >= 2):
        return float(statistics.median(body))
    return max(float(cross) * 0.92, 1.0)


def _align(
    spans: list[tuple[int, int]],
    tokens: list[_Token],
    em: float,
    length: int,
) -> list[tuple[int, int, str | None, float | None, bool]]:
    if not spans:
        return _layout(tokens, length, em)
    span_count, token_count = len(spans), len(tokens)
    inf = 1e12
    cost = [[inf] * (token_count + 1) for _span in range(span_count + 1)]
    prev: list[list[tuple[str, int, int] | None]] = [
        [None] * (token_count + 1) for _span in range(span_count + 1)
    ]
    cost[0][0] = 0.0

    def relax(
        span_index: int,
        token_index: int,
        value: float,
        source: tuple[str, int, int],
    ) -> None:
        if value < cost[span_index][token_index]:
            cost[span_index][token_index] = value
            prev[span_index][token_index] = source

    for span_index in range(span_count + 1):
        for token_index in range(token_count + 1):
            base = cost[span_index][token_index]
            if base >= inf:
                continue
            if span_index < span_count and token_index < token_count:
                added = _match_cost(spans[span_index], tokens[token_index], em)
                relax(span_index + 1, token_index + 1, base + added, ("M", span_index, token_index))
            if span_index + 1 < span_count and token_index < token_count:
                added = _merge_cost(
                    spans[span_index], spans[span_index + 1], tokens[token_index], em
                )
                relax(span_index + 2, token_index + 1, base + added, ("G", span_index, token_index))
            if span_index < span_count and token_index + 1 < token_count:
                added = _split_cost(
                    spans[span_index], tokens[token_index], tokens[token_index + 1], em
                )
                relax(span_index + 1, token_index + 2, base + added, ("S", span_index, token_index))
            width = 0 if span_index >= span_count else spans[span_index][1] - spans[span_index][0]
            if span_index < span_count and width <= max(2.0, 0.12 * em):
                relax(span_index + 1, token_index, base + 0.7, ("D", span_index, token_index))
            if token_index < token_count:
                relax(span_index, token_index + 1, base + 1.3, ("X", span_index, token_index))
    if cost[span_count][token_count] >= inf:
        return []
    return _trace(prev, spans, tokens, em, span_count, token_count)


def _layout(
    tokens: list[_Token], length: int, em: float
) -> list[tuple[int, int, str | None, float | None, bool]]:
    total = sum(token.weight for token in tokens) or 1.0
    cursor = 0
    pieces = []
    for index, token in enumerate(tokens):
        if index == len(tokens) - 1:
            end = length
        else:
            end = min(length, cursor + max(1, int(round(length * token.weight / total))))
        if end <= cursor:
            end = min(length, cursor + 1)
        pieces.append((cursor, end, token.char, token.score, True, token.cell))
        cursor = end
    return [piece for piece in pieces if piece[1] > piece[0]]


def _match_cost(span: tuple[int, int], token: _Token, em: float) -> float:
    width = span[1] - span[0]
    expected = max(token.weight * em, 1.0)
    width_pen = abs(width - expected) / em
    center = (span[0] + span[1]) / 2
    prior_pen = 0.0 if token.prior is None else abs(center - token.prior) / em
    return width_pen * 1.2 + prior_pen * 0.35


def _merge_cost(
    left: tuple[int, int], right: tuple[int, int], token: _Token, em: float
) -> float:
    combined = right[1] - left[0]
    expected = max(token.weight * em, 1.0)
    gap = right[0] - left[1]
    gap_pen = 0.0 if gap < 0.25 * em else gap / em
    return abs(combined - expected) / em * 1.2 + gap_pen + 0.15


def _split_cost(
    span: tuple[int, int], first: _Token, second: _Token, em: float
) -> float:
    expected = max((first.weight + second.weight) * em, 1.0)
    width = span[1] - span[0]
    return abs(width - expected) / em * 1.2 + 0.15


def _width_off(width: int, token: _Token, em: float) -> bool:
    expected = max(token.weight * em, 1.0)
    ratio = width / expected
    return ratio < 0.5 or ratio > 1.7


def _trace(
    prev: list[list[tuple[str, int, int] | None]],
    spans: list[tuple[int, int]],
    tokens: list[_Token],
    em: float,
    span_index: int,
    token_index: int,
) -> list[tuple[int, int, str | None, float | None, bool]]:
    pieces: list[tuple[int, int, str | None, float | None, bool]] = []
    guard = 0
    while (span_index or token_index) and guard < len(spans) + len(tokens) + 2:
        source = prev[span_index][token_index]
        if source is None:
            return []
        op, span_at, token_at = source
        if op == "M":
            start, end = spans[span_at]
            token = tokens[token_at]
            pieces.append((
                start,
                end,
                token.char,
                token.score,
                _width_off(end - start, token, em),
                token.cell,
            ))
        elif op == "G":
            token = tokens[token_at]
            pieces.append((
                spans[span_at][0],
                spans[span_at + 1][1],
                token.char,
                token.score,
                True,
                token.cell,
            ))
        elif op == "S":
            start, end = spans[span_at]
            first, second = tokens[token_at], tokens[token_at + 1]
            cut = _weighted_cut(start, end, first.weight, second.weight)
            pieces.append((cut, end, second.char, second.score, True, second.cell))
            pieces.append((start, cut, first.char, first.score, True, first.cell))
        elif op == "X":
            token = tokens[token_at]
            width = max(1, int(round(token.weight * em)))
            center = token.prior if token.prior is not None else _span_center(spans, span_at)
            start = int(round(center - width / 2))
            end = start + width
            pieces.append((start, end, token.char, token.score, True, token.cell))
        span_index, token_index = span_at, token_at
        guard += 1
    pieces.reverse()
    return pieces


def _span_center(spans: list[tuple[int, int]], span_at: int) -> float:
    if not spans:
        return 0.0
    index = min(span_at, len(spans) - 1)
    return (spans[index][0] + spans[index][1]) / 2


def _weighted_cut(start: int, end: int, first: float, second: float) -> int:
    cut = start + max(1, int(round((end - start) * first / (first + second))))
    return min(max(cut, start + 1), end - 1)


def _emit(
    pieces: list[tuple[int, int, str | None, float | None, bool, Quad | None]],
    proj: np.ndarray,
    binary: np.ndarray,
    inverse: np.ndarray,
    tall: bool,
    line: Line,
    ids: _Ids,
) -> list[Glyph]:
    glyphs: list[Glyph] = []
    limit = len(proj)
    for start, end, char, score, conflict, cell in pieces:
        start = min(max(0, start), limit - 1)
        end = min(max(start + 1, end), limit)
        start, end = _tighten_axis(proj, start, end)
        cross_start, cross_end, ink = _cross_span(binary, start, end, tall)
        if tall:
            corners = [
                (cross_start, start),
                (cross_end, start),
                (cross_end, end),
                (cross_start, end),
            ]
        else:
            corners = [
                (start, cross_start),
                (end, cross_start),
                (end, cross_end),
                (start, cross_end),
            ]
        glyphs.append(
            _glyph(
                ids,
                line,
                tuple(map_through(corners, inverse)),  # type: ignore[arg-type]
                char,
                score,
                uncertain=conflict or not ink,
                cell=cell,
            )
        )
    return glyphs


def _open_line(
    image: np.ndarray, line: Line
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray | None] | None:
    upright, inverse = warp_quad(image, line.quad, margin=0.0)
    gray = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY) if upright.ndim == 3 else upright
    if min(gray.shape[:2]) < 2:
        return None
    _level, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape[:2]
    tall = height >= width
    proj = (binary > 0).sum(axis=1 if tall else 0).astype(np.float64)
    ok, forward = cv2.invert(inverse)
    return proj, binary, inverse, tall, forward if ok else None


def _priors(
    words: list[WordBox] | None,
    forward: np.ndarray | None,
    tall: bool,
    length: int,
) -> list[float] | None:
    if not words or forward is None:
        return None
    centers: list[float] = []
    last = 0.0
    for word in words:
        points = map_through([(float(x), float(y)) for x, y in word.quad], forward)
        axis = [point[1] if tall else point[0] for point in points]
        center = min(max(sum(axis) / len(axis), 0.0), float(length - 1))
        center = max(center, last)
        centers.append(center)
        last = center
    return centers


def _smooth(proj: np.ndarray, expected: float) -> np.ndarray:
    # 核要小于字间距，否则相邻字之间的空白会被抹平。
    kernel = 3 if expected < 40 else 5
    if kernel >= len(proj):
        return proj
    window = np.ones(kernel, dtype=np.float64) / kernel
    return np.convolve(proj, window, mode="same")


def _fit_intervals(
    proj: np.ndarray, count: int, priors: list[float] | None
) -> list[tuple[int, int]]:
    length = len(proj)
    expected = length / count
    smooth = _smooth(proj, expected)
    candidates = _candidate_cuts(proj, smooth, count, expected)
    chosen = _choose_cuts(proj, smooth, candidates, count, priors, expected)
    if chosen is None:
        chosen = [int(round(index * length / count)) for index in range(count + 1)]
        chosen[0], chosen[-1] = 0, length
    intervals = []
    for start, end in zip(chosen, chosen[1:]):
        if end <= start:
            return []
        intervals.append((start, end))
    if len(intervals) != count:
        return []
    return intervals


def _candidate_cuts(
    proj: np.ndarray, smooth: np.ndarray, count: int, expected: float
) -> list[int]:
    length = len(smooth)
    points = {0, length}
    _add_gap_midpoints(points, proj <= 0, length)
    peak = float(smooth.max()) if length else 0.0
    if peak > 0:
        _add_gap_midpoints(points, smooth <= 0.4 * peak, length)
    for index in range(1, count):
        left = max(1, int((index - 0.5) * expected))
        right = min(length - 1, int((index + 0.5) * expected))
        if right > left:
            points.add(left + int(np.argmin(smooth[left:right])))
    return sorted(points)


def _add_gap_midpoints(points: set[int], low: np.ndarray, length: int) -> None:
    start: int | None = None
    for index, on in enumerate(low):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start >= 2 and 0 < start and index < length:
                points.add((start + index) // 2)
            start = None


def _choose_cuts(
    proj: np.ndarray,
    smooth: np.ndarray,
    candidates: list[int],
    count: int,
    priors: list[float] | None,
    expected: float,
) -> list[int] | None:
    total = len(candidates)
    if total < count + 1:
        return None
    peak = float(smooth.max()) or 1.0
    inf = 1e12
    cost = [[inf] * total for _k in range(count + 1)]
    prev = [[-1] * total for _k in range(count + 1)]
    cost[0][0] = 0.0
    for placed in range(count):
        for left in range(total):
            if cost[placed][left] >= inf:
                continue
            for right in range(left + 1, total):
                added = _segment_cost(
                    proj,
                    smooth,
                    candidates[left],
                    candidates[right],
                    placed,
                    priors,
                    expected,
                    peak,
                )
                total_cost = cost[placed][left] + added
                if total_cost < cost[placed + 1][right]:
                    cost[placed + 1][right] = total_cost
                    prev[placed + 1][right] = left
    if cost[count][total - 1] >= inf:
        return None
    cuts = [candidates[-1]]
    cursor = total - 1
    for placed in range(count, 0, -1):
        cursor = prev[placed][cursor]
        if cursor < 0:
            return None
        cuts.append(candidates[cursor])
    cuts.reverse()
    return cuts


def _segment_cost(
    proj: np.ndarray,
    smooth: np.ndarray,
    start: int,
    end: int,
    index: int,
    priors: list[float] | None,
    expected: float,
    peak: float,
) -> float:
    width = max(end - start, 1)
    ink = float(proj[start:end].sum())
    width_cost = abs(width - expected) / expected
    empty = 8.0 if ink < max(1.0, 0.01 * peak * width) else 0.0
    cut = 0.0 if end >= len(smooth) else float(smooth[min(end, len(smooth) - 1)]) / peak
    prior_cost = 0.0
    if priors is not None:
        prior_cost = abs((start + end) / 2 - priors[index]) / expected
    return width_cost * 0.6 + prior_cost * 1.4 + cut * 1.2 + empty


def _tighten_axis(proj: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """沿阅读方向把切口收到这段里的墨迹，并留一点边。"""
    ink = proj[start:end] > 0
    if not ink.any():
        return start, end
    found = np.flatnonzero(ink)
    pad = max(1, int(0.06 * (int(found[-1]) - int(found[0]) + 1)))
    new_start = max(0, start + int(found[0]) - pad)
    new_end = min(len(proj), start + int(found[-1]) + 1 + pad)
    if new_end <= new_start:
        return start, end
    return new_start, new_end


def _separate(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = [[start, end] for start, end in spans]
    for index in range(len(ordered) - 1):
        if ordered[index][1] <= ordered[index + 1][0]:
            continue
        mid = (ordered[index][1] + ordered[index + 1][0]) // 2
        mid = min(max(mid, ordered[index][0] + 1), ordered[index + 1][1] - 1)
        ordered[index][1] = mid
        ordered[index + 1][0] = mid
    return [(start, end) for start, end in ordered if end > start]


def _cross_span(
    binary: np.ndarray, start: int, end: int, tall: bool
) -> tuple[float, float, bool]:
    """在这一字的阅读切片里，把另一方向收到墨迹。"""
    height, width = binary.shape[:2]
    if tall:
        patch = binary[start:end, :]
        limit = width
        ink = (patch > 0).any(axis=0) if patch.size else np.array([], dtype=bool)
    else:
        patch = binary[:, start:end]
        limit = height
        ink = (patch > 0).any(axis=1) if patch.size else np.array([], dtype=bool)
    if patch.size == 0 or not ink.any():
        return 0.0, float(limit), False
    found = np.flatnonzero(ink)
    pad = max(1, int(0.08 * (int(found[-1]) - int(found[0]) + 1)))
    cross_start = max(0, int(found[0]) - pad)
    cross_end = min(limit, int(found[-1]) + 1 + pad)
    return float(cross_start), float(cross_end), True
