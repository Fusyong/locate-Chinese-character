"""按墨迹把竖排页面分成列，再在列内切开单字。不调用识别模型。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import cv2
import numpy as np

from hanloc.geometry import angle_deg
from hanloc.types import Glyph, Line

_LONG_SIDE = 4000


@dataclass
class _Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    def union(self, other: _Box) -> _Box:
        x0 = min(self.x, other.x)
        y0 = min(self.y, other.y)
        x1 = max(self.right, other.right)
        y1 = max(self.bottom, other.bottom)
        return _Box(x0, y0, x1 - x0, y1 - y0)


def _quad(box: _Box, scale: float) -> tuple[tuple[float, float], ...]:
    x0 = box.x / scale
    y0 = box.y / scale
    x1 = box.right / scale
    y1 = box.bottom / scale
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _ink_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    short = min(binary.shape[:2])
    kernel_size = max(3, int(short * 0.002) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _drop_rules(boxes: list[_Box]) -> list[_Box]:
    """去掉又细又长的界格线。字身接近方形，不会被去掉。"""
    if len(boxes) < 2:
        return boxes
    typical = statistics.median(box.w for box in boxes) or 1.0
    kept = []
    for box in boxes:
        thin_vertical = box.h > 8 * max(box.w, 1) and box.w < 0.4 * typical
        thin_horizontal = box.w > 8 * max(box.h, 1) and box.h < 0.4 * typical
        if not thin_vertical and not thin_horizontal:
            kept.append(box)
    return kept


def _components(mask: np.ndarray) -> list[_Box]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(16, int(0.00008 * mask.shape[0] * mask.shape[1]))
    boxes = [
        _Box(int(stats[index, 0]), int(stats[index, 1]), int(stats[index, 2]), int(stats[index, 3]))
        for index in range(1, count)
        if int(stats[index, 4]) >= min_area
    ]
    return boxes


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.6)
    radius = max(3, int(sigma * 3) | 1)
    blurred = cv2.GaussianBlur(
        values.astype(np.float32).reshape(1, -1),
        (radius, 1),
        sigmaX=sigma,
    )
    return blurred.ravel()


def _level(values: np.ndarray) -> float:
    median = float(np.median(values))
    upper = values[values >= median]
    if upper.size == 0:
        return median
    return 0.55 * float(np.median(upper))


def _runs(values: np.ndarray, level: float, min_length: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(values > level):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start >= min_length:
                runs.append((start, index))
            start = None
    if start is not None and len(values) - start >= min_length:
        runs.append((start, len(values)))
    return runs


def _projection_boxes(mask: np.ndarray) -> list[_Box]:
    """列距清楚时，用投影把整列切开。连通域被石花连住时走这里。"""
    _height, width = mask.shape[:2]
    ink = (mask > 0).astype(np.float32)
    smoothed = _smooth(ink.mean(axis=0), width * 0.012)
    columns = _runs(smoothed, _level(smoothed), max(4, width // 80))
    boxes: list[_Box] = []
    for x0, x1 in columns:
        if x1 - x0 < width * 0.025:
            continue
        if (x0 <= 1 or x1 >= width - 1) and (x1 - x0) < width * 0.06:
            continue
        column = ink[:, x0:x1]
        raw = column.mean(axis=1)
        rows = _smooth(raw, max(1.0, (x1 - x0) * 0.15))
        min_char = max(4, int((x1 - x0) * 0.35))
        band = mask[:, x0:x1]
        for y0, y1 in _runs(rows, _level(rows), min_char):
            patch = band[y0:y1]
            ys, xs = np.nonzero(patch)
            if xs.size == 0:
                continue
            boxes.append(
                _Box(
                    x0 + int(xs.min()),
                    y0 + int(ys.min()),
                    int(xs.max() - xs.min()) + 1,
                    int(ys.max() - ys.min()) + 1,
                )
            )
    return boxes


def _stuck_columns(boxes: list[_Box], page_height: int) -> bool:
    """有整列被石花连成一条时，改用投影切开。"""
    return any(
        box.h > 0.62 * page_height and box.h > 2.2 * max(box.w, 1) for box in boxes
    )


def _cluster_columns(boxes: list[_Box]) -> list[list[_Box]]:
    """横向范围叠在一起的墨迹算同一列。列与列之间几乎不重叠。"""
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda box: box.cx)
    groups: list[list[_Box]] = [[ordered[0]]]
    for box in ordered[1:]:
        group = groups[-1]
        left = min(item.x for item in group)
        right = max(item.right for item in group)
        overlap = min(box.right, right) - max(box.x, left)
        narrow = min(box.w, right - left)
        if narrow > 0 and overlap / narrow >= 0.25:
            group.append(box)
        else:
            groups.append([box])
    return groups


def _y_overlap(above: _Box, below: _Box) -> float:
    overlap = min(above.bottom, below.bottom) - max(above.y, below.y)
    narrow = min(above.h, below.h)
    if narrow <= 0:
        return 0.0
    return overlap / narrow


def _merge_parts(boxes: list[_Box]) -> list[_Box]:
    """左右部件、以及贴得很近的碎块，并成一个字。上下分开的字保持切开。"""
    ordered = sorted(boxes, key=lambda box: box.y)
    if len(ordered) < 2:
        return ordered
    heights = [box.h for box in ordered]
    typical = statistics.median(heights) or 1.0
    merged: list[_Box] = []
    for box in ordered:
        if not merged:
            merged.append(box)
            continue
        previous = merged[-1]
        gap = box.y - previous.bottom
        fragment = min(previous.h, box.h) < 0.55 * typical
        close = 0 <= gap <= 0.22 * typical
        if _y_overlap(previous, box) > 0.35 or (fragment and close):
            merged[-1] = previous.union(box)
        else:
            merged.append(box)
    return merged


def _mark_uncertain(glyphs: list[Glyph]) -> None:
    if len(glyphs) < 2:
        return
    sizes = []
    for glyph in glyphs:
        width = max(point[0] for point in glyph.quad) - min(point[0] for point in glyph.quad)
        height = max(point[1] for point in glyph.quad) - min(point[1] for point in glyph.quad)
        sizes.append((width, height))
    typical = statistics.median(min(width, height) for width, height in sizes) or 1.0
    for glyph, (width, height) in zip(glyphs, sizes, strict=True):
        if height > 1.7 * typical or width > 1.8 * typical or min(width, height) < 0.4 * typical:
            glyph.uncertain = True


def segment_columns(image: np.ndarray) -> tuple[list[Line], list[Glyph]]:
    """在已经整理好的页面上分列、切字。调用前应先反相拓片、去掉朱印。"""
    height, width = image.shape[:2]
    long_side = max(height, width)
    scale = 1.0 if long_side <= _LONG_SIDE else _LONG_SIDE / long_side
    work = image
    if scale != 1.0:
        work = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    mask = _ink_mask(work)
    boxes = _drop_rules(_components(mask))
    if _stuck_columns(boxes, mask.shape[0]):
        boxes = _projection_boxes(mask)
    lines: list[Line] = []
    glyphs: list[Glyph] = []
    glyph_count = 0
    for index, column in enumerate(_cluster_columns(boxes), start=1):
        parts = _merge_parts(column)
        if not parts:
            continue
        union = parts[0]
        for part in parts[1:]:
            union = union.union(part)
        line_quad = _quad(union, scale)
        line_id = f"line-{index}"
        lines.append(
            Line(
                id=line_id,
                quad=line_quad,  # type: ignore[arg-type]
                angle_deg=angle_deg(line_quad),  # type: ignore[arg-type]
            )
        )
        for part in parts:
            glyph_count += 1
            quad = _quad(part, scale)
            glyphs.append(
                Glyph(
                    id=f"glyph-{glyph_count}",
                    line_id=line_id,
                    quad=quad,  # type: ignore[arg-type]
                    angle_deg=angle_deg(quad),  # type: ignore[arg-type]
                )
            )
    _mark_uncertain(glyphs)
    return lines, glyphs
