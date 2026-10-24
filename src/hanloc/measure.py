"""从检出的字里量一个典型字宽。白边不算在内。"""

from __future__ import annotations

import statistics

import cv2
import numpy as np

from hanloc.types import Glyph, Quad


def uniform_middle(values: list[float]) -> tuple[float, int] | None:
    """去掉过大、过小的，再取中段里彼此接近的那些，求平均。"""
    usable = [float(value) for value in values if value >= 2]
    if not usable:
        return None
    if len(usable) < 5:
        return statistics.mean(usable), len(usable)
    ordered = sorted(usable)
    cut = max(1, len(ordered) // 5)
    middle = ordered[cut : len(ordered) - cut]
    if len(middle) < 3:
        middle = ordered
    median = statistics.median(middle)
    band = [value for value in middle if abs(value - median) <= 0.15 * median]
    chosen = band if len(band) >= 3 else middle
    return statistics.mean(chosen), len(chosen)


def ink_width(image: np.ndarray, quad: Quad) -> float | None:
    """框内去掉四周空白后，墨迹在页面上的水平宽度。"""
    height, width = image.shape[:2]
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    x0 = max(0, int(np.floor(min(xs))))
    y0 = max(0, int(np.floor(min(ys))))
    x1 = min(width, int(np.ceil(max(xs))))
    y1 = min(height, int(np.ceil(max(ys))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    patch = gray[y0:y1, x0:x1]
    border = np.concatenate((patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]))
    background = float(np.median(border))
    difference = np.abs(patch.astype(np.float32) - background)
    if float(difference.max()) < 12:
        return float(x1 - x0)
    level = max(18.0, 0.35 * float(difference.max()))
    columns = np.flatnonzero((difference >= level).any(axis=0))
    if columns.size == 0:
        return None
    return float(int(columns[-1]) - int(columns[0]) + 1)


def typical_ink_width(image: np.ndarray, glyphs: list[Glyph]) -> tuple[int, int] | None:
    """一页上比较均匀的那些字，去掉白边后的平均字宽。返回像素和参与平均的字数。"""
    widths = []
    for glyph in glyphs:
        width = ink_width(image, glyph.quad)
        if width is not None:
            widths.append(width)
    found = uniform_middle(widths)
    if found is None:
        return None
    average, count = found
    return int(round(average)), count
