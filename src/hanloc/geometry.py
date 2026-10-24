"""四点框、朝向、旋转映回与相对间距。不依赖 OCR。"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


def as_quad(points: Sequence[Sequence[float]]) -> Quad:
    materialized = list(points)
    if len(materialized) != 4:
        raise ValueError("四点框需要四个角点")
    return tuple((float(p[0]), float(p[1])) for p in materialized)  # type: ignore[return-value]


def edge_length(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def quad_width(quad: Quad) -> float:
    """第一条边与对边中较长的一条，视为框的宽。"""
    return max(edge_length(quad[0], quad[1]), edge_length(quad[2], quad[3]))


def quad_height(quad: Quad) -> float:
    """两条侧边中较长的一条，视为框的高。"""
    return max(edge_length(quad[1], quad[2]), edge_length(quad[3], quad[0]))


def angle_deg(quad: Quad) -> float:
    """第一条边相对图像横轴的角度，y 向下，向右为 0°。"""
    x0, y0 = quad[0]
    x1, y1 = quad[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def center(quad: Quad) -> Point:
    return (
        sum(p[0] for p in quad) / 4.0,
        sum(p[1] for p in quad) / 4.0,
    )


def top(quad: Quad) -> float:
    return min(p[1] for p in quad)


def left(quad: Quad) -> float:
    return min(p[0] for p in quad)


def x_interval(quad: Quad) -> tuple[float, float]:
    xs = [p[0] for p in quad]
    return min(xs), max(xs)


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def centers_separated(a: Quad, b: Quad, ratio: float = 1.5) -> bool:
    """中心距大于二者高度均值的 ratio 倍时，不把它们看成同一字列。"""
    ax, ay = center(a)
    bx, by = center(b)
    dist = math.hypot(ax - bx, ay - by)
    mean_h = (quad_height(a) + quad_height(b)) / 2.0
    if mean_h <= 1e-6:
        return dist > 0
    return dist > ratio * mean_h


def rotate_point_cw90(x: float, y: float, height: int) -> Point:
    """顺时针转 90° 后的坐标。height 是旋转前图像的高度。"""
    return (height - 1 - y, x)


def map_point_from_cw90(x_rot: float, y_rot: float, height: int) -> Point:
    """把顺时针转 90° 之后图像上的点映回原图。"""
    return (y_rot, height - 1 - x_rot)


def map_quad_from_cw90(quad: Quad, height: int) -> Quad:
    return as_quad(map_point_from_cw90(x, y, height) for x, y in quad)
