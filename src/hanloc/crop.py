"""按四点框把字拉正裁出。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hanloc.geometry import Quad, as_quad, quad_height, quad_width
from hanloc.types import Glyph


def load_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    path = Path(image)
    data = np.fromfile(path, dtype=np.uint8)
    loaded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if loaded is None:
        raise ValueError(f"无法读取图片：{path}")
    return loaded


def warp_quad(
    image: np.ndarray, quad: Quad, margin: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """拉正四点框。返回裁图，以及从裁图像素映回原图的矩阵。"""
    src = np.array(as_quad(quad), dtype=np.float32)
    width = max(quad_width(quad), 1.0)
    height = max(quad_height(quad), 1.0)
    pad_x = margin * width
    pad_y = margin * height
    out_w = max(int(round(width + 2 * pad_x)), 1)
    out_h = max(int(round(height + 2 * pad_y)), 1)
    dst = np.array(
        [
            [pad_x, pad_y],
            [pad_x + width, pad_y],
            [pad_x + width, pad_y + height],
            [pad_x, pad_y + height],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return warped, inverse


def map_through(points: list[tuple[float, float]], matrix: np.ndarray) -> list[tuple[float, float]]:
    array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(array, matrix)
    return [(float(p[0][0]), float(p[0][1])) for p in mapped]


def crop_quad(image: np.ndarray, quad: Quad, margin: float = 0.15) -> np.ndarray:
    warped, _inverse = warp_quad(image, quad, margin=margin)
    return warped


def crop_glyph(
    image: str | Path | np.ndarray, glyph: Glyph, margin: float = 0.15
) -> np.ndarray:
    """按字形四点框透视拉正，四周按框尺寸的比例留白。"""
    return crop_quad(load_image(image), glyph.quad, margin=margin)
