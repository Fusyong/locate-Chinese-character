"""检测前的缩放。识别器把一行收成约 48 像素高，单字大约也是这个高度。"""

from __future__ import annotations

from hanloc.types import Line

# 与核对页脚本里的 SCALE 保持一致。
TARGET_CHAR_PX = 48
DEFAULT_LONG_SIDE = 2000
MIN_LONG_SIDE = 960
MAX_LONG_SIDE = 2400
_NEARBY_LONG_SIDES = (1600, 2400)


def clamp_long_side(value: int) -> int:
    """长边限制在检测还能稳定出框的范围内。"""
    return max(MIN_LONG_SIDE, min(MAX_LONG_SIDE, int(value)))


def long_side_for_char(page_long: int, char_px: float) -> int:
    """让页面上 char_px 高的字，缩进检测器后大约高 48 像素。"""
    if page_long < 1 or char_px <= 0:
        raise ValueError("页面边长和字高需为正数")
    return clamp_long_side(round(TARGET_CHAR_PX * page_long / char_px))


def expected_char_px(page_long: int, max_side_len: int) -> int:
    """当前长边限制下，页面上大约多高的字会落在 48 像素。不足长边的页面不再放大。"""
    side = max(int(page_long), 1)
    cap = max(int(max_side_len), 1)
    if side <= cap:
        return TARGET_CHAR_PX
    return int(round(TARGET_CHAR_PX * side / cap))


def attempt_sides(primary: int) -> list[int]:
    """先用指定长边。整页检不出时，再试邻近的两档。"""
    sides = [clamp_long_side(primary)]
    for extra in _NEARBY_LONG_SIDES:
        if extra not in sides:
            sides.append(extra)
    return sides


def _covers_page(line: Line, width: int, height: int) -> bool:
    xs = [point[0] for point in line.quad]
    ys = [point[1] for point in line.quad]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    return box_width > 0.45 * width and box_height > 0.55 * height


def detection_failed(lines: list[Line], width: int, height: int) -> bool:
    """没有框，或少数几个框铺满整页。细高的一列不算失败。"""
    if width <= 0 or height <= 0 or not lines:
        return True
    if len(lines) > 6:
        return False
    covered = sum(1 for line in lines if _covers_page(line, width, height))
    return covered >= len(lines)
