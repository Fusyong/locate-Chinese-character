"""送进检测器之前的页面整理。几何不变，坐标仍是原图像素。"""

from __future__ import annotations

import cv2
import numpy as np

from hanloc.types import Profile


def invert_dark_ground(image: np.ndarray) -> np.ndarray:
    """只反相黑色碑面，四周白边保持为纸色。整页反相会把白边变成黑框，把各列连在一起。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = mask == 0
    coords = np.argwhere(dark)
    if coords.size == 0:
        return cv2.bitwise_not(image)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    inset = max(1, int(0.004 * min(image.shape[:2])))
    y0 = min(y0 + inset, y1)
    x0 = min(x0 + inset, x1)
    y1 = max(y1 - inset, y0)
    x1 = max(x1 - inset, x0)
    out = np.full_like(image, 255)
    out[y0 : y1 + 1, x0 : x1 + 1] = cv2.bitwise_not(image[y0 : y1 + 1, x0 : x1 + 1])
    return out


def light_ink(image: np.ndarray) -> bool:
    """中心区域里，浅色笔画占少数、底子又很暗时，这是拓片。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    y0, y1 = int(height * 0.2), int(height * 0.8)
    x0, x1 = int(width * 0.2), int(width * 0.8)
    if y1 <= y0 or x1 <= x0:
        return False
    center = gray[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(center, (5, 5), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = mask > 0
    if float(bright.mean()) >= 0.5:
        return False
    dark = center[~bright]
    light = center[bright]
    if dark.size == 0 or light.size == 0:
        return False
    dark_mean = float(dark.mean())
    return dark_mean < 100 and float(light.mean()) - dark_mean > 40


def suppress_red(image: np.ndarray) -> np.ndarray:
    """把朱印涂成周围纸色。黑墨的红通道不高，不会被涂掉。"""
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    mask = (red > 90) & (red > green + 45) & (red > blue + 45)
    if not np.any(mask):
        return image
    kept = image[~mask]
    paper = np.median(kept, axis=0) if kept.size else np.array([255, 255, 255])
    out = image.copy()
    out[mask] = np.clip(paper, 0, 255).astype(np.uint8)
    return out


def prepare_page(image: np.ndarray, profile: Profile) -> np.ndarray:
    """按版式反相拓片、去掉朱印。返回的图与原图同尺寸。"""
    page = image
    invert = profile.polarity == "light" or (profile.polarity == "auto" and light_ink(page))
    if invert:
        page = invert_dark_ground(page)
    if profile.suppress_red:
        page = suppress_red(page)
    return page
