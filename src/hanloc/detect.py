"""RapidOCR 适配。几何模块不导入这里。"""

from __future__ import annotations

import cv2
import numpy as np

from hanloc.geometry import Quad, angle_deg, as_quad, map_quad_from_cw90
from hanloc.split import WordBox
from hanloc.types import HanlocError, Line, Profile

_ENGINES: dict[tuple[bool, int, float], object] = {}


def _score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        numbers = [float(item) for item in np.ravel(np.asarray(value))]
        if not numbers:
            return None
        return sum(numbers) / len(numbers)
    return float(value)  # type: ignore[arg-type]


def _quad_of(box: object, original_height: int, rotated: bool) -> Quad:
    quad = as_quad(box)  # type: ignore[arg-type]
    if rotated:
        return map_quad_from_cw90(quad, original_height)
    return quad


def parse_result(
    result: object, original_height: int, rotated: bool
) -> list[tuple[Line, list[WordBox]]]:
    """把 RapidOCR 的行、行文本、行置信度和单字结果收成行与字框。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    array = np.asarray(boxes)
    if array.size == 0:
        return []
    texts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [])
    word_results = list(getattr(result, "word_results", None) or [])
    # 某一行没有字框时，RapidOCR 可能少返回一行，此时整页改走等分，避免字和行对错。
    words_aligned = len(word_results) == len(array)
    parsed: list[tuple[Line, list[WordBox]]] = []
    for index, box in enumerate(array):
        quad = _quad_of(box, original_height, rotated)
        text = texts[index] if index < len(texts) else None
        score = _score(scores[index]) if index < len(scores) else None
        raw_words = word_results[index] if words_aligned and word_results[index] else []
        words: list[WordBox] = []
        for item in raw_words:
            char, confidence, word_box = item[0], item[1], item[2]
            words.append(
                WordBox(
                    char=str(char),
                    score=_score(confidence),
                    quad=_quad_of(word_box, original_height, rotated),
                )
            )
        line = Line(
            id=f"line-{index + 1}",
            quad=quad,
            angle_deg=angle_deg(quad),
            text=None if text is None else str(text),
            score=score,
        )
        parsed.append((line, words))
    return parsed


def _engine(profile: Profile) -> object:
    key = (profile.use_cls, profile.max_side_len, float(profile.unclip_ratio))
    engine = _ENGINES.get(key)
    if engine is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise HanlocError('未安装识别依赖。请执行：pip install -e ".[ocr]"') from exc
        engine = RapidOCR(
            params={
                "Global.use_det": True,
                "Global.use_cls": profile.use_cls,
                "Global.use_rec": True,
                "Global.return_word_box": True,
                "Global.max_side_len": profile.max_side_len,
                "Det.unclip_ratio": profile.unclip_ratio,
            }
        )
        _ENGINES[key] = engine
    return engine


def detect_lines(image: np.ndarray, profile: Profile) -> list[tuple[Line, list[WordBox]]]:
    """整页检测。第二期可在此改为重叠切块后再把坐标映回全图。"""
    original_height = int(image.shape[0])
    view = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if profile.rotate90 else image
    result = _engine(profile)(view)
    if result is None:
        return []
    return parse_result(result, original_height, profile.rotate90)
