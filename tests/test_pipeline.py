import numpy as np

from hanloc.pipeline import locate
from hanloc.types import Line


def test_a_usable_line_is_not_retried(monkeypatch):
    calls: list[int] = []

    def fake(image: np.ndarray, profile):
        calls.append(profile.max_side_len)
        quad = ((5.0, 5.0), (40.0, 5.0), (40.0, 24.0), (5.0, 24.0))
        return [(Line("line-1", quad, 0.0, text="甲", score=0.9), [])]

    monkeypatch.setattr("hanloc.pipeline.detect_lines", fake)
    image = np.full((80, 120, 3), 255, np.uint8)
    page = locate(image, "book")
    assert calls == [2000]
    assert len(page.lines) == 1


def test_a_page_filling_box_retries_a_nearby_side(monkeypatch):
    calls: list[int] = []

    def fake(image: np.ndarray, profile):
        calls.append(profile.max_side_len)
        if profile.max_side_len == 1600:
            quad = ((10.0, 10.0), (40.0, 10.0), (40.0, 28.0), (10.0, 28.0))
        else:
            quad = ((2.0, 2.0), (118.0, 2.0), (118.0, 78.0), (2.0, 78.0))
        return [(Line("line-1", quad, 0.0), [])]

    monkeypatch.setattr("hanloc.pipeline.detect_lines", fake)
    image = np.full((80, 120, 3), 255, np.uint8)
    page = locate(image, "book")
    assert calls == [2000, 1600]
    assert page.lines[0].quad[1][0] == 40.0


def test_empty_detection_falls_back_to_ink_columns(monkeypatch):
    calls: list[int] = []

    def fake(image: np.ndarray, profile):
        calls.append(profile.max_side_len)
        return []

    monkeypatch.setattr("hanloc.pipeline.detect_lines", fake)
    image = np.full((80, 140, 3), 255, np.uint8)
    image[12:48, 16:40] = 0
    image[12:48, 96:124] = 0
    page = locate(image, "book")
    assert calls == [2000, 1600, 2400]
    assert len(page.glyphs) >= 2
