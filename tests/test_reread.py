import numpy as np
import pytest

from hanloc.reread import reread_glyphs
from hanloc.types import Glyph, HanlocError


def _glyph() -> Glyph:
    quad = ((2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0))
    return Glyph("a", "l1", quad, 0.0, char="甲", score=0.4)


def test_reread_writes_candidates_and_keeps_char():
    image = np.full((30, 30, 3), 255, np.uint8)
    glyph = _glyph()

    def reader(crops):
        assert len(crops) == 1
        assert crops[0].ndim == 3
        assert crops[0].shape[0] > 0
        return [(("书", 0.7), ("画", 0.1))]

    reread_glyphs(image, [glyph], reader=reader)
    assert glyph.char == "甲"
    assert glyph.candidates == (("书", 0.7), ("画", 0.1))


def test_reread_of_no_glyphs_does_nothing():
    image = np.full((8, 8, 3), 255, np.uint8)

    def reader(crops):
        raise AssertionError("空页不应识别")

    reread_glyphs(image, [], reader=reader)


def test_missing_torch_explains_the_extra(monkeypatch):
    def missing():
        raise ImportError("no torch")

    monkeypatch.setattr("hanloc.reread._import_model", missing)
    monkeypatch.setattr("hanloc.reread._ENGINE", None)
    with pytest.raises(HanlocError, match="calligraphy"):
        reread_glyphs(np.full((30, 30, 3), 255, np.uint8), [_glyph()])
