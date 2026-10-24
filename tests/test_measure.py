import numpy as np

from hanloc.measure import ink_width, typical_ink_width, uniform_middle
from hanloc.types import Glyph


def test_uniform_middle_drops_the_tiny_and_the_huge():
    average, count = uniform_middle([8, 40, 41, 39, 42, 40, 43, 38, 200, 12])
    assert count >= 3
    assert 38 <= average <= 43


def test_ink_width_ignores_the_white_margin():
    image = np.full((40, 70, 3), 255, np.uint8)
    image[8:32, 20:44] = 0
    quad = ((4.0, 2.0), (66.0, 2.0), (66.0, 38.0), (4.0, 38.0))
    assert ink_width(image, quad) == 24


def test_typical_width_uses_the_even_characters():
    image = np.full((80, 220, 3), 240, np.uint8)
    glyphs = []
    sizes = ((10, 8), (30, 22), (60, 22), (90, 21), (120, 23), (160, 50))
    for index, (x, size) in enumerate(sizes):
        image[20 : 20 + size, x : x + size] = 10
        quad = (
            (float(x), 20.0),
            (float(x + size), 20.0),
            (float(x + size), float(20 + size)),
            (float(x), float(20 + size)),
        )
        glyphs.append(Glyph(f"g{index}", "line-1", quad, 0.0))
    measured, count = typical_ink_width(image, glyphs)
    assert count == 4
    assert 20 <= measured <= 24
