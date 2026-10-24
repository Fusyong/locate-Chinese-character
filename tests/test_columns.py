import numpy as np

from hanloc.columns import segment_columns
from hanloc.flow import build_flows


def _page(*rects: tuple[int, int, int, int], width: int = 200, height: int = 200) -> np.ndarray:
    image = np.full((height, width, 3), 255, np.uint8)
    for x, y, w, h in rects:
        image[y : y + h, x : x + w] = 0
    return image


def test_two_columns_are_split_and_read_from_the_right():
    image = _page(
        (140, 20, 30, 30),
        (140, 70, 30, 30),
        (30, 20, 30, 30),
        (30, 70, 30, 30),
    )
    lines, glyphs = segment_columns(image)
    assert len(lines) == 2
    assert len(glyphs) == 4
    assert all(not glyph.uncertain for glyph in glyphs)
    flows = build_flows(lines, glyphs, "vertical_rtl")
    assert len(flows) == 2

    def centers(flow):
        chosen = [item for item in glyphs if item.id in flow.glyph_ids]
        xs = [sum(point[0] for point in glyph.quad) / 4 for glyph in chosen]
        tops = [min(point[1] for point in glyph.quad) for glyph in chosen]
        return xs, tops

    right_x, right_top = centers(flows[0])
    left_x, left_top = centers(flows[1])
    assert all(value > 100 for value in right_x)
    assert all(value < 80 for value in left_x)
    assert right_top[0] < right_top[1]
    assert left_top[0] < left_top[1]


def test_side_by_side_parts_of_one_character_merge():
    image = _page((20, 20, 14, 40), (36, 20, 14, 40), (20, 90, 30, 40))
    _lines, glyphs = segment_columns(image)
    assert len(glyphs) == 2
    heights = [
        max(point[1] for point in glyph.quad) - min(point[1] for point in glyph.quad)
        for glyph in glyphs
    ]
    assert all(30 <= height <= 50 for height in heights)


def test_projection_splits_a_noisy_column_into_characters():
    image = np.full((220, 80, 3), 255, np.uint8)
    image[15:50, 20:55] = 0
    image[80:115, 20:55] = 0
    image[145:180, 20:55] = 0
    # 石花把三字连成一条，连通域不再能切开。
    image[50:80, 36:40] = 0
    image[115:145, 36:40] = 0
    _lines, glyphs = segment_columns(image)
    assert len(glyphs) >= 3
    assert sum(not glyph.uncertain for glyph in glyphs) >= 3


def test_a_column_stuck_together_is_uncertain():
    image = _page((20, 15, 30, 40), (80, 15, 30, 150))
    _lines, glyphs = segment_columns(image)
    flags = {glyph.uncertain for glyph in glyphs}
    assert True in flags
    assert False in flags
