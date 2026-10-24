import numpy as np

from hanloc.detect import parse_result
from hanloc.geometry import map_point_from_cw90
from hanloc.split import (
    WordBox,
    _Ids,
    assign_recognition,
    glyphs_from_words,
    ink_split,
    segment_line,
)
from hanloc.types import Glyph, Line


def _line():
    quad = ((10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0))
    return Line("line-1", quad, 0.0, text="文", score=0.5)


def test_parse_result_keeps_line_and_character_recognition():
    class Result:
        boxes = np.array([[[0, 0], [20, 0], [20, 10], [0, 10]]], dtype=np.float32)
        txts = ("文",)
        scores = (0.91,)
        word_results = ([("文", 0.8, [[0, 0], [20, 0], [20, 10], [0, 10]])],)

    lines = parse_result(Result(), original_height=100, rotated=False)
    line, words = lines[0]
    assert line.text == "文"
    assert line.score == 0.91
    assert words[0].char == "文"
    assert words[0].score == 0.8


def test_parse_result_maps_rotated_boxes_back():
    height = 50

    def rot(x, y):
        return (height - 1 - y, x)

    box = [rot(2, 4), rot(12, 4), rot(12, 9), rot(2, 9)]

    class Result:
        boxes = np.array([box], dtype=np.float32)
        txts = ("甲",)
        scores = (0.5,)
        word_results = ([("甲", 0.4, box)],)

    line, words = parse_result(Result(), original_height=height, rotated=True)[0]
    assert line.quad[0] == (2.0, 4.0)
    assert words[0].quad[2] == (12.0, 9.0)
    assert map_point_from_cw90(box[0][0], box[0][1], height) == (2, 4)


def test_parse_result_ignores_word_boxes_when_line_count_differs():
    class Result:
        boxes = np.array(
            [
                [[0, 0], [10, 0], [10, 4], [0, 4]],
                [[0, 8], [10, 8], [10, 12], [0, 12]],
            ],
            dtype=np.float32,
        )
        txts = ("甲", "乙")
        scores = (0.9, 0.8)
        word_results = ([("甲", 0.9, [[0, 0], [10, 0], [10, 4], [0, 4]])],)

    parsed = parse_result(Result(), original_height=20, rotated=False)
    assert [line.text for line, _words in parsed] == ["甲", "乙"]
    assert all(words == [] for _line, words in parsed)


def test_grid_fallback_assigns_each_character():
    quad = ((0.0, 0.0), (30.0, 0.0), (30.0, 10.0), (0.0, 10.0))
    line = Line("line-1", quad, 0.0, text="上下", score=0.7)
    glyphs = glyphs_from_words(line, [], _Ids())
    assert [glyph.char for glyph in glyphs] == ["上", "下"]
    assert glyphs[0].score == 0.7
    assert glyphs[0].quad[1][0] < glyphs[1].quad[0][0] + 1


def test_mismatched_ink_boxes_keep_character_empty():
    glyphs = [
        Glyph("g1", "line-1", ((0, 0), (1, 0), (1, 1), (0, 1)), 0.0),
        Glyph("g2", "line-1", ((0, 2), (1, 2), (1, 3), (0, 3)), 0.0),
    ]
    words = [WordBox("只", 0.9, glyphs[0].quad)]
    assigned = assign_recognition(glyphs, words)
    assert assigned[0].char is None
    assert assigned[1].char is None


def test_matching_count_copies_recognition_onto_ink_boxes():
    glyphs = [Glyph("g1", "line-1", ((0, 0), (1, 0), (1, 1), (0, 1)), 0.0)]
    words = [WordBox("墨", 0.66, ((9, 9), (10, 9), (10, 10), (9, 10)))]
    assigned = assign_recognition(glyphs, words)
    assert assigned[0].char == "墨"
    assert assigned[0].score == 0.66
    assert assigned[0].quad == glyphs[0].quad


def test_ink_split_merges_a_close_fragment_and_keeps_a_separated_mark():
    image = np.full((240, 80, 3), 255, np.uint8)
    image[20:60, 20:55] = 0
    image[80:92, 20:55] = 0
    image[94:106, 20:55] = 0
    image[126:166, 20:55] = 0
    image[186:196, 20:40] = 0
    quad = ((15.0, 10.0), (60.0, 10.0), (60.0, 220.0), (15.0, 220.0))
    line = Line("line-1", quad, 0.0, text="一二三", score=0.4)
    glyphs = ink_split(image, line, _Ids(), merge_fragments=True)
    assert len(glyphs) == 4
    assert [glyph.uncertain for glyph in glyphs] == [False, False, False, True]


def test_ink_split_separates_three_blobs_and_marks_the_wide_one():
    image = np.full((200, 80, 3), 255, np.uint8)
    image[20:50, 20:50] = 0
    image[70:100, 20:50] = 0
    image[120:190, 20:50] = 0
    quad = ((15.0, 10.0), (55.0, 10.0), (55.0, 195.0), (15.0, 195.0))
    line = Line("line-1", quad, 0.0, text="一二三", score=0.4)
    glyphs = ink_split(image, line, _Ids())
    assert len(glyphs) == 3
    assert glyphs[0].uncertain is False
    assert glyphs[1].uncertain is False
    assert glyphs[2].uncertain is True
    tops = [min(point[1] for point in glyph.quad) for glyph in glyphs]
    assert tops == sorted(tops)


def _center(glyph, axis: int) -> float:
    return sum(point[axis] for point in glyph.quad) / 4


def test_segment_line_snaps_shifted_priors_onto_the_ink():
    image = np.full((20, 100, 3), 255, np.uint8)
    image[4:16, 8:22] = 0
    image[4:16, 28:42] = 0
    quad = ((0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0))
    line = Line("line-1", quad, 0.0, text="上下", score=0.9)
    words = [
        WordBox("上", 0.8, ((0.0, 0.0), (50.0, 0.0), (50.0, 20.0), (0.0, 20.0))),
        WordBox("下", 0.4, ((50.0, 0.0), (100.0, 0.0), (100.0, 20.0), (50.0, 20.0))),
    ]
    glyphs = segment_line(image, line, words, _Ids())
    assert [glyph.char for glyph in glyphs] == ["上", "下"]
    assert [glyph.score for glyph in glyphs] == [0.8, 0.4]
    assert _center(glyphs[0], 0) < 30
    assert 25 < _center(glyphs[1], 0) < 50
    assert glyphs[0].cell == words[0].quad
    assert glyphs[1].cell == words[1].quad


def test_segment_line_tightens_a_single_character_to_its_ink():
    image = np.full((40, 80, 3), 255, np.uint8)
    image[12:28, 24:48] = 0
    quad = ((0.0, 0.0), (80.0, 0.0), (80.0, 40.0), (0.0, 40.0))
    line = Line("line-1", quad, 0.0, text="中", score=0.7)
    words = [WordBox("中", 0.7, quad)]
    glyph = segment_line(image, line, words, _Ids())[0]
    xs = [point[0] for point in glyph.quad]
    ys = [point[1] for point in glyph.quad]
    assert glyph.char == "中"
    assert min(xs) > 10
    assert max(xs) < 60
    assert min(ys) > 5
    assert max(ys) < 35


def test_extra_ink_piece_is_merged_into_the_character_it_belongs_to():
    image = np.full((24, 90, 3), 255, np.uint8)
    image[4:20, 6:26] = 0
    image[4:20, 34:44] = 0
    image[4:20, 46:56] = 0
    quad = ((0.0, 0.0), (90.0, 0.0), (90.0, 24.0), (0.0, 24.0))
    line = Line("line-1", quad, 0.0, text="甲乙", score=0.9)
    words = [
        WordBox("甲", 0.9, ((0.0, 0.0), (40.0, 0.0), (40.0, 24.0), (0.0, 24.0))),
        WordBox("乙", 0.8, ((40.0, 0.0), (90.0, 0.0), (90.0, 24.0), (40.0, 24.0))),
    ]
    glyphs = segment_line(image, line, words, _Ids())
    assert [glyph.char for glyph in glyphs] == ["甲", "乙"]
    assert glyphs[0].uncertain is False
    assert glyphs[1].uncertain is True
    assert _center(glyphs[0], 0) < 30
    assert _center(glyphs[1], 0) > 30


def test_wide_ink_splits_between_a_character_and_punctuation():
    image = np.full((30, 80, 3), 255, np.uint8)
    image[6:24, 8:48] = 0
    quad = ((0.0, 0.0), (80.0, 0.0), (80.0, 30.0), (0.0, 30.0))
    line = Line("line-1", quad, 0.0, text="甲，", score=0.6)
    words = [
        WordBox("甲", 0.9, ((0.0, 0.0), (50.0, 0.0), (50.0, 30.0), (0.0, 30.0))),
        WordBox("，", 0.4, ((50.0, 0.0), (80.0, 0.0), (80.0, 30.0), (50.0, 30.0))),
    ]
    glyphs = segment_line(image, line, words, _Ids())
    assert [glyph.char for glyph in glyphs] == ["甲", "，"]
    assert all(glyph.uncertain for glyph in glyphs)
    han = max(point[0] for point in glyphs[0].quad) - min(point[0] for point in glyphs[0].quad)
    punct = max(point[0] for point in glyphs[1].quad) - min(point[0] for point in glyphs[1].quad)
    assert punct < han


def test_punctuation_keeps_a_narrow_box_when_the_count_matches():
    image = np.full((24, 80, 3), 255, np.uint8)
    image[4:20, 4:24] = 0
    image[4:20, 28:34] = 0
    image[4:20, 40:60] = 0
    quad = ((0.0, 0.0), (80.0, 0.0), (80.0, 24.0), (0.0, 24.0))
    line = Line("line-1", quad, 0.0, text="上，下", score=0.8)
    glyphs = segment_line(image, line, [], _Ids())
    assert [glyph.char for glyph in glyphs] == ["上", "，", "下"]
    assert [glyph.uncertain for glyph in glyphs] == [False, False, False]


def test_segment_line_keeps_text_when_the_line_has_no_ink():
    image = np.full((20, 40, 3), 255, np.uint8)
    quad = ((0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0))
    line = Line("line-1", quad, 0.0, text="甲乙", score=0.3)
    glyphs = segment_line(image, line, [], _Ids())
    assert [glyph.char for glyph in glyphs] == ["甲", "乙"]
    assert all(glyph.uncertain for glyph in glyphs)
