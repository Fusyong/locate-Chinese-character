from hanloc.scale import (
    attempt_sides,
    detection_failed,
    expected_char_px,
    long_side_for_char,
)
from hanloc.types import Line


def _line(width: float, height: float) -> Line:
    quad = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    return Line("line", quad, 0.0)


def test_a_measured_character_sets_the_long_side():
    assert long_side_for_char(4000, 96) == 2000
    assert long_side_for_char(4000, 200) == 960
    assert long_side_for_char(4000, 40) == 2400
    assert expected_char_px(4000, 2000) == 96
    assert expected_char_px(1500, 2000) == 48


def test_nearby_sides_follow_the_first_attempt():
    assert attempt_sides(2000) == [2000, 1600, 2400]
    assert attempt_sides(1600) == [1600, 2400]
    assert attempt_sides(5000) == [2400, 1600]


def test_only_a_page_sized_box_counts_as_failure():
    assert detection_failed([], 100, 100)
    assert detection_failed([_line(90, 90)], 100, 100)
    assert not detection_failed([_line(20, 90)], 100, 100)
    assert not detection_failed([_line(20, 20) for _ in range(7)], 100, 100)
