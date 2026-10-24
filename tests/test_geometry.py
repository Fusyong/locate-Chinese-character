import cv2
import numpy as np

from hanloc.geometry import (
    angle_deg,
    centers_separated,
    map_point_from_cw90,
    map_quad_from_cw90,
    rotate_point_cw90,
)


def test_angle_of_horizontal_edge_is_zero():
    quad = ((0, 0), (10, 0), (10, 4), (0, 4))
    assert angle_deg(quad) == 0


def test_centers_separated_uses_relative_height():
    near = ((0, 0), (10, 0), (10, 10), (0, 10))
    beside = ((12, 0), (22, 0), (22, 10), (12, 10))
    far = ((40, 0), (50, 0), (50, 10), (40, 10))
    assert centers_separated(near, beside, ratio=1.5) is False
    assert centers_separated(near, far, ratio=1.5) is True


def test_clockwise_rotation_roundtrip_and_matches_opencv():
    height, width = 30, 40
    x, y = 7, 11
    rotated = rotate_point_cw90(x, y, height)
    back = map_point_from_cw90(rotated[0], rotated[1], height)
    assert back == (x, y)

    image = np.zeros((height, width), np.uint8)
    image[y, x] = 255
    turned = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    ys, xs = np.where(turned == 255)
    assert (xs[0], ys[0]) == rotated

    quad = ((1.0, 2.0), (8.0, 2.0), (8.0, 6.0), (1.0, 6.0))
    mapped = map_quad_from_cw90(
        tuple(rotate_point_cw90(px, py, height) for px, py in quad),
        height,
    )
    for original, restored in zip(quad, mapped, strict=True):
        assert restored == original
