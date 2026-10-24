import cv2
import numpy as np

from hanloc.crop import crop_glyph, crop_quad
from hanloc.types import Glyph


def _rotated_card(color):
    image = np.full((160, 180, 3), 255, np.uint8)
    quad = ((30, 40), (120, 28), (132, 90), (42, 104))
    cv2.fillConvexPoly(image, np.array(quad, np.int32), color)
    return image, quad


def test_crop_makes_rotated_rectangle_fill_the_frame():
    image, quad = _rotated_card((0, 0, 210))
    cropped = crop_quad(image, quad, margin=0)
    assert cropped.shape[0] > 20 and cropped.shape[1] > 20
    red = cropped[:, :, 2]
    mask = red > 150
    ys, xs = np.where(mask)
    assert ys.min() <= 3
    assert xs.min() <= 3
    assert ys.max() >= cropped.shape[0] - 4
    assert xs.max() >= cropped.shape[1] - 4
    assert red[cropped.shape[0] // 2, cropped.shape[1] // 2] > 150


def test_crop_margin_leaves_a_blank_border():
    image, quad = _rotated_card((0, 0, 210))
    cropped = crop_quad(image, quad, margin=0.2)
    assert cropped[0, 0, 0] > 240
    assert cropped[0, 0, 2] > 240
    center = cropped[cropped.shape[0] // 2, cropped.shape[1] // 2]
    assert int(center[2]) > 150
    assert int(center[0]) < 40


def test_crop_glyph_accepts_the_same_image_array():
    image, quad = _rotated_card((0, 180, 0))
    glyph = Glyph(id="glyph-1", line_id="line-1", quad=quad, angle_deg=0)
    patch = crop_glyph(image, glyph, margin=0)
    assert patch.shape[0] > 10 and patch.shape[1] > 10
