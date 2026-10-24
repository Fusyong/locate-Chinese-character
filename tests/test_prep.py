import numpy as np

from hanloc.prep import light_ink, prepare_page, suppress_red
from hanloc.profiles import MODELBOOK, STELE, WOODBLOCK, get_profile
from hanloc.types import Profile


def test_light_ink_detects_a_rubbing_and_ignores_dark_strokes():
    rubbing = np.full((80, 60, 3), 15, np.uint8)
    rubbing[20:28, 15:45] = 230
    rubbing[40:48, 15:45] = 230
    assert light_ink(rubbing)
    page = np.full((80, 60, 3), 230, np.uint8)
    page[20:28, 15:45] = 15
    assert not light_ink(page)


def test_prepare_page_inverts_only_light_ink():
    rubbing = np.full((40, 40, 3), 10, np.uint8)
    rubbing[8:14, 8:30] = 240
    profile = Profile(
        name="trial",
        reading="vertical_rtl",
        rotate90=True,
        use_cls=False,
        max_side_len=400,
        unclip_ratio=1.2,
        split="ink",
        polarity="auto",
    )
    prepared = prepare_page(rubbing, profile)
    assert int(prepared[0, 0, 0]) > 200
    assert int(prepared[10, 16, 0]) < 40
    dark = np.full((40, 40, 3), 240, np.uint8)
    dark[8:14, 8:30] = 10
    assert np.array_equal(prepare_page(dark, profile), dark)


def test_suppress_red_replaces_seals_and_keeps_ink():
    image = np.full((20, 20, 3), (110, 160, 190), np.uint8)
    image[2:6, 2:6] = (20, 20, 20)
    image[10:16, 10:16] = (40, 50, 210)
    cleaned = suppress_red(image)
    assert tuple(int(v) for v in cleaned[3, 3]) == (20, 20, 20)
    seal = cleaned[12, 12].astype(int)
    assert seal[2] < seal[1] + 45
    assert get_profile("stele") is STELE
    assert STELE.max_side_len == 2000
    assert STELE.locate_mode == "lines"
    assert STELE.polarity == "auto"
    assert WOODBLOCK.max_side_len == 2000
    assert get_profile("modelbook").suppress_red is True
    assert MODELBOOK.max_side_len == 2000
    assert MODELBOOK.locate_mode == "lines"