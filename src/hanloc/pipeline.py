"""从图片到页面结果。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hanloc.crop import load_image
from hanloc.detect import detect_lines
from hanloc.flow import build_flows, centerfold_line_ids
from hanloc.profiles import get_profile
from hanloc.split import _Ids, segment_line
from hanloc.types import Glyph, Page, Profile


def locate(image: str | Path | np.ndarray, profile: str | Profile = "book") -> Page:
    """按版式定位一页。识别文字和置信度随框保存。"""
    chosen = profile if isinstance(profile, Profile) else get_profile(profile)
    loaded = load_image(image)
    height, width = loaded.shape[:2]
    detected = detect_lines(loaded, chosen)
    lines = [line for line, _words in detected]
    fold_ids = centerfold_line_ids(lines, width, chosen.centerfold)
    ids = _Ids()
    glyphs: list[Glyph] = []
    for line, words in detected:
        # 回推字框只用来对齐字数和顺序，外框改按这一行的墨迹来切。
        glyphs.extend(segment_line(loaded, line, words, ids))

    body_lines = [line for line in lines if line.id not in fold_ids]
    body_glyphs = [glyph for glyph in glyphs if glyph.line_id not in fold_ids]
    flows = build_flows(
        body_lines,
        body_glyphs,
        chosen.reading,
        commentary_ratio=chosen.commentary_ratio if chosen.mark_commentary else None,
    )
    fold_flows = build_flows(
        [line for line in lines if line.id in fold_ids],
        [glyph for glyph in glyphs if glyph.line_id in fold_ids],
        chosen.reading,
        role="centerfold",
        id_start=len(flows) + 1,
    )
    return Page(
        profile=chosen.name,
        width=width,
        height=height,
        lines=lines,
        glyphs=glyphs,
        flows=flows + fold_flows,
    )
