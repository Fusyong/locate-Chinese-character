"""从图片到页面结果。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from hanloc.columns import segment_columns
from hanloc.crop import load_image
from hanloc.detect import detect_lines
from hanloc.flow import build_flows, centerfold_line_ids, column_size_ids
from hanloc.prep import prepare_page
from hanloc.profiles import get_profile
from hanloc.scale import attempt_sides, detection_failed
from hanloc.split import WordBox, _Ids, assign_recognition, ink_split, segment_line
from hanloc.types import Flow, Glyph, Line, Page, Profile, Role


def _glyphs_of(
    image: np.ndarray,
    line: Line,
    words: list[WordBox],
    ids: _Ids,
    profile: Profile,
) -> list[Glyph]:
    if profile.align_chars:
        # 回推字框只用来对齐字数和顺序，外框改按这一行的墨迹来切。
        return segment_line(image, line, words, ids)
    pieces = ink_split(image, line, ids, merge_fragments=True)
    return assign_recognition(pieces, words)


def _flows_for(
    lines: list[Line],
    glyphs: list[Glyph],
    profile: Profile,
    role: Role,
    id_start: int,
    commentary: bool,
) -> list[Flow]:
    return build_flows(
        lines,
        glyphs,
        profile.reading,
        role=role,
        commentary_ratio=profile.commentary_ratio if commentary else None,
        id_start=id_start,
    )


def _from_lines(
    image: np.ndarray,
    detected: list[tuple[Line, list[WordBox]]],
    profile: Profile,
) -> tuple[list[Line], list[Glyph]]:
    lines = [line for line, _words in detected]
    ids = _Ids()
    glyphs: list[Glyph] = []
    for line, words in detected:
        glyphs.extend(_glyphs_of(image, line, words, ids, profile))
    return lines, glyphs


def _locate_lines(
    image: np.ndarray, profile: Profile, width: int, height: int
) -> tuple[list[Line], list[Glyph]]:
    """先按版式的长边检测。整页没有可用的框时，改试邻近长边，仍不行再按墨迹分列。"""
    best_lines: list[Line] = []
    best_glyphs: list[Glyph] = []
    for side in attempt_sides(profile.max_side_len):
        detected = detect_lines(image, replace(profile, max_side_len=side))
        lines, glyphs = _from_lines(image, detected, profile)
        if not detection_failed(lines, width, height):
            return lines, glyphs
        if len(glyphs) > len(best_glyphs):
            best_lines, best_glyphs = lines, glyphs
    column_lines, column_glyphs = segment_columns(image)
    if len(column_glyphs) > len(best_glyphs):
        return column_lines, column_glyphs
    return best_lines, best_glyphs


def locate(image: str | Path | np.ndarray, profile: str | Profile = "book") -> Page:
    """按版式定位一页。识别文字和置信度随框保存。"""
    chosen = profile if isinstance(profile, Profile) else get_profile(profile)
    loaded = load_image(image)
    height, width = loaded.shape[:2]
    prepared = prepare_page(loaded, chosen)
    if chosen.locate_mode == "columns":
        lines, glyphs = segment_columns(prepared)
    else:
        lines, glyphs = _locate_lines(prepared, chosen, width, height)
    fold_ids = centerfold_line_ids(lines, width, chosen.centerfold)
    narrow_ids, wide_ids = column_size_ids(
        lines,
        narrow_ratio=chosen.marginal_ratio if chosen.mark_marginalia else 0.0,
        wide_ratio=chosen.title_ratio if chosen.mark_title else 0.0,
    )
    narrow_ids -= fold_ids
    wide_ids -= fold_ids | narrow_ids

    def picked(id_set: set[str]) -> tuple[list[Line], list[Glyph]]:
        return (
            [line for line in lines if line.id in id_set],
            [glyph for glyph in glyphs if glyph.line_id in id_set],
        )

    aside = fold_ids | narrow_ids | wide_ids
    flows: list[Flow] = []
    groups: tuple[tuple[set[str], Role, bool], ...] = (
        (wide_ids, "title", False),
        ({line.id for line in lines if line.id not in aside}, "body", chosen.mark_commentary),
        (narrow_ids, "marginalia", False),
        (fold_ids, "centerfold", False),
    )
    for id_set, role, commentary in groups:
        part_lines, part_glyphs = picked(id_set)
        flows.extend(
            _flows_for(part_lines, part_glyphs, chosen, role, len(flows) + 1, commentary)
        )
    return Page(
        profile=chosen.name,
        width=width,
        height=height,
        lines=lines,
        glyphs=glyphs,
        flows=flows,
    )
