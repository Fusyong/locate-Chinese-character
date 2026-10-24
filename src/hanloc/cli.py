"""命令行：写出 JSON、叠加图和字图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hanloc.crop import crop_glyph, load_image
from hanloc.pipeline import locate
from hanloc.profiles import PROFILES
from hanloc.types import Centerfold, Page, Profile, page_to_dict


def _draw_overlay(image: np.ndarray, page: Page) -> np.ndarray:
    canvas = image.copy()
    for line in page.lines:
        points = np.array(line.quad, dtype=np.int32)
        cv2.polylines(canvas, [points], True, (160, 90, 20), 2)
    order = {glyph_id: str(index) for index, glyph_id in enumerate(_reading_ids(page), start=1)}
    for glyph in page.glyphs:
        color = (20, 90, 180) if glyph.uncertain else (40, 130, 60)
        points = np.array(glyph.quad, dtype=np.int32)
        cv2.polylines(canvas, [points], True, color, 1)
        label = order.get(glyph.id)
        if label:
            x, y = int(glyph.quad[0][0]), int(glyph.quad[0][1])
            cv2.putText(
                canvas,
                label,
                (x, max(y - 2, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def _reading_ids(page: Page) -> list[str]:
    ids: list[str] = []
    for flow in page.flows:
        ids.extend(flow.glyph_ids)
    return ids


def write_page(image: np.ndarray, page: Page, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "page.json").write_text(
        json.dumps(page_to_dict(page), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlay = _draw_overlay(image, page)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("无法写出叠加图")
    encoded.tofile(str(out_dir / "overlay.png"))
    crops = out_dir / "crops"
    crops.mkdir(exist_ok=True)
    for glyph in page.glyphs:
        patch = crop_glyph(image, glyph)
        ok, encoded = cv2.imencode(".png", patch)
        if not ok:
            continue
        encoded.tofile(str(crops / f"{glyph.id}.png"))


def _locate_command(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    chosen: Profile | str = args.profile
    if args.centerfold:
        from dataclasses import replace

        from hanloc.profiles import get_profile

        base = get_profile(args.profile)
        fold: Centerfold = args.centerfold
        chosen = replace(base, centerfold=fold)
    page = locate(image, chosen)
    write_page(image, page, Path(args.out))
    print(
        f"写出 {len(page.lines)} 行、{len(page.glyphs)} 字、{len(page.flows)} 条语流到 {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hanloc", description="定位页面上的汉字并裁字")
    sub = parser.add_subparsers(dest="command", required=True)
    locate_parser = sub.add_parser("locate", help="定位一页，写出 JSON、叠加图和字图")
    locate_parser.add_argument("image", help="页面图片路径")
    locate_parser.add_argument(
        "--profile",
        default="book",
        choices=sorted(PROFILES),
        help="版式：book、woodblock、manuscript",
    )
    locate_parser.add_argument("--out", required=True, help="输出目录")
    locate_parser.add_argument(
        "--centerfold",
        choices=["none", "gutter", "left", "right"],
        default=None,
        help="版心位置。默认沿用版式配置",
    )
    locate_parser.set_defaults(func=_locate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
