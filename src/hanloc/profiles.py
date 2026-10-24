"""三种版式只差在检测参数上。单字框都按行内墨迹切开。"""

from hanloc.types import Profile

BOOK = Profile(
    name="book",
    reading="horizontal_ltr",
    rotate90=False,
    use_cls=True,
    max_side_len=2000,
    unclip_ratio=1.6,
    split="ink",
)

WOODBLOCK = Profile(
    name="woodblock",
    reading="vertical_rtl",
    rotate90=True,
    use_cls=False,
    max_side_len=4000,
    unclip_ratio=1.3,
    split="ink",
    mark_commentary=True,
    commentary_ratio=0.7,
)

MANUSCRIPT = Profile(
    name="manuscript",
    reading="vertical_rtl",
    rotate90=True,
    use_cls=False,
    max_side_len=4000,
    unclip_ratio=1.3,
    split="ink",
)

PROFILES = {
    BOOK.name: BOOK,
    WOODBLOCK.name: WOODBLOCK,
    MANUSCRIPT.name: MANUSCRIPT,
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = "、".join(PROFILES)
        raise ValueError(f"未知版式：{name}。可选：{known}") from None
