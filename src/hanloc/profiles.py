"""版式只差在检测和切字参数上。"""

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

# 列距稳定的碑、墓志。浅字拓片会反相，再按墨迹分列、列内切字。
STELE = Profile(
    name="stele",
    reading="vertical_rtl",
    rotate90=False,
    use_cls=False,
    max_side_len=2200,
    unclip_ratio=1.2,
    split="ink",
    polarity="auto",
    align_chars=False,
    mark_marginalia=True,
    mark_title=True,
    locate_mode="columns",
)

# 法帖、信札、手卷。先去掉朱印，再按墨迹分列。连笔保持整块并标不确定。
MODELBOOK = Profile(
    name="modelbook",
    reading="vertical_rtl",
    rotate90=False,
    use_cls=False,
    max_side_len=2200,
    unclip_ratio=1.3,
    split="ink",
    polarity="auto",
    suppress_red=True,
    align_chars=False,
    mark_marginalia=True,
    mark_title=True,
    locate_mode="columns",
)

PROFILES = {
    BOOK.name: BOOK,
    WOODBLOCK.name: WOODBLOCK,
    MANUSCRIPT.name: MANUSCRIPT,
    STELE.name: STELE,
    MODELBOOK.name: MODELBOOK,
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = "、".join(PROFILES)
        raise ValueError(f"未知版式：{name}。可选：{known}") from None
