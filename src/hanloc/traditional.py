"""简体转繁体。"""

from __future__ import annotations

from hanloc.types import HanlocError

_CONVERTER = None


def _load():
    from opencc import OpenCC

    return OpenCC("s2t")


def _converter():
    global _CONVERTER
    if _CONVERTER is None:
        try:
            _CONVERTER = _load()
        except ImportError as exc:
            raise HanlocError('简转繁需要 OpenCC。请执行：pip install -e ".[view]"') from exc
    return _CONVERTER


def to_traditional(text: str) -> str:
    """用 OpenCC 的 s2t 把简体转为繁体。"""
    return _converter().convert(text)
