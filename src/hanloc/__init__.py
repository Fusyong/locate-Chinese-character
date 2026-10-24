"""hanloc：定位页面上的汉字。"""

from hanloc.crop import crop_glyph
from hanloc.pipeline import locate

__all__ = ["crop_glyph", "locate"]
__version__ = "0.1.0"
