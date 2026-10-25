"""书法单字二次识别。只写入候选，不改原来的字。"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from hanloc.crop import crop_glyph
from hanloc.types import Glyph, HanlocError

_TOP_K = 5
_BATCH = 16
_INPUT = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MIN_WEIGHT_BYTES = 20_000_000
_MODEL_NAME = "best_model.pth"
_MAP_NAME = "char_map.json"
_SOURCES = (
    "https://huggingface.co/xiajingfeng/chinese-calligraphy-recognition-v1/resolve/main/",
    "https://hf-mirror.com/xiajingfeng/chinese-calligraphy-recognition-v1/resolve/main/",
)

_LOCK = threading.Lock()
_ENGINE: _Shufa | None = None


def _cache_dir() -> Path:
    root = os.environ.get("HANLOC_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "hanloc"
    return base / "shufa"


def _download(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    errors: list[str] = []
    for source in _SOURCES:
        try:
            request = urllib.request.Request(
                source + name,
                headers={"User-Agent": "hanloc"},
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as handle,
            ):
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temporary.replace(dest)
            return
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            errors.append(str(exc))
    detail = errors[-1] if errors else "没有可用的下载地址"
    raise HanlocError(f"无法下载书法识别模型 {name}：{detail}")


def _ready(path: Path, minimum: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum


def _ensure_files() -> tuple[Path, Path]:
    folder = _cache_dir()
    weights = folder / _MODEL_NAME
    mapping = folder / _MAP_NAME
    if not _ready(mapping, 1000):
        _download(_MAP_NAME, mapping)
    if not _ready(mapping, 1000):
        raise HanlocError(f"书法识别的字表不完整：{mapping}")
    if not _ready(weights, _MIN_WEIGHT_BYTES):
        _download(_MODEL_NAME, weights)
    if not _ready(weights, _MIN_WEIGHT_BYTES):
        raise HanlocError(f"书法识别模型不完整：{weights}")
    return weights, mapping


def _load_labels(path: Path) -> dict[int, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HanlocError(f"书法识别的字表无法读取：{path}") from exc
    if not isinstance(raw, dict) or len(raw) < 1000:
        raise HanlocError(f"书法识别的字表不完整：{path}")
    labels: dict[int, str] = {}
    for char, index in raw.items():
        if not isinstance(char, str) or char == "":
            continue
        labels[int(index)] = char
    if len(labels) < 1000:
        raise HanlocError(f"书法识别的字表不完整：{path}")
    return labels


def _import_model():
    import torch
    from torchvision.models import resnet50

    return torch, resnet50


def _as_tensor(torch, image: np.ndarray):
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shrink = min(rgb.shape[0], rgb.shape[1]) > _INPUT
    resized = cv2.resize(
        rgb,
        (_INPUT, _INPUT),
        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
    )
    array = resized.astype(np.float32) / 255.0
    array = (array - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class _Shufa:
    def __init__(self) -> None:
        try:
            torch, resnet50 = _import_model()
        except ImportError as exc:
            message = '未安装书法识别依赖。请执行：pip install -e ".[calligraphy]"'
            raise HanlocError(message) from exc
        self._torch = torch
        weights, mapping = _ensure_files()
        self._labels = _load_labels(mapping)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = resnet50(weights=None)
        except TypeError:
            model = resnet50(pretrained=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.fc.in_features, len(self._labels)),
        )
        try:
            state = torch.load(weights, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(weights, map_location=self._device)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise HanlocError(f"书法识别模型与字表不匹配：{weights}") from exc
        model.to(self._device)
        model.eval()
        self._model = model

    def read(self, images: list[np.ndarray]) -> list[tuple[tuple[str, float], ...]]:
        torch = self._torch
        found: list[tuple[tuple[str, float], ...]] = []
        top_k = min(_TOP_K, len(self._labels))
        for start in range(0, len(images), _BATCH):
            batch = images[start : start + _BATCH]
            tensor = torch.stack([_as_tensor(torch, image) for image in batch]).to(self._device)
            with torch.inference_mode():
                probabilities = torch.softmax(self._model(tensor), dim=1)
                scores, indices = torch.topk(probabilities, k=top_k)
            for row_scores, row_indices in zip(scores.tolist(), indices.tolist(), strict=True):
                found.append(
                    tuple(
                        (self._labels.get(int(index), "?"), float(score))
                        for score, index in zip(row_scores, row_indices, strict=True)
                    )
                )
        return found


def _engine() -> _Shufa:
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = _Shufa()
        return _ENGINE


def reread_glyphs(
    image: np.ndarray,
    glyphs: list[Glyph],
    reader=None,
) -> None:
    """给每个字写入最多 5 个书法候选。``char`` 保持不动。"""
    if not glyphs:
        return
    recognize = _engine().read if reader is None else reader
    crops = [crop_glyph(image, glyph) for glyph in glyphs]
    found = recognize(crops)
    if len(found) != len(glyphs):
        raise HanlocError("书法识别返回的字数和单字框不一致")
    for glyph, item in zip(glyphs, found, strict=True):
        glyph.candidates = tuple((str(char), float(score)) for char, score in item)
