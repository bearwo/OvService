from __future__ import annotations

from pathlib import Path

import openvino as ov
from PIL import Image


def load_image_as_tensor(image_path: str | Path) -> ov.Tensor:
    img = Image.open(image_path).convert("RGB")
    import numpy as np
    arr = np.array(img)[None]
    return ov.Tensor(arr)


def load_image_from_bytes(data: bytes) -> ov.Tensor:
    import io
    img = Image.open(io.BytesIO(data)).convert("RGB")
    import numpy as np
    arr = np.array(img)[None]
    return ov.Tensor(arr)


def get_image_info(image_path: str | Path) -> dict:
    img = Image.open(image_path)
    return {
        "path": str(image_path),
        "format": img.format,
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
    }
