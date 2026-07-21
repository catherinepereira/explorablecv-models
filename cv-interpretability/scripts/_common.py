from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model.dataset import imagenette_wnid_to_label  # noqa: E402


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path: Path) -> torch.Tensor:
    """One sample image as a normalized [1,3,H,W] tensor, matching training."""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def quantize(map2d: np.ndarray, size: int = 64, scale: float = 255.0) -> list[list[int]]:
    """Mean-pool a 2D map down to size x size and emit uint8 rows for JSON.

    scale converts the input range to 0-255: the default expects [0,1] maps,
    pass scale=1.0 for maps already in 0-255.
    """
    h, w = map2d.shape
    factor_h = h // size
    factor_w = w // size
    down = map2d[: factor_h * size, : factor_w * size].reshape(size, factor_h, size, factor_w).mean(axis=(1, 3))
    return (down * scale).astype(np.uint8).tolist()


def imagenette_root(data_dir: str | Path) -> Path:
    p = Path(data_dir)
    candidate = p / "imagenette2"
    return candidate if candidate.exists() else p


def rename_wnid_dirs(root: Path):
    mapping = imagenette_wnid_to_label()
    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name in mapping:
                new_name = mapping[child.name]
                target = child.with_name(new_name)
                if not target.exists():
                    child.rename(target)
