from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.dataset import imagenette_wnid_to_label  # noqa: E402


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
