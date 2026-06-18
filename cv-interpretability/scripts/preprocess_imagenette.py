import argparse
from pathlib import Path

from _common import imagenette_root, rename_wnid_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    args = parser.parse_args()

    root = imagenette_root(args.data)
    if not (root / "train").exists() or not (root / "val").exists():
        raise FileNotFoundError(f"expected train/ and val/ under {root}; run download_data.py first")
    rename_wnid_dirs(root)

    train_count = sum(1 for _ in (root / "train").rglob("*.JPEG"))
    val_count = sum(1 for _ in (root / "val").rglob("*.JPEG"))
    print(f"train: {train_count} images, val: {val_count} images")
    print(f"classes: {sorted(p.name for p in (root / 'train').iterdir() if p.is_dir())}")


if __name__ == "__main__":
    main()
