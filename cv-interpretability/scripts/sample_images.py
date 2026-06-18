import argparse
import json
import shutil
from pathlib import Path

import random

from _common import imagenette_root
from model.constants import IMAGENETTE_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--out", default="exports/samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = imagenette_root(args.data) / "val"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for cls in IMAGENETTE_CLASSES:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"skip missing class {cls}")
            continue
        candidates = list(cls_dir.glob("*.JPEG"))
        rng.shuffle(candidates)
        picked = candidates[: args.per_class]
        for src in picked:
            dst_name = f"{cls.replace(' ', '_')}__{src.stem}.jpg"
            dst = out / dst_name
            shutil.copyfile(src, dst)
            manifest.append({"id": dst.stem, "file": dst.name, "class": cls})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} samples to {out}")


if __name__ == "__main__":
    main()
