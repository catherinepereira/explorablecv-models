"""Build the image index the embedding pass reads, from COCO instance annotations.

Each image is labeled by its largest annotated instance, so the label describes
what dominates the frame rather than whatever happens to be present. Images
whose largest instance covers less than --min-area of the frame are dropped,
since a scene with no dominant object lands in no cluster in particular.

COCO is heavily skewed (a person appears in about half of all images), so the
per-class cap balances the sample. Without it the plot is one enormous person
blob with 79 specks around it.

Reads instances_train2017.json straight out of annotations_trainval2017.zip, so
no extra download is needed when the zip is already in the shared data cache.
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

_repo = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = _repo.parents[0] / "data" / "coco-val2017" / "annotations_trainval2017.zip"
OUT_DIR = _repo / "data"

MEMBER = "annotations/instances_train2017.json"


def load_annotations(zip_path: Path, member: str) -> dict:
    if zip_path.suffix == ".json":
        return json.loads(zip_path.read_text())
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=str(DEFAULT_ZIP),
                        help="annotations_trainval2017.zip, or an extracted instances_*.json")
    parser.add_argument("--member", default=MEMBER, help="path inside the zip")
    parser.add_argument("--out", default=str(OUT_DIR / "index.parquet"))
    parser.add_argument("--per-class", type=int, default=400,
                        help="max images per category, to balance COCO's skew")
    parser.add_argument("--min-area", type=float, default=0.10,
                        help="largest instance must cover at least this fraction of the frame")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading {args.annotations}")
    data = load_annotations(Path(args.annotations), args.member)

    categories = {c["id"]: c for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    print(f"  {len(images)} images, {len(categories)} categories")

    # Largest annotated instance per image, ignoring crowd regions whose area is
    # a loose polygon estimate rather than one object
    best: dict[int, dict] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        prev = best.get(ann["image_id"])
        if prev is None or ann["area"] > prev["area"]:
            best[ann["image_id"]] = ann
    print(f"  {len(best)} images with at least one instance")

    rows = []
    for image_id, ann in best.items():
        im = images[image_id]
        frame = im["width"] * im["height"]
        if frame <= 0 or ann["area"] / frame < args.min_area:
            continue
        cat = categories[ann["category_id"]]
        rows.append({
            "image_id": image_id,
            "coco_url": im["coco_url"],
            "flickr_url": im.get("flickr_url"),
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
            "category": cat["name"],
            "supercategory": cat["supercategory"],
            "area_frac": round(ann["area"] / frame, 4),
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} images clear the {args.min_area:.0%} dominance floor")

    # Balance, taking the most dominant instances first so each class is
    # represented by its clearest examples
    df = df.sort_values("area_frac", ascending=False)
    df = df.groupby("category", group_keys=False).head(args.per_class)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    counts = df["category"].value_counts()
    print(f"\n{len(df)} images across {len(counts)} categories")
    print(f"  per class: min {counts.min()}, median {int(counts.median())}, max {counts.max()}")
    thin = counts[counts < 50]
    if len(thin):
        print(f"  {len(thin)} categories under 50 images: {', '.join(thin.index[:8])}")

    by_super: dict[str, int] = defaultdict(int)
    for s, n in df["supercategory"].value_counts().items():
        by_super[s] = n
    print("  supercategories: " + ", ".join(f"{s} {n}" for s, n in sorted(by_super.items())))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nSaved index -> {out}")


if __name__ == "__main__":
    main()
