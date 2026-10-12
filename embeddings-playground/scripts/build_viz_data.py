"""Reduce the CLIP embeddings to 2D and 3D with UMAP and write the frontend's viz.json.

Also writes class_vectors.json, the L2-normalized mean embedding per COCO
category, which the frontend shows as the average embedding for a class.

Colors come from the category's supercategory, so related things share a hue
(all vehicles blue, all animals green). With 80 categories an arbitrary palette
is unreadable, while hue families let you see the broad structure of the plot
before reading a single label.
"""

import argparse
import colorsys
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import umap

_repo = Path(__file__).resolve().parents[1]
OUT_DIR = _repo / "data"

# COCO's own coco_url is http-only and images.cocodataset.org presents a
# certificate for a different host, so an https page cannot hotlink it. The
# path-style S3 URL is the same bucket with a valid certificate
IMAGE_BASE = "https://s3.amazonaws.com/images.cocodataset.org/train2017"

# One hue per COCO supercategory, spread around the wheel with related groups
# kept apart. Categories inside a group fan out in lightness from these
SUPERCATEGORY_HUES = {
    "person": 0.95,
    "vehicle": 0.58,
    "outdoor": 0.45,
    "animal": 0.28,
    "accessory": 0.80,
    "sports": 0.12,
    "kitchen": 0.08,
    "food": 0.04,
    "furniture": 0.70,
    "electronic": 0.52,
    "appliance": 0.88,
    "indoor": 0.17,
}
FALLBACK_HUE = 0.0


def build_palette(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Map each (category, supercategory) to a hex color.

    Members of a supercategory share its hue and fan out in lightness, so a
    cluster's family reads at a glance and neighbors inside it stay separable.
    """
    groups: dict[str, list[str]] = {}
    for category, supercategory in sorted(pairs):
        groups.setdefault(supercategory, []).append(category)

    colors: dict[str, str] = {}
    for supercategory, members in groups.items():
        hue = SUPERCATEGORY_HUES.get(supercategory, FALLBACK_HUE)
        n = len(members)
        for i, category in enumerate(members):
            # Spread lightness across the group, holding back from pure white
            # or black so every swatch stays visible on both themes
            light = 0.38 + (0.34 * (i / max(1, n - 1)) if n > 1 else 0.14)
            sat = 0.62 - 0.18 * (i % 2)
            r, g, b = colorsys.hls_to_rgb(hue, light, sat)
            colors[category] = "#{:02X}{:02X}{:02X}".format(
                round(r * 255), round(g * 255), round(b * 255)
            )
    return colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings",   default=str(OUT_DIR / "embeddings.npz"))
    parser.add_argument("--index",        default=str(OUT_DIR / "index.parquet"))
    parser.add_argument("--out",          default=str(OUT_DIR / "viz.json"))
    parser.add_argument("--n-neighbors",  type=int,   default=15)
    parser.add_argument("--min-dist",     type=float, default=0.1)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    print("Loading embeddings...")
    data = np.load(args.embeddings)
    ids: np.ndarray        = data["ids"]
    embeddings: np.ndarray = data["embeddings"].astype(np.float32)
    print(f"  {embeddings.shape[0]} images, dim={embeddings.shape[1]}")

    print("Loading index...")
    index = pd.read_parquet(args.index).set_index("image_id")

    umap_kwargs = dict(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                       metric="cosine", random_state=args.random_state, verbose=True)

    print("Running UMAP 2D...")
    reducer2d = umap.UMAP(n_components=2, **umap_kwargs)
    coords2d  = reducer2d.fit_transform(embeddings)

    print("Running UMAP 3D...")
    reducer3d = umap.UMAP(n_components=3, **umap_kwargs)
    coords3d  = reducer3d.fit_transform(embeddings)

    rows = index.loc[[int(i) for i in ids.tolist()]]
    categories = rows["category"].tolist()
    supercategories = rows["supercategory"].tolist()

    counts = dict(Counter(categories))
    category_colors = build_palette(list({(c, s) for c, s in zip(categories, supercategories)}))
    supercategory_of = dict(zip(categories, supercategories))

    print("  Categories:")
    for c, n in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {c:25s} {n}")
    print(f"    ... {len(counts)} categories total")

    print("Building output...")
    points = []
    for i, raw_id in enumerate(ids.tolist()):
        image_id = int(raw_id)
        row = rows.loc[image_id]
        points.append({
            "id":            image_id,
            "x":             round(float(coords3d[i, 0]), 4),
            "y":             round(float(coords3d[i, 1]), 4),
            "z":             round(float(coords3d[i, 2]), 4),
            "x2":            round(float(coords2d[i, 0]), 4),
            "y2":            round(float(coords2d[i, 1]), 4),
            "category":      categories[i],
            "supercategory": supercategories[i],
            "color":         category_colors[categories[i]],
            "image_url":     f"{IMAGE_BASE}/{row['file_name']}",
            "width":         int(row["width"]),
            "height":        int(row["height"]),
        })

    out = {
        "points": points,
        "category_colors": category_colors,
        "supercategories": supercategory_of,
        "meta": {
            "n_images": len(points),
            "umap_n_neighbors": args.n_neighbors,
            "umap_min_dist": args.min_dist,
            "embedding_dim": int(embeddings.shape[1]),
            "model": "openai/clip-vit-base-patch32",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {len(points)} points -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("Building category averages...")
    category_arr = np.array(categories)
    class_vectors = {}
    for c in sorted(set(categories)):
        mean = embeddings[category_arr == c].mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        class_vectors[c] = [round(float(v), 4) for v in mean]
    cv_path = out_path.parent / "class_vectors.json"
    cv_path.write_text(json.dumps(
        {"dim": int(embeddings.shape[1]), "categories": class_vectors},
        separators=(",", ":"),
    ))
    print(f"Saved {len(class_vectors)} category averages -> {cv_path}")

    for name, model in [("umap_model.pkl", reducer3d), ("umap_model_2d.pkl", reducer2d)]:
        path = out_path.parent / name
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved UMAP model to {path}")


if __name__ == "__main__":
    main()
