import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap

from _common import device, imagenette_root
from model.builders import build_model, penultimate_features, MODEL_NAMES
from model.dataset import ImageNetteDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="exports/umap")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=2000)
    args = parser.parse_args()

    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, pretrained=False).to(dev)
    ckpt = args.checkpoint or f"exports/{args.model}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    root = imagenette_root(args.data)
    ds = ImageNetteDataset(root, args.split)
    if len(ds) > args.max_images:
        idx = np.random.RandomState(0).choice(len(ds), args.max_images, replace=False)
        ds.samples = [ds.samples[i] for i in idx]
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    feats, labels, paths = [], [], []
    with torch.no_grad():
        for x, y, p in tqdm(loader, desc=args.model):
            x = x.to(dev)
            f = penultimate_features(model, args.model, x).cpu().numpy()
            feats.append(f)
            labels.extend(y.tolist())
            paths.extend(p)
    feats = np.concatenate(feats, axis=0)

    reducer = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist, random_state=0)
    coords = reducer.fit_transform(feats)

    class_names = sorted(ds.class_to_idx, key=lambda k: ds.class_to_idx[k])
    points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "label": class_names[labels[i]],
            "thumb": Path(paths[i]).name,
        }
        for i in range(len(labels))
    ]
    (out / f"{args.model}.json").write_text(json.dumps({"model": args.model, "points": points}))
    print(f"wrote {len(points)} UMAP points")


if __name__ == "__main__":
    main()
