import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from lime import lime_image
from skimage.segmentation import slic

from _common import device
from model.builders import build_model, MODEL_NAMES
from model.constants import IMAGENETTE_CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def quantize(map2d: np.ndarray, size: int = 64) -> list[list[int]]:
    h, w = map2d.shape
    factor_h = h // size
    factor_w = w // size
    down = map2d[: factor_h * size, : factor_w * size].reshape(size, factor_h, size, factor_w).mean(axis=(1, 3))
    return down.astype(np.uint8).tolist()


def make_predict_fn(model, dev):
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    def predict(images: np.ndarray) -> np.ndarray:
        batch = (images.astype(np.float32) / 255.0 - mean) / std
        batch = batch.transpose(0, 3, 1, 2)
        x = torch.from_numpy(batch).to(dev)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        return probs

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", default="exports/samples")
    parser.add_argument("--out", default="exports/lime")
    parser.add_argument("--num-samples", type=int, default=500)
    args = parser.parse_args()

    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, pretrained=False).to(dev)
    ckpt = args.checkpoint or f"exports/{args.model}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_predict_fn(model, dev)
    manifest = json.loads((Path(args.samples) / "manifest.json").read_text())

    records = []
    for item in tqdm(manifest, desc=args.model):
        img = np.asarray(Image.open(Path(args.samples) / item["file"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
        explanation = explainer.explain_instance(
            img,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=args.num_samples,
            segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10, sigma=1, start_label=0),
        )
        label = explanation.top_labels[0]
        segments = explanation.segments
        weights = dict(explanation.local_exp[label])
        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, w in weights.items():
            heatmap[segments == seg_id] = w
        mag = np.abs(heatmap).max() + 1e-8
        heatmap = heatmap / mag
        heatmap_u8 = (((heatmap + 1) / 2) * 255).astype(np.uint8)
        records.append({
            "id": item["id"],
            "model": args.model,
            "pred_class": IMAGENETTE_CLASSES[int(label)],
            "heatmap": quantize(heatmap_u8),
        })

    (out / f"{args.model}.json").write_text(json.dumps(records))
    print(f"wrote {len(records)} LIME records")


if __name__ == "__main__":
    main()
