import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import device, imagenette_root
from model.builders import build_model, MODEL_NAMES
from model.dataset import ImageNetteDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default="exports")
    args = parser.parse_args()

    dev = device()
    out_dir = Path(args.out)
    ckpt = args.checkpoint or str(out_dir / f"{args.model}.pt")

    model = build_model(args.model, pretrained=False).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    root = imagenette_root(args.data)
    val_ds = ImageNetteDataset(root, "val")
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    correct, total = 0, 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            preds = model(x).argmax(1)
            for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1
            correct += (preds == y).sum().item()
            total += x.size(0)

    acc = correct / total
    per_class_acc = [c / t if t else 0.0 for c, t in zip(per_class_correct, per_class_total)]
    result = {"model": args.model, "val_accuracy": acc, "per_class_accuracy": per_class_acc}
    (out_dir / f"{args.model}.eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
