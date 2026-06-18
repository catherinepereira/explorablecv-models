"""
Evaluate a trained SmallCNN on the CIFAR-10 test set.

Reports overall accuracy, per-class accuracy, and writes a confusion matrix to confusion.json,
so the notebook / viz site can plot it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from cnn_scratch import SmallCNN
from cnn_scratch.data import CIFAR10_CLASSES, get_loaders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _, test_loader = get_loaders(args.data_root, batch_size=args.batch_size, workers=2)

    model = SmallCNN(num_classes=10).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    n_classes = len(CIFAR10_CLASSES)
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(args.device)
            targets = targets.to(args.device)
            preds = model(images).argmax(dim=1)
            for t, p in zip(targets.cpu().tolist(), preds.cpu().tolist()):
                confusion[t, p] += 1

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    overall = correct / total
    per_class = (confusion.diag() / confusion.sum(dim=1).clamp(min=1)).tolist()

    print(f"Overall accuracy: {overall:.4f}")
    print("Per-class accuracy:")
    for name, acc in zip(CIFAR10_CLASSES, per_class):
        print(f"  {name:12s} {acc:.4f}")

    (args.out / "confusion.json").write_text(json.dumps({
        "classes": list(CIFAR10_CLASSES),
        "matrix": confusion.tolist(),
        "overall_accuracy": overall,
        "per_class_accuracy": per_class,
    }, indent=2))


if __name__ == "__main__":
    main()
