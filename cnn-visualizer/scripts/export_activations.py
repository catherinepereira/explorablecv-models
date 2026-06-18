"""
Run a few sample CIFAR-10 test images through a trained SmallCNN and dump the intermediate activations as JSON

Each activation is converted to PNG tiles (one per channel, normalized to 0-255 grayscale) 
so the frontend doesn't have to do any tensor math.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image

from cnn_scratch import SmallCNN
from cnn_scratch.data import CIFAR10_CLASSES, denormalize, get_loaders


# Layer name -> (display label, expected shape note)
LAYER_INFO = {
    "input":  ("input image",            "3x32x32 RGB"),
    "conv1a": ("conv1a (3 → 32, 3x3)",   "32 channels, 32x32"),
    "conv1b": ("conv1b (32 → 32, 3x3)",  "32 channels, 32x32"),
    "pool1":  ("maxpool 2x2",            "32 channels, 16x16"),
    "conv2a": ("conv2a (32 → 64, 3x3)",  "64 channels, 16x16"),
    "conv2b": ("conv2b (64 → 64, 3x3)",  "64 channels, 16x16"),
    "pool2":  ("maxpool 2x2",            "64 channels, 8x8"),
    "conv3":  ("conv3 (64 → 128, 3x3)",  "128 channels, 8x8"),
    "pool3":  ("maxpool 2x2",            "128 channels, 4x4"),
}


def tensor_to_grayscale_png(t: torch.Tensor, path: Path, upscale: int = 4) -> None:
    """Save a 2D tensor as a PNG, min-max normalized to 0-255."""
    arr = t.numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if upscale > 1:
        img = img.resize((arr.shape[1] * upscale, arr.shape[0] * upscale), Image.NEAREST)
    img.save(path)


def save_input_png(img_chw: torch.Tensor, path: Path, upscale: int = 4) -> None:
    """Save the denormalized RGB input image."""
    arr = (denormalize(img_chw).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if upscale > 1:
        img = img.resize((arr.shape[1] * upscale, arr.shape[0] * upscale), Image.NEAREST)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--wrong-fraction", type=float, default=0.3,
                    help="Fraction of samples that should be wrong predictions.")
    ap.add_argument("--max-channels", type=int, default=16,
                    help="Cap channels exported per layer (keeps the site light).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    model = SmallCNN(num_classes=10).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    # First pass: classify the whole test set so we can pick a balanced mix
    _, test_loader = get_loaders(args.data_root, batch_size=256, workers=0)
    all_imgs, all_targets, all_preds = [], [], []
    with torch.no_grad():
        for imgs, tgts in test_loader:
            preds = model(imgs.to(args.device)).argmax(dim=1).cpu()
            all_imgs.append(imgs); all_targets.append(tgts); all_preds.append(preds)
    all_imgs = torch.cat(all_imgs)
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)

    correct_mask = all_preds == all_targets
    n_wrong = max(1, int(round(args.num_samples * args.wrong_fraction)))
    n_correct = args.num_samples - n_wrong

    # Sample roughly evenly across classes within each pool
    g = torch.Generator().manual_seed(args.seed)
    chosen = []
    for pool_mask, k in [(correct_mask, n_correct), (~correct_mask, n_wrong)]:
        per_class: list[int] = []
        for cls in range(10):
            cls_idx = ((all_targets == cls) & pool_mask).nonzero(as_tuple=True)[0]
            if len(cls_idx) == 0:
                continue
            perm = torch.randperm(len(cls_idx), generator=g)
            per_class.append(cls_idx[perm[0]].item())
        # Round-robin across classes until we have k
        picked: list[int] = []
        i = 0
        while len(picked) < k and per_class:
            picked.append(per_class[i % len(per_class)])
            i += 1
            if i >= len(per_class):
                # Refill pool with fresh draws so we don't repeat
                refill: list[int] = []
                for cls in range(10):
                    cls_idx = ((all_targets == cls) & pool_mask).nonzero(as_tuple=True)[0]
                    cls_idx = [int(x) for x in cls_idx.tolist() if int(x) not in picked]
                    if cls_idx:
                        perm = torch.randperm(len(cls_idx), generator=g)
                        refill.append(cls_idx[int(perm[0].item())])
                per_class = refill
                i = 0
                if not per_class:
                    break
        chosen.extend(picked[:k])

    chosen_idx = torch.tensor(chosen)
    images = all_imgs[chosen_idx].to(args.device)
    targets = all_targets[chosen_idx]

    acts = model.forward_with_activations(images)

    samples = []
    for i in range(args.num_samples):
        sample_dir = args.out / f"sample_{i:02d}"
        sample_dir.mkdir(exist_ok=True)

        true_idx = int(targets[i].item())
        pred_idx = int(acts["logits"][i].argmax().item())
        probs = torch.softmax(acts["logits"][i], dim=0).tolist()

        save_input_png(acts["input"][i], sample_dir / "input.png")

        layer_meta = []
        for name, (label, shape_note) in LAYER_INFO.items():
            if name == "input":
                continue
            feat = acts[name][i]  # (C, H, W)
            c_total = feat.shape[0]
            n_export = min(args.max_channels, c_total)
            layer_dir = sample_dir / name
            layer_dir.mkdir(exist_ok=True)
            for c in range(n_export):
                tensor_to_grayscale_png(feat[c], layer_dir / f"ch_{c:02d}.png")
            layer_meta.append({
                "name": name,
                "label": label,
                "shape_note": shape_note,
                "channels_total": c_total,
                "channels_exported": n_export,
                "h": int(feat.shape[1]),
                "w": int(feat.shape[2]),
            })

        samples.append({
            "id": i,                              # asset folder index (sample_00, sample_01, ...)
            "dataset_id": int(chosen_idx[i].item()),  # row index in the CIFAR-10 test set
            "true_class": CIFAR10_CLASSES[true_idx],
            "pred_class": CIFAR10_CLASSES[pred_idx],
            "correct": true_idx == pred_idx,
            "probs": {cls: round(p, 4) for cls, p in zip(CIFAR10_CLASSES, probs)},
            "layers": layer_meta,
        })

    n_params = sum(p.numel() for p in model.parameters())
    test_correct = int((all_preds == all_targets).sum().item())
    test_total = int(all_targets.numel())

    per_class_acc: dict[str, float] = {}
    for cls in range(10):
        cls_mask = all_targets == cls
        denom = int(cls_mask.sum().item())
        if denom > 0:
            per_class_acc[CIFAR10_CLASSES[cls]] = round(
                int(((all_preds == all_targets) & cls_mask).sum().item()) / denom, 4
            )

    history_path = Path(args.ckpt).parent / "history.json"
    history_stats: dict[str, object] = {}
    if history_path.exists():
        h = json.loads(history_path.read_text())
        history_stats = {
            "epochs_trained": len(h.get("train_loss", [])),
            "best_val_acc": round(max(h.get("val_acc", [0])), 4),
            "final_train_acc": round(h["train_acc"][-1], 4) if h.get("train_acc") else None,
            "final_val_acc": round(h["val_acc"][-1], 4) if h.get("val_acc") else None,
        }

    stats = {
        "params": n_params,
        "test_accuracy": round(test_correct / test_total, 4),
        "test_correct": test_correct,
        "test_total": test_total,
        "per_class_accuracy": per_class_acc,
        **history_stats,
    }

    (args.out / "manifest.json").write_text(json.dumps({
        "classes": list(CIFAR10_CLASSES),
        "checkpoint": str(args.ckpt),
        "stats": stats,
        "samples": samples,
    }, indent=2))
    print(f"Exported {args.num_samples} samples to {args.out}")


if __name__ == "__main__":
    main()
