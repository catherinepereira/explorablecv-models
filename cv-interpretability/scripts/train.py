import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from _common import device, imagenette_root
from model.builders import build_model, MODEL_NAMES
from model.dataset import ImageNetteDataset, assert_classes_match_expected


def run_epoch(model, loader, criterion, optimizer, dev, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y, _ in tqdm(loader, leave=False):
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="exports")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2

    dev = device()
    root = imagenette_root(args.data)
    train_ds = ImageNetteDataset(root, "train")
    val_ds = ImageNetteDataset(root, "val")
    class_names = sorted(train_ds.class_to_idx, key=lambda k: train_ds.class_to_idx[k])
    assert_classes_match_expected(class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    history = []
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, dev, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, dev, train=False)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out / f"{args.model}.pt")

    meta = {
        "model": args.model,
        "labels": class_names,
        "input_shape": [3, 224, 224],
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "best_val_acc": best_acc,
        "trained_at": datetime.now().isoformat(),
        "training_examples": len(train_ds),
        "history": history,
    }
    (out / f"{args.model}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
