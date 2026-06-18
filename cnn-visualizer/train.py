"""Train SmallCNN on CIFAR-10

Usage:
    python train.py --out checkpoints/run1 --epochs 30

For each minibatch:
    1. forward pass   : feed images through the model, get logits
    2. loss           : compare logits to true labels via cross-entropy
    3. zero gradients : clear gradient buffers from the previous step
    4. backward pass  : autograd walks the graph and fills .grad on every parameter
    5. optimizer step : optimizer reads .grad and updates each parameter

We log train/val loss and accuracy per epoch into history.json so the notebook and viz site can plot the curves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from cnn_scratch import SmallCNN
from cnn_scratch.data import get_loaders


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Number of correct predictions in a batch."""
    return int((logits.argmax(dim=1) == targets).sum().item())


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(loader, desc=("train" if train else "val"), leave=False)
    with torch.set_grad_enabled(train):
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                # The three lines that make a neural net learn!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_correct += accuracy(logits, targets)
            total_seen += bs
            pbar.set_postfix(loss=f"{total_loss/total_seen:.3f}",
                             acc=f"{total_correct/total_seen:.3f}")

    return total_loss / total_seen, total_correct / total_seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_loaders(
        args.data_root, batch_size=args.batch_size, workers=args.workers,
    )

    model = SmallCNN(num_classes=10).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SmallCNN, {n_params/1e6:.2f}M parameters, device={args.device}")

    # Cross-entropy on raw logits. 
    # Internally this is log_softmax + NLL, combined for numerical stability.
    criterion = nn.CrossEntropyLoss()

    # Adam: adaptive learning rates per parameter using running estimates of the gradient's first and second moments.
    # Weight decay (L2 regularization) shrinks weights each step, which helps against overfitting.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine annealing: smoothly decay the learning rate from args.lr to close to 0 over training.
    # The intuition is "take big steps early to find a good basin, then small steps to settle into its bottom".
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, args.device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_acc": val_acc},
                       args.out / "best.pt")

        (args.out / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({"model": model.state_dict(),
                "epoch": args.epochs, "val_acc": val_acc},
               args.out / "last.pt")

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min. Best val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
