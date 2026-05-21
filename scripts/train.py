import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ARCHITECTURES

# Per-channel (R, G, B) mean/std of the CIFAR-10 train set, used to standardize inputs for stable optimization
# Must match the values baked into export_onnx.py so inference preprocessing stays in sync
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
EXPORT_DIR = Path(__file__).parent.parent / 'exports'


def get_loaders(batch_size: int, num_workers: int = 4):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[0]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train(arch_name: str, epochs: int, lr: float, batch_size: int, smoke: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training {arch_name} on {device}')

    Model = ARCHITECTURES[arch_name]
    model = Model().to(device)

    train_loader, test_loader = get_loaders(batch_size)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    if smoke:
        epochs = 2

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'epoch {epoch + 1}/{epochs}')
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)[0]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                pbar.set_postfix(loss=f'{loss.item():.3f}')
            if smoke and i >= 20:
                break
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'epoch {epoch + 1}: test acc {acc * 100:.2f}%')

    EXPORT_DIR.mkdir(exist_ok=True, parents=True)
    ckpt_path = EXPORT_DIR / f'{arch_name}.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'Saved checkpoint to {ckpt_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=list(ARCHITECTURES.keys()) + ['all'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--smoke', action='store_true', help='2 epochs, ~20 steps each')
    args = p.parse_args()

    archs = list(ARCHITECTURES.keys()) if args.arch == 'all' else [args.arch]
    for a in archs:
        train(a, args.epochs, args.lr, args.batch_size, args.smoke)


if __name__ == '__main__':
    main()
