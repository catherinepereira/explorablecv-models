"""CIFAR-10 data loading 

CIFAR-10 is 60,000 32x32 RGB images across 10 classes (50k train / 10k test).
We use torchvision's downloader, which fetches from https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane", 
    "automobile", 
    "bird", 
    "cat", 
    "deer", 
    "dog", 
    "frog", 
    "horse", 
    "ship", 
    "truck",
)

# These are the per-channel means/stds of the CIFAR-10 training set.
# Subtracting the mean and dividing by the std rescales each channel to roughly mean-0,
# std-1, which helps optimization (gradients on a similar scale across channels)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        # Augumentation
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(data_root: Path, batch_size: int = 128, workers: int = 2,
                download: bool = True) -> tuple[DataLoader, DataLoader]:
    train_set = datasets.CIFAR10(
        root=str(data_root), train=True, download=download,
        transform=build_transforms(train=True),
    )
    test_set = datasets.CIFAR10(
        root=str(data_root), train=False, download=download,
        transform=build_transforms(train=False),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Inverse of the Normalize transform, for visualization."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1)
