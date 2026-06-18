from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, IMAGENETTE_CLASSES


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageNetteDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root = Path(root) / split
        self.transform = transform or build_transforms(train=(split == "train"))
        self.class_to_idx = {c: i for i, c in enumerate(sorted(p.name for p in self.root.iterdir() if p.is_dir()))}
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in self.root.iterdir():
            if not cls_dir.is_dir():
                continue
            label = self.class_to_idx[cls_dir.name]
            for img_path in cls_dir.glob("*.JPEG"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, str(path)


def imagenette_wnid_to_label() -> dict[str, str]:
    return {
        "n01440764": "tench",
        "n02102040": "English springer",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03417042": "garbage truck",
        "n03425413": "gas pump",
        "n03445777": "golf ball",
        "n03888257": "parachute",
    }


def assert_classes_match_expected(class_names: list[str]):
    expected = set(IMAGENETTE_CLASSES)
    got = set(class_names)
    if expected != got:
        raise ValueError(f"class mismatch: expected {expected}, got {got}")
