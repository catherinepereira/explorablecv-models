import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torchvision
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
OUT_DIR = Path(__file__).parent.parent.parent / 'cnn-architecture-comparison' / 'public' / 'samples'

CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False)

    found = {}
    for img, label in test:
        name = CIFAR10_LABELS[label]
        if name in found:
            continue
        found[name] = img
        if len(found) == len(CIFAR10_LABELS):
            break

    for name, img in found.items():
        upscaled = img.resize((128, 128), Image.NEAREST)
        out_path = OUT_DIR / f'{name}.png'
        upscaled.save(out_path)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
