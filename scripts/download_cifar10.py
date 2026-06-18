"""Download CIFAR-10 into the shared data cache via torchvision.

Used by cnn-architecture-comparison and cnn-visualizer, which both train on CIFAR-10.
torchvision caches the dataset under the shared root so the two apps share one copy.
"""

import argparse

from data_paths import dataset_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    from torchvision import datasets

    out = dataset_dir("cifar10")
    datasets.CIFAR10(out, train=True, download=True)
    datasets.CIFAR10(out, train=False, download=True)
    print(f"data at {out}")


if __name__ == "__main__":
    main()
