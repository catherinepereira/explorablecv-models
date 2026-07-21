# explorablecv-models

Model training and ONNX export code for the [explorablecv](../explorablecv) computer-vision demos.

## Layout

| Folder | App | What it produces |
| --- | --- | --- |
| [cnn-architecture-comparison](cnn-architecture-comparison) | cnn-architecture-comparison | CIFAR-10 classifiers (LeNet, AlexNet, VGG, ResNet, ...) exported to ONNX |
| [cnn-visualizer](cnn-visualizer) | cnn-visualizer | A small CNN plus per-layer activation dumps for the visualizer |
| [cv-interpretability](cv-interpretability) | cv-interpretability | Imagenette classifier plus CAM / LIME / attention-rollout / UMAP bundles |
| [vit-playground](vit-playground) | vit-playground | ViT-tiny exported with logits, patch embeddings, per-layer attention |

## Shared scripts and data

[scripts](scripts) holds dataset downloads shared by more than one app:

- `download_cifar10.py`: CIFAR-10 (used by cnn-architecture-comparison, cnn-visualizer)
- `download_imagenette.py`: Imagenette / Imagewoof (used by cv-interpretability, vit-playground)
- `data_paths.py`: resolves the shared data root

Datasets land in a shared `data/` cache at the repo root so every app reads from one place. Override the
location with the `EXPLORABLECV_DATA` environment variable, or pass `--dest` for a private copy
(cv-interpretability does, since its preprocess renames class dirs in place). The shared scripts only
populate the cache. Each app keeps its own loaders, so no app depends on another.

```bash
python scripts/download_cifar10.py
python scripts/download_imagenette.py
```