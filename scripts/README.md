# Shared scripts

Dataset downloads shared by more than one app. They write into a single `data/` cache at the repo root
so a download is reused rather than repeated per app. Override the location with `EXPLORABLECV_DATA`.

| Script | Dataset | Apps |
| --- | --- | --- |
| `download_cifar10.py` | CIFAR-10 (torchvision) | cnn-architecture-comparison, cnn-visualizer |
| `download_imagenette.py` | Imagenette / Imagewoof (fast.ai) | cv-interpretability, vit-playground |
| `data_paths.py` | resolves the shared data root | (imported by the above) |

```bash
python scripts/download_cifar10.py
python scripts/download_imagenette.py                 # imagenette (default)
python scripts/download_imagenette.py --variant imagewoof
python scripts/download_imagenette.py --dest cv-interpretability/data/raw
```

cv-interpretability uses `--dest` for a private copy: its preprocess step renames
class directories in place, which must not touch the shared cache.

These scripts only populate the shared cache. Each app keeps its own data loaders and points them at the
shared root through `EXPLORABLECV_DATA`, so no app imports another app's code.
