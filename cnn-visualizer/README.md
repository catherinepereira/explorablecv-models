# cnn-from-scratch-model

Generates model and activations for [`cnn-visualizer`](../../explorablecv/apps/cnn-visualizer), a static site that visualizes the trained model's activations layer by layer.

## Important Files

- `cnn_scratch/model.py`: `SmallCNN`, 3 conv blocks plus a dense head, 600k params, with annotations
- `cnn_scratch/layers.py`: reference implementation of `Conv2d`, `Linear`, `ReLU`, `MaxPool2d`, `Flatten` written in terms of slicing, `unfold`, and matmul. Not used by `SmallCNN` (which uses stock `torch.nn` layers for training speed), but kept as a readable companion for understanding what those layers compute
- `cnn_scratch/data.py`: CIFAR-10 loading, normalization, augmentation.
- `train.py`: PyTorch training loop.
- `scripts/eval.py`: test-set accuracy, per-class accuracy, confusion matrix.
- `scripts/export_activations.py`: runs sample images through the trained model and dumps activation PNGs plus a manifest for the viz site.

## Quick start

```bash
python -m venv venv
. venv/Scripts/activate
pip install -r requirements.txt

python train.py --out checkpoints/run1 --epochs 30
python scripts/eval.py --ckpt checkpoints/run1/best.pt --out checkpoints/run1
python scripts/export_activations.py \
    --ckpt checkpoints/run1/best.pt \
    --out ../../explorablecv/apps/cnn-visualizer/public/activations
```

CIFAR-10 downloads automatically into `data/` on first run (~170 MB).