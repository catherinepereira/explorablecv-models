# cnn-architecture-comparison-model

Training repo for [cnn-architecture-comparison](../cnn-architecture-comparison). Trains six CNN architectures on CIFAR-10 and exports them to ONNX with intermediate feature map outputs.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CIFAR-10 downloads on first run via torchvision

## Train

```powershell
python scripts/train.py --arch all --epochs 50
```

Or one at a time:

```powershell
python scripts/train.py --arch resnet --epochs 80
```

Checkpoints land in `exports/<arch>.pt`

## Export to ONNX

```powershell
python scripts/export_onnx.py --arch all
```

Writes `<arch>.onnx` and `model_meta.json` to `exports/`. Each ONNX file exposes the final logits plus 2-5 intermediate feature maps, named in `<Model>.export_outputs()`.

## Verify ONNX parity

```powershell
python scripts/evaluate_onnx.py --arch all --n 500
```

Checks that argmax matches and prints max numerical drift between the PyTorch and ONNX outputs on 500 test images. Drift larger than 1e-4 means a fused op behaved differently.

## Deploy to the frontend

```powershell
copy exports\*.onnx ..\cnn-architecture-comparison\public\models\
copy exports\model_meta.json ..\cnn-architecture-comparison\public\models\
```

## License

MIT
