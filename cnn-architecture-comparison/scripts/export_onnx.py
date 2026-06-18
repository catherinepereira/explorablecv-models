import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from model import ARCHITECTURES

EXPORT_DIR = Path(__file__).parent.parent / 'exports'

CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
]


def export(arch_name: str, out_dir: Path):
    ckpt = EXPORT_DIR / f'{arch_name}.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f'No checkpoint at {ckpt}. Run train.py first.')

    Model = ARCHITECTURES[arch_name]
    model = Model()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    output_names = Model.export_outputs()
    dummy = torch.randn(1, 3, 32, 32)

    out_dir.mkdir(exist_ok=True, parents=True)
    onnx_path = out_dir / f'{arch_name}.onnx'

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=['input'],
        output_names=output_names,
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    print(f'Exported {onnx_path}')


def write_meta(out_dir: Path):
    meta = {
        'labels': CIFAR10_LABELS,
        'input_shape': [3, 32, 32],
        'normalization': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2470, 0.2435, 0.2616],
        },
        'exported_at': datetime.now().isoformat(),
    }
    (out_dir / 'model_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'Wrote {out_dir / "model_meta.json"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='all', choices=list(ARCHITECTURES.keys()) + ['all'])
    p.add_argument('--out', type=Path, default=EXPORT_DIR)
    args = p.parse_args()

    archs = list(ARCHITECTURES.keys()) if args.arch == 'all' else [args.arch]
    for a in archs:
        export(a, args.out)
    write_meta(args.out)


if __name__ == '__main__':
    main()
