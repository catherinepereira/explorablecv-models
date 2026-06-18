import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms as T

from model import ARCHITECTURES

EXPORT_DIR = Path(__file__).parent.parent / 'exports'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def evaluate(arch_name: str, n: int = 1000):
    onnx_path = EXPORT_DIR / f'{arch_name}.onnx'
    ckpt_path = EXPORT_DIR / f'{arch_name}.pt'

    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

    Model = ARCHITECTURES[arch_name]
    pt_model = Model()
    pt_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    pt_model.eval()

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    matches = 0
    drifts = []
    for i in range(n):
        x, _ = test[i]
        x_np = x.numpy()[None, :]
        with torch.no_grad():
            pt_logits = pt_model(torch.from_numpy(x_np))[0].numpy()[0]
        onnx_logits = session.run(['logits'], {'input': x_np})[0][0]
        drifts.append(float(np.max(np.abs(pt_logits - onnx_logits))))
        if pt_logits.argmax() == onnx_logits.argmax():
            matches += 1
    print(f'{arch_name}: {matches}/{n} argmax matches, max drift {max(drifts):.2e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='all', choices=list(ARCHITECTURES.keys()) + ['all'])
    p.add_argument('--n', type=int, default=500)
    args = p.parse_args()
    archs = list(ARCHITECTURES.keys()) if args.arch == 'all' else [args.arch]
    for a in archs:
        evaluate(a, args.n)


if __name__ == '__main__':
    main()
