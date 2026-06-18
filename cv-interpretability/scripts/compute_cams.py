import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from _common import device
from model.builders import build_model, target_layer_for_cam, MODEL_NAMES
from model.constants import IMAGENETTE_CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.activations = out.detach()

    def _bwd(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def close(self):
        self.h1.remove(); self.h2.remove()

    def __call__(self, x: torch.Tensor, target_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, target_idx]
        score.backward()
        a, g = self.activations, self.gradients
        if a.ndim == 4:
            weights = g.mean(dim=(2, 3), keepdim=True)
            cam = (weights * a).sum(dim=1, keepdim=True)
        else:
            tokens = a[:, 1:, :]
            gtokens = g[:, 1:, :]
            weights = gtokens.mean(dim=1, keepdim=True)
            cam_vec = (weights * tokens).sum(dim=-1)
            n = int(cam_vec.shape[1] ** 0.5)
            cam = cam_vec.reshape(1, 1, n, n)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def score_cam(model, x: torch.Tensor, target_layer, target_idx: int, dev) -> np.ndarray:
    activations = []
    h = target_layer.register_forward_hook(lambda _m, _i, o: activations.append(o.detach()))
    with torch.no_grad():
        model(x)
    h.remove()
    a = activations[0]
    if a.ndim == 4:
        a = F.interpolate(a, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        c = a.shape[1]
        maps = a[0]
        cam = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, device=dev)
        with torch.no_grad():
            for k in range(c):
                m = maps[k]
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                masked = x * m.unsqueeze(0).unsqueeze(0)
                w = F.softmax(model(masked), dim=1)[0, target_idx]
                cam += w * m
        cam = F.relu(cam).cpu().numpy()
    else:
        tokens = a[0, 1:, :]
        n = int(tokens.shape[0] ** 0.5)
        maps = tokens.T.reshape(-1, n, n).unsqueeze(0)
        maps = F.interpolate(maps, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0]
        cam = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, device=dev)
        with torch.no_grad():
            for k in range(maps.shape[0]):
                m = maps[k]
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                masked = x * m.unsqueeze(0).unsqueeze(0)
                w = F.softmax(model(masked), dim=1)[0, target_idx]
                cam += w * m
        cam = F.relu(cam).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def saliency_map(model, x: torch.Tensor, target_idx: int) -> np.ndarray:
    x = x.clone().requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits[0, target_idx].backward()
    grad = x.grad.detach()[0].abs().max(dim=0).values
    grad = grad.cpu().numpy()
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return grad


def quantize(map2d: np.ndarray, size: int = 64) -> list[list[int]]:
    h, w = map2d.shape
    factor_h = h // size
    factor_w = w // size
    down = map2d[: factor_h * size, : factor_w * size].reshape(size, factor_h, size, factor_w).mean(axis=(1, 3))
    arr = (down * 255).astype(np.uint8)
    return arr.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", default="exports/samples")
    parser.add_argument("--out", default="exports/cams")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "scorecam", "saliency"])
    args = parser.parse_args()

    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, pretrained=False).to(dev)
    ckpt = args.checkpoint or f"exports/{args.model}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    manifest = json.loads((Path(args.samples) / "manifest.json").read_text())
    target_layer = target_layer_for_cam(model, args.model)

    gradcam = GradCAM(model, target_layer)
    records = []
    try:
        for item in tqdm(manifest, desc=args.model):
            img_path = Path(args.samples) / item["file"]
            x = load_image(img_path).to(dev)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            entry = {
                "id": item["id"],
                "model": args.model,
                "true_class": item["class"],
                "pred_class": IMAGENETTE_CLASSES[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probs": probs.tolist(),
                "maps": {},
            }
            if "gradcam" in args.methods:
                entry["maps"]["gradcam"] = quantize(gradcam(x, pred_idx))
            if "scorecam" in args.methods:
                entry["maps"]["scorecam"] = quantize(score_cam(model, x, target_layer, pred_idx, dev))
            if "saliency" in args.methods:
                entry["maps"]["saliency"] = quantize(saliency_map(model, x, pred_idx))
            records.append(entry)
    finally:
        gradcam.close()

    (out / f"{args.model}.json").write_text(json.dumps(records))
    print(f"wrote {len(records)} entries to {out / f'{args.model}.json'}")


if __name__ == "__main__":
    main()
