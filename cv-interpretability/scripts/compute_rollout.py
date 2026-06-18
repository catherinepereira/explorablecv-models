import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from _common import device
from model.builders import build_model, MODEL_NAMES
from model.constants import IMAGENETTE_CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def patched_attn_forward(self, x: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = self.q_norm(q)
    k = self.k_norm(k)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    self._stashed_attn = attn.detach()
    attn = self.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def install_attn_capture(model):
    handles = []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        blk.attn._stashed_attn = None
        bound = patched_attn_forward.__get__(blk.attn, type(blk.attn))
        old = blk.attn.forward
        blk.attn.forward = bound
        handles.append((blk.attn, old))
    return handles


def restore_attn(handles):
    for attn, old in handles:
        attn.forward = old


def rollout(model, x: torch.Tensor, discard_ratio: float = 0.0):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    mats = []
    for blk in model.blocks:
        a = blk.attn._stashed_attn
        if a is None:
            continue
        a = a.mean(dim=1)  # average over heads -> [B, T, T]
        if discard_ratio > 0:
            flat = a.view(a.size(0), -1)
            n_drop = int(flat.size(1) * discard_ratio)
            if n_drop > 0:
                _, idx = flat.topk(n_drop, largest=False)
                flat.scatter_(1, idx, 0)
            a = flat.view_as(a)
        eye = torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        mats.append(a)
    out = mats[0]
    for m in mats[1:]:
        out = m @ out
    cls_row = out[0, 0, 1:]
    n = int(cls_row.shape[0] ** 0.5)
    cam = cls_row.reshape(1, 1, n, n)
    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs


def quantize(map2d: np.ndarray, size: int = 64) -> list[list[int]]:
    h, w = map2d.shape
    factor_h = h // size
    factor_w = w // size
    down = map2d[: factor_h * size, : factor_w * size].reshape(size, factor_h, size, factor_w).mean(axis=(1, 3))
    arr = (down * 255).astype(np.uint8)
    return arr.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_s", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", default="exports/samples")
    parser.add_argument("--out", default="exports/rollout")
    parser.add_argument("--discard-ratio", type=float, default=0.0)
    args = parser.parse_args()

    if args.model != "vit_s":
        raise SystemExit("attention rollout only applies to ViT models")

    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, pretrained=False).to(dev)
    ckpt = args.checkpoint or f"exports/{args.model}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    handles = install_attn_capture(model)

    manifest = json.loads((Path(args.samples) / "manifest.json").read_text())
    records = []
    try:
        for item in tqdm(manifest, desc=args.model):
            img_path = Path(args.samples) / item["file"]
            x = load_image(img_path).to(dev)
            cam, probs = rollout(model, x, discard_ratio=args.discard_ratio)
            pred_idx = int(probs.argmax())
            records.append({
                "id": item["id"],
                "model": args.model,
                "true_class": item["class"],
                "pred_class": IMAGENETTE_CLASSES[pred_idx],
                "rollout": quantize(cam),
            })
    finally:
        restore_attn(handles)

    (out / f"{args.model}.json").write_text(json.dumps(records))
    print(f"wrote {len(records)} rollout records")


if __name__ == "__main__":
    main()
