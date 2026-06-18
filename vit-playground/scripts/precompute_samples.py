"""Precompute ViT-tiny outputs for the bundled sample images.

The demo runs the model live for uploaded images, but the samples ship with
their results so picking one is instant and needs no model download. For each
sample this resizes it to 224x224, runs the exported ONNX model, and writes a
compact JSON with the logits, patch embeddings, and the rolled-up attention
matrix (the demo's attention section consumes the rollout, not the 12 raw
per-layer tensors, so shipping the rollout keeps each file ~155KB instead of
~5.6MB).
"""

import base64
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[3] / "explorables-mono/apps/vit-playground"
SAMPLES_DIR = ROOT / "public/samples"
MODEL_PATH = ROOT / "public/models/vit-tiny/model.onnx"
OUT_DIR = ROOT / "public/samples/precomputed"

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SAMPLES = ["springer", "french-horn", "chainsaw", "church", "garbage-truck", "golf-ball"]


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = (np.asarray(img).astype(np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]  # NCHW


def attention_rollout(attentions: list[np.ndarray]) -> np.ndarray:
    # Match the frontend: average heads, add identity, renormalize rows, then
    # multiply layers deepest-first (rolled = layer_l @ rolled)
    def step(att: np.ndarray) -> np.ndarray:
        a = att[0].mean(axis=0)  # average over heads -> [197, 197]
        a = a + np.eye(a.shape[0], dtype=a.dtype)
        return a / a.sum(axis=1, keepdims=True)

    rolled = step(attentions[0])
    for att in attentions[1:]:
        rolled = step(att) @ rolled
    return rolled


def f16_b64(arr: np.ndarray) -> str:
    # float16 halves the payload and is plenty for display; base64 keeps the
    # JSON compact and avoids a giant array of decimal strings
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


def main() -> None:
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sid in SAMPLES:
        x = preprocess(SAMPLES_DIR / f"{sid}.jpg")
        out = dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x})))
        attentions = [out[f"attention_{i}"] for i in range(12)]
        rolled = attention_rollout(attentions).astype(np.float32)

        record = {
            "logits": f16_b64(out["logits"][0]),
            "patchEmbeddings": f16_b64(out["patch_embeddings"][0].reshape(-1)),
            "rollout": f16_b64(rolled.reshape(-1)),
            "tokens": rolled.shape[0],
            "embedDim": out["patch_embeddings"].shape[-1],
        }
        path = OUT_DIR / f"{sid}.json"
        path.write_text(json.dumps(record))
        print(f"{sid}: {path.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
