"""Precompute YOLOv8n-seg instance results for the frontend's bundled samples.

Runs the exported ONNX model over every photo in the frontend's public/samples,
mirroring the browser decode in src/seg/instInference.ts (letterbox to 640,
conf 0.25, per-class NMS 0.45, top 12, sigmoid mask threshold 0.5, lowest score
painted first). Writes per sample:
  precomputed/<name>_inst.png   instance id + 1 per pixel, grayscale, 800x600
  precomputed/<name>_inst.json  [{classId, score}] ordered by descending score
Re-run after changing the sample photos or re-exporting the model.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT.parents[1] / "explorablecv" / "apps" / "cv-segmentation-playground"
SAMPLES_DIR = FRONTEND_DIR / "public" / "samples"
OUT_DIR = SAMPLES_DIR / "precomputed"
MODEL = FRONTEND_DIR / "public" / "models" / "yolov8n_seg" / "model.onnx"

# Must match the frontend's working buffer (IMG_W x IMG_H) and instInference.ts
IMG_W, IMG_H = 800, 600
INPUT_SIZE = 640
PROTO_SIZE = 160
NUM_CLASSES = 80
NUM_COEFFS = 32
CONF_THRESHOLD = 0.25
NMS_IOU = 0.45
MASK_THRESHOLD = 0.5
MAX_INSTANCES = 12


def letterbox(img: Image.Image) -> tuple[np.ndarray, float, int, int]:
    scale = min(INPUT_SIZE / img.width, INPUT_SIZE / img.height)
    draw_w, draw_h = round(img.width * scale), round(img.height * scale)
    pad_x = (INPUT_SIZE - draw_w) // 2
    pad_y = (INPUT_SIZE - draw_h) // 2
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(img.resize((draw_w, draw_h)), (pad_x, pad_y))
    x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, scale, pad_x, pad_y


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    iy = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def decode(head: np.ndarray) -> list[dict]:
    # head is [4 + classes + coeffs, 8400]
    scores = head[4 : 4 + NUM_CLASSES]
    best_class = scores.argmax(axis=0)
    best_score = scores.max(axis=0)
    keep = best_score >= CONF_THRESHOLD
    cands = [
        {
            "box": head[:4, a],
            "classId": int(best_class[a]),
            "score": float(best_score[a]),
            "coeffs": head[4 + NUM_CLASSES :, a],
        }
        for a in np.flatnonzero(keep)
    ]
    cands.sort(key=lambda c: -c["score"])

    kept: list[dict] = []
    for cand in cands:
        clash = any(
            k["classId"] == cand["classId"] and box_iou(k["box"], cand["box"]) >= NMS_IOU
            for k in kept
        )
        if not clash:
            kept.append(cand)
        if len(kept) == MAX_INSTANCES:
            break
    return kept


def build_map(kept: list[dict], protos: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    inst_map = np.zeros((IMG_H, IMG_W), np.uint8)
    proto_scale = PROTO_SIZE / INPUT_SIZE
    logit_floor = np.log(MASK_THRESHOLD / (1 - MASK_THRESHOLD))

    # Per-pixel prototype coordinates for the whole working image
    xs = np.minimum(PROTO_SIZE - 1, ((np.arange(IMG_W) * scale + pad_x) * proto_scale).astype(int))
    ys = np.minimum(PROTO_SIZE - 1, ((np.arange(IMG_H) * scale + pad_y) * proto_scale).astype(int))

    for i in range(len(kept) - 1, -1, -1):
        inst = kept[i]
        cx, cy, w, h = inst["box"]
        x0 = max(0, int((cx - w / 2 - pad_x) / scale))
        y0 = max(0, int((cy - h / 2 - pad_y) / scale))
        x1 = min(IMG_W, int(np.ceil((cx + w / 2 - pad_x) / scale)))
        y1 = min(IMG_H, int(np.ceil((cy + h / 2 - pad_y) / scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        logits = np.einsum(
            "k,kyx->yx",
            inst["coeffs"],
            protos[:, ys[y0:y1][:, None], xs[x0:x1][None, :]],
        )
        region = inst_map[y0:y1, x0:x1]
        region[logits >= logit_floor] = i + 1
    return inst_map


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    for photo in sorted(SAMPLES_DIR.glob("*.jpg")):
        img = Image.open(photo).convert("RGB").resize((IMG_W, IMG_H))
        x, scale, pad_x, pad_y = letterbox(img)
        head, protos = sess.run(None, {input_name: x})
        kept = decode(head[0])
        inst_map = build_map(kept, protos[0], scale, pad_x, pad_y)

        png = OUT_DIR / f"{photo.stem}_inst.png"
        Image.fromarray(inst_map, mode="L").save(png, optimize=True)
        meta = [{"classId": k["classId"], "score": round(k["score"], 4)} for k in kept]
        (OUT_DIR / f"{photo.stem}_inst.json").write_text(json.dumps(meta))
        labels = [f"{k['classId']}:{k['score']:.2f}" for k in kept]
        covered = int((inst_map > 0).sum()) * 100 // inst_map.size
        print(f"{photo.stem}: {len(kept)} instances [{', '.join(labels)}] {covered}% covered -> {png.name}")


if __name__ == "__main__":
    main()
