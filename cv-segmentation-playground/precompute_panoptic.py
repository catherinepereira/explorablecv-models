"""Precompute Mask2Former panoptic results for the frontend's bundled samples.

Runs the exported int8 ONNX over every photo in the frontend's public/samples,
mirroring the browser decode in src/seg/panopticInference.ts (512x512 input,
softmax class scores over 133 classes + no-object, sigmoid masks at 128x128,
per-pixel argmax of score x mask over kept queries, stuff segments fused per
class). Writes per sample:
  precomputed/<name>_pan.png   segment id per pixel, grayscale, 128x128
  precomputed/<name>_pan.json  [{classId, score}] per segment, id = index + 1
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
MODEL = FRONTEND_DIR / "public" / "models" / "mask2former_panoptic" / "model.onnx"

# Must match the frontend's panopticInference.ts
INPUT_SIZE = 512
MASK_SIZE = 128
NUM_CLASSES = 133
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
MIN_SEGMENT_PIXELS = 32
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode(class_logits: np.ndarray, mask_logits: np.ndarray):
    """Mirror of the frontend decode. Returns (seg_map 128x128, segments)."""
    probs = softmax(class_logits, axis=-1)  # [100, 134]
    labels = probs[:, :NUM_CLASSES].argmax(axis=-1)
    scores = probs[np.arange(len(labels)), labels]
    keep = scores > SCORE_THRESHOLD
    if not keep.any():
        return np.zeros((MASK_SIZE, MASK_SIZE), np.uint8), []

    kept_ids = np.flatnonzero(keep)
    masks = 1 / (1 + np.exp(-mask_logits[kept_ids]))  # [K, 128, 128]
    kept_scores = scores[kept_ids]
    kept_labels = labels[kept_ids]

    # Per-pixel best query by score-weighted mask probability. A pixel joins a
    # segment only when that query's mask probability clears the threshold
    weighted = masks * kept_scores[:, None, None]
    best_q = weighted.argmax(axis=0)
    best_prob = np.take_along_axis(masks, best_q[None], axis=0)[0]
    assigned = best_prob >= MASK_THRESHOLD

    # Fuse stuff queries of the same class into one segment, keep things apart
    seg_map = np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
    segments: list[dict] = []
    stuff_segment: dict[int, int] = {}
    for qi in range(len(kept_ids)):
        pix = assigned & (best_q == qi)
        if pix.sum() < MIN_SEGMENT_PIXELS:
            continue
        cls = int(kept_labels[qi])
        score = float(kept_scores[qi])
        if cls >= 80 and cls in stuff_segment:
            sid = stuff_segment[cls]
            segments[sid - 1]["score"] = max(segments[sid - 1]["score"], score)
        else:
            segments.append({"classId": cls, "score": round(score, 4)})
            sid = len(segments)
            if cls >= 80:
                stuff_segment[cls] = sid
        seg_map[pix] = sid
    return seg_map, segments


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    for photo in sorted(SAMPLES_DIR.glob("*.jpg")):
        img = Image.open(photo).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        class_logits, mask_logits = sess.run(
            None, {input_name: x.transpose(2, 0, 1)[None].astype(np.float32)}
        )
        seg_map, segments = decode(class_logits[0], mask_logits[0])

        png = OUT_DIR / f"{photo.stem}_pan.png"
        Image.fromarray(seg_map, mode="L").save(png, optimize=True)
        (OUT_DIR / f"{photo.stem}_pan.json").write_text(json.dumps(segments))
        desc = [f"{s['classId']}:{s['score']:.2f}" for s in segments]
        print(f"{photo.stem}: {len(segments)} segments [{', '.join(desc)}] -> {png.name}")


if __name__ == "__main__":
    main()
