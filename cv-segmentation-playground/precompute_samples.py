"""Precompute segmentation class maps for the frontend's bundled samples.

Runs the exported ONNX model over every photo in the frontend's public/samples
and writes each argmax class map as a grayscale PNG (class index in the single
channel) to public/samples/precomputed/<name>.png. The frontend loads these
instead of running the model, so picking a sample shows a result instantly.
Re-run after changing the sample photos or re-exporting the model.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT.parents[1] / "explorablecv" / "apps" / "cv-segmentation-playground"
SAMPLES_DIR = FRONTEND_DIR / "public" / "samples"
OUT_DIR = SAMPLES_DIR / "precomputed"
MODEL = FRONTEND_DIR / "public" / "models" / "lraspp_mobilenet_v3" / "model.onnx"

# Must match the frontend's SEG_INPUT_SIZE and ImageNet normalization
INPUT_SIZE = 520
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    for photo in sorted(SAMPLES_DIR.glob("*.jpg")):
        img = Image.open(photo).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        logits = sess.run(None, {input_name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        class_map = logits.argmax(axis=0).astype(np.uint8)
        out = OUT_DIR / f"{photo.stem}.png"
        Image.fromarray(class_map, mode="L").save(out, optimize=True)
        present = sorted(int(c) for c in np.unique(class_map))
        print(f"{photo.stem}: {class_map.shape} classes {present} -> {out.name} ({out.stat().st_size // 1024}K)")


if __name__ == "__main__":
    main()
