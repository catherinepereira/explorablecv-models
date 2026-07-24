# cv-detection-export

Exports YOLOv8n to ONNX for the [cv-detection-playground](../../explorablecv/apps/cv-detection-playground)
demo.

## What it does

`export_onnx.py` exports Ultralytics YOLOv8n (COCO-80) to ONNX at 640x640 with
`nms=False`, so the raw per-anchor output is preserved. The demo runs NMS itself,
which is what lets its confidence and NMS-IoU sliders work live. It writes
`model.onnx` into the demo's `public/models/yolov8n/`.

## ONNX contract

- Input: `images` `[1, 3, 640, 640]`, RGB, scaled by 1/255, NCHW. The demo
  letterboxes the source image (gray 114 padding, aspect preserved) before
  feeding it.
- Output: `output0` `[1, 84, 8400]`. Rows are `[cx, cy, w, h, class0..class79]`,
  columns are 8400 anchors. Box coordinates are center-form in letterboxed-input
  pixels (0..640). Class values are per-class confidences with sigmoid applied.
  No objectness row, no built-in NMS.

## Run

```bash
python export_onnx.py
```
