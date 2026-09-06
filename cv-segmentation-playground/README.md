# cv-segmentation-export

Exports the three models behind the [cv-segmentation-playground](../../explorablecv/apps/cv-segmentation-playground)
demo and precomputes results for its bundled sample photos.

## Semantic model

`export_onnx.py` loads torchvision's `lraspp_mobilenet_v3_large` with the
`COCO_WITH_VOC_LABELS_V1` weights (21 Pascal VOC classes), wraps it to return the
main logits tensor instead of the usual dict, and exports. The demo does argmax
and upsampling itself, so the aux classifier is dropped and no softmax is baked
in. It writes `model.onnx` into the demo's `public/models/lraspp_mobilenet_v3/`.

`deeplabv3_mobilenet_v3_large` is drop-in compatible (same I/O) if more accuracy
is wanted over speed.

- Input: `input` `[1, 3, 520, 520]`, ImageNet-normalized RGB, NCHW
- Output: `out` `[1, 21, H, W]`, raw per-pixel class logits. Index 0 is
  background. The demo argmaxes over the class axis and upsamples to the image.

## Instance model

`export_instance_onnx.py` exports Ultralytics YOLOv8n-seg with `nms=False` and
writes `model.onnx` into the demo's `public/models/yolov8n_seg/`. The demo
letterboxes to 640, decodes, and runs NMS itself.

- Input: `images` `[1, 3, 640, 640]`, RGB scaled by 1/255, NCHW
- Output `output0`: `[1, 116, 8400]`, rows are
  `[cx, cy, w, h, class0..class79, coeff0..coeff31]`, columns are candidates
- Output `output1`: `[1, 32, 160, 160]` mask prototypes. A kept box's mask is
  `sigmoid(coeffs . protos)` cropped to the box.

## Panoptic model

`export_panoptic_onnx.py` exports Mask2Former (swin-tiny, COCO panoptic) from
the HuggingFace checkpoint, casts the float64 GridSample grids the torch trace
emits to float32 (onnxruntime only registers GridSample for float), and
dynamically quantizes to int8. The fp32 intermediate stays in this folder,
only the quantized `model.onnx` lands in the demo's
`public/models/mask2former_panoptic/`.

- Input: `input` `[1, 3, 512, 512]`, ImageNet-normalized RGB, NCHW
- Output `class_logits`: `[1, 100, 134]`, one distribution per query over 133
  COCO panoptic classes plus no-object
- Output `mask_logits`: `[1, 100, 128, 128]` per-query mask logits. The demo
  keeps queries scoring over 0.5, assigns each pixel to the best
  score-weighted mask, and fuses stuff queries per class.

## Sample precompute

`precompute_samples.py` runs the semantic ONNX over the demo's
`public/samples/*.jpg` and writes each argmax class map as a grayscale PNG to
`public/samples/precomputed/`. `precompute_instances.py` and
`precompute_panoptic.py` do the same for the instance and panoptic models,
mirroring the browser decodes, and write `<name>_inst.png` / `<name>_inst.json`
and `<name>_pan.png` / `<name>_pan.json`. The demo loads these instead of
running the models, so picking a sample shows results instantly. Re-run all
three after changing the sample photos or re-exporting a model.

## Run

```bash
python export_onnx.py
python export_instance_onnx.py
python export_panoptic_onnx.py
python precompute_samples.py
python precompute_instances.py
python precompute_panoptic.py
```
