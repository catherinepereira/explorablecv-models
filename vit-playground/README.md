# vit-playground-export

Exports ViT-tiny to ONNX for the [vit-playground](../../explorables-mono/apps/vit-playground)
demo, with the intermediate tensors the demo visualizes added as named outputs.

## What it does

`scripts/export_onnx.py` loads `WinKawaks/vit-tiny-patch16-224` (a ViT with
a single CLS token, so the token grid is 197 = 1 CLS + 196 patches, matching what
the demo teaches).

It writes `model.onnx` and `imagenet_classes.json` into the demo's
`public/models/vit-tiny/`.

## ONNX outputs

- `logits`, `[1, 1000]`, ImageNet class scores
- `patch_embeddings`, `[1, 196, 192]`, the learned per-patch projection before
  the CLS token and positions are added (the demo shows these per patch)
- `attention_0` .. `attention_11`, `[1, 3, 197, 197]` each, the post-softmax
  attention per layer, used for attention rollout

Input: `pixel_values` `[1, 3, 224, 224]`, ImageNet-normalized RGB, NCHW.

## Run

```bash
python scripts/export_onnx.py
```
