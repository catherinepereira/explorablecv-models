# CV Interpretability Model

Training code for three ImageNette classifiers (Custom CNN, ResNet-18, ViT-S) plus the scripts that precompute interpretability outputs (Grad-CAM, Score-CAM, saliency, LIME, and attention rollout for ViT) and UMAP embeddings for the [cv-interpretability](../cv-interpretability) viewer.

The viewer is a static site, so this repo emits everything as JSON and a bundle script copies the artifacts into the frontend's `public/` directory.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CUDA is assumed for training. Inference paths in this repo will fall back to CPU, but don't try to train ViT-S on CPU.

## Data

ImageNette (160px or full-res variant, the loader resizes to 224 either way).

```bash
python scripts/download_data.py
python scripts/preprocess_imagenette.py
```

The preprocess step renames the WNID directories (`n01440764` etc.) to readable class names so the rest of the pipeline reads `data/raw/imagenette2/train/tench/...`.

## Train all three models

```bash
python scripts/train.py --model custom_cnn --epochs 20
python scripts/train.py --model resnet18   --epochs 15
python scripts/train.py --model vit_s      --epochs 10 --lr 1e-4 --batch 32
```

Each writes `exports/<model>.pt` and `exports/<model>.meta.json`. Add `--smoke` for a 2-epoch sanity run.

## Evaluate

```bash
python scripts/evaluate.py --model custom_cnn
python scripts/evaluate.py --model resnet18
python scripts/evaluate.py --model vit_s
```

Writes `exports/<model>.eval.json` with overall and per-class accuracy.

## Precompute interpretability outputs

These are the inputs the viewer renders. Run after training, in this order:

```bash
python scripts/sample_images.py --per-class 5

for model in custom_cnn resnet18 vit_s; do
  python scripts/compute_cams.py --model $model
  python scripts/compute_lime.py --model $model
  python scripts/compute_umap.py --model $model --max-images 2000
done

python scripts/compute_rollout.py --model vit_s
```

CAM outputs are quantized to a 64x64 uint8 grid per image, small enough that all three models' maps for 50 sample images fit comfortably in a single JSON file.

## Bundle for the frontend

```bash
python scripts/bundle_for_frontend.py --frontend ../cv-interpretability/public
```

This step copies:
- `samples/*.jpg`           thumbnail images
- `data/classes.json`       ordered class labels
- `data/samples.json`       sample manifest
- `data/models.json`        per-model accuracy and param counts
- `data/cams.json`          Grad-CAM / Score-CAM / saliency maps for every sample
- `data/lime.json`          LIME heatmaps for every sample
- `data/rollout.json`       attention rollout maps (ViT only)
- `data/umap.json`          UMAP point clouds for every model

## Architecture reference

| Model      | Input    | Backbone               | Params | Optimizer | Notes                  |
|------------|----------|------------------------|--------|-----------|------------------------|
| custom_cnn | 3x224x224| 4 conv blocks + GAP    | ~4.5M  | AdamW     | trained from scratch   |
| resnet18   | 3x224x224| torchvision ResNet-18  | ~11.2M | AdamW     | ImageNet pretrained    |
| vit_s      | 3x224x224| timm vit_small_patch16 | ~22.0M | AdamW     | ImageNet pretrained    |

## Other Notes

- **Preprocessing parity.** The viewer ships precomputed maps, so inference-time normalization is entirely in this repo (`model/constants.py` and `model/dataset.py`).
- **CAM hooks on ViT.** Grad-CAM on ViT-S targets `blocks[-1].norm1`, reshapes the CLS-token-excluded tokens into a 14x14 grid, then upsamples. Pick a different layer and the spatial layout assumption breaks.
- **Score-CAM is slow per image.** It does one forward pass per activation channel, which for ViT-S is ~384 forwards per image. Across 50 samples on a 4070 SUPER it runs in about 60-90 seconds per model. Larger sample sets scale linearly.
- **Custom CNN needs `inplace=False` on its ReLUs.** Grad-CAM attaches a backward hook on the last conv layer, and the in-place ReLU two layers downstream raises "Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace."
- **Attention rollout patches timm's `Attention.forward`** to disable `fused_attn` and stash the per-block attention matrix. The patched forward accepts arbitrary kwargs because newer timm passes `attn_mask` and `is_causal` through.
- **LIME randomness.** `lime_image` doesn't expose a seed, so results jitter run-to-run. Pin `num_samples` and accept it.
- **UMAP determinism.** `random_state=0` plus single-threaded UMAP gives stable layouts. Multi-threaded UMAP is faster but the layout shifts between runs, which is confusing when comparing.
