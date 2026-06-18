"""Export ViT-tiny to ONNX with attention and patch embeddings as named outputs.

The vit-playground frontend feeds a [1,3,224,224] ImageNet-normalized tensor and
reads back `logits`, the `patch_embeddings` [1,196,192] (the learned projection
of each patch, before CLS and positions), and 12 attention tensors named
attention_0..attention_11, each [1,3,197,197] (batch, heads, tokens, tokens =
1 CLS + 196 patches). The wrapper exposes the attentions and the patch-embedding
layer, which the model returns with output_attentions=True / as an intermediate
but does not surface as ONNX outputs on its own. ViT-tiny (not DeiT) is used so
there is a single CLS token and 197 tokens, matching the 197-token grid the
frontend teaches. DeiT adds a second distillation token.
"""

import json
from pathlib import Path

import torch
from transformers import ViTForImageClassification

MODEL_ID = "WinKawaks/vit-tiny-patch16-224"
OUT_DIR = Path(__file__).resolve().parent.parent / "exports"
FRONTEND_DIR = (
    Path(__file__).resolve().parents[3]
    / "explorables-mono/apps/vit-playground/public/models/vit-tiny"
)


class ViTWithAttention(torch.nn.Module):
    def __init__(self, model: ViTForImageClassification):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        # The learned per-patch projection, [1, 196, 192], before CLS/position
        patch_embeddings = self.model.vit.embeddings.patch_embeddings(
            pixel_values
        )
        out = self.model(pixel_values=pixel_values, output_attentions=True)
        return (out.logits, patch_embeddings, *out.attentions)


def main() -> None:
    # Eager attention so output_attentions returns the weight matrices. SDPA and
    # flash backends fuse the softmax and never materialize them.
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    )
    model.eval()
    wrapper = ViTWithAttention(model)

    n_layers = model.config.num_hidden_layers
    attn_names = [f"attention_{i}" for i in range(n_layers)]
    output_names = ["logits", "patch_embeddings", *attn_names]

    dummy = torch.randn(1, 3, 224, 224)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "model.onnx"

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (OUT_DIR / "imagenet_classes.json").write_text(json.dumps(labels))

    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"outputs: {output_names}")

    # Copy into the frontend's public dir so the demo can fetch it
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "imagenet_classes.json"):
        (FRONTEND_DIR / name).write_bytes((OUT_DIR / name).read_bytes())
    print(f"copied to {FRONTEND_DIR}")


if __name__ == "__main__":
    main()
