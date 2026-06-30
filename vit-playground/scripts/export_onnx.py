"""Export ViT-tiny to ONNX with attention, Q/K/V, and patch embeddings as outputs.

The vit-playground frontend feeds a [1,3,224,224] ImageNet-normalized tensor and
reads back `logits`, the `patch_embeddings` [1,196,192] (the learned projection
of each patch, before CLS and positions), 12 attention tensors named
attention_0..attention_11, each [1,3,197,197] (batch, heads, tokens, tokens =
1 CLS + 196 patches), and the per-head query/key/value tensors query_0..11,
key_0..11, value_0..11, each [1,3,197,64], that those attention maps come from.

The attention explorer uses Q and K to show the dot-product score behind one
query patch's weight: softmax(q . k^T / sqrt(64)) reproduces attention_L exactly.

The wrapper exposes the attentions and patch-embedding layer the model returns
as intermediates, and hooks the q/k/v projections to capture their outputs.
ViT-tiny (not DeiT) is used so there is a single CLS token and 197 tokens,
matching the 197-token grid the frontend teaches. DeiT adds a second token.
"""

import json
from pathlib import Path

import torch
from transformers import ViTForImageClassification

MODEL_ID = "WinKawaks/vit-tiny-patch16-224"
OUT_DIR = Path(__file__).resolve().parent.parent / "exports"
FRONTEND_DIR = (
    Path(__file__).resolve().parents[3]
    / "explorablecv/apps/vit-playground/public/models/vit-tiny"
)


class ViTWithAttention(torch.nn.Module):
    def __init__(self, model: ViTForImageClassification):
        super().__init__()
        self.model = model
        self.heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.heads

    # Split a projection's [1, tokens, hidden] output into per-head
    # [1, heads, tokens, head_dim], the layout the frontend reads and the one
    # softmax(q . k^T / sqrt(head_dim)) needs to match the attention maps
    def _per_head(self, proj: torch.Tensor) -> torch.Tensor:
        b, n, _ = proj.shape
        return proj.view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, pixel_values: torch.Tensor):
        # Capture each layer's q/k/v projection outputs as the forward runs
        captured: dict[str, torch.Tensor] = {}
        handles = []
        for i, block in enumerate(self.model.vit.layers):
            for role, module in (
                ("query", block.attention.q_proj),
                ("key", block.attention.k_proj),
                ("value", block.attention.v_proj),
            ):

                def hook(_m, _inp, out, key=f"{role}_{i}"):
                    captured[key] = out

                handles.append(module.register_forward_hook(hook))

        # The learned per-patch projection, [1, 196, 192], before CLS/position
        patch_embeddings = self.model.vit.embeddings.patch_embeddings(
            pixel_values
        )
        out = self.model(pixel_values=pixel_values, output_attentions=True)
        for h in handles:
            h.remove()

        n_layers = len(self.model.vit.layers)
        qkv = []
        for role in ("query", "key", "value"):
            for i in range(n_layers):
                qkv.append(self._per_head(captured[f"{role}_{i}"]))
        return (out.logits, patch_embeddings, *out.attentions, *qkv)


def main() -> None:
    # Eager attention so output_attentions returns the weight matrices. SDPA and
    # flash backends fuse the softmax and never materialize them
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    )
    model.eval()
    wrapper = ViTWithAttention(model)

    n_layers = model.config.num_hidden_layers
    attn_names = [f"attention_{i}" for i in range(n_layers)]
    qkv_names = [
        f"{role}_{i}"
        for role in ("query", "key", "value")
        for i in range(n_layers)
    ]
    output_names = ["logits", "patch_embeddings", *attn_names, *qkv_names]

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
