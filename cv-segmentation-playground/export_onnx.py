"""Export LR-ASPP MobileNetV3 (VOC, 21 classes) to ONNX for cv-segmentation.

The frontend feeds a [1,3,520,520] ImageNet-normalized tensor and reads back the
raw [1,21,H,W] class logits from output "out". It does argmax + upsample itself,
so the aux classifier is dropped and no softmax is exported.
"""

from pathlib import Path

import torch
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

SIZE = 520
FRONTEND_DIR = (
    Path(__file__).resolve().parents[2]
    / "explorablecv/apps/cv-segmentation-playground/public/models/lraspp_mobilenet_v3"
)


class SegWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The torchvision seg models return a dict, the frontend wants the main
        # logits tensor only
        return self.model(x)["out"]


def main() -> None:
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()
    wrapper = SegWrapper(model)

    dummy = torch.randn(1, 3, SIZE, SIZE)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = FRONTEND_DIR / "model.onnx"

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
