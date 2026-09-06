"""Export Mask2Former (swin-tiny, COCO panoptic) to ONNX for the panoptic section.

The frontend feeds a [1,3,512,512] ImageNet-normalized tensor and reads two
outputs: class_queries_logits [1,100,134] (133 COCO panoptic classes + no
object) and masks_queries_logits [1,100,128,128]. Panoptic post-processing
(per-pixel argmax of class score x mask sigmoid over kept queries) runs
client-side. The fp32 export is dynamically quantized to int8 to keep the
download reasonable.
"""

from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper, shape_inference
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import Mask2FormerForUniversalSegmentation

SIZE = 512
CHECKPOINT = "facebook/mask2former-swin-tiny-coco-panoptic"
FRONTEND_DIR = (
    Path(__file__).resolve().parents[2]
    / "explorablecv/apps/cv-segmentation-playground/public/models/mask2former_panoptic"
)


class PanopticWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(pixel_values=x)
        return out.class_queries_logits, out.masks_queries_logits


def cast_double_grids(path: Path) -> None:
    """Cast float64 GridSample grids to float32 in place.

    The torch trace emits the deformable-attention sampling grids as double,
    and onnxruntime only registers GridSample for float, so the session fails
    with NOT_IMPLEMENTED until the grid inputs are cast.
    """
    model = onnx.load(str(path))
    inferred = shape_inference.infer_shapes(model)
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in list(inferred.graph.value_info) + list(inferred.graph.input)
    }
    patched = 0
    nodes = list(model.graph.node)
    for idx, node in enumerate(nodes):
        if node.op_type != "GridSample":
            continue
        grid = node.input[1]
        if types.get(grid) != TensorProto.DOUBLE:
            continue
        cast_out = f"{grid}_f32"
        cast = helper.make_node(
            "Cast", [grid], [cast_out], to=TensorProto.FLOAT, name=f"{grid}_cast"
        )
        model.graph.node.insert(list(model.graph.node).index(node), cast)
        node.input[1] = cast_out
        patched += 1
    onnx.save(model, str(path))
    print(f"cast {patched} GridSample grids to float32")


def main() -> None:
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CHECKPOINT)
    model.eval()
    wrapper = PanopticWrapper(model)

    dummy = torch.randn(1, 3, SIZE, SIZE)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    # The fp32 intermediate stays out of the frontend's public dir, only the
    # quantized model ships
    fp32_path = Path(__file__).resolve().parent / "mask2former_fp32.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(fp32_path),
        input_names=["input"],
        output_names=["class_logits", "mask_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")
    cast_double_grids(fp32_path)

    int8_path = FRONTEND_DIR / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    print(f"wrote {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
