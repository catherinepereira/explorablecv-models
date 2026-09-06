"""Export YOLOv8n-seg to ONNX for cv-segmentation-playground's instance section.

The frontend feeds a letterboxed [1,3,640,640] RGB tensor (scaled by 1/255) and
reads two outputs: [1,116,8400] box rows ([cx,cy,w,h,class0..class79,coeff0..
coeff31]) and [1,32,160,160] mask prototypes. Each kept box's mask is
sigmoid(coeffs . protos) cropped to the box. NMS runs client-side, so nms=False.
"""

from pathlib import Path

from ultralytics import YOLO

FRONTEND_DIR = (
    Path(__file__).resolve().parents[2]
    / "explorablecv/apps/cv-segmentation-playground/public/models/yolov8n_seg"
)


def main() -> None:
    model = YOLO("yolov8n-seg.pt")
    onnx_path = model.export(
        format="onnx", imgsz=640, opset=12, nms=False, simplify=True
    )

    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    dest = FRONTEND_DIR / "model.onnx"
    dest.write_bytes(Path(onnx_path).read_bytes())
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
