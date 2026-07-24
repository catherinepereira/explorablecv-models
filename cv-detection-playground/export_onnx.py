"""Export YOLOv8n to ONNX for cv-detection-playground.

The frontend feeds a letterboxed [1,3,640,640] RGB tensor (scaled by 1/255) and
reads the raw [1,84,8400] output: rows are [cx,cy,w,h,class0..class79], columns
are 8400 anchors. NMS runs client-side so the confidence and NMS-IoU sliders
work, which is why nms=False here.
"""

from pathlib import Path

from ultralytics import YOLO

FRONTEND_DIR = (
    Path(__file__).resolve().parents[2]
    / "explorablecv/apps/cv-detection-playground/public/models/yolov8n"
)


def main() -> None:
    model = YOLO("yolov8n.pt")
    onnx_path = model.export(
        format="onnx", imgsz=640, opset=12, nms=False, simplify=True
    )

    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    dest = FRONTEND_DIR / "model.onnx"
    dest.write_bytes(Path(onnx_path).read_bytes())
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
