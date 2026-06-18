import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.builders import MODEL_NAMES
from model.constants import IMAGENETTE_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", default="exports")
    parser.add_argument("--frontend", default="../cv-interpretability/public")
    args = parser.parse_args()

    exports = Path(args.exports)
    fe = Path(args.frontend)
    data_dir = fe / "data"
    samples_dst = fe / "samples"
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_dst.mkdir(parents=True, exist_ok=True)

    (data_dir / "classes.json").write_text(json.dumps(IMAGENETTE_CLASSES))

    samples_src = exports / "samples"
    if samples_src.exists():
        for img in samples_src.glob("*.jpg"):
            shutil.copy(img, samples_dst / img.name)
        manifest = json.loads((samples_src / "manifest.json").read_text())
        (data_dir / "samples.json").write_text(json.dumps(manifest))

    model_stats = []
    for name in MODEL_NAMES:
        meta_path = exports / f"{name}.meta.json"
        eval_path = exports / f"{name}.eval.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        entry = {
            "model": name,
            "best_val_acc": meta.get("best_val_acc"),
            "training_examples": meta.get("training_examples"),
        }
        if eval_path.exists():
            entry["val_accuracy"] = json.loads(eval_path.read_text()).get("val_accuracy")
        model_stats.append(entry)

    (data_dir / "models.json").write_text(json.dumps(model_stats, indent=2))

    cams_combined = {}
    cams_dir = exports / "cams"
    if cams_dir.exists():
        for name in MODEL_NAMES:
            f = cams_dir / f"{name}.json"
            if f.exists():
                cams_combined[name] = json.loads(f.read_text())
        (data_dir / "cams.json").write_text(json.dumps(cams_combined))

    rollout_combined = {}
    rollout_dir = exports / "rollout"
    if rollout_dir.exists():
        for name in MODEL_NAMES:
            f = rollout_dir / f"{name}.json"
            if f.exists():
                rollout_combined[name] = json.loads(f.read_text())
        (data_dir / "rollout.json").write_text(json.dumps(rollout_combined))

    lime_combined = {}
    lime_dir = exports / "lime"
    if lime_dir.exists():
        for name in MODEL_NAMES:
            f = lime_dir / f"{name}.json"
            if f.exists():
                lime_combined[name] = json.loads(f.read_text())
        (data_dir / "lime.json").write_text(json.dumps(lime_combined))

    umap_combined = {}
    umap_dir = exports / "umap"
    if umap_dir.exists():
        for name in MODEL_NAMES:
            f = umap_dir / f"{name}.json"
            if f.exists():
                umap_combined[name] = json.loads(f.read_text())
        (data_dir / "umap.json").write_text(json.dumps(umap_combined))

    print(f"bundled artifacts to {fe}")


if __name__ == "__main__":
    main()
