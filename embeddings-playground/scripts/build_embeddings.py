"""Encode each indexed COCO image with CLIP and save the 512D vectors.

Images are fetched from images.cocodataset.org and embedded in one pass rather
than downloaded to disk first. The frontend hotlinks the same URLs, so the
pixels are only needed here, and ~20k images at full size is several GB that
would be deleted straight afterwards.

A thread pool does the fetching because the job is network-bound: the GPU
finishes a batch of 64 far faster than 64 images arrive.
"""

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_repo = Path(__file__).resolve().parents[1]
OUT_DIR = _repo / "data"

CLIP_MODEL = "openai/clip-vit-base-patch32"
TIMEOUT = 20


def fetch_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def embed_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # transformers 5.x returns a pooled output. pooler_output is the
        # projected 512D image embedding, matching CLAP's vector width
        out = model.get_image_features(**inputs)
    embs = out.pooler_output
    # CLIP leaves the projected features unnormalized, so normalize here and
    # cosine similarity in the frontend reduces to a dot product
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(OUT_DIR / "index.parquet"))
    parser.add_argument("--out", default=str(OUT_DIR / "embeddings.npz"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N images")
    parser.add_argument("--vector-format", default="float16", choices=["float16", "float32"],
                        help="dtype for the compact web vector blob")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CLIP model: {CLIP_MODEL}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    index = pd.read_parquet(args.index)
    if args.limit:
        index = index.head(args.limit)
    print(f"Images to embed: {len(index)}")

    ids = index["image_id"].tolist()
    urls = index["coco_url"].tolist()

    all_ids: list[int] = []
    all_embeddings: list[np.ndarray] = []
    failed: list[int] = []

    batch_ids: list[int] = []
    batch_images: list[Image.Image] = []

    def flush() -> None:
        if not batch_images:
            return
        embs = embed_batch(model, processor, batch_images, device)
        all_ids.extend(batch_ids)
        all_embeddings.append(embs)
        batch_ids.clear()
        batch_images.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for image_id, image in tqdm(
            zip(ids, pool.map(fetch_image, urls)), total=len(ids), desc="embedding"
        ):
            if image is None:
                failed.append(image_id)
                continue
            batch_ids.append(image_id)
            batch_images.append(image)
            if len(batch_ids) >= args.batch_size:
                flush()

    flush()

    if not all_embeddings:
        raise RuntimeError("No embeddings produced. Check network access to images.cocodataset.org")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    ids_arr = np.array(all_ids, dtype=np.int64)

    np.savez(out, ids=ids_arr, embeddings=embeddings)
    print(f"Saved {len(ids_arr)} embeddings to {out} with shape={embeddings.shape}")

    write_compact(out, ids_arr, embeddings, args.vector_format)

    if failed:
        fail_path = out.with_suffix(".failed.json")
        fail_path.write_text(json.dumps(failed))
        print(f"  {len(failed)} images failed at {fail_path}")


def write_compact(out_npz: Path, ids: np.ndarray, embeddings: np.ndarray, dtype: str) -> None:
    """Write a flat vector blob plus an id-order sidecar for the web frontend.

    The blob is row-major (N, dim) in little-endian `dtype`, row i belongs to
    ids[i]. The frontend reads the whole blob once and builds id -> row from the
    sidecar, so the blob layout does not need to match viz.json point order.
    """
    np_dtype = np.dtype(dtype).newbyteorder("<")
    vec = embeddings.astype(np_dtype, copy=False)
    blob = out_npz.with_name("vectors.bin")
    blob.write_bytes(vec.tobytes(order="C"))
    meta = {"dim": int(embeddings.shape[1]), "dtype": dtype, "count": int(len(ids)),
            "ids": ids.astype(int).tolist()}
    out_npz.with_name("vectors.meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"Saved compact {dtype} vectors to {blob} ({blob.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
