# embeddings-playground-export

Builds the embedding data behind the [embeddings-playground](../../explorablecv/apps/embeddings-playground)
demo: CLIP encodes each COCO image to 512D, UMAP reduces to 2D and 3D.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

### 1. `scripts/build_index.py`

Picks the images to embed from the COCO 2017 instance annotations. Each image is
labeled by its largest annotated instance, dropping images whose largest instance
covers under 10% of the frame. A per-class cap balances the sample, since a person
appears in about half of all COCO images.

Reads `instances_train2017.json` straight out of `annotations_trainval2017.zip` in
the shared `data/coco-val2017/` cache, so no extra download is needed. Writes
`data/index.parquet` with about 20,000 images across all 80 categories.

### 2. `scripts/build_embeddings.py`

Encodes each indexed image with CLIP (`openai/clip-vit-base-patch32`) to 512D.
Images are fetched from `images.cocodataset.org` and embedded in one streaming
pass rather than downloaded first, since the frontend hotlinks the same URLs and
the pixels are only needed here.

Outputs:

- `data/embeddings.npz`: full-precision embeddings + ids
- `data/vectors.bin`: flat row-major vector blob for the web frontend, float16 by default (`--vector-format float32` for full precision)
- `data/vectors.meta.json`: `{dim, dtype, count, ids}` so the frontend maps image id to row

### 3. `scripts/build_viz_data.py`

Reduces the CLIP embeddings to 2D and 3D with UMAP, maps categories to colors by
supercategory hue, writes `data/viz.json`, and saves the UMAP models to `data/`.
Also writes `data/class_vectors.json`, the L2-normalized mean embedding per category.

### 4. Copy to the frontend

The frontend serves these from its `public/` directory:

```bash
cp data/viz.json data/vectors.bin data/vectors.meta.json data/class_vectors.json \
  ../../explorablecv/apps/embeddings-playground/public/
```

## Credits

Images and annotations are from [COCO](https://cocodataset.org/). Annotations are
licensed CC BY 4.0, images are subject to their original Flickr terms. The demo
hotlinks the images rather than redistributing them.
