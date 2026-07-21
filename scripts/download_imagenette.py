"""Download Imagenette (or Imagewoof) into the shared data cache.

Imagenette is a 10-class subset of ImageNet from fast.ai. Used by cv-interpretability
and vit-playground. Pass --dest for a private copy: cv-interpretability's preprocess
step renames class dirs in place, which must not touch the shared cache.
"""

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from data_paths import dataset_dir

VARIANTS = {
    "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    "imagewoof": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
}


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract rejecting members that would escape dest (absolute paths,
    ``..`` traversal, or links). Uses the stdlib data filter where available
    (Python 3.12+) and falls back to a manual check on older interpreters."""
    if hasattr(tarfile, "data_filter"):
        tf.extractall(dest, filter="data")
        return
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if dest not in target.parents and target != dest:
            raise ValueError(f"unsafe path in archive: {member.name!r}")
        if member.islnk() or member.issym():
            raise ValueError(f"unsafe link in archive: {member.name!r}")
    tf.extractall(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="imagenette")
    parser.add_argument(
        "--dest",
        default=None,
        help="download here instead of the shared cache",
    )
    args = parser.parse_args()

    url = VARIANTS[args.variant]
    if args.dest:
        out = Path(args.dest)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = dataset_dir(args.variant)
    archive = out / url.rsplit("/", 1)[-1]
    extracted = out / archive.stem

    if not archive.exists():
        print(f"downloading {url}")
        # Download to a temp file and rename on success so an interrupted
        # download never leaves a truncated archive that later runs trust
        fd, tmp = tempfile.mkstemp(dir=out, suffix=".part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, archive)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        print("archive already present")

    if not extracted.exists():
        print("extracting")
        # Extract into a temp dir and rename so an interrupted extraction
        # doesn't leave a partial directory that the existence check trusts.
        # The tarball has one top-level dir matching the archive stem
        tmp_dir = Path(tempfile.mkdtemp(dir=out, suffix=".part"))
        try:
            with tarfile.open(archive) as tf:
                _safe_extract(tf, tmp_dir)
            os.replace(tmp_dir / extracted.name, extracted)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"data at {extracted}")


if __name__ == "__main__":
    main()
