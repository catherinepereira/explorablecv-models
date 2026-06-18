import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "imagenette2.tgz"

    if not archive.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    else:
        print("archive already present")

    extracted = out / "imagenette2"
    if not extracted.exists():
        print("extracting")
        with tarfile.open(archive) as tf:
            tf.extractall(out)
    print(f"data at {extracted}")


if __name__ == "__main__":
    main()
