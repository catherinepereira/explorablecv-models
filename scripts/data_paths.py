"""Shared dataset location for the explorablecv-models apps.

Every app reads datasets from one place so a download is shared rather than repeated.
The location defaults to a `data/` directory at the repo root and can be overridden
with the EXPLORABLECV_DATA environment variable.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Return the shared dataset root, creating it if needed."""
    env = os.environ.get("EXPLORABLECV_DATA")
    root = Path(env).expanduser() if env else REPO_ROOT / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_dir(name: str) -> Path:
    """Return (and create) the directory for a named dataset under the shared root."""
    path = data_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
