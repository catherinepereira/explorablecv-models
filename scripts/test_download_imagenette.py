"""Tests for the shared imagenette downloader, focused on extraction safety.

Run with: python -m pytest scripts/test_download_imagenette.py
The network is never hit. urlretrieve is monkeypatched with local tarballs.
"""

import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import download_imagenette as dl


def _make_tarball(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_symlink_tarball(path: Path, link_name: str, target: str) -> None:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(link_name)
        info.type = tarfile.SYMTYPE
        info.linkname = target
        tf.addfile(info)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPLORABLECV_DATA", str(tmp_path))
    return tmp_path


def test_happy_path_and_idempotent(sandbox, monkeypatch):
    src = sandbox / "fake.tgz"
    _make_tarball(src, {"imagenette2/hello.txt": b"hi"})
    monkeypatch.setattr(dl.urllib.request, "urlretrieve", lambda url, dst: dl.shutil.copy(src, dst))
    monkeypatch.setattr(sys, "argv", ["download_imagenette.py", "--variant", "imagenette"])

    dl.main()
    result = sandbox / "imagenette" / "imagenette2" / "hello.txt"
    assert result.read_text() == "hi"
    assert not list((sandbox / "imagenette").glob("*.part"))

    # second run must not re-download or re-extract, and must not raise
    dl.main()
    assert result.read_text() == "hi"


def test_rejects_path_traversal(sandbox, monkeypatch):
    src = sandbox / "evil.tgz"
    _make_tarball(src, {"../escape.txt": b"pwned"})
    monkeypatch.setattr(dl.urllib.request, "urlretrieve", lambda url, dst: dl.shutil.copy(src, dst))
    monkeypatch.setattr(sys, "argv", ["download_imagenette.py", "--variant", "imagenette"])

    with pytest.raises(Exception):
        dl.main()
    # nothing escaped the cache dir
    assert not (sandbox / "escape.txt").exists()
    assert not (sandbox.parent / "escape.txt").exists()


def test_rejects_symlink_member(sandbox, monkeypatch):
    src = sandbox / "link.tgz"
    _make_symlink_tarball(src, "imagenette2/evil", "/etc/hosts")
    monkeypatch.setattr(dl.urllib.request, "urlretrieve", lambda url, dst: dl.shutil.copy(src, dst))
    monkeypatch.setattr(sys, "argv", ["download_imagenette.py", "--variant", "imagenette"])

    with pytest.raises(Exception):
        dl.main()
    assert not (sandbox / "imagenette" / "imagenette2").exists()
