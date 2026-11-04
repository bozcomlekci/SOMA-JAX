"""Asset resolution: search path, Git LFS stubs, the slim archive, the manifest."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from soma_jax import assets

POINTER = ("version https://git-lfs.github.com/spec/v1\n"
           "oid sha256:{oid}\nsize {size}\n")
OID = "79f02176534ebc0671c76bf09203bbee8e1621f8b6d34cc63aec69b73460bba2"


def _pointer(path: Path, oid: str = OID, size: int = 344934193) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(POINTER.format(oid=oid, size=size))
    return path


def test_lfs_pointer_detected(tmp_path):
    assert assets.lfs_pointer(_pointer(tmp_path / "rig.usda")) == (OID, 344934193)


def test_real_files_and_dirs_are_not_pointers(tmp_path):
    real = tmp_path / "real.obj"
    real.write_text("v 0 0 0\n")
    big = tmp_path / "big.bin"
    big.write_bytes(b"version https://git-lfs.github.com/spec/v1\n" + b"x" * 4096)
    assert assets.lfs_pointer(real) is None
    assert assets.lfs_pointer(big) is None           # too large to be a stub
    assert assets.lfs_pointer(tmp_path) is None
    assert assets.lfs_pointer(tmp_path / "missing") is None


def test_env_roots_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv(assets.ASSETS_ENV_VAR,
                       os.pathsep.join([str(tmp_path / "a"), "", " ", str(tmp_path / "b")]))
    assert assets.env_asset_roots() == (tmp_path / "a", tmp_path / "b")
    monkeypatch.delenv(assets.ASSETS_ENV_VAR)
    assert assets.env_asset_roots() == ()


def test_resolve_skips_lfs_stub(monkeypatch, tmp_path):
    name = "selftest_probe/thing.usda"
    _pointer(tmp_path / "stubs" / name)
    real = tmp_path / "real" / name
    real.parent.mkdir(parents=True)
    real.write_text("#usda 1.0\n")
    monkeypatch.setenv(assets.ASSETS_ENV_VAR,
                       os.pathsep.join([str(tmp_path / "stubs"), str(tmp_path / "real")]))
    assert assets.resolve(name) == real


def test_resolve_names_the_stub_when_nothing_usable(monkeypatch, tmp_path):
    name = "selftest_probe/only_a_stub.usda"
    stub = _pointer(tmp_path / name)
    monkeypatch.setenv(assets.ASSETS_ENV_VAR, str(tmp_path))
    assert assets.resolve(name, required=False) is None
    with pytest.raises(FileNotFoundError, match="Git LFS pointer") as err:
        assets.resolve(name)
    assert str(stub) in str(err.value)
    assert "download_assets.py" in str(err.value)


def _npz(path: Path, keys) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.zeros(2) for k in keys})
    return path


def test_slim_soma_archive_is_skipped(monkeypatch, tmp_path):
    _npz(tmp_path / "slim" / "SOMA_neutral.npz", ["mean", "shapedirs", "triangles"])
    full = _npz(tmp_path / "full" / "SOMA_neutral.npz",
                ["mean", "shapedirs", "bind_shape", "bind_pose_world", "joint_names"])
    monkeypatch.setenv(assets.ASSETS_ENV_VAR,
                       os.pathsep.join([str(tmp_path / "slim"), str(tmp_path / "full")]))
    assert assets.resolve("SOMA_neutral.npz") == full


def test_env_root_takes_precedence(monkeypatch, tmp_path):
    name = "selftest_probe/override.json"
    (tmp_path / "selftest_probe").mkdir()
    (tmp_path / name).write_text("{}")
    monkeypatch.setenv(assets.ASSETS_ENV_VAR, str(tmp_path))
    assert assets.resolve(name) == tmp_path / name


def test_manifest_matches_submodule_pointers():
    """SOMA_X_ASSET_FILES must track the pinned submodule (catches a stale bump)."""
    checked = 0
    for rel, (sha, size, _is_lfs) in assets.SOMA_X_ASSET_FILES.items():
        path = assets.SUBMODULE_ASSETS / rel
        if not path.exists():
            continue
        pointer = assets.lfs_pointer(path)
        if pointer is not None:
            assert pointer == (sha, size), f"{rel}: submodule pointer {pointer} != manifest"
            checked += 1
        elif path.is_file():
            assert path.stat().st_size == size, f"{rel}: size {path.stat().st_size} != {size}"
            checked += 1
    if not checked:
        pytest.skip("third_party/SOMA-X not checked out")


def test_manifest_commit_matches_gitlink():
    git = shutil.which("git")
    if git is None or not (assets.REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run([git, "-C", str(assets.REPO_ROOT), "ls-tree", "HEAD",
                          "third_party/SOMA-X"], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():
        pytest.skip("no submodule gitlink")
    assert out.stdout.split()[2] == assets.SOMA_X_COMMIT


def test_data_root_without_symlinks(monkeypatch, tmp_path):
    """Where symlinks are not permitted (Windows without the privilege) the view
    falls back to hardlinks/copies — and must not redo that on every call."""
    from pathlib import Path as _Path
    name = "selftest_probe/x.json"
    (tmp_path / "src" / "selftest_probe").mkdir(parents=True)
    (tmp_path / "src" / name).write_text('{"a": 1}')
    monkeypatch.setenv(assets.ASSETS_ENV_VAR, str(tmp_path / "src"))
    monkeypatch.setattr(assets, "REQUIRED_FILES", (name,))
    monkeypatch.setattr(assets, "DATA_ROOT", tmp_path / "view")

    def no_symlinks(self, *a, **k):
        raise OSError("symlinks unavailable")
    monkeypatch.setattr(_Path, "symlink_to", no_symlinks)

    view = assets.data_root()
    first = (view / name).stat()
    assert (view / name).read_text() == '{"a": 1}'
    assets.data_root()
    second = (view / name).stat()
    assert (first.st_ino, first.st_mtime_ns) == (second.st_ino, second.st_mtime_ns)


def test_data_root_view(tmp_path):
    """data_root() links whatever resolves, and never links an LFS stub."""
    root = assets.data_root(materialise=True)
    for rel in assets.REQUIRED_FILES:
        link = root / rel
        if link.exists():
            assert assets.lfs_pointer(link) is None, f"{rel} links to an LFS stub"
