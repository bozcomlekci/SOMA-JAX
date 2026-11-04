"""Asset location for SOMA-JAX.

Upstream: ``soma/assets.py``
    Same role — locate the model data the layer needs — but resolved locally
    rather than via a HuggingFace snapshot, because most of it already ships in
    the vendored submodule.

Layout
======

Assets come from these places, searched in this order:

0. **``$SOMA_JAX_ASSETS``** — optional, an ``os.pathsep``-separated list of
   directories. Needed when the package is installed non-editably
   (``pip install .`` / ``pip install git+...``), because the repository-relative
   locations below then point into ``site-packages``; also handy for keeping one
   shared copy of the assets on a cluster.
1. **``third_party/SOMA-X/assets/``** — the vendored upstream submodule. It
   already carries the template rig, procedural-transform JSON, correctives
   checkpoint, and the MHR / Anny / SMPL / SMPL-X / GarmentMeasurements packs,
   so there is no reason to keep a second copy of ~1 GB of identical files.
   ``git submodule update --init --recursive`` is all that is needed — **with
   git-lfs active**, since upstream stores every one of those files in Git LFS.
2. **``assets/third_party/``** — everything the submodule does *not* carry,
   fetched by ``tools/download_assets.py``. Git-ignored. The same script also
   fills in submodule files that arrived as Git LFS pointer stubs.

Derived artefacts that this repo builds (notably
``SOMA_neutral_fixed.npz``, see ``docs/INSTALL.md`` §4.2) live directly in
``assets/``.

Git LFS pointer stubs
=====================

Cloning without git-lfs installed (or without ``git lfs install`` having been
run) leaves ~130-byte text stubs in place of every LFS file, so the paths exist
but hold no data. :func:`resolve` skips such stubs instead of handing them to a
loader that would then fail with an opaque parse error, and reports them by
name when nothing usable is found.

The one asset that must be downloaded
=====================================

``SOMA_neutral.npz`` **cannot** be taken from the submodule. That copy is the
slim v0.2.1 archive and is missing all eleven rig keys the conversion needs
(``bind_pose_world``, ``bind_pose_local``, ``bind_shape``, ``joint_names``,
``joint_parent_ids``, ``t_pose_world``, ``t_pose_local`` and the four
``skinning_weights_*`` arrays). :func:`resolve` enforces this rather than
silently handing back an archive that fails later with a missing-key error —
for the submodule copy and for a slim archive found anywhere else.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Environment variable holding extra asset directories (``os.pathsep``-separated),
#: searched before the repository-relative locations.
ASSETS_ENV_VAR = "SOMA_JAX_ASSETS"

#: Where each upstream/third-party asset family comes from, for
#: ``tools/download_assets.py`` and for anyone tracking provenance.
SOURCE_REPOSITORIES = {
    "SOMA-X": "https://github.com/NVlabs/SOMA-X",
    "MHR": "https://github.com/facebookresearch/MHR",
    "Anny": "https://github.com/naver/anny",
}

#: HuggingFace repo holding the full-schema SOMA archive.
HF_REPO_ID = "nvidia/SOMA-X"
#: Immutable revision pinned by ``docs/INSTALL.md`` §4.1.
HF_REVISION = "466879a83d57eabf3d875ded2d869f2075f90348"

SUBMODULE_ASSETS = REPO_ROOT / "third_party" / "SOMA-X" / "assets"
DOWNLOAD_ASSETS = REPO_ROOT / "assets" / "third_party"
BUILT_ASSETS = REPO_ROOT / "assets"

#: Relative paths the vendored submodule provides as-is.
FROM_SUBMODULE = (
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "correctives_model.pt",
    "example_animation.npy",
    "MHR",
    "Anny",
    "SMPL",
    "SMPLX",
    "GarmentMeasurements",
)

#: Assets that must be downloaded; the submodule copy is unusable or absent.
MUST_DOWNLOAD = {
    "SOMA_neutral.npz": (
        "the submodule ships the slim v0.2.1 archive, which omits the rig "
        "arrays this port needs (bind_pose_world, skinning_weights_*, "
        "joint_names, t_pose_world, ...)"
    ),
    "GarmentMeasurements/point.npz": (
        "the submodule ships only mean.obj and SOMA_wrap.obj for this pack; "
        "the PCA archive is not vendored"
    ),
}

#: Individual files upstream's ``data_root`` contract expects. Checking whole
#: directories is not enough — an empty ``GarmentMeasurements/`` would pass.
REQUIRED_FILES = (
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "correctives_model.pt",
    "MHR/base_body_lod1.obj",
    "MHR/mhr_model_lod1.pt",
    "MHR/SOMA_wrap_lod1.obj",
    "SMPL/base_body.obj",
    "SMPL/SOMA_wrap.obj",
    "SMPLX/base_body.obj",
    "SMPLX/SOMA_wrap.obj",
    "Anny/base_body.obj",
    "Anny/SOMA_wrap.obj",
    "GarmentMeasurements/mean.obj",
    "GarmentMeasurements/SOMA_wrap.obj",
    "GarmentMeasurements/point.npz",
)

#: The ``third_party/SOMA-X`` commit the superproject pins (upstream v0.2.1).
SOMA_X_COMMIT = "86632764684281dc98f31ab9c4aac36a4cdbc428"

#: ``relative path -> (sha256, size in bytes, stored in Git LFS)`` for the
#: submodule assets at :data:`SOMA_X_COMMIT`. Lets ``tools/download_assets.py``
#: fetch and verify them without git-lfs — or without the submodule at all,
#: e.g. from a GitHub "Download ZIP" checkout. Keep in sync with the gitlink
#: when the submodule is bumped (``git lfs ls-files -l`` lists the LFS oids).
SOMA_X_ASSET_FILES = {
    "SOMA_template_rig.usda": (
        "79f02176534ebc0671c76bf09203bbee8e1621f8b6d34cc63aec69b73460bba2", 344934193, True),
    "SOMA_procedural_transforms.json": (
        "5c135532d77932b547050b131a8fa0e67b63e9541c3cd621889ba483d946fb00", 20045, False),
    "correctives_model.pt": (
        "411d061a651051a7a4ec941543495d35eac21e5a7a69c8ce042b89cd12772a43", 70549157, True),
    "example_animation.npy": (
        "f424385dae3128278e74e81e0789ee4048753199d04fef77c27ba5457bd3819d", 5601024, True),
    "MHR/SOMA_wrap_lod1.obj": (
        "9ab7a2328d5b026c17ef1d09ed692ffeb68f96d3fc74970cba72e667dbe0f728", 1816801, True),
    "MHR/base_body_lod1.obj": (
        "e8bbb3f3681d76737635c10a9e6f6a058ad3d9be5a75547f84a578cea572e916", 2277452, True),
    "MHR/base_body_lod6.obj": (
        "cd46620ec46e968658527f25e2b54b8f2292bfd486eac59c596f8b5d5229b9e0", 39006, True),
    "MHR/mhr_model_lod1.pt": (
        "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc", 696110248, True),
    "MHR/mhr_model_lod6.pt": (
        "28839b7ee0eb38eb3943cf26cff61a306de78f86227a0f17bb044252c7bc06ac", 2080751, True),
    "SMPL/SOMA_wrap.obj": (
        "d56c7d1452a57b3f1950cfe9b81fec483b3a8e7a9a85bdc4ea83f2996e1a58da", 2731479, True),
    "SMPL/base_body.obj": (
        "08996c762b0adfe45aeb4d6297c9b93d460484b09815674d36a3a544d64e0c43", 457369, True),
    "SMPL/smpl_anim.npy": (
        "1126c0b377f00b2678a91a2e0dd0eb712d27ed700eea6edc76ed46e779769a84", 1009136, True),
    "SMPLX/SOMA_wrap.obj": (
        "7498e3a7873d7b4d8f1d3455a8af945589b1c494fbff99b14bc905a691208b6f", 2630564, True),
    "SMPLX/base_body.obj": (
        "f03f95e5e303c8dc070afa09387e9dde861eeb1567ccf61a70bf53a5f7fd2b70", 700964, True),
    "Anny/SOMA_wrap.obj": (
        "0e9caa78848b27c32f69601d289aeea8835237c38d02c8b0c0fce8f6ee699549", 2613345, True),
    "Anny/base_body.obj": (
        "7bf31cbc0897670de51b7cd6b1a3da55a35fe56ef1d7b2e6aae2f95a11196504", 1182571, True),
    "GarmentMeasurements/SOMA_wrap.obj": (
        "c1871742042290cb6de7641e7e29fc1c42f331e8be1b4f9a019437d244986baa", 2192530, True),
    "GarmentMeasurements/mean.obj": (
        "4536d92bc379e818a4b29b83ab7ab19b3036cd7ceaf3469fba5883b88eb33c13", 3552167, True),
}

#: Keys whose absence marks a ``SOMA_neutral.npz`` as the slim archive.
_FULL_SCHEMA_KEYS = ("bind_shape", "bind_pose_world", "joint_names")

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
#: Real LFS pointer files are ~130 bytes; anything larger is real content.
_LFS_POINTER_MAX_BYTES = 1024


def lfs_pointer(path) -> tuple[str, int] | None:
    """Return ``(sha256, size)`` when ``path`` is a Git LFS pointer stub.

    A clone made without git-lfs active contains these ~130-byte text files in
    place of the real data. Returns ``None`` for real files, directories and
    unreadable paths.
    """
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size > _LFS_POINTER_MAX_BYTES:
            return None
        with open(path, "rb") as f:
            head = f.read(_LFS_POINTER_MAX_BYTES)
    except OSError:
        return None
    if not head.startswith(_LFS_POINTER_PREFIX):
        return None
    oid, size = None, -1
    for line in head.decode("ascii", "replace").splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            try:
                size = int(line.split()[1])
            except (IndexError, ValueError):
                pass
    return (oid, size) if oid else None


def _is_slim_soma_archive(path: Path) -> bool:
    """True for a ``SOMA_neutral.npz`` that lacks the full-schema rig keys.

    Reads only the zip directory, so it is cheap even for the 27 MB archive.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = {n[:-4] if n.endswith(".npy") else n for n in zf.namelist()}
    except (OSError, zipfile.BadZipFile):
        return False   # not ours to judge — let the loader report it
    return not all(k in names for k in _FULL_SCHEMA_KEYS)


def env_asset_roots() -> tuple[Path, ...]:
    """Directories listed in ``$SOMA_JAX_ASSETS``, in order."""
    raw = os.environ.get(ASSETS_ENV_VAR, "")
    return tuple(Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip())


def _search_roots() -> tuple[Path, ...]:
    return env_asset_roots() + (
        SUBMODULE_ASSETS, DOWNLOAD_ASSETS, DOWNLOAD_ASSETS / "hf", BUILT_ASSETS)


def resolve(name: str, *, required: bool = True) -> Path | None:
    """Locate an asset by relative name.

    Git LFS pointer stubs are skipped, as is a slim ``SOMA_neutral.npz``, so
    the first *usable* copy wins.

    Args:
        name: e.g. ``"SOMA_template_rig.usda"``, ``"MHR/base_body_lod1.obj"``,
            ``"SOMA_neutral.npz"``.
        required: raise when missing instead of returning ``None``.

    Returns:
        Absolute path, or ``None`` when absent and ``required`` is False.

    Raises:
        FileNotFoundError: when required and not found anywhere.
    """
    roots = _search_roots()
    if name in MUST_DOWNLOAD:
        # Skip the submodule for these: a copy exists but is the wrong one.
        roots = tuple(r for r in roots if r != SUBMODULE_ASSETS)

    pointers: list[Path] = []
    slim: list[Path] = []
    for root in roots:
        candidate = root / name
        if not candidate.exists():
            continue
        if lfs_pointer(candidate) is not None:
            pointers.append(candidate)
            continue
        if name == "SOMA_neutral.npz" and _is_slim_soma_archive(candidate):
            slim.append(candidate)
            continue
        return candidate

    if not required:
        return None
    hint = MUST_DOWNLOAD.get(name)
    detail = f" ({hint})" if hint else ""
    found = ""
    if pointers:
        found += (
            " Found only Git LFS pointer stub(s), not the data: "
            + ", ".join(str(p) for p in pointers)
            + ". The repository was cloned without git-lfs active; run "
            "`git lfs install && git submodule foreach --recursive git lfs pull`, "
            "or `python tools/download_assets.py`, which fetches the files "
            "without git-lfs."
        )
    if slim:
        found += (
            " Found only the slim SOMA_neutral.npz (no rig arrays): "
            + ", ".join(str(p) for p in slim)
            + ". Download the full-schema archive (docs/INSTALL.md §4.1)."
        )
    raise FileNotFoundError(
        f"Asset {name!r} not found{detail}.{found} Searched: "
        + ", ".join(str(r) for r in roots)
        + f". Run `git submodule update --init --recursive` for submodule assets, "
        f"or `python tools/download_assets.py` for the rest — see docs/INSTALL.md. "
        f"Assets kept elsewhere can be added to the search path with "
        f"${ASSETS_ENV_VAR}."
    )


def get_assets_dir() -> Path:
    """Directory holding the vendored upstream assets (submodule)."""
    return SUBMODULE_ASSETS


#: Materialised view satisfying upstream's single-directory contract.
DATA_ROOT = REPO_ROOT / "assets" / "data_root"


def data_root(materialise: bool = True) -> Path:
    """A single directory satisfying upstream's ``SOMALayer(data_root=...)``.

    Upstream wants **one** directory holding ``SOMA_neutral.npz`` next to the
    template rig, procedural JSON, correctives checkpoint and the per-model
    packs. Our assets are deliberately split — most live in the submodule, the
    rest are downloaded — so no existing directory satisfies that contract.

    This assembles one out of symlinks (falling back to hardlink/copy where
    symlinks are unavailable), pointing at whatever :func:`resolve` finds. It
    is cheap, adds no duplicate bytes, and is refreshed when a link dangles.

    Args:
        materialise: build/refresh the directory. Pass False to get the path
            without touching the filesystem.

    Returns:
        Path to the assembled data root.
    """
    if not materialise:
        return DATA_ROOT

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_FILES:
        src = resolve(rel, required=False)
        dst = DATA_ROOT / rel
        if src is None:
            # Drop a link left from an earlier run (it may point at an LFS
            # stub or a file that has since gone); upstream must not load it.
            if dst.is_symlink():
                dst.unlink()
            continue
        if dst.is_symlink() or dst.exists():
            try:
                if dst.resolve() == src.resolve() or dst.samefile(src):
                    continue                       # symlink or hardlink to src
                s, d = src.stat(), dst.stat()
                if (not dst.is_symlink() and d.st_size == s.st_size
                        and d.st_mtime >= s.st_mtime):
                    continue                       # an up-to-date copy
            except OSError:
                pass
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
        except OSError:            # e.g. Windows without symlink privilege
            try:
                os.link(src, dst)  # same volume: no privilege needed, no bytes copied
            except OSError:
                import shutil
                shutil.copy2(src, dst)
    return DATA_ROOT


def missing_assets() -> list[str]:
    """Required files that cannot be resolved anywhere."""
    return [rel for rel in REQUIRED_FILES if resolve(rel, required=False) is None]
