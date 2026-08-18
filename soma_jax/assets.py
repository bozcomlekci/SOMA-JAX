"""Asset location for SOMA-JAX.

Upstream: ``soma/assets.py``
    Same role — locate the model data the layer needs — but resolved locally
    rather than via a HuggingFace snapshot, because most of it already ships in
    the vendored submodule.

Layout
======

Assets come from two places, searched in this order:

1. **``third_party/SOMA-X/assets/``** — the vendored upstream submodule. It
   already carries the template rig, procedural-transform JSON, correctives
   checkpoint, and the MHR / Anny / SMPL / SMPL-X / GarmentMeasurements packs,
   so there is no reason to keep a second copy of ~1 GB of identical files.
   ``git submodule update --init --recursive`` is all that is needed.
2. **``assets/third_party/``** — everything the submodule does *not* carry,
   fetched by ``tools/download_assets.py``. Git-ignored.

Derived artefacts that this repo builds (notably
``SOMA_neutral_fixed.npz``, see ``docs/INSTALL.md`` §4.2) live directly in
``assets/``.

The one asset that must be downloaded
=====================================

``SOMA_neutral.npz`` **cannot** be taken from the submodule. That copy is the
slim v0.2.1 archive and is missing all eleven rig keys the conversion needs
(``bind_pose_world``, ``bind_pose_local``, ``bind_shape``, ``joint_names``,
``joint_parent_ids``, ``t_pose_world``, ``t_pose_local`` and the four
``skinning_weights_*`` arrays). :func:`resolve` enforces this rather than
silently handing back an archive that fails later with a missing-key error.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def _search_roots() -> tuple[Path, ...]:
    return (SUBMODULE_ASSETS, DOWNLOAD_ASSETS, DOWNLOAD_ASSETS / "hf", BUILT_ASSETS)


def resolve(name: str, *, required: bool = True) -> Path | None:
    """Locate an asset by relative name.

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

    for root in roots:
        candidate = root / name
        if candidate.exists():
            return candidate

    if not required:
        return None
    hint = MUST_DOWNLOAD.get(name)
    detail = f" ({hint})" if hint else ""
    raise FileNotFoundError(
        f"Asset {name!r} not found{detail}. Searched: "
        + ", ".join(str(r) for r in roots)
        + ". Run `git submodule update --init --recursive` for submodule assets, "
        "or `python tools/download_assets.py` for the rest — see docs/INSTALL.md."
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
            continue
        if dst.is_symlink() or dst.exists():
            try:
                if dst.resolve() == src.resolve():
                    continue
            except OSError:
                pass
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
        except OSError:            # e.g. filesystems without symlink support
            import shutil
            shutil.copy2(src, dst)
    return DATA_ROOT


def missing_assets() -> list[str]:
    """Required files that cannot be resolved anywhere."""
    return [rel for rel in REQUIRED_FILES if resolve(rel, required=False) is None]
