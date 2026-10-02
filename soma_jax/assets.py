"""Asset location for SOMA-JAX.

Upstream: ``soma/assets.py``
    Same role — locate the model data the layer needs — but resolved locally
    rather than via a HuggingFace snapshot. Upstream's ``get_assets_dir()``
    downloads the ``nvidia/soma-x`` HF repo at the immutable tag matching the
    package release (``revision="v0.3.3"``); that tag's file set is exactly the
    vendored submodule's ``assets/`` directory (upstream
    ``tools/ci/hf_assets_v0.3.json``), so the submodule *is* that snapshot.

Layout
======

Assets come from two places, searched in this order:

1. **``third_party/SOMA-X/assets/``** — the vendored upstream submodule, at the
   same release as the code. It carries everything upstream's HF snapshot
   does: ``SOMA_neutral.npz``, the template rig, the procedural-transform JSON,
   the correctives checkpoint, ``SOMAHand.npz`` and the MHR / Anny / SMPL /
   SMPL-X / MANO / GarmentMeasurements packs.
   ``git submodule update --init --recursive`` is all that is needed (with
   ``git-lfs`` on ``PATH`` — without it the checkout aborts mid-filter).
2. **``assets/third_party/``** — what the submodule does *not* carry, i.e.
   ``GarmentMeasurements/point.npz``. Git-ignored.

Derived artefacts that this repo builds (notably ``SOMA_neutral_fixed.npz``,
a SOMA-JAX-only cache — see ``docs/INSTALL.md``) live directly in ``assets/``.

The v0.3 asset contract
=======================

Since SOMA-X v0.3 ``SOMA_neutral.npz`` is a **shape/topology asset only**:
shape PCA, topology, UVs, LOD maps, segments and the historical reference
T-poses behind ``get_reference_pose(version=...)``. Its metadata's
``asset_contract.removed_npz_rig_fields`` lists the eleven rig keys it no longer
carries (``bind_pose_*``, ``bind_shape``, ``joint_names``, ``joint_parent_ids``,
``t_pose_*`` and the four ``skinning_weights_*``); the rig, bind pose, bind shape
and skinning come from ``SOMA_template_rig.usda`` alone, and the public joint
names from ``SOMA_procedural_transforms.json``. :mod:`soma_jax.rig_build` does
that merge.

``GarmentMeasurements/point.npz`` is not part of upstream's asset set at all:
upstream's ``docs/data_assets.md`` has users generate it from the public
``point.pca`` of the GarmentMeasurements repo with
``tools/convert_gm_pca_to_npz.py`` (here ``tools/convert/convert_gm_pca_to_npz.py``).
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

#: HuggingFace asset repo, upstream's ``soma.assets.REPO_ID``.
REPO_ID = "nvidia/soma-x"
#: SOMA-JAX's earlier name for :data:`REPO_ID`.
HF_REPO_ID = REPO_ID
#: Upstream's default ``get_assets_dir(revision=...)``: the immutable asset tag
#: matching the package release. Kept in step with the submodule's version.
UPSTREAM_ASSET_REVISION = "v0.3.3"
#: Older immutable HF revision used ONLY by ``tools/download_assets.py`` as a
#: convenience fallback for ``GarmentMeasurements/point.npz``, which upstream
#: v0.3 no longer ships (it asks users to convert it locally). SOMA-JAX extra.
HF_REVISION = "466879a83d57eabf3d875ded2d869f2075f90348"

SUBMODULE_ASSETS = REPO_ROOT / "third_party" / "SOMA-X" / "assets"
DOWNLOAD_ASSETS = REPO_ROOT / "assets" / "third_party"
BUILT_ASSETS = REPO_ROOT / "assets"

#: Relative paths the vendored submodule provides as-is — upstream's v0.3 HF
#: asset set (``tools/ci/hf_assets_v0.3.json``) minus docs images.
FROM_SUBMODULE = (
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "SOMAHand.npz",
    "correctives_model.pt",
    "example_animation.npy",
    "MHR",
    "Anny",
    "MANO",
    "SMPL",
    "SMPLX",
    "GarmentMeasurements",
)

#: Assets that must come from outside the submodule.
MUST_DOWNLOAD = {
    "GarmentMeasurements/point.npz": (
        "upstream does not ship this PCA archive; generate it from the public "
        "GarmentMeasurements point.pca with tools/convert/convert_gm_pca_to_npz.py "
        "(upstream docs/data_assets.md)"
    ),
}

#: Individual files upstream's full-body ``data_root`` contract expects.
#: Checking whole directories is not enough — an empty ``GarmentMeasurements/``
#: would pass.
REQUIRED_FILES = (
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "correctives_model.pt",
    "MHR/base_body_lod1.obj",
    "MHR/mhr_model_lod1.pt",
    "MHR/SOMA_wrap_lod1.obj",
    # Body MHR backend at ``low_lod`` (upstream `MHRIdentityModel`:
    # ``lod = "lod1" if not low_lod else "lod6"``).
    "MHR/base_body_lod6.obj",
    "MHR/mhr_model_lod6.pt",
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

#: Files the SOMA Hand layer (upstream ``soma.hand``, v0.3.0) reads from the
#: same ``data_root``: the hand mapping/PCA archive and the per-side MANO
#: correspondence pair. Its MHR backend reuses the body's ``MHR/*_lod1`` files.
#: Kept separate so a body-only setup is not reported as incomplete;
#: :func:`data_root` links both sets.
HAND_REQUIRED_FILES = (
    "SOMAHand.npz",
    "MANO/base_hand_left.obj",
    "MANO/base_hand_right.obj",
    "MANO/SOMA_wrap_left.obj",
    "MANO/SOMA_wrap_right.obj",
)

#: Example inputs that ship with the assets and that upstream's tools read from
#: ``data_root``: the demo motion (``tools/hand/demo_soma_hand_vis.py``) and the
#: SMPL clip ``tools/convert/smpl2soma.py`` converts. Linked by :func:`data_root`,
#: never required.
EXAMPLE_FILES = (
    "example_animation.npy",
    "SMPL/smpl_anim.npy",
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
        # Never vendored upstream, so there is no point searching the submodule.
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


def get_assets_dir(revision: str = UPSTREAM_ASSET_REVISION, cache_dir=None) -> Path:
    """The SOMA asset directory for ``revision`` (upstream ``get_assets_dir``).

    Upstream downloads (or reuses from the HuggingFace cache) the
    :data:`REPO_ID` snapshot at ``revision``. The vendored submodule's
    ``assets/`` *is* that snapshot for the release this port tracks, so the
    default revision returns it without network access when it is checked out;
    any other revision, an explicit ``cache_dir``, or a missing submodule goes
    through ``huggingface_hub.snapshot_download`` exactly as upstream does.

    Args:
        revision: git revision (branch, tag or commit) of the asset repo.
        cache_dir: override the default HuggingFace cache directory.

    Returns:
        Path of the local asset directory.
    """
    if (revision == UPSTREAM_ASSET_REVISION and cache_dir is None
            and (SUBMODULE_ASSETS / "SOMA_neutral.npz").is_file()):
        return SUBMODULE_ASSETS
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=REPO_ID, repo_type="model", revision=revision,
                                  cache_dir=cache_dir))


#: Materialised view satisfying upstream's single-directory contract.
DATA_ROOT = REPO_ROOT / "assets" / "data_root"

#: Licensed model files the user provides (registration required, never
#: shipped): ``data/<dir>/`` here -> ``<data_root>/<MODEL>/`` where upstream's
#: loaders look (``SMPL/SMPL_NEUTRAL.npz``, ``MANO/MANO_LEFT.pkl``, ...).
LICENSED_MODEL_DIRS = {"SMPL": "smpl", "SMPLH": "smplh", "SMPLX": "smplx", "MANO": "mano"}


def _licensed_model_files():
    """``(data-root relative path, source)`` for each licensed file under ``data/``."""
    for model, local in LICENSED_MODEL_DIRS.items():
        folder = REPO_ROOT / "data" / local
        if not folder.is_dir():
            continue
        for src in sorted(folder.iterdir()):
            if (src.is_file() and src.suffix in (".npz", ".pkl")
                    and src.name.startswith(model + "_")):
                yield f"{model}/{src.name}", src


def data_root(materialise: bool = True) -> Path:
    """A single directory satisfying upstream's ``SOMALayer(data_root=...)``.

    Upstream wants **one** directory holding ``SOMA_neutral.npz`` next to the
    template rig, procedural JSON, correctives checkpoint and the per-model
    packs. Our assets are deliberately split — most live in the submodule, the
    rest are downloaded — so no existing directory satisfies that contract.

    This assembles one out of symlinks (falling back to hardlink/copy where
    symlinks are unavailable), pointing at whatever :func:`resolve` finds, plus
    the licensed model files placed under ``data/smpl``, ``data/smplh``,
    ``data/smplx`` and ``data/mano`` (upstream names, e.g. ``SMPL_NEUTRAL.npz``),
    linked where upstream's loaders look for them. It is cheap, adds no
    duplicate bytes, and is refreshed when a link dangles.

    Args:
        materialise: build/refresh the directory. Pass False to get the path
            without touching the filesystem.

    Returns:
        Path to the assembled data root.
    """
    if not materialise:
        return DATA_ROOT

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    links = [(rel, resolve(rel, required=False))
             for rel in REQUIRED_FILES + HAND_REQUIRED_FILES + EXAMPLE_FILES]
    links += list(_licensed_model_files())
    for rel, src in links:
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


def missing_assets(*, hand: bool = False) -> list[str]:
    """Required files that cannot be resolved anywhere.

    Args:
        hand: also check :data:`HAND_REQUIRED_FILES` (the SOMA Hand layer's).
    """
    files = REQUIRED_FILES + (HAND_REQUIRED_FILES if hand else ())
    return [rel for rel in files if resolve(rel, required=False) is None]
