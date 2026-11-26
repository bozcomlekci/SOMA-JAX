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

Assets come from these places, searched in this order:

0. **``$SOMA_JAX_ASSETS``** — optional, an ``os.pathsep``-separated list of
   directories. Needed when the package is installed non-editably
   (``pip install .`` / ``pip install git+...``), because the repository-relative
   locations below then point into ``site-packages``; also handy for keeping one
   shared copy of the assets on a cluster.
1. **``third_party/SOMA-X/assets/``** — the vendored upstream submodule, at the
   same release as the code. It carries everything upstream's HF snapshot
   does: ``SOMA_neutral.npz``, the template rig, the procedural-transform JSON,
   the correctives checkpoint, ``SOMAHand.npz`` and the MHR / Anny / SMPL /
   SMPL-X / MANO / GarmentMeasurements packs.
   ``git submodule update --init --recursive`` is all that is needed, with
   git-lfs active: upstream stores those files in Git LFS.
2. **``assets/third_party/``** — what the submodule does *not* carry, i.e.
   ``GarmentMeasurements/point.npz``, fetched by ``tools/download_assets.py``.
   Git-ignored. The same script fills in submodule files that arrived as Git
   LFS pointer stubs, without git-lfs.

Derived artefacts that this repo builds (notably ``SOMA_neutral_fixed.npz``,
a SOMA-JAX-only cache — see ``docs/INSTALL.md``) live directly in ``assets/``.

Git LFS pointer stubs
=====================

Cloning without git-lfs installed (or without ``git lfs install`` having been
run) leaves ~130-byte text stubs in place of every LFS file, so the paths exist
but hold no data. :func:`resolve` skips such stubs instead of handing them to a
loader that would then fail with an opaque parse error, and reports them by
name when nothing usable is found.

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

import os
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

#: The ``third_party/SOMA-X`` commit the superproject pins (upstream v0.3.3).
#: ``SOMA_X_ASSET_FILES`` maps each of its ``assets/`` files to
#: ``(sha256, size in bytes, stored in Git LFS)``. Lets ``tools/download_assets.py``
#: fetch and verify them without git-lfs — or without the submodule at all,
#: e.g. from a GitHub "Download ZIP" checkout. Regenerate both with
#: ``python tools/download_assets.py --print-table`` when the gitlink moves.
SOMA_X_COMMIT = "d6aa640f7787498009c4e3d57fcc14a243905d10"
SOMA_X_ASSET_FILES = {
    "Anny/SOMA_wrap.obj": (
        "15e1500a13d3fac2db17b622ab54292cfb2e58d9efaa1e86cc11e46071bffb2e", 2613343, True),
    "Anny/base_body.obj": (
        "7bf31cbc0897670de51b7cd6b1a3da55a35fe56ef1d7b2e6aae2f95a11196504", 1182571, True),
    "GarmentMeasurements/SOMA_wrap.obj": (
        "c1871742042290cb6de7641e7e29fc1c42f331e8be1b4f9a019437d244986baa", 2192530, True),
    "GarmentMeasurements/mean.obj": (
        "4536d92bc379e818a4b29b83ab7ab19b3036cd7ceaf3469fba5883b88eb33c13", 3552167, True),
    "MANO/SOMA_wrap_left.obj": (
        "e198a37e9d33e32176d6325d47d48c153b5429dce5d248769ac03959b9c28aaa", 194628, True),
    "MANO/SOMA_wrap_right.obj": (
        "5c5f8dc18bd8386aacee78dfa4437f3cd1cfbad066ae3772d2f666db1d047771", 194585, True),
    "MANO/base_hand_left.obj": (
        "fd8b895d18e38cad9f082e4e7f8c66534c70253eedebdf77c5cd4987fcad54b8", 49391, True),
    "MANO/base_hand_right.obj": (
        "9d83807a5852cf35a85da421e32d6168de6e33c2ce6f42dd531aecc88febc3b6", 49408, True),
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
    "SOMAHand.npz": (
        "233e9225c1da05c6a3dd98107b9f027f2366e3b972bbe67e0d164e5d28619fff", 993833, True),
    "SOMA_neutral.npz": (
        "5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63", 26608121, True),
    "SOMA_procedural_transforms.json": (
        "9853e19ab51625ffaa7d9e7ccf0e443272c2740837604e6bacd66604e8edf509", 20045, False),
    "SOMA_template_rig.usda": (
        "a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856", 28306038, True),
    "correctives_model.pt": (
        "411d061a651051a7a4ec941543495d35eac21e5a7a69c8ce042b89cd12772a43", 70549157, True),
    "example_animation.npy": (
        "f424385dae3128278e74e81e0789ee4048753199d04fef77c27ba5457bd3819d", 5601024, True),
}

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


def env_asset_roots() -> tuple[Path, ...]:
    """Directories listed in ``$SOMA_JAX_ASSETS``, in order."""
    raw = os.environ.get(ASSETS_ENV_VAR, "")
    return tuple(Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip())


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
    return env_asset_roots() + (
        SUBMODULE_ASSETS, DOWNLOAD_ASSETS, DOWNLOAD_ASSETS / "hf", BUILT_ASSETS)


def _pinned_size(name: str) -> int | None:
    """Byte size the pinned submodule commit records for ``name``, when the
    checkout does not hold the file itself (``None`` otherwise)."""
    sub = SUBMODULE_ASSETS / name
    pointer = lfs_pointer(sub)
    if pointer is not None:
        return pointer[1]
    if not sub.exists() and name in SOMA_X_ASSET_FILES:
        return SOMA_X_ASSET_FILES[name][1]
    return None


def resolve(name: str, *, required: bool = True) -> Path | None:
    """Locate an asset by relative name.

    Git LFS pointer stubs are skipped, so the first *usable* copy wins.

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

    # When the checked-out submodule holds a pointer stub for this file — or
    # has no copy at all (a "Download ZIP" checkout) — a copy found elsewhere
    # must be the pinned release's file: one fetched for an earlier submodule
    # pin (another size than the pointer, or SOMA_X_ASSET_FILES, records) is
    # stale. A real file in the submodule needs no such check.
    pinned_size = _pinned_size(name)
    pointers: list[Path] = []
    stale: list[Path] = []
    for root in roots:
        candidate = root / name
        if not candidate.exists():
            continue
        if lfs_pointer(candidate) is not None:
            pointers.append(candidate)
            continue
        if (pinned_size is not None and candidate.is_file()
                and candidate.stat().st_size != pinned_size):
            stale.append(candidate)
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
    if stale:
        found += (
            " Found only copies from another release (their size differs from the "
            "pinned submodule commit's): " + ", ".join(str(p) for p in stale)
            + ". Run `python tools/download_assets.py` to fetch this release's file."
        )
    raise FileNotFoundError(
        f"Asset {name!r} not found{detail}.{found} Searched: "
        + ", ".join(str(r) for r in roots)
        + f". Run `git submodule update --init --recursive` for submodule assets, "
        f"or `python tools/download_assets.py` for the rest — see docs/INSTALL.md. "
        f"Assets kept elsewhere can be added to the search path with "
        f"${ASSETS_ENV_VAR}."
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
            # Drop a link left from an earlier run (it may point at an LFS
            # stub or a file that has since gone); upstream must not load it.
            if dst.is_symlink():
                dst.unlink(missing_ok=True)
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
            dst.unlink(missing_ok=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
        except FileExistsError:    # another process (e.g. pytest -n) got there first
            pass
        except OSError:            # e.g. Windows without symlink privilege
            try:
                os.link(src, dst)  # same volume: no privilege needed, no bytes copied
            except FileExistsError:
                pass
            except OSError:
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
