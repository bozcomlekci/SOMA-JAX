"""Download SOMA assets from HuggingFace — port of upstream ``tools/download_assets.py``.

Upstream: ``tools/download_assets.py`` (SOMA-X v0.3.3).
:func:`download_assets` and the ``--target-dir`` / ``--revision`` options are
upstream's: they fetch the ``nvidia/soma-x`` asset snapshot through
:func:`soma_jax.assets.get_assets_dir` (``huggingface_hub.snapshot_download``)
into the HuggingFace cache, or into ``--target-dir`` used as that cache.

SOMA-JAX does not need that download: everything upstream ships lives in the
**`third_party/SOMA-X` submodule** (`third_party/SOMA-X/assets/`), at the same
release as the code, and is used in place (``git submodule update --init
--recursive``, with `git-lfs` on ``PATH``). Two SOMA-JAX extras serve that
layout:

* ``--check`` reports which of the files SOMA-JAX reads are present (submodule
  or download), re-hashing downloads with a known checksum, and downloads
  nothing.
* ``--extras`` fetches the one file upstream does not ship either,
  **`GarmentMeasurements/point.npz`** (the GarmentMeasurements shape PCA), from the
  older immutable `nvidia/SOMA-X` HuggingFace revision that published it, checks
  its sha256 and writes it to ``assets/third_party/`` (git-ignored). Upstream's
  own instruction (`docs/data_assets.md`) is to convert it from the public
  `point.pca` of https://github.com/mbotsch/GarmentMeasurements::

    python tools/convert/convert_gm_pca_to_npz.py /path/to/point.pca \
        assets/third_party/GarmentMeasurements/point.npz

Derived artefacts that this repo *builds* — notably the optional runtime archive
``assets/SOMA_neutral_fixed.npz`` (``tools/pipeline/build_soma_rig.py``) — live
in ``assets/``.

Usage::

    python tools/download_assets.py                    # upstream: HF snapshot (revision main)
    python tools/download_assets.py --target-dir DIR --revision v0.3.3
    python tools/download_assets.py --check            # SOMA-JAX: report what is missing
    python tools/download_assets.py --extras [--force] # SOMA-JAX: GarmentMeasurements/point.npz
"""
from __future__ import annotations
import argparse
import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from soma_jax.assets import (  # noqa: E402
    DOWNLOAD_ASSETS,
    HAND_REQUIRED_FILES,
    HF_REPO_ID,
    HF_REVISION,
    REQUIRED_FILES,
    SOURCE_REPOSITORIES,
    SUBMODULE_ASSETS,
    resolve,
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

# (filename, sha256 or None, description)
DOWNLOADS = [
    # The submodule ships only mean.obj / SOMA_wrap.obj for this pack; the
    # checksum is the LFS object id HuggingFace reports for HF_REVISION.
    ("GarmentMeasurements/point.npz",
     "6ae75a1ab7a8ae4f46bac503146976ba59cf918bca8aefe7320a3ed02aa2416a",
     "GarmentMeasurements shape PCA (not shipped by upstream)"),
]


def download_assets(target_dir=None, revision="main"):
    """Download SOMA assets from HuggingFace (upstream ``download_assets``).

    Args:
        target_dir: If provided, used as the HuggingFace cache directory.
            The actual assets will be stored in a subdirectory managed by
            huggingface_hub.  If None, uses the default HF cache
            (``~/.cache/huggingface/hub/``).
        revision: Git revision (branch, tag, or commit hash) to download.

    Returns:
        Path to the downloaded assets directory.
    """
    from soma_jax.assets import get_assets_dir

    path = get_assets_dir(revision=revision, cache_dir=target_dir)
    logger.info(f"Assets downloaded to: {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download SOMA assets from HuggingFace",
        epilog="Source repositories: "
               + ", ".join(f"{k} {v}" for k, v in SOURCE_REPOSITORIES.items()),
    )
    # Upstream's options.
    p.add_argument(
        "--target-dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface/hub/)",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Git revision to download (default: main)",
    )
    # SOMA-JAX extras.
    p.add_argument("--check", action="store_true",
                   help="SOMA-JAX: report which assets are present and exit without downloading")
    p.add_argument("--extras", action="store_true",
                   help="SOMA-JAX: fetch the files the submodule does not ship "
                        "(GarmentMeasurements/point.npz) instead of the HF snapshot")
    p.add_argument("--output-dir", default=str(DOWNLOAD_ASSETS),
                   help=f"--extras download target (default: {DOWNLOAD_ASSETS})")
    p.add_argument("--repo-id", default=HF_REPO_ID,
                   help=f"--extras HuggingFace repository (default: {HF_REPO_ID})")
    p.add_argument("--extras-revision", default=HF_REVISION,
                   help="--extras immutable revision (default: the one carrying point.npz)")
    p.add_argument("--force", action="store_true", help="--extras: re-download even if present")
    add_logging_args(p)
    args = p.parse_args()
    configure_logging(args)
    return args


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check() -> int:
    """Report per-FILE availability. Returns a process exit code.

    Enumerates individual files rather than family directories: an empty
    ``GarmentMeasurements/`` would otherwise be reported healthy. Existing
    downloads are re-hashed where a checksum is known, so a corrupt file is
    reported rather than silently skipped on the next run.
    """
    expected = {name: sha for name, sha, _ in DOWNLOADS}
    missing, corrupt = [], []

    logger.info("required assets (body):")
    for rel in REQUIRED_FILES:
        path = resolve(rel, required=False)
        if path is None:
            logger.info(f"  [MISSING] {rel}")
            missing.append(rel)
            continue
        origin = "submodule" if str(SUBMODULE_ASSETS) in str(path) else "download"
        sha = expected.get(rel)
        if sha is not None:
            digest = _sha256(path)
            if digest != sha:
                logger.warning(f"  [CORRUPT] {rel}  ({origin}) sha256 {digest[:12]}... != {sha[:12]}...")
                corrupt.append(rel)
                continue
            logger.info(f"  [ok     ] {rel}  ({origin}, sha256 verified)")
        else:
            logger.info(f"  [ok     ] {rel}  ({origin})")

    logger.info("\nSOMA Hand assets (optional, for soma_jax.hand):")
    for rel in HAND_REQUIRED_FILES:
        path = resolve(rel, required=False)
        logger.info(f"  [{'ok     ' if path is not None else 'MISSING'}] {rel}")

    if missing:
        logger.info("\n  -> git submodule update --init --recursive   (for vendored assets)")
        logger.info("  -> python tools/download_assets.py --extras  (for the rest)")
    if corrupt:
        logger.info("\n  -> python tools/download_assets.py --extras --force   (re-download corrupt files)")
    if not missing and not corrupt:
        logger.info("\nall required assets present and verified.")
    return 1 if (missing or corrupt) else 0


def download(repo_id: str, revision: str, filename: str, out_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub is not installed.\n  pip install huggingface_hub")
    logger.info(f"Downloading {filename} from {repo_id}@{revision[:8]} ...")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                revision=revision, local_dir=str(out_dir)))


def fetch_extras(args) -> None:
    """SOMA-JAX ``--extras``: fetch what neither the submodule nor upstream ships."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloadable = {name for name, _, _ in DOWNLOADS}
    missing_sub = [rel for rel in REQUIRED_FILES
                   if rel not in downloadable and resolve(rel, required=False) is None]
    if missing_sub:
        logger.warning(f"NOTE: {len(missing_sub)} vendored asset(s) missing "
              f"({', '.join(missing_sub[:3])}{'...' if len(missing_sub) > 3 else ''}).")
        logger.warning("      These are not downloaded — run:")
        logger.warning("      git submodule update --init --recursive\n")

    failures = []
    for filename, sha256, description in DOWNLOADS:
        target = out_dir / filename
        existing = resolve(filename, required=False)
        if existing and not args.force:
            if sha256 is not None and _sha256(existing) != sha256:
                logger.warning(f"{filename}: present at {existing} but CHECKSUM MISMATCH — re-downloading")
            else:
                logger.info(f"Skipping {filename} (present at {existing}; --force to re-download)")
                continue

        logger.info(f"\n{filename}: {description}")
        try:
            got = download(args.repo_id, args.extras_revision, filename, out_dir)
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            logger.warning(f"  Download manually to {target}, or convert it from point.pca with "
                  "tools/convert/convert_gm_pca_to_npz.py — see docs/INSTALL.md")
            failures.append(filename)
            continue

        if got != target:
            got.replace(target)
        if sha256 is not None:
            digest = _sha256(target)
            if digest != sha256:
                logger.warning(f"  CHECKSUM MISMATCH\n    expected {sha256}\n    got      {digest}")
                failures.append(filename)
                continue
            logger.info("  sha256 OK")
        logger.info(f"  Saved to {target}")

    logger.info(f"\nDownload directory: {out_dir.resolve()}")
    logger.info(f"Vendored assets:    {SUBMODULE_ASSETS}")
    if failures:
        sys.exit(f"FAILED: {', '.join(failures)}")
    logger.info("Next (optional): python tools/pipeline/build_soma_rig.py "
          "-> assets/SOMA_neutral_fixed.npz, the torch/usd-free runtime archive for "
          "SOMALayer.load(). SOMALayer.from_upstream_assets() needs nothing further.")


def main():
    args = parse_args()
    if args.check:
        sys.exit(check())
    if args.extras:
        fetch_extras(args)
        return
    download_assets(target_dir=args.target_dir, revision=args.revision)


if __name__ == "__main__":
    main()
