"""Fetch the assets SOMA-JAX needs that are not already vendored.

Most model data ships inside the **`third_party/SOMA-X` submodule**
(`third_party/SOMA-X/assets/`): the template rig, the procedural-transform
JSON, the correctives checkpoint, and the MHR / Anny / SMPL / SMPL-X /
GarmentMeasurements packs. Those are used in place — this script does not copy
them. Run ``git submodule update --init --recursive`` to get them.

What actually needs downloading is the **full-schema `SOMA_neutral.npz`**. The
submodule's copy is the slim v0.2.1 archive and omits every rig array the
conversion needs (`bind_pose_world`, `bind_pose_local`, `bind_shape`,
`joint_names`, `joint_parent_ids`, `t_pose_world`, `t_pose_local` and the four
`skinning_weights_*` entries), so `SOMALayer.load` would fail with a
missing-key error. It is fetched from an immutable revision of
`nvidia/SOMA-X` on HuggingFace and checksum-verified.

Downloads land in ``assets/third_party/`` (git-ignored). Derived artefacts that
this repo *builds* — notably ``assets/SOMA_neutral_fixed.npz``, see
``docs/INSTALL.md`` §4.2 — live one level up in ``assets/``.

Upstream source repositories, for provenance and for fetching anything further:

    SOMA-X   https://github.com/NVlabs/SOMA-X
    MHR      https://github.com/facebookresearch/MHR
    Anny     https://github.com/naver/anny

Usage::

    python tools/download_assets.py              # -> assets/third_party/
    python tools/download_assets.py --check      # report what is missing, download nothing
    python tools/download_assets.py --force
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from soma_jax.assets import (  # noqa: E402
    DOWNLOAD_ASSETS,
    HF_REPO_ID,
    HF_REVISION,
    REQUIRED_FILES,
    SOURCE_REPOSITORIES,
    SUBMODULE_ASSETS,
    resolve,
)

# (filename, sha256 or None, description)
DOWNLOADS = [
    ("SOMA_neutral.npz",
     "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f",
     "full-schema SOMA rig — mean shape, PCA basis, sparse skinning, joint tree"),
    # The submodule ships only mean.obj / SOMA_wrap.obj for this pack.
    ("GarmentMeasurements/point.npz", None,
     "CAESARS garment PCA archive (not vendored in the submodule)"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Download the assets not vendored in third_party/SOMA-X.",
        epilog="Source repositories: "
               + ", ".join(f"{k} {v}" for k, v in SOURCE_REPOSITORIES.items()),
    )
    p.add_argument("--output-dir", default=str(DOWNLOAD_ASSETS),
                   help=f"download target (default: {DOWNLOAD_ASSETS})")
    p.add_argument("--repo-id", default=HF_REPO_ID,
                   help=f"HuggingFace repository (default: {HF_REPO_ID})")
    p.add_argument("--revision", default=HF_REVISION,
                   help="immutable revision to pin (default: the one INSTALL.md verifies)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--check", action="store_true",
                   help="report availability and exit without downloading")
    return p.parse_args()


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

    print("required assets:")
    for rel in REQUIRED_FILES:
        path = resolve(rel, required=False)
        if path is None:
            print(f"  [MISSING] {rel}")
            missing.append(rel)
            continue
        origin = "submodule" if str(SUBMODULE_ASSETS) in str(path) else "download"
        sha = expected.get(rel)
        if sha is not None:
            digest = _sha256(path)
            if digest != sha:
                print(f"  [CORRUPT] {rel}  ({origin}) sha256 {digest[:12]}... != {sha[:12]}...")
                corrupt.append(rel)
                continue
            print(f"  [ok     ] {rel}  ({origin}, sha256 verified)")
        else:
            print(f"  [ok     ] {rel}  ({origin})")

    if missing:
        print("\n  -> git submodule update --init --recursive   (for vendored assets)")
        print("  -> python tools/download_assets.py           (for the rest)")
    if corrupt:
        print("\n  -> python tools/download_assets.py --force   (re-download corrupt files)")
    if not missing and not corrupt:
        print("\nall required assets present and verified.")
    return 1 if (missing or corrupt) else 0


def download(repo_id: str, revision: str, filename: str, out_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub is not installed.\n  pip install huggingface_hub")
    print(f"Downloading {filename} from {repo_id}@{revision[:8]} ...")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                revision=revision, local_dir=str(out_dir)))


def main():
    args = parse_args()
    if args.check:
        sys.exit(check())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloadable = {name for name, _, _ in DOWNLOADS}
    missing_sub = [rel for rel in REQUIRED_FILES
                   if rel not in downloadable and resolve(rel, required=False) is None]
    if missing_sub:
        print(f"NOTE: {len(missing_sub)} vendored asset(s) missing "
              f"({', '.join(missing_sub[:3])}{'...' if len(missing_sub) > 3 else ''}).")
        print("      These are not downloaded — run:")
        print("      git submodule update --init --recursive\n")

    failures = []
    for filename, sha256, description in DOWNLOADS:
        target = out_dir / filename
        existing = resolve(filename, required=False)
        if existing and not args.force:
            if sha256 is not None and _sha256(existing) != sha256:
                print(f"{filename}: present at {existing} but CHECKSUM MISMATCH — re-downloading")
            else:
                print(f"Skipping {filename} (present at {existing}; --force to re-download)")
                continue

        print(f"\n{filename}: {description}")
        try:
            got = download(args.repo_id, args.revision, filename, out_dir)
        except Exception as e:
            print(f"  Failed: {e}")
            print(f"  Download manually to {target} — see docs/INSTALL.md §4.1")
            failures.append(filename)
            continue

        if got != target:
            got.replace(target)
        if sha256 is not None:
            digest = _sha256(target)
            if digest != sha256:
                print(f"  CHECKSUM MISMATCH\n    expected {sha256}\n    got      {digest}")
                failures.append(filename)
                continue
            print("  sha256 OK")
        print(f"  Saved to {target}")

    print(f"\nDownload directory: {out_dir.resolve()}")
    print(f"Vendored assets:    {SUBMODULE_ASSETS}")
    if failures:
        sys.exit(f"FAILED: {', '.join(failures)}")
    print("Next: build the runtime archive — docs/INSTALL.md §4.2 "
          "-> assets/SOMA_neutral_fixed.npz")


if __name__ == "__main__":
    main()
