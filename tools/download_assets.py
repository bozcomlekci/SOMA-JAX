"""Fetch the assets SOMA-JAX needs that a checkout does not already provide.

Most model data ships inside the **`third_party/SOMA-X` submodule**
(`third_party/SOMA-X/assets/`): the template rig, the procedural-transform
JSON, the correctives checkpoint, and the MHR / Anny / SMPL / SMPL-X /
GarmentMeasurements packs. Those are used in place — this script does not copy
them. Run ``git submodule update --init --recursive`` to get them.

Upstream stores all of those files in **Git LFS**. A clone made without git-lfs
active gets ~130-byte pointer stubs instead, and a GitHub "Download ZIP" gets no
submodule at all. For either case this script fetches the real files for the
pinned submodule commit and verifies them against their sha256 — no git-lfs
needed. Pass ``--no-lfs`` to skip that step.

What always needs downloading is the **full-schema `SOMA_neutral.npz`**. The
submodule's copy is the slim v0.2.1 archive and omits every rig array the
conversion needs (`bind_pose_world`, `bind_pose_local`, `bind_shape`,
`joint_names`, `joint_parent_ids`, `t_pose_world`, `t_pose_local` and the four
`skinning_weights_*` entries), so `SOMALayer.load` would fail with a
missing-key error. It is fetched from an immutable revision of
`nvidia/SOMA-X` on HuggingFace and checksum-verified.

Downloads land in ``assets/third_party/`` (git-ignored). Derived artefacts that
this repo *builds* — notably ``assets/SOMA_neutral_fixed.npz``, see
``docs/INSTALL.md`` §4.2 — live one level up in ``assets/``.

The script needs only the Python standard library: it does not import JAX (so
``--check`` works on a machine where nothing is installed yet), and it uses
``huggingface_hub`` when available but falls back to plain HTTPS otherwise.
``HF_ENDPOINT`` (a HuggingFace mirror) and ``HF_TOKEN`` are honoured either way.

Upstream source repositories, for provenance and for fetching anything further:

    SOMA-X   https://github.com/NVlabs/SOMA-X
    MHR      https://github.com/facebookresearch/MHR
    Anny     https://github.com/naver/anny

Usage::

    python tools/download_assets.py              # -> assets/third_party/
    python tools/download_assets.py --minimal    # only what SOMALayer.from_upstream_assets() needs
    python tools/download_assets.py --check      # report what is missing, download nothing
    python tools/download_assets.py --force
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_assets_module():
    """Import ``soma_jax/assets.py`` on its own.

    ``import soma_jax.assets`` would run ``soma_jax/__init__``, which imports
    JAX; this script has to work before the package — or JAX — is installed.
    """
    spec = importlib.util.spec_from_file_location(
        "_soma_jax_assets", REPO_ROOT / "soma_jax" / "assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load_assets_module()

# (filename, sha256, size, description) — fetched from HuggingFace at --revision.
HF_DOWNLOADS = [
    ("SOMA_neutral.npz",
     "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f", 27488638,
     "full-schema SOMA rig — mean shape, PCA basis, sparse skinning, joint tree"),
    # The submodule ships only mean.obj / SOMA_wrap.obj for this pack.
    ("GarmentMeasurements/point.npz",
     "6ae75a1ab7a8ae4f46bac503146976ba59cf918bca8aefe7320a3ed02aa2416a", 8626920,
     "CAESARS garment PCA archive (not vendored in the submodule)"),
]

#: What ``SOMALayer.from_upstream_assets()`` reads; ``--minimal`` fetches only these.
MINIMAL_FILES = ("SOMA_neutral.npz", "SOMA_template_rig.usda",
                 "SOMA_procedural_transforms.json")

#: A HuggingFace publish of nvidia/SOMA-X whose files are byte-identical to the
#: pinned submodule for most assets — a second source when GitHub is
#: unreachable. Every download is sha256-verified, so a file that differs there
#: (the v0.2.2 template rig, for one) is rejected, never used.
HF_MIRROR_REVISION = "4cfe361e6bf093b4918deab851f97c74dd061356"

_USER_AGENT = "soma-jax-download-assets"


def parse_args():
    p = argparse.ArgumentParser(
        description="Download the SOMA-JAX assets a checkout does not provide.",
        epilog="Source repositories: "
               + ", ".join(f"{k} {v}" for k, v in A.SOURCE_REPOSITORIES.items()),
    )
    p.add_argument("--output-dir", default=str(A.DOWNLOAD_ASSETS),
                   help=f"download target (default: {A.DOWNLOAD_ASSETS}). A directory "
                        f"outside the checkout must be added to ${A.ASSETS_ENV_VAR}.")
    p.add_argument("--repo-id", default=A.HF_REPO_ID,
                   help=f"HuggingFace repository (default: {A.HF_REPO_ID})")
    p.add_argument("--revision", default=A.HF_REVISION,
                   help="immutable revision to pin (default: the one INSTALL.md verifies)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--check", action="store_true",
                   help="report availability and exit without downloading")
    p.add_argument("--minimal", action="store_true",
                   help="fetch only what SOMALayer.from_upstream_assets() needs "
                        f"({', '.join(MINIMAL_FILES)})")
    p.add_argument("--no-lfs", action="store_true",
                   help="do not fetch submodule files that are missing or Git LFS stubs")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _submodule_commit() -> str:
    """Commit the submodule is checked out at; the pinned one when unknown."""
    sub = REPO_ROOT / "third_party" / "SOMA-X"
    try:
        out = subprocess.run(["git", "-C", str(sub), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30, check=True)
        commit = out.stdout.strip()
        # A submodule that was never initialised resolves to the superproject.
        top = subprocess.run(["git", "-C", str(sub), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=30, check=True)
        if Path(top.stdout.strip()).resolve() == sub.resolve() and len(commit) == 40:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return A.SOMA_X_COMMIT


def _expected(rel: str) -> tuple[str | None, int | None, bool]:
    """``(sha256, size, is_lfs)`` for a submodule file.

    The submodule's own LFS pointer is authoritative when present (it tracks
    the commit actually checked out); the pinned manifest covers the rest.
    """
    pointer = A.lfs_pointer(A.SUBMODULE_ASSETS / rel)
    if pointer is not None:
        return pointer[0], pointer[1], True
    sha, size, is_lfs = A.SOMA_X_ASSET_FILES.get(rel, (None, None, True))
    return sha, size, is_lfs


def _fetch(url: str, dest: Path, sha256: str | None = None, size: int | None = None,
           headers: dict | None = None, retries: int = 3) -> None:
    """Stream ``url`` into ``dest`` atomically, verifying size and sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT,
                                                       **(headers or {})})
            with urllib.request.urlopen(req, timeout=120) as r:
                total = int(r.headers.get("Content-Length") or 0)
                if size and total and total != size:
                    raise ValueError(f"server reports {total} bytes, expected {size}")
                h, done, step = hashlib.sha256(), 0, max(total // 10, 1 << 26)
                mark = step
                with open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if total and done >= mark:
                            print(f"    {_human(done)} / {_human(total)}", flush=True)
                            mark += step
            if size and done != size:
                raise ValueError(f"received {done} bytes, expected {size}")
            if sha256 and h.hexdigest() != sha256:
                raise ValueError(f"sha256 {h.hexdigest()[:12]}... != expected {sha256[:12]}...")
            tmp.replace(dest)
            return
        except ValueError as e:          # wrong content: retrying the same URL won't help
            tmp.unlink(missing_ok=True)
            raise RuntimeError(str(e)) from None
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            last_err = e
            if e.code in (401, 403, 404):
                break
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            tmp.unlink(missing_ok=True)
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{last_err}")


def _hf_base() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_hf(repo_id: str, revision: str, filename: str, out_dir: Path,
                sha256: str | None, size: int | None) -> None:
    """One file from HuggingFace into ``out_dir / filename``.

    Uses ``huggingface_hub`` when installed, plain HTTPS otherwise.
    """
    target = out_dir / filename
    print(f"  from {repo_id}@{revision[:8]} (HuggingFace) ...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    if hf_hub_download is not None:
        got = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                   revision=revision, local_dir=str(out_dir)))
        if got.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            got.replace(target)
        if sha256 is not None and _sha256(target) != sha256:
            target.unlink(missing_ok=True)
            raise RuntimeError("sha256 mismatch after download")
        return
    _fetch(f"{_hf_base()}/{repo_id}/resolve/{revision}/{filename}", target,
           sha256=sha256, size=size, headers=_hf_headers())


def download_submodule_file(rel: str, target: Path, commit: str, repo_id: str) -> None:
    """A submodule asset at ``commit``: GitHub first, then the HF mirror."""
    sha256, size, is_lfs = _expected(rel)
    if sha256 is None:
        raise RuntimeError(f"no checksum known for {rel}; cannot verify a download")
    host = "media.githubusercontent.com/media" if is_lfs else "raw.githubusercontent.com"
    sources = [
        ("GitHub", f"https://{host}/NVlabs/SOMA-X/{commit}/assets/{rel}", {}),
        ("HuggingFace", f"{_hf_base()}/{repo_id}/resolve/{HF_MIRROR_REVISION}/{rel}",
         _hf_headers()),
    ]
    errors = []
    for label, url, headers in sources:
        print(f"  from {label} ({_human(size) if size else '?'}) ...", flush=True)
        try:
            _fetch(url, target, sha256=sha256, size=size, headers=headers)
            return
        except RuntimeError as e:
            print(f"    {label} failed: {e}")
            errors.append(f"{label}: {e}")
    raise RuntimeError("; ".join(errors))


def _status(rel: str) -> tuple[str, Path | None]:
    """``ok`` / ``lfs-stub`` / ``slim`` / ``missing`` for one required file."""
    path = A.resolve(rel, required=False)
    if path is not None:
        return "ok", path
    for root in A._search_roots():
        candidate = root / rel
        if candidate.exists() and A.lfs_pointer(candidate) is not None:
            return "lfs-stub", candidate
    if rel == "SOMA_neutral.npz" and (A.SUBMODULE_ASSETS / rel).exists():
        return "slim", A.SUBMODULE_ASSETS / rel
    return "missing", None


def check(files) -> int:
    """Report per-FILE availability. Returns a process exit code.

    Enumerates individual files rather than family directories: an empty
    ``GarmentMeasurements/`` would otherwise be reported healthy. Files with a
    known checksum are re-hashed, so a corrupt one is reported rather than
    silently skipped on the next run.
    """
    expected = {name: sha for name, sha, _, _ in HF_DOWNLOADS}
    missing, corrupt, stubs = [], [], []
    roots = A.env_asset_roots()
    if roots:
        print(f"${A.ASSETS_ENV_VAR}: " + os.pathsep.join(str(r) for r in roots))

    print("required assets:")
    for rel in files:
        state, path = _status(rel)
        if state == "lfs-stub":
            print(f"  [LFS-STUB] {rel}  ({path} is a Git LFS pointer, not the data)")
            stubs.append(rel)
            continue
        if state != "ok":
            print(f"  [MISSING ] {rel}" + ("  (only the slim submodule archive exists)"
                                          if state == "slim" else ""))
            missing.append(rel)
            continue
        origin = "submodule" if str(A.SUBMODULE_ASSETS) in str(path) else "download"
        sha = expected.get(rel) or A.SOMA_X_ASSET_FILES.get(rel, (None,))[0]
        if sha is not None and (origin == "download" or rel in expected):
            digest = _sha256(path)
            if digest != sha:
                print(f"  [CORRUPT ] {rel}  ({origin}) sha256 {digest[:12]}... != {sha[:12]}...")
                corrupt.append(rel)
                continue
            print(f"  [ok      ] {rel}  ({origin}, sha256 verified)")
        else:
            print(f"  [ok      ] {rel}  ({origin})")

    if stubs:
        print("\n  Git LFS stubs: the checkout was made without git-lfs active. Either")
        print("  -> git lfs install && git submodule foreach --recursive git lfs pull")
        print("  -> python tools/download_assets.py           (no git-lfs needed)")
    if missing:
        print("\n  -> git submodule update --init --recursive   (for vendored assets)")
        print("  -> python tools/download_assets.py           (for the rest)")
    if corrupt:
        print("\n  -> python tools/download_assets.py --force   (re-download corrupt files)")
    if not (missing or corrupt or stubs):
        print("\nall required assets present and verified.")
    return 1 if (missing or corrupt or stubs) else 0


def main():
    args = parse_args()
    files = MINIMAL_FILES if args.minimal else A.REQUIRED_FILES
    if args.check:
        sys.exit(check(files))

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for filename, sha256, size, description in HF_DOWNLOADS:
        if filename not in files:
            continue
        target = out_dir / filename
        existing = A.resolve(filename, required=False)
        if existing and not args.force:
            if _sha256(existing) != sha256:
                print(f"{filename}: present at {existing} but CHECKSUM MISMATCH — re-downloading")
            else:
                print(f"Skipping {filename} (present at {existing}; --force to re-download)")
                continue
        print(f"\n{filename}: {description}")
        try:
            download_hf(args.repo_id, args.revision, filename, out_dir, sha256, size)
        except Exception as e:  # noqa: BLE001 — report every failure, then continue
            print(f"  Failed: {e}")
            print(f"  Download manually to {target} — see docs/INSTALL.md §4.1")
            failures.append(filename)
            continue
        print(f"  sha256 OK — saved to {target}")

    hf_names = {name for name, _, _, _ in HF_DOWNLOADS}
    pending = [rel for rel in files if rel not in hf_names
               and (args.force or _status(rel)[0] != "ok")]
    if pending and args.no_lfs:
        print(f"\nNOTE: {len(pending)} submodule asset(s) missing or Git LFS stubs "
              f"({', '.join(pending[:3])}{'...' if len(pending) > 3 else ''}); "
              "skipped because of --no-lfs.")
    elif pending:
        commit = _submodule_commit()
        print(f"\nSubmodule assets missing or stored as Git LFS stubs: {len(pending)} "
              f"(fetching NVlabs/SOMA-X@{commit[:8]} without git-lfs)")
        for rel in pending:
            target = out_dir / rel
            print(f"\n{rel}")
            try:
                download_submodule_file(rel, target, commit, args.repo_id)
            except Exception as e:  # noqa: BLE001
                print(f"  Failed: {e}")
                failures.append(rel)
                continue
            print(f"  sha256 OK — saved to {target}")

    print(f"\nDownload directory: {out_dir.resolve()}")
    print(f"Vendored assets:    {A.SUBMODULE_ASSETS}")
    outside = out_dir.resolve() not in {r.resolve() for r in A._search_roots()}
    if outside:
        print(f"NOTE: {out_dir} is not on the asset search path; "
              f"export {A.ASSETS_ENV_VAR}={out_dir.resolve()}")
    if failures:
        sys.exit(f"FAILED: {', '.join(failures)}")
    print("All requested assets are in place. SOMALayer.from_upstream_assets() "
          "needs nothing else (docs/INSTALL.md §4.2 is optional).")


if __name__ == "__main__":
    main()
