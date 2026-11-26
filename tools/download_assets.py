"""Fetch the assets SOMA-JAX needs that a checkout does not already provide.

Upstream: ``tools/download_assets.py``. Upstream's command downloads the
``nvidia/soma-x`` HuggingFace asset snapshot for its release into the
HuggingFace cache; :func:`download_assets` and the ``--target-dir`` /
``--revision`` options do exactly that here too (through
:func:`soma_jax.assets.get_assets_dir`, so ``huggingface_hub`` is needed).

SOMA-JAX reads its assets from the **`third_party/SOMA-X` submodule** instead
(`third_party/SOMA-X/assets/`), which carries that same snapshot at the release
the code tracks — ``git submodule update --init --recursive`` is all it needs.
So the bare command does what *this* layout needs:

* Upstream stores the submodule's assets in **Git LFS**. A clone made without
  git-lfs active gets ~130-byte pointer stubs instead, and a GitHub "Download
  ZIP" gets no submodule at all. For either case the files of the pinned
  submodule commit are fetched and verified against their sha256 — no git-lfs
  needed (``--no-lfs`` skips this). A copy fetched for an earlier submodule pin
  is refetched.
* **`GarmentMeasurements/point.npz`** (the GarmentMeasurements shape PCA) ships
  with neither upstream nor the submodule; it is fetched from the older
  immutable `nvidia/SOMA-X` HuggingFace revision that published it. Upstream's
  own instruction (`docs/data_assets.md`) is to convert it from the public
  `point.pca` of https://github.com/mbotsch/GarmentMeasurements::

      python tools/convert/convert_gm_pca_to_npz.py /path/to/point.pca \\
          assets/third_party/GarmentMeasurements/point.npz

Downloads land in ``assets/third_party/`` (git-ignored). Derived artefacts that
this repo *builds* — notably the optional runtime archive
``assets/SOMA_neutral_fixed.npz`` (``tools/pipeline/build_soma_rig.py``) — live
one level up in ``assets/``.

The default path needs only the Python standard library: it does not import
JAX (so ``--check`` works on a machine where nothing is installed yet), and it
uses ``huggingface_hub`` when available but falls back to plain HTTPS
otherwise. ``HF_ENDPOINT`` (a HuggingFace mirror) and ``HF_TOKEN`` are honoured
either way.

Usage::

    python tools/download_assets.py              # what the submodule checkout lacks
    python tools/download_assets.py --minimal    # only what SOMALayer.from_upstream_assets() needs
    python tools/download_assets.py --check      # report what is missing, download nothing
    python tools/download_assets.py --extras     # only GarmentMeasurements/point.npz
    python tools/download_assets.py --only SOMAHand.npz MANO/base_hand_left.obj
    python tools/download_assets.py --force
    python tools/download_assets.py --target-dir DIR --revision v0.3.6   # upstream's download
    python tools/download_assets.py --print-table   # maintainers: manifest for a submodule bump
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

# (filename, sha256, size, description) — fetched from HuggingFace at A.HF_REVISION.
HF_DOWNLOADS = [
    # The submodule ships only mean.obj / SOMA_wrap.obj for this pack.
    ("GarmentMeasurements/point.npz",
     "6ae75a1ab7a8ae4f46bac503146976ba59cf918bca8aefe7320a3ed02aa2416a", 8626920,
     "GarmentMeasurements shape PCA (shipped by neither upstream nor the submodule)"),
]

#: What ``SOMALayer.from_upstream_assets()`` reads; ``--minimal`` fetches only these.
MINIMAL_FILES = ("SOMA_neutral.npz", "SOMA_template_rig.usda",
                 "SOMA_procedural_transforms.json", "correctives_model.pt")

_USER_AGENT = "soma-jax-download-assets"


def download_assets(target_dir=None, revision="main"):
    """Download the SOMA asset snapshot from HuggingFace (upstream ``download_assets``).

    Args:
        target_dir: If provided, used as the HuggingFace cache directory.
            The actual assets will be stored in a subdirectory managed by
            huggingface_hub.  If None, uses the default HF cache
            (``~/.cache/huggingface/hub/``).
        revision: Git revision (branch, tag, or commit hash) to download.

    Returns:
        Path to the downloaded assets directory.
    """
    path = A.get_assets_dir(revision=revision, cache_dir=target_dir)
    print(f"Assets downloaded to: {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download the SOMA-JAX assets a checkout does not provide.",
        epilog="Source repositories: "
               + ", ".join(f"{k} {v}" for k, v in A.SOURCE_REPOSITORIES.items()),
    )
    # Upstream's options: the HuggingFace snapshot, as upstream downloads it.
    p.add_argument("--target-dir", default=None,
                   help="upstream: download the HuggingFace snapshot, using DIR as the "
                        "HuggingFace cache")
    p.add_argument("--revision", default=None,
                   help="upstream: download the HuggingFace snapshot at this revision "
                        "(upstream's default: main)")
    # SOMA-JAX's options: fill in what the vendored layout lacks.
    p.add_argument("--output-dir", default=str(A.DOWNLOAD_ASSETS),
                   help=f"download target (default: {A.DOWNLOAD_ASSETS}). A directory "
                        f"outside the checkout must be added to ${A.ASSETS_ENV_VAR}.")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--check", action="store_true",
                   help="report availability and exit without downloading")
    p.add_argument("--minimal", action="store_true",
                   help="fetch only what SOMALayer.from_upstream_assets() needs "
                        f"({', '.join(MINIMAL_FILES)})")
    p.add_argument("--hand", action="store_true",
                   help="with --minimal: also the SOMA Hand layer's files (SOMAHand.npz, "
                        "MANO packs), which the default run already includes")
    p.add_argument("--extras", action="store_true",
                   help="fetch only the files the submodule does not ship "
                        "(GarmentMeasurements/point.npz)")
    p.add_argument("--no-lfs", action="store_true",
                   help="do not fetch submodule files that are missing or Git LFS stubs")
    p.add_argument("--only", nargs="+", metavar="FILE", default=None,
                   help="fetch (or --check) just these relative paths, e.g. "
                        "Anny/base_body.obj — any required asset or any file of the "
                        "pinned submodule, such as example_animation.usd")
    p.add_argument("--print-table", nargs="?", const="HEAD", metavar="COMMIT",
                   help="maintainers: print soma_jax/assets.py's SOMA_X_COMMIT and "
                        "SOMA_X_ASSET_FILES for a submodule commit (default: the "
                        "checked-out one) and exit")
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


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT / "third_party" / "SOMA-X"), *args],
                          capture_output=True, timeout=120, check=True)


def _submodule_commit() -> str:
    """Commit the submodule is checked out at; the pinned one when unknown."""
    sub = REPO_ROOT / "third_party" / "SOMA-X"
    try:
        commit = _git("rev-parse", "HEAD").stdout.decode().strip()
        # A submodule that was never initialised resolves to the superproject.
        top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
        if Path(top).resolve() == sub.resolve() and len(commit) == 40:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return A.SOMA_X_COMMIT


def print_table(commit: str) -> int:
    """Print the ``SOMA_X_COMMIT`` / ``SOMA_X_ASSET_FILES`` block for ``commit``.

    Reads the submodule's git objects: an LFS pointer gives the real file's
    sha256 and size; any other file is hashed as stored.
    """
    try:
        commit = _git("rev-parse", f"{commit}^{{commit}}").stdout.decode().strip()
        listing = _git("ls-tree", "-r", "-l", commit, "assets/").stdout.decode()
    except (OSError, subprocess.SubprocessError) as e:
        print(f"cannot read the submodule at {commit}: {e}", file=sys.stderr)
        return 1
    rows = []
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        _, kind, blob, _size = meta.split()
        if kind != "blob":
            continue
        rel = path[len("assets/"):]
        if rel.startswith("images/"):          # README images, not model data
            continue
        data = _git("cat-file", "blob", blob).stdout
        if data.startswith(A._LFS_POINTER_PREFIX):
            fields = dict(ln.split(" ", 1) for ln in data.decode().splitlines() if " " in ln)
            rows.append((rel, fields["oid"].split(":", 1)[1], int(fields["size"]), True))
        else:
            rows.append((rel, hashlib.sha256(data).hexdigest(), len(data), False))
    print(f'SOMA_X_COMMIT = "{commit}"')
    print("SOMA_X_ASSET_FILES = {")
    for rel, sha, size, is_lfs in sorted(rows):
        print(f'    "{rel}": (\n        "{sha}", {size}, {is_lfs}),')
    print("}")
    return 0


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


def download_submodule_file(rel: str, target: Path, commit: str) -> None:
    """A submodule asset at ``commit``: GitHub first, then HuggingFace.

    The HuggingFace snapshot at the release tag holds the same files as the
    submodule at that release; every download is sha256-verified, so a file
    that differs there is rejected, never used.
    """
    sha256, size, is_lfs = _expected(rel)
    if sha256 is None:
        raise RuntimeError(f"no checksum known for {rel}; cannot verify a download")
    host = "media.githubusercontent.com/media" if is_lfs else "raw.githubusercontent.com"
    sources = [
        ("GitHub", f"https://{host}/NVlabs/SOMA-X/{commit}/assets/{rel}", {}),
        ("HuggingFace",
         f"{_hf_base()}/{A.REPO_ID}/resolve/{A.UPSTREAM_ASSET_REVISION}/{rel}",
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
    """``ok`` / ``lfs-stub`` / ``stale`` / ``missing`` for one file."""
    path = A.resolve(rel, required=False)
    if path is not None:
        return "ok", path
    pinned = A.lfs_pointer(A.SUBMODULE_ASSETS / rel)
    for root in A._search_roots():
        candidate = root / rel
        if not candidate.exists():
            continue
        if A.lfs_pointer(candidate) is None and pinned is not None:
            return "stale", candidate
    if pinned is not None:
        return "lfs-stub", A.SUBMODULE_ASSETS / rel
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

    print("assets:")
    for rel in files:
        state, path = _status(rel)
        if state == "lfs-stub":
            print(f"  [LFS-STUB] {rel}  ({path} is a Git LFS pointer, not the data)")
            stubs.append(rel)
            continue
        if state == "stale":
            print(f"  [STALE   ] {rel}  ({path} is another release's file)")
            stubs.append(rel)
            continue
        if state != "ok":
            print(f"  [MISSING ] {rel}")
            missing.append(rel)
            continue
        origin = "submodule" if str(A.SUBMODULE_ASSETS) in str(path) else "download"
        sha = expected.get(rel) or _expected(rel)[0]
        if sha is not None and origin == "download":
            digest = _sha256(path)
            if digest != sha:
                print(f"  [CORRUPT ] {rel}  ({origin}) sha256 {digest[:12]}... != {sha[:12]}...")
                corrupt.append(rel)
                continue
            print(f"  [ok      ] {rel}  ({origin}, sha256 verified)")
        else:
            print(f"  [ok      ] {rel}  ({origin})")

    if stubs:
        print("\n  Git LFS stubs or another release's files. Either")
        print("  -> git lfs install && git submodule foreach --recursive git lfs pull")
        print("  -> python tools/download_assets.py           (no git-lfs needed)")
    if missing:
        print("\n  -> git submodule update --init --recursive   (for vendored assets)")
        print("  -> python tools/download_assets.py           (for the rest)")
    if corrupt:
        print("\n  -> python tools/download_assets.py --force   (re-download corrupt files)")
    if not (missing or corrupt or stubs):
        print("\nall assets present and verified.")
    return 1 if (missing or corrupt or stubs) else 0


def main():
    args = parse_args()
    if args.print_table is not None:
        sys.exit(print_table(args.print_table))
    if args.target_dir is not None or args.revision is not None:
        download_assets(target_dir=args.target_dir, revision=args.revision or "main")
        return

    hf_names = tuple(name for name, _, _, _ in HF_DOWNLOADS)
    if args.only:
        known = (set(A.REQUIRED_FILES) | set(A.HAND_REQUIRED_FILES)
                 | set(A.SOMA_X_ASSET_FILES) | set(hf_names))
        unknown = [f for f in args.only if f not in known]
        if unknown:
            sys.exit(f"unknown asset(s): {', '.join(unknown)}\n"
                     f"choose from: {', '.join(sorted(known))}")
        files = tuple(dict.fromkeys(args.only))
    elif args.extras:
        files = hf_names
    else:
        # Default: everything the package and the ported tools read — the
        # body's data_root contract, the hand layer's files and the example
        # clips (the USD demo motion, the SMPL clip smpl2soma converts).
        files = (MINIMAL_FILES if args.minimal else
                 A.REQUIRED_FILES + A.HAND_REQUIRED_FILES + A.EXAMPLE_FILES)
        if args.hand:
            files = files + A.HAND_REQUIRED_FILES
        files = tuple(dict.fromkeys(files))
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
            download_hf(A.HF_REPO_ID, A.HF_REVISION, filename, out_dir, sha256, size)
        except Exception as e:  # noqa: BLE001 — report every failure, then continue
            print(f"  Failed: {e}")
            print(f"  Download manually to {target} — see docs/INSTALL.md §4.2")
            failures.append(filename)
            continue
        print(f"  sha256 OK — saved to {target}")

    pending = [rel for rel in files if rel not in hf_names
               and (args.force or _status(rel)[0] != "ok")]
    if pending and args.no_lfs:
        print(f"\nNOTE: {len(pending)} submodule asset(s) missing or Git LFS stubs "
              f"({', '.join(pending[:3])}{'...' if len(pending) > 3 else ''}); "
              "skipped because of --no-lfs.")
    elif pending:
        commit = _submodule_commit()
        print(f"\nSubmodule assets missing, stored as Git LFS stubs, or from another "
              f"release: {len(pending)} (fetching NVlabs/SOMA-X@{commit[:8]} without git-lfs)")
        for rel in pending:
            target = out_dir / rel
            print(f"\n{rel}")
            try:
                download_submodule_file(rel, target, commit)
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
    print("All requested assets are in place.")
    if all(_status(rel)[0] == "ok" for rel in MINIMAL_FILES):
        print("SOMALayer.from_upstream_assets() needs nothing else "
              "(the runtime archive of docs/INSTALL.md §4.3 is optional).")


if __name__ == "__main__":
    main()
