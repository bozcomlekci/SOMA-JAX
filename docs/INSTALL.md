# Installation

SOMA-JAX is a pure-JAX library. The core install needs only JAX + a few
scientific-Python packages; visualization and SOMA-X parity/benchmarks add
optional dependencies. Python **3.10+** is required.

## 1. Clone (with submodules)

The repo pulls two Git submodules: upstream `SOMA-X` (the parity reference)
and [`SMPL-JAX`](https://github.com/bozcomlekci/SMPL-JAX) (pure-JAX SMPL /
SMPL-X, which the retargeting tools and two test modules import as
`smpl_jax`):

```bash
git clone --recurse-submodules https://github.com/bozcomlekci/SOMA-JAX.git
cd SOMA-JAX
# if you already cloned without submodules:
git submodule update --init --recursive
```

**Git LFS.** Upstream SOMA-X stores every file under `assets/` in Git LFS. With
git-lfs installed *and initialised* (`git lfs install`, once per machine) the
clone fetches them; otherwise each one arrives as a ~130-byte pointer stub.
SOMA-JAX detects stubs rather than handing them to a loader, and
`python tools/download_assets.py` (§4) fetches the real files for the pinned
submodule commit without git-lfs, verifying each against its sha256 — the same
script also covers a GitHub "Download ZIP" checkout, which has no submodules at
all. To use git-lfs instead:

```bash
git lfs install && git submodule foreach --recursive git lfs pull
```

## 2. Create an environment

Any Python 3.10+ environment works (conda/mamba, `venv`, `uv`, …):

```bash
conda create -n soma-jax python=3.10 -y && conda activate soma-jax
# or:  python -m venv .venv && source .venv/bin/activate
```

## 3. Install the package

```bash
pip install -e ".[dev,vis,usd]"      # or ".[all]", which adds the plotting extra
```

This installs:

- **core** — `jax`, `jaxlib`, `equinox`, `numpy`, `scipy`, `optax`
- **`dev`** — `pytest`, `pytest-xdist`
- **`vis`** — `trimesh`, `rtree`, `pyrender` (OBJ/PLY export, topology transfer
  for the non-SOMA identity backends, offscreen rendering, GIFs). `rtree` backs
  trimesh's closest-point queries; without it they fail, or
  `compute_barycentric_coords` falls back to a nearest-centroid approximation
  and warns.
- **`usd`** — `usd-core`. `SOMALayer.from_upstream_assets()` reads
  `SOMA_template_rig.usda` through it, so the quickstart needs it; so does
  USD import/export (`soma_jax.usd_io`). `SOMALayer.load()` of a prebuilt
  archive (§4.2) does not.
- **`plots`** — `matplotlib`, `tueplots`, for `benchmarks/plot_*.py`.

Every dependency ships binary wheels for Linux (x86-64, aarch64), macOS on
Apple silicon and Windows (x86-64), on Python 3.10–3.14 (current jaxlib has no
Intel-macOS wheels). The dependency floors are the oldest versions verified to work
together (`jax` 0.4.26, `optax` 0.2.2, `numpy` 1.24.1; see the comment in
`pyproject.toml`).

A **non-editable** install (`pip install .`, or `pip install
git+https://github.com/bozcomlekci/SOMA-JAX`) works too, but the package then
lives in `site-packages` and cannot see the checkout's asset directories.
Point it at them, or at any shared copy, with `SOMA_JAX_ASSETS`
(`os.pathsep`-separated; searched first):

```bash
export SOMA_JAX_ASSETS=/data/soma/assets                  # e.g. a cluster-wide copy
python tools/download_assets.py --output-dir "$SOMA_JAX_ASSETS"
```

### SMPL-JAX (optional)

`tools/pipeline/` and two test modules (`test_body_model_io.py`'s cross-check,
`test_motion_retarget.py`) use **`smpl_jax`**. It is not on PyPI under that
name here — install the submodule:

```bash
pip install -e third_party/SMPL-JAX
```

Without it the `tools/pipeline/` scripts fall back to `soma_jax.body_models`,
which gives bit-identical SMPL / SMPL-X vertices and joints, so they work on a
plain install; the two test modules skip (`pytest.importorskip`), which is part
of the gap between the minimal and full test runs in §5. Either way the SMPL
family's model files are registration-only and are not fetched for you.

By default `pip` installs the **CPU** build of JAX. For NVIDIA GPUs install a
CUDA build instead (see the [JAX install guide](https://docs.jax.dev/en/latest/installation.html)
for the current command), e.g.:

```bash
pip install -U "jax[cuda12]"
```

**Blackwell cards (RTX 50-series, `sm_120`) need CUDA ≥ 12.8.** The
`nvidia-*-cu12` wheels are pinned only as `>=`, so an environment assembled
around an older CUDA can leave you on cuBLAS 12.4 — which carries no `sm_120`
kernels and fails with `INTERNAL: the library was not initialized`, sometimes
only for particular shapes, which makes it look like a bug in the caller. Check
and fix with:

```bash
python -c "import jax; print(jax.devices())"
pip list | grep nvidia-cublas-cu12          # want >= 12.8
pip install -U nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
               nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12 \
               nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
```

## 4. Obtain and prepare the SOMA model asset

Model archives are deliberately excluded from Git, so a clone does **not**
contain any model data. One command fetches whatever this checkout lacks:

```bash
python tools/download_assets.py            # everything (~1.2 GB if no LFS data arrived)
python tools/download_assets.py --minimal  # only what from_upstream_assets() needs (~370 MB)
python tools/download_assets.py --only Anny/base_body.obj Anny/SOMA_wrap.obj   # just these
python tools/download_assets.py --check    # report, download nothing
```

It needs only the Python standard library — not JAX, not `huggingface_hub`
(used when installed; `HF_ENDPOINT` mirrors and `HF_TOKEN` are honoured either
way) — so it also runs before the package is installed. §4.1 below is the
manual equivalent for the one NVIDIA file that must come from Hugging Face; from
there `SOMALayer.from_upstream_assets()` builds the rig directly and nothing
further is required.

§4.2 additionally bakes those inputs into a single self-contained archive:

```text
assets/SOMA_neutral_fixed.npz
```

That archive is what `SOMALayer.load(...)` takes, and it is **not**
byte-compatible with NVIDIA's upstream `SOMA_neutral.npz`: SOMA-JAX uses
`v_template`, dense `weights`, `parents`, and `J_regressor`, while the upstream
archive uses `mean`, sparse skinning arrays, and `joint_parent_ids`. Passing the
upstream file straight to `load(...)` raises a `ValueError` that says so — use
`from_upstream_assets(npz_path=...)` for that file, or build the archive per §4.2.

### 4.0 Where assets live

Most model data already ships inside the **`third_party/SOMA-X` submodule**
(`third_party/SOMA-X/assets/`) — the template rig, procedural-transform JSON,
correctives checkpoint, and the MHR / Anny / SMPL / SMPL-X /
GarmentMeasurements packs. SOMA-JAX uses those in place rather than keeping a
second copy of ~1 GB of identical files, so `git submodule update --init
--recursive` covers them.

| Location | Contents | Tracked? |
|---|---|---|
| `$SOMA_JAX_ASSETS` | optional extra directories (`os.pathsep`-separated), searched first | — |
| `third_party/SOMA-X/assets/` | vendored upstream assets, used in place | submodule (Git LFS) |
| `assets/third_party/` | downloads (`tools/download_assets.py`), incl. stand-ins for LFS stubs | git-ignored |
| `assets/` | archives this repo *builds*, e.g. `SOMA_neutral_fixed.npz` | tracked (large binaries excluded by extension) |

`soma_jax.assets.resolve()` searches those in order, so code and tests never
hardcode a layout. It skips Git LFS pointer stubs and the slim
`SOMA_neutral.npz` wherever they are found, and names them in the error when
nothing usable exists. Check what is present with:

```bash
python tools/download_assets.py --check
```

Source repositories, if you need to fetch anything further:
[SOMA-X](https://github.com/NVlabs/SOMA-X) ·
[MHR](https://github.com/facebookresearch/MHR) ·
[Anny](https://github.com/naver/anny)

### 4.1 Download the public NVIDIA source asset

`tools/download_assets.py` does this for you. By hand: install the Hugging Face
client, then download the full-schema archive from an immutable revision of the
official [`nvidia/SOMA-X`](https://huggingface.co/nvidia/SOMA-X) repository:

```bash
pip install huggingface_hub
mkdir -p assets/third_party
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="nvidia/SOMA-X",
    filename="SOMA_neutral.npz",
    revision="466879a83d57eabf3d875ded2d869f2075f90348",
    local_dir="assets/third_party",
)
print(path)
PY
```

The downloaded file should have this checksum:

```bash
echo "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f  assets/third_party/SOMA_neutral.npz" \
  | sha256sum --check
```

Do not substitute `third_party/SOMA-X/assets/SOMA_neutral.npz` here. That is the
newer slim v0.2.1 archive: it has 30 keys where the full archive has 41, and is
missing every rig array this conversion needs — `bind_pose_world`,
`bind_pose_local`, `bind_shape`, `joint_names`, `joint_parent_ids`,
`t_pose_world`, `t_pose_local` and the four `skinning_weights_*` entries.
`soma_jax.assets.resolve()` refuses to return it for this reason. It is the
*only* asset that must be downloaded when the submodule's LFS data is present;
everything else comes from the submodule.

Do not substitute a newer `nvidia/SOMA-X` revision for the submodule's files
either: from v0.2.2 on, upstream ships a simplified `SOMA_template_rig.usda`
("removes unused helper joints", 28 MB instead of 345 MB) and a revised
procedural JSON, which are not the v0.2.1 rig this port is pinned and
parity-tested against. `download_assets.py` verifies every file against the
pinned commit's sha256 and rejects anything else.

### 4.2 Build the SOMA-JAX archive (optional)

> **You do not need this.** `SOMALayer.from_upstream_assets()` builds both of
> upstream's rigs directly from `SOMA_neutral.npz` + `SOMA_template_rig.usda` —
> no PyTorch, no upstream package — including upstream's *default* 122-joint
> procedural rig, whose 78-joint public half it derives with the ported
> `derive_soma_rig_without_procedural_joints` (it matches this archive to
> float32 round-off):
>
> ```python
> from soma_jax import SOMALayer
> layer = SOMALayer.from_upstream_assets()                   # procedural (default)
> layer = SOMALayer.from_upstream_assets(procedural=False)   # 78-joint public rig
> ```
>
> Build the archive below when you want a single self-contained file to ship, or
> as an independent check of the merge: the two routes are compared by
> `tests/test_rig_build.py`.

Run this once from the repository root. It reads the rig through the
**upstream SOMA-X layer** so the skinning weights and bind transforms come from
the canonical `SOMA_template_rig.usda` merge (SOMA-X overrides the npz rig with
the template USD — using npz arrays alone diverges at the leg weights), then
converts centimeters to meters, reshapes the PCA basis, and fits the affine
joint regressor. Requires the `third_party/SOMA-X` submodule plus its `torch`
and `pxr` (usd-core) dependencies:

```bash
python - <<'PY'
from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch

sys.path.insert(0, "third_party/SOMA-X")
from soma.soma import SOMALayer as TorchSOMA
from soma_jax.assets import data_root, resolve

spec = importlib.util.spec_from_file_location("bsr", "tools/pipeline/build_soma_rig.py")
bsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsr)

# The upstream layer is the authority on the merged rig (npz + template USD).
# data_root() is one directory over the submodule and the downloads, which is
# the layout upstream expects.
tl = TorchSOMA(data_root=str(data_root()), identity_model_type="soma", device="cpu",
               mode="dense", enable_procedural_transforms=False,
               correctives_model_path=None)
src = dict(np.load(resolve("SOMA_neutral.npz"), allow_pickle=False))

weights = tl.skeleton_transfer.skinning_weights.numpy().astype(np.float32)
bind_world = tl.skeleton_transfer.bind_world_transforms.numpy().astype(np.float32)
bind_local = tl.skeleton_transfer.bind_local_transforms.numpy().astype(np.float32)
parents = np.asarray([int(p) for p in tl.skeleton_transfer.joint_parent_ids], np.int32)

vertex_count = src["mean"].shape[0]
component_count = src["eigenvalues"].shape[0]
fit_parents = parents.astype(int).copy(); fit_parents[0] = 0
joint_count = weights.shape[1]
children = {j: [c for c in range(joint_count) if fit_parents[c] == j and c != j]
            for j in range(joint_count)}
joint_regressor = bsr.build_regressor(
    src["bind_shape"].astype(np.float64), bind_world[:, :3, 3].astype(np.float64),
    weights, fit_parents, children,
)

rig = dict(tl.rig_data)
asset = {
    "v_template": src["mean"].astype(np.float32) / 100.0,
    "faces": src["triangles"].astype(np.int32),
    "parents": parents,
    "joint_names": src["joint_names"],
    "weights": weights,                              # template-merged (production rig)
    "J_regressor": joint_regressor.astype(np.float32),
    "shapedirs": (src["shapedirs"]
                  .reshape(component_count, vertex_count, 3)
                  .transpose(1, 2, 0) / 100.0).astype(np.float32),
    # SOMA-X weights identity coefficients by sqrt(eigenvalues) before the
    # basis matmul; SOMAIdentityModel applies the same scaling via this key.
    "eigenvalues": src["eigenvalues"].astype(np.float32),
    # Canonical bind-pose mesh (native cm): enables the faithful per-identity
    # skeleton fit (SkeletonTransfer RBF + two-stage Kabsch) inside SOMALayer.
    "bind_shape": src["bind_shape"].astype(np.float32),
    "bind_pose_world": bind_world,                   # template-merged
    "bind_pose_local": bind_local,                   # template-merged
    "t_pose_world": np.asarray(rig.get("t_pose_world", src["t_pose_world"]), np.float32),
    "t_pose_local": np.asarray(rig.get("t_pose_local", src["t_pose_local"]), np.float32),
}
for key in ("mirror_vert_indices", "segment_eye_bags", "segment_mouth_bag",
            "lod_mid_to_low", "triangles_low"):
    asset[key] = src[key]

Path("assets").mkdir(exist_ok=True)
np.savez_compressed("assets/SOMA_neutral_fixed.npz", **asset)
print("Wrote assets/SOMA_neutral_fixed.npz")
PY
```

With this asset, the SOMA-JAX forward (`SOMALayer.__call__`) reproduces the
upstream SOMA-X forward to float32 precision (≈0.0003 cm max vertex
difference).

The two similarly named files have different purposes:

| Path | Purpose |
|---|---|
| `assets/third_party/SOMA_neutral.npz` | Unmodified NVIDIA source asset; used for conversion and SOMA-X parity tools. |
| `assets/SOMA_neutral_fixed.npz` | Generated SOMA-JAX runtime asset; pass this to `SOMALayer.load(...)`. |
| `third_party/SOMA-X/assets/SOMA_neutral.npz` | Slim v0.2.1 upstream asset; use only with its companion USD/JSON assets and the original SOMA-X package. |

The full Hugging Face snapshot is optional and is only needed for the other
identity backends, conversion tools, and parity benchmarks:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/SOMA-X",
    revision="466879a83d57eabf3d875ded2d869f2075f90348",
    local_dir="assets/third_party",
)
PY
```

Some tools also expect SMPL/SMPL-X `.npz`/`.pkl` model files under
`data/smpl/` and `data/smplx/`. Download those from the official
[SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/)
project pages (registration required) and point the tools at them, or symlink
an existing checkout into `data/`.

## 5. Verify

```bash
python -c "import soma_jax; print('SOMA-JAX OK')"
python - <<'PY'
from soma_jax import SOMALayer

# Straight from the §4.1 download — no §4.2 archive needed.
layer = SOMALayer.from_upstream_assets()
print(f"SOMA rig OK: {layer.v_template.shape[0]} vertices, "
      f"{len(layer.public_joint_names)} posable joints")
# ...or, if you built the §4.2 archive:
# layer = SOMALayer.load("assets/SOMA_neutral_fixed.npz")
PY
python -m pytest tools/selftest -q  # portable self-test, shipped with the repo
```

`tools/selftest` checks *this* machine end to end: every module imports, asset
resolution (Git LFS stubs, the slim archive, `SOMA_JAX_ASSETS`), the geometry
kernels and their gradients on a synthetic rig, the layer on the real assets
(README quickstart, procedural rig, `jit`/`vmap`/`grad`, low LOD, bone scales,
pose inversion, NPZ and USD round trips, two non-SOMA identity backends) and the
command-line tools, including an offscreen render. With torch + `warp-lang`
installed it also compares the forward — and the corrective checkpoint — against
upstream SOMA-X from the submodule. Tests whose assets or
optional packages are missing skip with the reason; `-m "not slow"` keeps the
asset-free subset (about a minute). JAX runs on its default backend —
`SOMA_JAX_TEST_PLATFORM=cpu|gpu` pins it.

The developer suite in `tests/` is kept local (see
[`DESCRIPTION.md`](DESCRIPTION.md#testing)); from a checkout that has it:

```bash
python -m pytest tests/ -q          # 240 passed / 29 skipped with .[dev,vis];
                                    # 457 collected with torch+SOMA-X+usd-core.
                                    # Runs on CPU; SOMA_JAX_TEST_PLATFORM=gpu to opt in
                                    # (456 passed / 1 skipped there).
```

## Offscreen / headless rendering

`pyrender` renders through OpenGL. With no display, the render and demo
scripts pick a headless backend themselves (`tools/_common.py`): **EGL** when
`libEGL` is installed — Mesa's software EGL works without a GPU — then
**OSMesa**. An explicit `PYOPENGL_PLATFORM` always wins.

```bash
sudo apt install libegl1            # Debian/Ubuntu (Fedora: dnf install mesa-libEGL)
export PYOPENGL_PLATFORM=egl        # optional — chosen automatically without a display
```

OSMesa (`apt install libosmesa6`, `PYOPENGL_PLATFORM=osmesa`) needs a newer
PyOpenGL than the 3.1.0 that `pyrender` pins — 3.1.0 fails with `cannot import
name 'OSMesaCreateContextAttribs'`:

```bash
pip install "PyOpenGL>=3.1.5"       # pip reports pyrender's pin as a conflict; it works
```

A virtual X display works with no extra libraries: `xvfb-run -a python
tools/vis/vis_pyrender.py ...`.

## Optional: SOMA-X parity & benchmarks (PyTorch + Warp)

The parity tests (`tests/test_soma_x_parity.py`) and the `benchmarks/` scripts
compare against the original SOMA-X, which runs on **PyTorch + NVIDIA Warp**:

```bash
pip install torch warp-lang          # match your CUDA; see the PyTorch install matrix
# CPU only:  pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(The Warp package on PyPI is `warp-lang`; there is no `nvidia-warp`.) The
scripts import upstream from the pinned `third_party/SOMA-X` submodule unless a
`soma` package is installed. Warp's JAX kernel — the "hybrid" pipeline — runs
only with JAX on a CUDA GPU; elsewhere the benchmark skips that row and
`tools/compare_render/pose_somajax.py` falls back to the pure-JAX SVD. The
SOMA-X side runs on CPU as well (`pose_somax.py --device cpu`).

Note the two backends ship **different CUDA runtimes** — JAX bundles CUDA 12,
while a recent PyTorch build may need CUDA 13 NVRTC. Loading both into one
process clashes, so `benchmarks/run_runtime.sh`, `run_memory.sh`, and
`tools/compare_render/run.sh` run each backend in its own subprocess with the
right `LD_LIBRARY_PATH`. Those scripts:

- derive the repo root from their own location (no absolute paths to edit);
- use `python` by default — override with `PY=/path/to/python bash …` (or
  `PYTHON=…` for the render scripts) to point at the env that has torch/jax/warp;
- auto-detect the Torch CUDA-13 NVRTC libs from the `nvidia-cu13` wheel
  (override with `TORCH_CUDA_LIBS=…`).

The BVH motion clips used by some render scripts are not included; point
`BVH_ROOT` at your own SOMA-skeleton BVH directory, e.g.
`BVH_ROOT=/path/to/bvh/clips bash tools/pipeline/render_bvh.sh`.

## Troubleshooting

- **`jax` uses CPU on a GPU box** — you installed the CPU wheel; reinstall with
  `pip install -U "jax[cuda12]"`.
- **`OpenGL`/`EGL` errors when rendering** — install EGL (`apt install
  libegl1`) or see [Offscreen / headless rendering](#offscreen--headless-rendering);
  `NoSuchDisplayException: Cannot connect to "None"` means no display and no
  headless backend was selected.
- **Submodule dirs empty** — run `git submodule update --init --recursive`.
- **"Found only Git LFS pointer stub(s)"** — the clone was made without git-lfs
  active; run `python tools/download_assets.py` (or `git lfs install && git
  submodule foreach --recursive git lfs pull`).
- **Asset not found** — complete step 4; `python tools/download_assets.py
  --check` lists what is missing. With a non-editable install, set
  `SOMA_JAX_ASSETS`.
- **`ValueError: ... is NVIDIA's upstream SOMA_neutral.npz`** (formerly
  `KeyError: 'v_template'`) — an upstream archive was passed to
  `SOMALayer.load`; use `SOMALayer.from_upstream_assets(npz_path=...)`.
- **`ImportError: USD I/O requires the 'usd-core' package`** —
  `from_upstream_assets()` reads the template USD: `pip install -e ".[usd]"`.
- **`No module named 'rtree'`** — trimesh's closest-point query needs it:
  `pip install -e ".[vis]"` (or `pip install rtree`).
- **`AttributeError: _ARRAY_API not found`** importing JAX — a jaxlib built for
  NumPy 1.x (≤ 0.4.25) under NumPy 2; upgrade JAX (`pip install -U jax jaxlib`).
- **`AttributeError: module 'jax.random' has no attribute 'KeyArray'`** — an
  old chex/optax pair; `pip install -U optax chex`.
- **`No FFI handler registered ... on a platform Host`** — Warp's JAX kernel
  was used with JAX on CPU; it needs a CUDA GPU (use `rotation_backend="jax"`).
- **`KeyError: 'bind_shape'` during conversion** — the slim v0.2.1 GitHub asset
  was downloaded. Repeat step 4.1 using the pinned full-schema Hugging Face
  revision.
