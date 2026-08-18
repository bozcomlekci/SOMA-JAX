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

## 2. Create an environment

Any Python 3.10+ environment works (conda/mamba, `venv`, `uv`, …):

```bash
conda create -n soma-jax python=3.10 -y && conda activate soma-jax
# or:  python -m venv .venv && source .venv/bin/activate
```

## 3. Install the package

```bash
pip install -e ".[dev,vis]"
```

This installs:

- **core** — `jax`, `jaxlib`, `equinox`, `numpy`, `scipy`, `optax`
- **`dev`** — `pytest`, `pytest-xdist`
- **`vis`** — `trimesh`, `pyrender` (OBJ/PLY export, offscreen rendering, GIFs)

USD import/export (`soma_jax.usd_io`) is optional; add it with
`pip install -e ".[usd]"` (or `pip install usd-core`). Without it the rest of
the package works normally and only USD calls raise.

### SMPL-JAX (optional)

`tools/pipeline/` and two test modules (`test_body_model_io.py`'s cross-check,
`test_motion_retarget.py`) import **`smpl_jax`**. It is not on PyPI under that
name here — install the submodule:

```bash
pip install -e third_party/SMPL-JAX
```

Without it those tools raise `ImportError` and the two test modules skip
(`pytest.importorskip`), which is part of the gap between the minimal and full
test runs in §5.

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
contain any model data. §4.1 downloads the one NVIDIA source file that must be
fetched; from there `SOMALayer.from_upstream_assets()` builds the rig directly
and nothing further is required.

§4.2 additionally bakes those inputs into a single self-contained archive:

```text
assets/SOMA_neutral_fixed.npz
```

That archive is what `SOMALayer.load(...)` takes, and it is **not**
byte-compatible with NVIDIA's upstream `SOMA_neutral.npz`: SOMA-JAX uses
`v_template`, dense `weights`, `parents`, and `J_regressor`, while the upstream
archive uses `mean`, sparse skinning arrays, and `joint_parent_ids`. Passing the
upstream file straight to `load(...)` therefore fails with a missing-key error —
use `from_upstream_assets()` for that file, or build the archive per §4.2.

### 4.0 Where assets live

Most model data already ships inside the **`third_party/SOMA-X` submodule**
(`third_party/SOMA-X/assets/`) — the template rig, procedural-transform JSON,
correctives checkpoint, and the MHR / Anny / SMPL / SMPL-X /
GarmentMeasurements packs. SOMA-JAX uses those in place rather than keeping a
second copy of ~1 GB of identical files, so `git submodule update --init
--recursive` covers them.

| Location | Contents | Tracked? |
|---|---|---|
| `third_party/SOMA-X/assets/` | vendored upstream assets, used in place | submodule |
| `assets/third_party/` | downloads (`tools/download_assets.py`) | git-ignored |
| `assets/` | archives this repo *builds*, e.g. `SOMA_neutral_fixed.npz` | tracked (large binaries excluded by extension) |

`soma_jax.assets.resolve()` searches those in order, so code and tests never
hardcode a layout. Check what is present with:

```bash
python tools/download_assets.py --check
```

Source repositories, if you need to fetch anything further:
[SOMA-X](https://github.com/NVlabs/SOMA-X) ·
[MHR](https://github.com/facebookresearch/MHR) ·
[Anny](https://github.com/naver/anny)

### 4.1 Download the public NVIDIA source asset

Install the Hugging Face client, then download the full-schema archive from an
immutable revision of the official [`nvidia/SOMA-X`](https://huggingface.co/nvidia/SOMA-X)
repository:

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
*only* asset that must be downloaded; everything else comes from the submodule.

### 4.2 Build the SOMA-JAX archive (optional)

> **You probably do not need this.** Since the torch-free rig merge landed,
> `SOMALayer.from_upstream_assets()` builds the rig directly from the two files
> §4.1 downloaded — no PyTorch, no `third_party/SOMA-X`, and it is the only route
> to upstream's *default* 122-joint procedural rig:
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

spec = importlib.util.spec_from_file_location("bsr", "tools/pipeline/build_soma_rig.py")
bsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsr)

# The upstream layer is the authority on the merged rig (npz + template USD).
tl = TorchSOMA(data_root="assets/third_party", identity_model_type="soma", device="cpu",
               mode="dense", enable_procedural_transforms=False,
               load_correctives_model=False)
src = dict(np.load("assets/third_party/SOMA_neutral.npz", allow_pickle=False))

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
python -m pytest tests/ -q          # 240 passed / 29 skipped with .[dev,vis];
                                    # 457 collected with torch+SOMA-X+usd-core.
                                    # Runs on CPU; SOMA_JAX_TEST_PLATFORM=gpu to opt in
                                    # (456 passed / 1 skipped there).
```

## Offscreen / headless rendering

`pyrender` renders through OpenGL. On a headless machine use the EGL backend:

```bash
export PYOPENGL_PLATFORM=egl        # the render/demo scripts set this for you
```

If EGL is unavailable, install `osmesa` and set `PYOPENGL_PLATFORM=osmesa`.

## Optional: SOMA-X parity & benchmarks (PyTorch + Warp)

The parity tests (`tests/test_soma_x_parity.py`) and the `benchmarks/` scripts
compare against the original SOMA-X, which runs on **PyTorch + NVIDIA Warp**:

```bash
pip install torch nvidia-warp        # match your CUDA; see the PyTorch install matrix
```

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
- **`OpenGL`/`EGL` errors when rendering** — set `PYOPENGL_PLATFORM=egl` (or
  `osmesa`); confirm a GPU/driver is visible.
- **Submodule dirs empty** — run `git submodule update --init --recursive`.
- **Asset not found** — complete step 4 and check that
  `assets/SOMA_neutral_fixed.npz` exists.
- **`KeyError: 'v_template'`** — an upstream NVIDIA archive was passed directly
  to SOMA-JAX; load `assets/SOMA_neutral_fixed.npz` instead.
- **`KeyError: 'bind_shape'` during conversion** — the slim v0.2.1 GitHub asset
  was downloaded. Repeat step 4.1 using the pinned full-schema Hugging Face
  revision.
