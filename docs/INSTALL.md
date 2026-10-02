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
git lfs install                     # SOMA-X's assets are git-lfs objects
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

Two more extras are optional: **`usd`** (`usd-core`, for `soma_jax.usd_io`;
without it the rest of the package works and only USD calls raise) and
**`anny`** (the `anny` package upstream's Anny backend imports). Add them with
`pip install -e ".[usd,anny]"`.

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

## 4. Model assets

Model data is excluded from this repository. Everything SOMA-X ships lives in
the **`third_party/SOMA-X` submodule** (`third_party/SOMA-X/assets/`), at the
same release as the code — the slim `SOMA_neutral.npz`, `SOMA_template_rig.usda`,
the procedural-transform JSON, the corrective checkpoint, `SOMAHand.npz` and the
MHR / Anny / SMPL / SMPL-X / MANO / GarmentMeasurements packs. SOMA-JAX reads
them in place; `git submodule update --init --recursive` fetches them (with
`git-lfs` installed — without it the asset files are LFS pointer stubs).

Since SOMA-X v0.3 the rig comes from `SOMA_template_rig.usda` and
`SOMA_neutral.npz` carries shape and topology only, so nothing about the body
rig needs downloading:

```python
from soma_jax import SOMALayer
layer = SOMALayer.from_upstream_assets()                   # 110-joint procedural rig (default)
layer = SOMALayer.from_upstream_assets(procedural=False)   # 78-joint legacy rig
```

### 4.1 Where assets live

| Location | Contents | Tracked? |
|---|---|---|
| `third_party/SOMA-X/assets/` | upstream's assets, used in place | submodule (git-lfs) |
| `assets/third_party/` | downloads (`tools/download_assets.py`) | git-ignored |
| `assets/` | files this repo *builds*, e.g. `SOMA_neutral_fixed.npz` | tracked (large binaries excluded by extension) |
| `data/smpl/`, `data/smplx/`, `data/smplh/`, `data/mano/` | licensed SMPL-family / MANO model files you provide | git-ignored |

`soma_jax.assets.resolve()` searches these in order and `soma_jax.assets.data_root()`
materialises an upstream-layout view over them (`assets/data_root/`), so code
and tests never hardcode a layout. Check what is present with:

```bash
python tools/download_assets.py --check
```

### 4.2 What still needs fetching

* **`GarmentMeasurements/point.npz`** (the GarmentMeasurement identity backend
  only). Upstream does not ship it either; its docs have users convert the
  public `point.pca` from [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements):

  ```bash
  python tools/convert/convert_gm_pca_to_npz.py /path/to/point.pca \
      assets/third_party/GarmentMeasurements/point.npz
  ```

  `python tools/download_assets.py --extras` instead downloads the copy an
  older immutable `nvidia/SOMA-X` Hugging Face revision published and checks its
  sha256 (a SOMA-JAX convenience). Without flags the script does what
  upstream's does — download the whole Hugging Face asset snapshot
  (`--target-dir`, `--revision`) — which the submodule already provides.
* **SMPL / SMPL-X model files** (the SMPL-family backends and some tools), from
  the [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/)
  project pages (registration required). Pass them as upstream does —
  `identity_model_kwargs={"model_path": ...}` — or place them under
  `data/smpl/`, `data/smplx/`, `data/smplh/` or `data/mano/` with upstream's file
  names (`SMPL_NEUTRAL.npz` / `.pkl`, `SMPLX_FEMALE.npz`, `MANO_LEFT.pkl`, …):
  `soma_jax.assets.data_root()` links them where upstream's loaders look
  (`<data_root>/SMPL/SMPL_NEUTRAL.npz`, …).

### 4.3 The SOMA-JAX runtime archive (optional)

`SOMALayer.load(path)` reads a single-file archive of the 78-joint legacy rig,
so a runtime can skip `usd-core`. It is a SOMA-JAX-only cache, built from the
submodule's assets without PyTorch:

```bash
python tools/pipeline/build_soma_rig.py          # -> assets/SOMA_neutral_fixed.npz
```

Rebuild it whenever the submodule moves: a cache built from an older template
reproduces that template's rig. Upstream's own `SOMA_neutral.npz` is a
different schema and cannot be passed to `load()` — use `from_upstream_assets()`
for it.

## 5. Verify

```bash
python -c "import soma_jax; print('SOMA-JAX OK')"
python - <<'PY'
from soma_jax import SOMALayer

layer = SOMALayer.from_upstream_assets()
print(f"SOMA rig OK: {layer.v_template.shape[0]} vertices, "
      f"{len(layer.public_joint_names)} public joints, "
      f"{len(layer.target_joint_names)} skinning joints")
PY
python -m pytest tests/ -q -n 4     # local checkout only; see DESCRIPTION.md#testing
```

The test suite runs on CPU by default; `SOMA_JAX_TEST_PLATFORM=gpu` opts into
the accelerator path.

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
pip install torch warp-lang          # match your CUDA; see the PyTorch install matrix
```

Note the two backends ship **different CUDA runtimes** — JAX bundles CUDA 12,
while a recent PyTorch build may need CUDA 13 NVRTC. Loading both into one
process clashes, so `benchmarks/run_runtime.sh`, `run_memory.sh`, and
`tools/compare_render/run.sh` run each backend in its own subprocess with the
right `LD_LIBRARY_PATH`. Those scripts:

- derive the repo root from their own location (no absolute paths to edit);
- use `python` by default — override with `PY=/path/to/python bash …`
  (`PYTHON=…` for `tools/pipeline/render_bvh.sh`) to point at the env that has
  torch/jax/warp;
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
- **Asset not found / unreadable `.npz` or `.usda`** — the submodule's assets
  are git-lfs objects: install `git-lfs`, then `git lfs pull` inside
  `third_party/SOMA-X` (or re-run `git submodule update --init --recursive`).
  `python tools/download_assets.py --check` lists what is missing.
- **`KeyError: 'v_template'`** — upstream's `SOMA_neutral.npz` was passed to
  `SOMALayer.load(...)`; use `SOMALayer.from_upstream_assets()` for it, or build
  the runtime archive (§4.3).
