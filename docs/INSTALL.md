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
  archive (§4.3) does not.
- **`plots`** — `matplotlib`, `tueplots`, for `benchmarks/plot_*.py`.

Every dependency ships binary wheels for Linux (x86-64, aarch64), macOS on
Apple silicon and Windows (x86-64), on Python 3.10–3.14 (current jaxlib has no
Intel-macOS wheels). The dependency floors are the oldest versions verified to work
together (`jax` 0.4.26, `optax` 0.2.2, `numpy` 1.24.1; see the comment in
`pyproject.toml`).

One more extra is optional: **`anny`** (the `anny` package upstream's Anny
backend imports), `pip install -e ".[anny]"`.

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

## 4. Model assets

Model data is excluded from this repository. Everything SOMA-X ships lives in
the **`third_party/SOMA-X` submodule** (`third_party/SOMA-X/assets/`), at the
same release as the code — the slim `SOMA_neutral.npz`, `SOMA_template_rig.usda`,
the procedural-transform JSON, the corrective checkpoint, `SOMAHand.npz` and the
MHR / Anny / SMPL / SMPL-X / MANO / GarmentMeasurements packs. SOMA-JAX reads
them in place; `git submodule update --init --recursive` fetches them (with
`git-lfs` active — without it each asset arrives as an LFS pointer stub).

One command fetches whatever this checkout lacks — the real files behind LFS
stubs, the whole asset set for a "Download ZIP" checkout, and
`GarmentMeasurements/point.npz`, which neither upstream nor the submodule ships:

```bash
python tools/download_assets.py            # everything the checkout lacks
python tools/download_assets.py --minimal  # only what from_upstream_assets() needs
python tools/download_assets.py --only SOMAHand.npz MANO/base_hand_left.obj   # just these
python tools/download_assets.py --check    # report, download nothing
```

It needs only the Python standard library — not JAX, not `huggingface_hub`
(used when installed; `HF_ENDPOINT` mirrors and `HF_TOKEN` are honoured either
way) — so it also runs before the package is installed. Every file is checked
against the sha256 the pinned submodule commit records for it.

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
| `$SOMA_JAX_ASSETS` | optional extra directories (`os.pathsep`-separated), searched first | — |
| `third_party/SOMA-X/assets/` | upstream's assets, used in place | submodule (git-lfs) |
| `assets/third_party/` | downloads (`tools/download_assets.py`), incl. stand-ins for LFS stubs | git-ignored |
| `assets/` | files this repo *builds*, e.g. `SOMA_neutral_fixed.npz` | tracked (large binaries excluded by extension) |
| `data/smpl/`, `data/smplx/`, `data/smplh/`, `data/mano/` | licensed SMPL-family / MANO model files you provide | git-ignored |

`soma_jax.assets.resolve()` searches the first four in order and
`soma_jax.assets.data_root()` materialises an upstream-layout view over them
(`assets/data_root/`, with the `data/` files linked in), so code and tests never
hardcode a layout. It skips Git LFS pointer stubs, and downloads made for an
earlier submodule pin, wherever they are found, and names them in the error
when nothing usable exists. Check what is present with:

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

  `python tools/download_assets.py` (`--extras` for this file alone) instead
  downloads the copy an older immutable `nvidia/SOMA-X` Hugging Face revision
  published and checks its sha256 (a SOMA-JAX convenience). Upstream's own
  command — download the whole Hugging Face asset snapshot — is
  `python tools/download_assets.py --target-dir DIR --revision TAG`; the
  submodule already provides that snapshot.
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
python -m pytest tools/selftest -q  # portable self-test, shipped with the repo
```

`tools/selftest` checks *this* machine end to end: every module imports, asset
resolution (Git LFS stubs, stale downloads, `SOMA_JAX_ASSETS`), the geometry
kernels and their gradients on a synthetic rig, the layer on the real assets
(README quickstart, procedural rig, `jit`/`vmap`/`grad`, low LOD, bone scales,
pose inversion, NPZ and USD round trips, two non-SOMA identity backends) and the
command-line tools, including an offscreen render. With torch + `warp-lang`
installed it also compares SOMA-JAX against upstream SOMA-X from the submodule.
Tests whose assets or optional packages are missing skip with the reason;
`-m "not slow"` keeps the asset-free subset (about a minute). JAX runs on its
default backend — `SOMA_JAX_TEST_PLATFORM=cpu|gpu` pins it.

The developer suite in `tests/` is kept local (see
[`DESCRIPTION.md`](DESCRIPTION.md#testing)); from a checkout that has it:

```bash
python -m pytest tests/ -q -n 4
```

The test suite runs on CPU by default; `SOMA_JAX_TEST_PLATFORM=gpu` opts into
the accelerator path.

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
- **`OpenGL`/`EGL` errors when rendering** — install EGL (`apt install
  libegl1`) or see [Offscreen / headless rendering](#offscreen--headless-rendering);
  `NoSuchDisplayException: Cannot connect to "None"` means no display and no
  headless backend was selected.
- **Submodule dirs empty** — run `git submodule update --init --recursive`.
- **"Found only Git LFS pointer stub(s)"**, or an unreadable `.npz` / `.usda`
  — the clone was made without git-lfs active; run
  `python tools/download_assets.py` (or `git lfs install && git submodule
  foreach --recursive git lfs pull`).
- **Asset not found** — complete step 4; `python tools/download_assets.py
  --check` lists what is missing. With a non-editable install, set
  `SOMA_JAX_ASSETS`.
- **`ValueError: ... is NVIDIA's upstream SOMA_neutral.npz`** (formerly
  `KeyError: 'v_template'`) — upstream's archive was passed to
  `SOMALayer.load(...)`; use `SOMALayer.from_upstream_assets()` for it, or build
  the runtime archive (§4.3).
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
