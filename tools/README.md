# tools/

Command-line scripts (run from the repo root, e.g. `python tools/<sub>/<script>.py`).
These are **not** imported by the `soma_jax` package. They are grouped by role:
the `pipeline/` scripts import each other, so they live together; the rest are
standalone.

## pipeline/ — the interdependent SOMA build / retarget / render pipeline

| Script | Purpose |
|---|---|
| `demo_soma_vis.py` | Interactive multi-model demo: rest / posed / animation, incl. BVH-driven side-by-side renders and skeleton overlay. |
| `demo_walk.py` | A real walk: downloads a CMU motion-capture trial (ASF/AMC), retargets it onto the SOMA skeleton and renders a GIF + pose strip; also saves the motion as a SOMA NPZ. |
| `build_soma_rig.py` | Build the SOMA rig / per-joint affine regressor from the bind data. |
| `build_identity_packs.py` | Assemble per-identity coefficient packs for the identity backends. |
| `build_lowlod.py` | Downsample the mid-LOD mesh to the low-LOD set. |
| `bvh_parser.py` | Parse SOMA-skeleton BVH motion clips → poses / rotmats / translation. |
| `soma_x_skinning.py` | NumPy reference SOMA-X skinning (bind-world FK + LBS) for parity. |
| `motion_pipeline.py` | Retarget SMPL-X motion → SOMA skeleton (inverse-LBS). |
| `correctives_jax.py` | Load / apply the pose-corrective MLP. |
| `mhr_jax.py` | MHR identity rig loader and rest-shape. |
| `soma_to_smplx.py` | SOMA → SMPL-X bridge. |
| `motion2soma.py` | SMPL-X motion → SOMA NPZ. |
| `pose_converter.py` | Export retargeted SOMA motion in the SOMA-X NPZ format. |
| `render_bvh.{sh,clips}` | Batch-render a set of SOMA-skeleton BVH motion clips. |

## compare_render/ — SOMA-X vs SOMA-JAX comparison

Side-by-side comparison GIF (`gen_motion.py` → `pose_somax.py` / `pose_somajax.py`
→ `render_compare.py`, orchestrated by `run.sh`). Imports the `pipeline/` renderer.

`render_tf32_teaser.py` builds the float32→TF32 speedup teaser
(`assets/media/soma_jax_tf32_teaser.gif`) from the same cached pose outputs. Each
column's top-left panel *is* a progress bar (method name at the left, precision
at the right end) whose length reads the speedup; the runtime multiplier sits in
the centre. It is one continuous motion pass: the body switches float32→TF32 at
the middle of the motion, so the SOMA-JAX bar leaps 1.6×→2.7× the SOMA-X bar and
its animation gets smoother while SOMA-X stays choppy. TF32 is JAX-only and set
apart — float32 is the fair comparison.

## convert/ — standalone format converters

| Script | Purpose |
|---|---|
| `smpl2soma.py` | SMPL / SMPL-X identity → SOMA. |
| `mhr2soma.py` | MHR → SOMA. |
| `shape_convert.py` | Cross-model identity-coefficient conversion. |
| `convert_correctives_pt_to_npz.py` | Corrective-model `.pt` → `.npz`. |
| `convert_gm_pca_to_npz.py` | GarmentMeasurement PCA → `.npz`. |

## vis/ — standalone viewers / exporters

| Script | Purpose |
|---|---|
| `vis_mesh_export.py` | Export rest / animation meshes to OBJ / PLY. |
| `vis_pyrender.py` | Static or interactive PyRender viewer. |

## top-level utilities

| Script | Purpose |
|---|---|
| `audit_soma_features.py` | Audit feature parity of SOMA-JAX vs upstream SOMA-X. |
| `download_assets.py` | Fetch what the checkout lacks into `assets/third_party/`: the full-schema rig from HuggingFace, plus any submodule asset that is missing or a Git LFS pointer stub (from the pinned upstream commit). Standard library only — runs before JAX is installed; every file sha256-verified. `--minimal` fetches only what `SOMALayer.from_upstream_assets()` needs, `--only FILE...` just the named files; `--check` reports. |
| `_common.py` | Helpers the scripts share (not a CLI): `ensure_headless_gl()` picks EGL/OSMesa for pyrender when there is no display; `load_layer()` builds a layer from no asset (resolved upstream files), NVIDIA's `SOMA_neutral.npz`, or a SOMA-JAX archive. |

## selftest/ — portable install check

```bash
python -m pytest tools/selftest -q                 # full (~6 min on 4 CPU cores)
python -m pytest tools/selftest -q -m "not slow"   # asset-free subset (~1 min)
```

Verifies an install on the machine at hand: imports and metadata, asset
resolution (Git LFS stubs, the slim archive, `SOMA_JAX_ASSETS`), geometry kernels
and their gradients on a synthetic rig, the layer on the real assets (README
quickstart, procedural rig, `jit`/`vmap`/`grad`, low LOD, bone scales, pose
inversion, NPZ/USD round trips, identity backends) and these tools, including an
offscreen render whose shading is checked. Where torch + `warp-lang` are
installed, `test_upstream_parity.py` also checks the forward and the corrective
checkpoint against upstream SOMA-X (5e-5 m / 1e-4 m guards). Missing assets or optional packages
skip with the reason. Unlike the local developer suite in `tests/`, this one
ships with the repository.
