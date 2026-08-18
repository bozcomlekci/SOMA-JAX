# tools/

Command-line scripts (run from the repo root, e.g. `python tools/<sub>/<script>.py`).
These are **not** imported by the `soma_jax` package. They are grouped by role:
the `pipeline/` scripts import each other, so they live together; the rest are
standalone.

## pipeline/ — the interdependent SOMA build / retarget / render pipeline

| Script | Purpose |
|---|---|
| `demo_soma_vis.py` | Interactive multi-model demo: rest / posed / animation, incl. BVH-driven side-by-side renders and skeleton overlay. |
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
| `download_assets.py` | Fetch model weights / rigs from HuggingFace into `assets/`. |
