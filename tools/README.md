# tools/

Command-line scripts (run from the repo root, e.g. `python tools/<sub>/<script>.py`).
They are **not** imported by the `soma_jax` package.

SOMA-X v0.3.3's tools are ported under their own names, with upstream's
arguments — upstream's `third_party/SOMA-X/docs/tools.md` documents their usage —
and each script's docstring names its upstream file and anything it does
differently. Where upstream fails on its own default (procedural) layer — its
conversion tools apply the 110-joint skinning rig's orient and joint names to
the 78 public rotations, and `PoseInversion`'s default low-LOD path fails to
import — the ports do what the code intends; see
[`docs/FAITHFULNESS.md`](../docs/FAITHFULNESS.md#upstream-defects-not-reproduced).

## Ports of upstream's tools

| SOMA-JAX | Upstream `tools/` | Purpose |
|---|---|---|
| `convert/smpl2soma.py` | `smpl2soma.py` | SMPL-family animation → SOMA NPZ (pose inversion) |
| `convert/convert_amass_to_soma.py` | `convert_amass_to_soma.py` | AMASS clips → SOMA NPZ |
| `convert/mhr2soma.py` | `mhr2soma.py` | MHR animation → SOMA NPZ |
| `convert/pose_converter.py` | `pose_converter.py` | SOMA poses → native SMPL-family pose parameters |
| `convert/shape_convert.py` | `shape_convert.py` | identity-coefficient conversion between backends |
| `convert/convert_identity_backend.py` | `convert_identity_backend.py` | re-express a SOMA NPZ animation under another identity backend |
| `convert/identity_conversion.py` | `identity_conversion.py` | shared optimisation for the identity conversion |
| `convert/convert_gm_pca_to_npz.py` | `convert_gm_pca_to_npz.py` | GarmentMeasurement `point.pca` → `point.npz` |
| `hand/mano2soma.py`, `hand/soma2mano.py` | `hand/` | MANO ↔ SOMA Hand |
| `hand/convert_identity_backend.py` | `hand/convert_identity_backend.py` | hand animation identity conversion |
| `hand/sample_soma_hand_pose_pca.py` | `hand/sample_soma_hand_pose_pca.py` | hand pose PCA sampling |
| `hand/demo_soma_hand_vis.py` | `hand/demo_soma_hand_vis.py` | hand demo |
| `demo_soma_vis.py` | `demo_soma_vis.py` | every identity backend posed with one motion |
| `rig_soma_body_mesh.py` | `rig_soma_body_mesh.py` | rig a custom SOMA body template mesh, export UsdSkel |
| `conversion_utils.py`, `soma_rig_assets.py`, `logging_utils.py` | same names | helpers the tools share |
| `vis_pyrender.py` | `vis_pyrender.py` | PyRender helpers, vendored unchanged |
| `make_teaser.py` | `make_teaser.py` | README teaser compositor, vendored (import path only) |
| `download_assets.py` | `download_assets.py` | upstream's HuggingFace asset download (`--target-dir`, `--revision`); SOMA-JAX adds `--check` and `--extras`, which report and fetch what the submodule does not ship (`docs/INSTALL.md` §4.2) |

Arguments follow upstream's, with two kinds of difference: defaults that point
at upstream's `./assets` (`--data-root`, `--motion-file`, `--hand-asset`)
resolve through `soma_jax.assets.data_root()` instead, and `--device` is
accepted and ignored. A few ports add options (`--seed`, `--smpl-model-path`);
each script's docstring lists them.

Not ported: `ci/` and `docker-entrypoint.sh` (upstream's release
infrastructure) and the `soma_procedural_blender` / `soma_procedural_maya` DCC
plugins.

## SOMA-JAX additions

### pipeline/ — the interdependent build / retarget / render pipeline

| Script | Purpose |
|---|---|
| `build_soma_rig.py` | build the optional runtime archive `assets/SOMA_neutral_fixed.npz` (torch-free) |
| `build_identity_packs.py` | assemble per-identity coefficient packs for `soma_jax.identity_packs` |
| `build_lowlod.py` | derive the low-LOD vertex subset |
| `demo_soma_vis.py` | multi-model demo: rest / posed / animation, BVH-driven side-by-side renders, skeleton overlay |
| `bvh_parser.py` | parse SOMA-skeleton BVH clips → poses / rotmats / translation |
| `soma_x_skinning.py` | NumPy reference SOMA-X skinning (bind-world FK + LBS) for parity |
| `motion_pipeline.py` | retarget SMPL-X motion → SOMA skeleton (inverse LBS) |
| `correctives_jax.py` | load / apply the pose-corrective MLP |
| `mhr_jax.py` | MHR identity rig loader and rest shape |
| `soma_to_smplx.py` | SOMA → SMPL-X bridge |
| `motion2soma.py` | SMPL-X motion → SOMA NPZ |
| `pose_converter.py` | export retargeted SOMA motion in the SOMA-X NPZ format |
| `render_bvh.{sh,clips}` | batch-render a set of SOMA-skeleton BVH clips |

### compare_render/ — SOMA-X vs SOMA-JAX comparison

Side-by-side comparison GIF (`gen_motion.py` → `pose_somax.py` / `pose_somajax.py`
→ `render_compare.py`, orchestrated by `run.sh`). Imports the `pipeline/` renderer.
The motion is a SOMA-skeleton BVH clip when one is reachable (`BVH=` or
`BVH_ROOT=`), otherwise the clip SOMA-X ships for its own demo
(`third_party/SOMA-X/assets/example_animation.npy`). Both columns skin the same
public rig; the SOMA-JAX column is the JAX + Warp `svd3` hybrid. The speed ratio
drawn is the benchmark's (`benchmarks/results/runtime.json`, batch 2048), not the
capture's own timing.

`render_tf32_teaser.py` (also run by `run.sh`) builds the float32→TF32 speedup
teaser (`assets/media/soma_jax_tf32_teaser.gif`) from the same pose outputs. Each
column's top-left panel *is* a progress bar (method name at the left, precision
at the right end) whose length reads the speedup; the runtime multiplier sits in
the centre. It is one continuous motion pass: the body switches float32→TF32 at
the middle of the motion, so the SOMA-JAX bar extends from the float32 to the
TF32 ratio of the SOMA-X bar (2.6× → 2.8× with the committed results) and its
animation gets smoother while SOMA-X stays choppy. TF32 is JAX-only and set
apart — float32 is the fair comparison.

### vis/ — standalone viewers / exporters (take the runtime archive)

| Script | Purpose |
|---|---|
| `vis_mesh_export.py` | export rest / animation meshes to OBJ / PLY |
| `vis_pyrender.py` | static or interactive PyRender viewer |

### Others

| Script | Purpose |
|---|---|
| `audit_soma_features.py` | audit SOMA-JAX features against the SOMA paper / reference |
| `convert/convert_correctives_pt_to_npz.py` | corrective checkpoint `.pt` → `.npz` |
| `convert/pack_gm_matrices_to_npz.py` | pack GarmentMeasurement matrices |
| `convert/shape_space_convert.py` | identity conversion through the identity packs |
