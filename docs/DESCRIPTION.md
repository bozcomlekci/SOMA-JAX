# SOMA-JAX

**A faithful JAX port of NVIDIA [SOMA-X](https://github.com/NVlabs/SOMA-X) — the
Skeleton-Oriented Mean Avatar universal body-model pivot.**

SOMA-JAX reimplements SOMA-X **v0.3.3** in pure JAX (`jax.jit` / `jax.vmap` /
`jax.grad` + `equinox`), replacing the upstream PyTorch + NVIDIA Warp backend.
It reads the same assets, exposes the same API under the same names, and
reproduces upstream's results: the body forward matches at every LOD, on both
rigs, with the pose-corrective network, to ≤ 3.1 µm — while being end-to-end
differentiable and hardware-portable (NVIDIA GPU / CPU / TPU).

This document is the extended description; see the top-level
[`README.md`](../README.md) for the quick start,
[`INSTALL.md`](INSTALL.md) for setup and [`FAITHFULNESS.md`](FAITHFULNESS.md)
for the module-by-module correspondence with upstream.

---

## What SOMA is

SOMA unifies parametric human body models (SMPL, SMPL-X, MHR, Anny,
GarmentMeasurement and SOMA's own 128-coefficient PCA) under a single canonical
body topology and a shared 78-joint skeleton, so identity sources and pose data
can be mixed and matched at inference time. Its pipeline has three
abstractions:

1. **Mesh topology abstraction** — barycentric transfer of any source body mesh
   onto the canonical SOMA topology.
2. **Skeletal abstraction** — per-joint RBF position regression + a two-stage
   Kabsch/Procrustes rotation fit to place the SOMA skeleton in any body shape.
3. **Pose abstraction** — inverse-LBS pose recovery with Newton–Schulz
   orthogonalization.

SOMA-JAX implements all three, plus the forward pose path (FK + linear blend
skinning on the 110-joint twist rig), pose correctives, the SOMA Hand and MANO
layers, and SOMA-X's fitting and conversion tools.

---

## What is implemented

| Area | Module(s) | Notes |
|------|-----------|-------|
| **Body layer** | `body/soma.py` (`SOMALayer`) | identity → skeleton fit → FK + LBS at the mid / low / xlo LOD, on the 110-joint procedural rig (default) or the 78-joint legacy rig, with pose correctives and bone scales |
| **Identity backends** | `body/identity_model.py`, `identity_model.py` | SOMA PCA, MHR, Anny, SMPL / SMPL-X, GarmentMeasurement, built from the asset root as upstream builds them; `identity_packs.py` adds a pack-based route |
| **Hand layers** | `hand/` | `SOMAHandLayer` and `MANOLayer`, with the hand identity model and reference poses |
| **SMPL-family rigs** | `smpl/` | `SMPLLayer` / `SMPLXLayer` rigs and cross-topology pose transfer |
| **Mesh topology transfer** | `geometry/barycentric_interp.py`, `geometry/laplacian.py` | tetrahedral barycentric transfer onto the SOMA topology, Laplacian re-solve of the inner-face vertices the source lacks |
| **Skeletal abstraction** | `geometry/skeleton_transfer.py`, `geometry/interpolate.py` | RBF joint regression + two-stage Kabsch |
| **Procedural rig** | `procedural_transforms.py` | the twist-joint definition and parameter transform that drive 32 twist joints from the 78 public ones |
| **Pose inversion** | `fitting/` (`pose_inversion.py`, `pose_inversion_mhr.py`) | SOMA-X's multi-stage solver (inverse-LBS Procrustes refit → Lie-algebra Gauss–Newton → optional Adam FK refinement) and the native-MHR inverter |
| **Pose smoothing** | `fitting/rts_smoothing.py` | SO(3) Rauch–Tung–Striebel smoothing of pose trajectories |
| **Reference poses** | `reference_poses.py` | the reference-pose history, aliases and convention conversion |
| **Pose correctives** | `correctives_model.py` | the masked corrective MLP and its checkpoint format |
| **FK + skinning** | `geometry/lbs.py`, `geometry/batched_skinning.py` | level-order FK, dense and sparse top-K LBS, `BatchedSkinning` |
| **I/O** | `io.py`, `usd_io.py` | SOMA `.npz` clips (interchangeable with SOMA-X's), UsdSkel rig / animation I/O (optional `usd-core`) |
| **Standalone body models** | `body_models/` | SMPL, SMPL-H, SMPL-X, MHR and Anny forwards (a SOMA-JAX addition) |

The **forward path** is `jit`/`vmap`-compilable and differentiable end to end —
a single JAX graph, so thousands of subjects batch through `vmap` at once.
Asset loading and one-off precomputation (rig assembly, topology
correspondences, the skeleton transfer's RBF systems, Laplacian factorization)
run on the host with NumPy/SciPy and are not traced.

---

## Fidelity to SOMA-X

- **Parity tests.** Upstream's torch implementation and the JAX one run on
  identical inputs for the body layer (every LOD, both rigs, correctives on and
  off, every identity backend), pose inversion (all three stages, and the MHR
  inverter), the procedural transform, skeleton transfer, alignment, skinning,
  topology transfer, the hand layers, smoothing, reference poses and I/O. The
  measured agreement is tabulated in [`FAITHFULNESS.md`](FAITHFULNESS.md).
- **Upstream's tests.** SOMA-X's own test files are ported and run against
  SOMA-JAX (`tests/test_upstream_*.py` and peers), except those that test
  torch, Warp or upstream's release CI.
- **API surface.** Upstream's package layout (`soma_jax.body.soma`,
  `soma_jax.fitting.pose_inversion`, …, with the pre-0.3 paths as aliases), and
  every public name SOMA-X exports under the same name, save four pieces of
  torch/Warp machinery; constructors and functions take upstream's parameters
  in upstream's order; `SOMALayer`'s attributes carry upstream's meaning. Differences by design (immutable layers, no torch device
  state, classmethod construction) and the upstream defects SOMA-JAX does not
  reproduce are listed in [`FAITHFULNESS.md`](FAITHFULNESS.md).
- **Same inputs.** `SOMALayer.from_upstream_assets()` reads upstream's own
  `SOMA_neutral.npz` and `SOMA_template_rig.usda` from the `third_party/SOMA-X`
  submodule, as upstream's constructor does — no PyTorch involved.

---

## Performance

Benchmarked head-to-head against SOMA-X (PyTorch + Warp) on an RTX 5080 — the
full forward (identity blend → skeleton fit → FK + LBS) on the same 78-joint
rig, matched **float32**. Which SOMA-JAX pipeline you pick sets the margin, so
both are stated:

| Pipeline | B=1 | B=128 | B=2048 | Needs |
|---|---:|---:|---:|---|
| **Pure JAX** (the faithful port) | 5.6× faster | 3.6× | **2.0×** (15.0 vs 30.0 ms) | nothing beyond JAX |
| **Hybrid** (JAX + one Warp `svd3` kernel) | 24× | 6.0× | **2.65×** (11.3 ms) | optional `warp-lang`; approximates upstream's `auto` rotation solve |

The pure-JAX path reproduces SOMA-X's posed meshes to 0.0027 mm. The hybrid buys
its extra speed with an optional dependency and a plain-Kabsch rotation step
that departs from upstream's `auto` on ill-conditioned joints: 0.69 mm max /
1.6 µm mean against SOMA-X's meshes.

On **peak GPU memory** — CUDA context plus requested-bytes high-water, with
NVIDIA Warp's allocations counted on both sides — SOMA-X carries the slightly
smaller fixed baseline (1.01 vs 1.08 GiB), but SOMA-JAX grows **3.5× more
slowly** with batch size (0.624 vs 2.215 MiB/sample, against a 0.207 MiB/sample
output-buffer floor). The two cross between B=32 and B=64; by B=4096 SOMA-JAX
is 2.8× lighter (3.58 vs 9.88 GiB), and SOMA-X OOMs at B=8192 on the 16 GB card
while every JAX pipeline still fits. See
[`benchmarks/README.md`](../benchmarks/README.md) for the measurement method,
the fairness checks and the precision (float32 vs TF32) discussion.


---

## Quick start

See the [README](../README.md#usage) for the canonical forward pass and pose
inversion. Notes that matter when you go past it:

* `SOMALayer.from_upstream_assets()` is upstream's `SOMALayer(...)` constructor:
  upstream's parameters in upstream's order, building upstream's layer from the
  submodule's assets. Its default identity backend is SOMA's PCA
  (`identity_model_type="soma"`), where upstream's is `"mhr"`, because the MHR
  backend needs `torch` to read its TorchScript archive.
* `SOMALayer.load(path)` takes the optional single-file runtime archive
  ([`INSTALL.md`](INSTALL.md) §4.3), not upstream's `SOMA_neutral.npz`.
* `identity_coeffs` is 128-wide for the SOMA PCA backend; other backends take
  their own width (`layer.identity_model.num_identity_coeffs`).
* `layer(SOMAParams(...))` returns all 78 joints, Root included;
  `layer.forward(poses, identity_coeffs, ...)` has upstream's signature and
  returns upstream's 77. `layer.pose(...)` is a lower-level, SOMA-JAX-shaped
  entry point (see [`FAITHFULNESS.md`](FAITHFULNESS.md#differences-by-design));
  code written for upstream's `pose()` should call `forward()`.

Install: `pip install -e ".[dev,vis]"` — see [`INSTALL.md`](INSTALL.md).
Core dependencies are `jax`, `jaxlib`, `equinox`, `numpy`, `scipy`, `optax`.

---

## Working with the library

### Pose inversion

Recover SOMA skeleton rotations from posed mesh vertices with upstream's
solver (`soma_jax.fitting.PoseInversion`, also exported as `SOMAPoseInversion`):

```python
from soma_jax import SOMALayer, SOMAPoseInversion

layer = SOMALayer.from_upstream_assets()
inv = SOMAPoseInversion(layer)                        # low_lod=True, as upstream
inv.prepare_identity(identity_coeffs)

result = inv.fit(posed_vertices)                      # analytical + Lie-GN
result = inv.fit(posed_vertices, lie_iters=0)         # analytical only
result = inv.fit(posed_vertices, autograd_iters=10)   # + autograd FK refinement

result.rotations          # (B, J, 3, 3) absolute local rotations
result.root_translation   # (B, 3)
result.per_vertex_error   # (B, V)
```

Weight extremities and add a pose prior when contact accuracy matters:

```python
result = inv.fit(
    posed_vertices,
    leaf_weight={"head": 2, "hands": 2, "feet": 5, "heels": 10},
    autograd_iters=20, autograd_pose_prior=1e-3,
)
```

`soma_jax.fitting.MHRPoseInversion` inverts native MHR meshes to MHR pose and
model parameters (it needs MHR assets that SOMA-X's public release does not
include — see [`FAITHFULNESS.md`](FAITHFULNESS.md)), and
`soma_jax.fitting.smooth_pose` smooths the resulting trajectories:

```python
from soma_jax.fitting import smooth_pose

rotations, root_translation = smooth_pose(result.rotations, result.root_translation,
                                          soma_layer=layer, fps=30.0)
```

`soma_jax.PoseInversion` (top level) is a different, SOMA-JAX-only lightweight
inverter (single Kabsch init + one autograd refine) with explicit 1-DOF hinge
constraints:

```python
from soma_jax import PoseInversion

inverter = PoseInversion(
    rest_verts=rest_verts, weights=weights,
    rest_joints=rest_joints, parents=parents,
    dof_constraints={
        4: jnp.array([1.0, 0.0, 0.0]),   # left knee — hinge around X
        5: jnp.array([1.0, 0.0, 0.0]),   # right knee
        18: jnp.array([0.0, 1.0, 0.0]),  # left elbow — hinge around Y
        19: jnp.array([0.0, 1.0, 0.0]),  # right elbow
    },
)
rotmats = inverter.fit(posed_verts, mode="combined", num_refine_iters=50)
```

### Bone scales

`scale_params` stretch individual limb and finger bones. The 60 active controls
are listed by `scale_param_names`, each naming a `(parent, child)` edge in
`scale_param_segments`:

```python
coeffs = jnp.zeros(128)                                    # one identity, unbatched
rest, joints, binds = layer.prepare_identity(coeffs, return_bind_transforms=True)

scales = jnp.ones((1, layer.num_bone_scale_params))
scales = scales.at[0, layer.scale_param_names.index("LeftForeArm")].set(1.5)

out = layer.pose(rotmats, transl, rest[None], joints[None],
                 bind_transforms=binds[None], bone_scales=scales)
out.transforms   # (B, J, 4, 4) world joint transforms
```

Pass `fk_only=True` to skip skinning and get joints/transforms only.

### USD export

Requires the optional `usd-core` package (`pip install usd-core`):

```python
from soma_jax import export_soma_usd

export_soma_usd("anim.usda", layer, result.rotations, result.root_translation,
                bind_transforms_world=binds, rest_shape=rest, fps=30.0)
```

### Visualization

The `tools/vis/` scripts take the runtime archive
(`python tools/pipeline/build_soma_rig.py`, [`INSTALL.md`](INSTALL.md) §4.3):

```bash
# Export rest mesh to OBJ
python tools/vis/vis_mesh_export.py \
    --soma-model assets/SOMA_neutral_fixed.npz --output rest.obj

# Export full animation as PLY frames
python tools/vis/vis_mesh_export.py \
    --soma-model assets/SOMA_neutral_fixed.npz --animation anim.soma.npz \
    --output-dir frames/ --format ply --all-frames

# Static render with PyRender
python tools/vis/vis_pyrender.py \
    --soma-model assets/SOMA_neutral_fixed.npz --output rest.png

# Interactive viewer
python tools/vis/vis_pyrender.py \
    --soma-model assets/SOMA_neutral_fixed.npz --animation anim.soma.npz --interactive

# Demo: rest / posed / multi-model meshes + animated GIF
python tools/pipeline/demo_soma_vis.py \
    --soma-model assets/SOMA_neutral_fixed.npz \
    --smpl-model SMPL_NEUTRAL.pkl \
    --smplx-model SMPLX_NEUTRAL.npz \
    --output-dir demo_renders/ --gif demo.gif --num-frames 30
```

SOMA-X's own demo, `tools/demo_soma_vis.py`, is ported under that name and
takes upstream's arguments; upstream's `tools/vis_pyrender.py` helpers are
vendored unchanged.

### Tools

SOMA-X's tools are ported under their own names; see
[`tools/README.md`](../tools/README.md) for the full list and what SOMA-JAX adds.

| Tool | Purpose |
|------|---------|
| `tools/convert/smpl2soma.py`, `convert_amass_to_soma.py` | SMPL-family animation → SOMA NPZ |
| `tools/convert/mhr2soma.py` | MHR animation → SOMA NPZ |
| `tools/convert/pose_converter.py` | SOMA poses → native SMPL-family pose parameters |
| `tools/convert/shape_convert.py` | identity-coefficient conversion between backends |
| `tools/convert/convert_identity_backend.py`, `identity_conversion.py` | re-express a SOMA NPZ animation under another identity backend |
| `tools/convert/convert_gm_pca_to_npz.py` | GarmentMeasurement PCA packager |
| `tools/hand/` | MANO ↔ SOMA Hand conversion, hand identity conversion, hand pose PCA, hand demo |
| `tools/rig_soma_body_mesh.py` | rig a custom SOMA body template mesh and export it as UsdSkel |
| `tools/demo_soma_vis.py` | upstream's demo: every identity backend posed with one motion |
| `tools/download_assets.py` | upstream's HuggingFace asset download; SOMA-JAX's `--check` / `--extras` report and fetch what the submodule lacks (INSTALL.md §4.2) |
| `tools/pipeline/build_soma_rig.py` | build the optional runtime archive |

### Architecture

```
soma_jax/
├── __init__.py              # upstream's top-level exports (+ SOMA-JAX extras)
├── body/                    # soma.body
│   ├── soma.py              #   SOMALayer — from_upstream_assets / load / forward / pose
│   └── identity_model.py    #   identity backends (SOMA, MHR, Anny, SMPL-family, GM)
├── hand/                    # SOMAHandLayer, MANOLayer, hand identity model
├── smpl/                    # SMPLLayer / SMPLXLayer rigs, cross-topology pose transfer
├── fitting/                 # soma.fitting
│   ├── pose_inversion.py    #   PoseInversion (top-level alias SOMAPoseInversion)
│   ├── pose_inversion_mhr.py  # MHRPoseInversion
│   └── rts_smoothing.py     #   SO(3) RTS smoothing
├── reference_poses.py       # reference-pose history, aliases, conversion
├── procedural_transforms.py # twist-joint definition + parameter transform (78 → 110)
├── correctives_model.py     # CorrectivesMLP
├── identity_model.py        # BaseIdentityModel, coordinate transforms, create_identity_model
├── identity_packs.py        # pack-based identity backends (SOMA-JAX extra)
├── rig_build.py             # torch-free template-rig assembly
├── io.py                    # NPZ clips
├── usd_io.py                # UsdSkel rig / animation I/O (optional usd-core)
├── assets.py                # asset discovery / data_root
├── units.py, types.py, _smpl_family_loader.py
├── pose_inversion_lite.py   # lightweight inverter, top-level PoseInversion (SOMA-JAX extra)
├── body_models/             # standalone SMPL / SMPL-H / SMPL-X / MHR / Anny (SOMA-JAX extra)
│   ├── mhr_native.py        #   the MHR TorchScript archive's forward, transcribed to JAX
│   └── anny_native.py       #   Anny rest-shape evaluation
└── geometry/
    ├── transforms.py        # SO(3)/SE(3), alignment (Kabsch, Newton–Schulz, auto)
    ├── lbs.py               # FK + LBS (dense, sparse top-K)
    ├── batched_skinning.py  # BatchedSkinning, FKTopology
    ├── rig_utils.py         # hierarchy helpers, world↔local, joint orient, pose mirroring
    ├── skeleton_transfer.py # joint fitting
    ├── interpolate.py       # RBF interpolation
    ├── barycentric_interp.py, laplacian.py  # topology transfer
    ├── chamfer.py           # ChamferLoss (+ vertex-set chamfer)
    └── warp_kabsch.py       # optional Warp svd3 kernel (SOMA-JAX extra)
```

### Testing

The pytest suite is **developed and run locally, not distributed** — `tests/` is
git-ignored, so a clone of this repository does not contain it. The parity
figures quoted in [`FAITHFULNESS.md`](FAITHFULNESS.md) come from it; the
citations there name the module that produced each number even though the file
is not in the published tree.

```bash
python -m pytest tests/ -q -n 4      # local checkout only
```

With torch + the `third_party/SOMA-X` submodule + its assets + `usd-core` +
`warp-lang` installed, and the licensed SMPL / SMPL-X files under `data/`, the
suite collects **1,103 tests: 1,088 pass and 15 skip**. The skips are
environment-gated: the licensed MANO files (5), the MHR inversion assets
upstream does not ship (5), upstream's pose-clip mirror checks, which read clips
named by environment variables (2), CUDA-only Warp kernels on a CPU run (2), and
one case documenting upstream's `-1` parent-index behaviour. Without the optional parity
dependencies the upstream-parity modules skip rather than fail.

Tests run on **CPU** by default — they are parity tests against a float32
PyTorch/Warp reference, and CPU is the backend that reproduces it bit-stably.
Set `SOMA_JAX_TEST_PLATFORM=gpu` to exercise the accelerator path. A CUDA build
of JAX older than **12.8** cannot serve a Blackwell card — its cuBLAS carries no
`sm_120` kernels and fails with `INTERNAL: the library was not initialized`;
upgrade the `nvidia-*-cu12` wheels if you hit that.

---

## Scope

SOMA-JAX reproduces **what SOMA-X does**, faithfully, in JAX, and also exposes a
few explicitly-labelled JAX-only alternatives where a cheaper or simpler route
is useful (the linear skeleton fit, the lightweight `PoseInversion`, the
optional Warp Kabsch kernel, the standalone body models) — each listed in
[`FAITHFULNESS.md`](FAITHFULNESS.md#soma-jax-additions). Things outside SOMA-X's
scope (IK solvers, 3DGS avatar synthesis, etc.) are intentionally not part of
this repo.

---

## License & citation

SOMA-JAX is licensed under **[Apache-2.0](../LICENSE)**, matching upstream
[SOMA-X](https://github.com/NVlabs/SOMA-X), from which it derives; see
[`NOTICE`](../NOTICE) for attribution, the summary of changes, and the separate
(often research-only) terms governing the model assets, which the Apache grant
does **not** cover.

If you use SOMA-JAX, please cite the original SOMA work
([arXiv:2603.16858](https://arxiv.org/abs/2603.16858)) and, where relevant,
SMPL-X ([Pavlakos et al., CVPR 2019](https://smpl-x.is.tue.mpg.de/)).
