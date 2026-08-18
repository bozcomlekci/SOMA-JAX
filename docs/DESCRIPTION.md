# SOMA-JAX

**A faithful JAX port of NVIDIA [SOMA-X](https://github.com/NVlabs/SOMA-X) — the
Skeleton-Oriented Mean Avatar universal body-model pivot.**

SOMA-JAX reimplements SOMA-X's pipeline in pure JAX (`jax.jit` / `jax.vmap` /
`jax.grad` + `equinox`), replacing the upstream PyTorch + NVIDIA Warp backend.
It reproduces the same rig, the same three abstractions, and the same forward
results — verified against upstream SOMA-X to 3.2e-6 m on the LBS-only forward
(correctives are implemented but not upstream-parity-tested) — while being
end-to-end differentiable and hardware-portable (NVIDIA GPU / CPU / TPU).

This document is the extended description; see the top-level
[`README.md`](../README.md) for the quick start and [`docs/INSTALL.md`](INSTALL.md)
for setup.

---

## What SOMA is

SOMA unifies parametric human body models (SMPL, SMPL-X, SMPL-H, MHR, Anny, and
SOMA's own 128-coefficient PCA) under a single canonical body topology and a
shared 78-joint skeleton, so identity sources and pose data can be mixed and
matched at inference time. Its pipeline has three abstractions:

1. **Mesh topology abstraction** — barycentric transfer of any source body mesh
   onto the canonical SOMA topology.
2. **Skeletal abstraction** — per-joint RBF position regression + a two-stage
   Kabsch/Procrustes rotation fit to place the SOMA skeleton in any body shape.
3. **Pose abstraction** — inverse-LBS pose recovery with Newton–Schulz
   orthogonalization.

SOMA-JAX implements all three, plus the forward pose path (FK + linear blend
skinning) and pose-corrective displacements.

---

## What is implemented

Every piece below is in the package today. Test coverage varies — see
[`docs/FAITHFULNESS.md`](FAITHFULNESS.md) for what is parity-tested against
upstream and what is not.

| Area | Module(s) | Notes |
|------|-----------|-------|
| **SOMA forward** | `soma_jax/soma.py` (`SOMALayer`, `eqx.Module`) | identity blend → skeleton fit → FK + LBS |
| **Mesh topology transfer** | `geometry/barycentric_interp.py`, `geometry/laplacian.py` | face-index + barycentric gather to canonical SOMA topology, then a Laplacian re-solve of the inner-face vertices the source mesh lacks |
| **Skeletal abstraction** | `geometry/skeleton_transfer.py` | RBF joint regression + two-stage Kabsch (`jnp.linalg.svd` or a Warp `svd3` kernel); `PoseMirror` |
| **Pose abstraction** | `pose_inversion_soma.py`, `geometry/transforms.py` | SOMA-X's multi-stage solver: inverse-LBS Procrustes refit → Lie-algebra Gauss–Newton → optional Adam (`optax`) FK refinement; 1-DOF hinge constraints |
| **Pose abstraction (alt.)** | `pose_inversion.py` | lightweight Kabsch + Newton–Schulz init with a single Adam refine |
| **FK + skinning** | `geometry/lbs.py`, `geometry/batched_skinning.py` | level-order FK, dense and sparse top-K LBS |
| **Identity models** | `identity_model.py` | `soma`, `smpl`, `smplx`, `smplh`, `mhr`, `anny`, `garment_measurement` |
| **Standalone body models** | `body_models/` | full SMPL / SMPL-X / SMPL-H / MHR / Anny forward passes with pose blend shapes |
| **Pose correctives** | `correctives_model.py` (**equinox** MLP) | pose-dependent vertex displacements |
| **Animation I/O** | `io.py` | the SOMA `.npz` format (identity + poses + translation + metadata); SOMA-X's `save_soma_npz` semantics, verified interchangeable in both directions |
| **Procedural transforms** | `procedural_transforms.py` | SOMA-X procedural limb/finger bone-scale parameters |

The **forward path** is `jit`/`vmap`-compilable and differentiable end to end —
a single JAX graph, so thousands of subjects batch through `vmap` at once.
Offline setup steps are not: topology-transfer preprocessing
(`barycentric_interp`) uses NumPy/trimesh, and `laplacian` uses a SciPy sparse
solve. Neither runs inside the traced forward.

---

## Fidelity to SOMA-X

- **Parity tests.** `tests/test_soma_x_parity.py` checks JAX geometry against
  the upstream `third_party/SOMA-X` implementation (rig load, identity blend,
  skeleton transfer, FK/LBS); `tests/test_layer_parity.py` covers the
  end-to-end forward and `tests/test_pose_inversion_parity.py` the multi-stage
  inversion. Run `pytest tests/ -q` for the current suite size.
- **API surface.** `SOMALayer` mirrors SOMA-X's `SomaLayer`; the animation
  `.npz` matches SOMA-X's `io.save_soma_npz` field names *and* defaults —
  clips round-trip between the two implementations (see
  [`docs/FAITHFULNESS.md`](FAITHFULNESS.md)).
- **Same inputs.** SOMA-JAX builds its runtime archive from the upstream
  `SOMA_neutral.npz` rig plus the canonical template USD; see
  [`docs/INSTALL.md`](INSTALL.md) §4.2. The upstream archive is not loadable
  as-is (different key schema).

---

## Performance

Benchmarked head-to-head against SOMA-X (PyTorch + Warp) on an RTX 5080 — full
forward at batch 2048, matched **float32**. Which SOMA-JAX pipeline you pick
decides the answer, so both are stated:

| Pipeline | vs SOMA-X at B=2048 | Needs |
|---|---|---|
| **Hybrid** (JAX + one Warp `svd3` kernel) | **1.68× faster** (18.8 vs 31.6 ms) | optional `warp-lang`; approximates upstream's `auto` rotation solve |
| **Pure JAX** (the faithful port) | **0.61×** — i.e. 1.65× *slower* (52.0 ms) | nothing beyond JAX |

The pure-JAX path wins below B≈256 and loses above it; the whole gap is
`jnp.linalg.svd` over `B·78` tiny 3×3 matrices, which is XLA's weak spot. So the
headline "faster than SOMA-X" belongs to the hybrid row, and it buys that speed
with an optional dependency and a rotation-solve approximation measured at
210 µm max / 0.94 µm mean on the posed mesh.

On **peak GPU memory** — CUDA context plus live allocator high-water, with
NVIDIA Warp's allocations counted on both sides — SOMA-JAX carries the larger
fixed baseline (1.86 vs 1.01 GiB) but grows **5.1× more slowly** with batch
size (0.432 vs 2.215 MiB/sample, against a 0.207 MiB/sample output-buffer
floor). The two cross between B=256 and B=512; by B=4096 SOMA-JAX is 3.0×
lighter (3.32 vs 9.88 GiB), and SOMA-X OOMs at B=8192 on the 16 GB card while
every JAX pipeline still fits. See
[`benchmarks/README.md`](../benchmarks/README.md) for the measurement method
and the precision (float32 vs TF32) discussion.

---

## Quick start

See the [README](../README.md#usage) for the canonical forward pass and pose
inversion. Two notes that matter when you go past it:

* `SOMALayer.from_upstream_assets()` builds upstream's rig straight from the two
  NVIDIA source files and needs no PyTorch. `SOMALayer.load(...)` takes the
  single-file archive that [`INSTALL.md`](INSTALL.md) §4.2 bakes, which is
  optional. Passing NVIDIA's `SOMA_neutral.npz` to `load` fails — different key
  schema.
* `identity_coeffs` is 128-wide for the SOMA PCA backend; other backends take
  their own width (`smpl`/`smplx` betas, MHR identity params, …).

Install: `pip install -e ".[dev,vis]"` — see [`INSTALL.md`](INSTALL.md).
Core dependencies are `jax`, `jaxlib`, `equinox`, `numpy`, `scipy`, `optax`.

---

## Working with the library

The README is a general overview; this is the practical reference for
everything it does not cover.

### Pose Inversion

Recover SOMA skeleton rotations from posed mesh vertices. `SOMAPoseInversion`
is the faithful SOMA-X solver:

```python
from soma_jax import SOMALayer, SOMAPoseInversion

layer = SOMALayer.load("assets/SOMA_neutral_fixed.npz")
inv = SOMAPoseInversion(layer)
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

`PoseInversion` is a lighter-weight alternative (single Kabsch init + one
autograd refine) with explicit 1-DOF hinge constraints:

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

### Bone Scales

`scale_params` stretch individual limb and finger bones. The 56 active controls
are listed by `scale_param_names`, each naming a `(parent, child)` edge:

```python
rest, joints, binds = layer.prepare_identity(coeffs, return_bind_transforms=True)

scales = jnp.ones((1, layer.num_bone_scale_params))
scales = scales.at[0, layer.scale_param_names.index("LeftForeArm")].set(1.5)

out = layer.pose(rotmats, transl, rest[None], joints[None],
                 bind_transforms=binds[None], bone_scales=scales)
out.transforms   # (B, J, 4, 4) world joint transforms
```

Pass `fk_only=True` to skip skinning and get joints/transforms only.

### USD Export

Requires the optional `usd-core` package (`pip install usd-core`):

```python
from soma_jax import export_soma_usd

export_soma_usd("anim.usda", layer, result.rotations, result.root_translation,
                bind_transforms_world=binds, rest_shape=rest, fps=30.0)
```

### Visualization

```bash
# Export rest mesh to OBJ
python tools/vis/vis_mesh_export.py \
    --soma-model SOMA_neutral.npz --output rest.obj

# Export full animation as PLY frames
python tools/vis/vis_mesh_export.py \
    --soma-model SOMA_neutral.npz --animation anim.soma.npz \
    --output-dir frames/ --format ply --all-frames

# Static render with PyRender
python tools/vis/vis_pyrender.py \
    --soma-model SOMA_neutral.npz --output rest.png

# Interactive viewer
python tools/vis/vis_pyrender.py \
    --soma-model SOMA_neutral.npz --animation anim.soma.npz --interactive

# Demo: comparison of rest / posed / multi-model meshes + animated GIF
python tools/pipeline/demo_soma_vis.py \
    --soma-model SOMA_neutral.npz \
    --smpl-model SMPL_NEUTRAL.pkl \
    --smplx-model SMPLX_NEUTRAL.npz \
    --output-dir demo_renders/ --gif demo.gif --num-frames 30
```

### Conversion Tools

| Tool | Purpose |
|------|---------|
| `tools/convert/smpl2soma.py` | SMPL animation → SOMA NPZ |
| `tools/convert/mhr2soma.py` | MHR animation → SOMA NPZ |
| `tools/pipeline/motion2soma.py` | SMPL-X motion → SOMA NPZ |
| `tools/convert/shape_convert.py` | Cross-model identity coefficient conversion |
| `tools/convert/convert_gm_pca_to_npz.py` | GarmentMeasurement PCA packager |
| `tools/download_assets.py` | fetch the NVIDIA source rig from HuggingFace (INSTALL.md §4.1) |
| `tools/vis/vis_mesh_export.py` | OBJ / PLY mesh export |
| `tools/vis/vis_pyrender.py` | 3D rendering (static / interactive) |
| `tools/pipeline/demo_soma_vis.py` | Interactive demo script |

### Architecture

```
soma_jax/
├── soma.py               # SOMALayer — main entry point (load / from_upstream_assets)
├── rig_build.py          # torch-free npz + template-USD rig merge, joint pruning
├── procedural_transforms.py  # 78 → 122 twist-rig expansion (upstream's default)
├── identity_model.py     # SOMA identity model wrappers (7 backends)
├── correctives_model.py  # CorrectivesMLP (equinox Module)
├── pose_inversion_soma.py# SOMA-X multi-stage inversion (analytical/Lie-GN/autograd)
├── pose_inversion.py     # Lightweight Kabsch + Adam inversion + DOF constraints
├── assets.py             # asset discovery / data_root resolution
├── io.py                 # NPZ animation I/O
├── usd_io.py             # UsdSkel rig / animation I/O (optional usd-core)
├── units.py              # Unit enum
├── types.py              # SOMAParams, SOMAOutput
├── smpl/                 # SMPL-family topology bridge + cross-layer pose transfer
├── body_models/          # Standalone parametric body models
│   ├── _base.py          #   BaseBodyModel + blend shape helpers
│   ├── smpl.py           #   SMPL (24 joints)
│   ├── smplx.py          #   SMPL-X (55 joints, expressions)
│   ├── smplh.py          #   SMPL-H (52 joints, MANO hands)
│   ├── mhr.py            #   MHR (body-part scales)
│   ├── mhr_native.py     #   MHR TorchScript forward, transcribed to JAX
│   ├── anny.py           #   Anny (children, Z-up)
│   ├── anny_native.py    #   Anny rest-shape forward
│   └── model_io.py       #   .pkl / .npz loaders
└── geometry/
    ├── transforms.py     # SO(3) utilities (Kabsch, Newton-Schulz, 6D)
    ├── lbs.py            # Forward kinematics + LBS (dense + sparse top-K)
    ├── barycentric_interp.py  # Topology transfer
    ├── laplacian.py      # Laplacian mesh editing
    ├── skeleton_transfer.py   # Joint fitting + PoseMirror
    ├── rig_utils.py      # Joint hierarchy, world↔local, joint orient
    ├── batched_skinning.py    # Standalone BatchedSkinning Module
    └── chamfer.py        # Chamfer distance (pure JAX)
```

### Testing

The pytest suite is **developed and run locally, not distributed** — `tests/` is
git-ignored, so a clone of this repository does not contain it. The numbers below
record what it reports here, and every parity figure quoted in
[`FAITHFULNESS.md`](FAITHFULNESS.md) comes from it; the citations there name the
module that produced each number even though the file is not in the published
tree.

```bash
python -m pytest tests/ -q      # local checkout only
```

Counts depend on what is installed. With `.[dev,vis]` alone the run reports
**240 passed, 29 skipped** — the upstream-parity and USD modules skip because
they need extra dependencies, and nothing fails. Installing torch + the
`third_party/SOMA-X` submodule + the SOMA assets + `usd-core` brings the suite
to **457 collected**, adding `test_soma_x_parity.py`
(54), `test_soma_x_parity_modules.py` (34), `test_rig_build.py` (26),
`test_mhr_native.py` (26), `test_smpl_transfer.py` (18),
`test_procedural_parity.py` (17), `test_usd_io.py` (17),
`test_anny_native.py` (14) and the layer/pose-inversion parity modules. The
suite covers geometry, identity models, body models, pose inversion, I/O,
visualization tools, and gradient flow.

Tests run on **CPU** by default — they are parity tests against a float32
PyTorch/Warp reference, and CPU is the backend that reproduces it bit-stably.
Set `SOMA_JAX_TEST_PLATFORM=gpu` to exercise the accelerator path, which is also
verified green: **456 passed, 1 skipped** on an RTX 5080 (the extra pass is the
Warp `svd3` case, which needs CUDA and skips on CPU). A CUDA build of JAX older
than **12.8** cannot serve a Blackwell card — its cuBLAS carries no `sm_120`
kernels and fails with `INTERNAL: the library was not initialized`; upgrade the
`nvidia-*-cu12` wheels if you hit that.

---

## Scope

SOMA-JAX aims to reproduce **what SOMA-X does**, faithfully, in JAX, and also
exposes a few explicitly-labelled JAX-only alternatives where a cheaper or
simpler route is useful (the linear skeleton fit, the lightweight
`PoseInversion`, the optional Warp Kabsch kernel) — each flagged as such in
[`docs/FAITHFULNESS.md`](FAITHFULNESS.md). Beyond that it is not a superset:
things outside SOMA-X's scope (IK solvers, 3DGS avatar synthesis, etc.) are
intentionally not part of this repo.

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
