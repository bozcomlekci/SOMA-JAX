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
  submodule's assets, with no PyTorch (it does need `usd-core`, the `usd` extra,
  to read the template USD). Its default identity backend is SOMA's PCA
  (`identity_model_type="soma"`), where upstream's is `"mhr"`, because the MHR
  backend needs `torch` to read its TorchScript archive.
* `SOMALayer.load(path)` takes the optional single-file runtime archive
  ([`INSTALL.md`](INSTALL.md) §4.3), not upstream's `SOMA_neutral.npz`: passing
  that file raises a `ValueError` pointing at
  `from_upstream_assets(npz_path=...)` — the key schema differs.
* `identity_coeffs` is 128-wide for the SOMA PCA backend; other backends take
  their own width (`layer.identity_model.num_identity_coeffs`).
* `layer(SOMAParams(...))` returns all 78 joints, Root included;
  `layer.forward(poses, identity_coeffs, ...)` has upstream's signature and
  returns upstream's 77. `layer.pose(...)` is a lower-level, SOMA-JAX-shaped
  entry point (see [`FAITHFULNESS.md`](FAITHFULNESS.md#differences-by-design));
  code written for upstream's `pose()` should call `forward()`.

Install: `pip install -e ".[dev,vis,usd]"` — see [`INSTALL.md`](INSTALL.md).
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
inv.prepare_identity(identity_coeffs)                 # (1, 128): one subject

result = inv.fit(posed_vertices)                      # (B, V, 3) frames; analytical + Lie-GN
result = inv.fit(posed_vertices, lie_iters=0)         # analytical only
result = inv.fit(posed_vertices, autograd_iters=10)   # + autograd FK refinement

result.rotations          # (B, J, 3, 3) absolute local rotations
result.root_translation   # (B, 3)
result.per_vertex_error   # (B, V)
```

The autograd stage is an Adam refinement started from the Lie-GN result, as
upstream's is, and it behaves the same way: its first steps *raise* the error
before it drops below the starting point. On random poses (σ = 0.2 rad) the
mean vertex error goes from 2.37 mm without it to 16.2 / 6.3 / 2.4 mm after
5 / 20 / 60 steps (upstream: 16.3 / 6.6 / 2.1 mm), so give it on the order of
60+ steps or leave it off.

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
constraints. It solves rotations only, so the posed mesh must keep the rest
mesh's root placement — SOMA's `transl` is the hips' world position, so pose
with `transl=joints[:, 1]`. A hinge constrains a joint relative to its parent,
with the axis in the rest pose's frame; SOMA's rest pose is an A-pose with bent
elbows, so derive elbow axes from the rest skeleton:

```python
from soma_jax import PoseInversion

rest, joints = layer.prepare_identity(identity_coeffs)       # (1, V, 3), (1, J, 3)
names, J = list(layer.joint_names), joints[0]

def hinge(a, b, c):                   # normal of the rest bone pair a -> b -> c
    n = jnp.cross(J[names.index(b)] - J[names.index(a)],
                  J[names.index(c)] - J[names.index(b)])
    return n / jnp.linalg.norm(n)

inverter = PoseInversion(
    rest_verts=rest[0], weights=layer.weights,
    rest_joints=J, parents=layer.parents,
    dof_constraints={
        names.index("LeftShin"): jnp.array([1.0, 0.0, 0.0]),    # knees — hinge around X
        names.index("RightShin"): jnp.array([1.0, 0.0, 0.0]),
        names.index("LeftForeArm"): hinge("LeftArm", "LeftForeArm", "LeftHand"),
        names.index("RightForeArm"): hinge("RightArm", "RightForeArm", "RightHand"),
    },
)
world_R = inverter.fit(posed_vertices, mode="combined", num_refine_iters=50)
# (B, J, 3, 3) world rotations; parent-relative: world_R[:, p].mT @ world_R[:, j]
```

`inverter.fit_hierarchical(posed_vertices)` refines joint by joint from the
root down instead — slower, and usually a little more accurate.

### Bone scales

`scale_params` stretch individual limb and finger bones. The 60 active controls
are listed by `scale_param_names`: each names the joint whose offset from its
parent it stretches (`scale_param_segments` gives the `(parent, child)` edge,
so `"LeftForeArm"` lengthens the upper arm):

```python
coeffs = jnp.zeros(128)                                    # one identity, unbatched
rest, joints, binds = layer.prepare_identity(coeffs, return_bind_transforms=True)

scales = jnp.ones((rest.shape[0], layer.num_bone_scale_params))
scales = scales.at[:, layer.scale_param_names.index("LeftForeArm")].set(1.5)

out = layer.pose(rotmats, transl, rest, joints,      # (B, J, 3, 3) local, (B, 3)
                 bind_transforms=binds, bone_scales=scales)
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

`--soma-model` is optional in every visualization tool: without it the rig is
built from the resolved upstream assets (as `SOMALayer.from_upstream_assets()`
does); it also accepts NVIDIA's `SOMA_neutral.npz` or the runtime archive
(`python tools/pipeline/build_soma_rig.py`, [`INSTALL.md`](INSTALL.md) §4.3).
Offscreen rendering picks EGL/OSMesa by itself on a machine without a display
(see [`INSTALL.md`](INSTALL.md#offscreen--headless-rendering)).

```bash
# Export rest mesh to OBJ
python tools/vis/vis_mesh_export.py --output rest.obj

# Export full animation as PLY frames
python tools/vis/vis_mesh_export.py \
    --animation anim.soma.npz \
    --output-dir frames/ --format ply --all-frames

# Static render with PyRender (an explicit asset, e.g. the downloaded one)
python tools/vis/vis_pyrender.py \
    --soma-model assets/SOMA_neutral_fixed.npz --output rest.png

# Interactive viewer (needs a display)
python tools/vis/vis_pyrender.py --animation anim.soma.npz --interactive

# Demo: rest / custom shape / posed stills + animated GIF
python tools/pipeline/demo_soma_vis.py \
    --output-dir demo_renders/ --gif demo_renders/demo.gif --num-frames 30

# Demo: real motion capture — CMU trials retargeted onto the SOMA skeleton
# (downloaded on first use), each rendered to a GIF plus a strip of poses. A
# motion that travels is filmed from the side by a tracking camera, one in
# place from a fixed three-quarter view (--camera follow/fixed to choose)
python tools/pipeline/demo_motion.py                     # a walk: demo_renders/walk/
python tools/pipeline/demo_motion.py --list              # the 12 examples
python tools/pipeline/demo_motion.py --motion backflip   # one of them
python tools/pipeline/demo_motion.py --gallery           # all, plus demo_renders/gallery/
python tools/pipeline/demo_motion.py --gallery --gallery-width 1024   # README's mocap_gallery.gif
python tools/pipeline/demo_motion.py --trial 14_01 --start 5 --seconds 4   # any CMU trial

# Demo: every identity model on one SOMA skeleton and motion, skeleton drawn
# inside each body. MHR/Anny/Garment packs come from
# tools/pipeline/build_identity_packs.py (MHR needs torch); SMPL-family columns
# need the registration-only SMPL / SMPL-X model files.
python tools/pipeline/demo_soma_vis.py --side-by-side --soma-skeleton-overlay \
    --mhr-identity assets/identity/identity_mhr.npz \
    --anny-identity assets/identity/identity_anny.npz \
    --garment-identity assets/identity/identity_garment.npz \
    --smpl-model SMPL_NEUTRAL.pkl --smplx-model SMPLX_NEUTRAL.npz \
    --output-dir demo_renders/ --gif demo_renders/side_by_side.gif
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
| `tools/download_assets.py` | upstream's HuggingFace asset download (`--target-dir`, `--revision`); without them, fetches what the checkout lacks — submodule assets that are missing or Git LFS stubs, and `GarmentMeasurements/point.npz` — stdlib only, sha256-verified (INSTALL.md §4) |
| `tools/pipeline/build_soma_rig.py` | build the optional runtime archive |
| `tools/vis/vis_mesh_export.py` | OBJ / PLY mesh export |
| `tools/vis/vis_pyrender.py` | 3D rendering (static / interactive) |
| `tools/pipeline/demo_soma_vis.py` | multi-model demo: rest / posed / animation, side-by-side renders |
| `tools/pipeline/demo_motion.py` | real motion capture (CMU: walk, run, cartwheel, dance, sports, ...) retargeted onto SOMA and rendered |

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

A separate, portable **self-test does ship**: `python -m pytest tools/selftest
-q` checks an install on any machine — imports, asset resolution, the geometry
kernels and gradients on a synthetic rig, the layer on the real assets, the
command-line tools and, where torch + `warp-lang` are installed, parity with
upstream SOMA-X (see [`INSTALL.md`](INSTALL.md#5-verify)). It skips what the
machine lacks instead of failing; CI runs it on every push.

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
