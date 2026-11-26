# Faithfulness to SOMA-X

How SOMA-JAX corresponds to upstream [SOMA-X](https://github.com/NVlabs/SOMA-X)
**v0.3.6**, its latest release (the `third_party/SOMA-X` submodule, release
commit `d7bac2a`): what is ported, what is measured against upstream and how
closely, what differs by design, which upstream defects are not reproduced,
what SOMA-JAX adds, and what has no JAX counterpart.

**On the test citations below.** Figures marked **shipped** come from
`tools/selftest/test_upstream_parity.py`, which ships with the repository and
runs on every push in CI's `parity` job (CPU torch + Warp, upstream imported
from the submodule); `python -m pytest tools/selftest/test_upstream_parity.py -s`
prints them. The other citations name modules of the developer suite under
`tests/`, which is run locally rather than distributed (`tests/` is
git-ignored), so a clone will not contain them. Those were measured against
v0.3.3; the modules they cover did not change in v0.3.4–v0.3.6, which changed
the corrective weights, the hand correctives' wrist input, the rotation-vector
conversions, the NPZ export tools and PoseInversion's internals, and added the
USD animation API — all measured against v0.3.6 by the shipped suite.

## Summary

* **Scope.** Every upstream package module has a SOMA-JAX counterpart at the
  same path (`soma_jax.body.soma`, `soma_jax.fitting.pose_inversion`, …), the
  pre-0.3 paths resolve as upstream's do (`soma_jax.soma`,
  `soma_jax.pose_inversion`, `soma_jax.pose_inversion_mhr`,
  `soma_jax.rts_smoothing`), and every public name upstream exports exists here
  under the same name, except four pieces of torch/Warp machinery
  ([Not ported](#not-ported)). Upstream's test files are ported, except the
  cases that test torch, Warp or upstream's release CI (listed there too).
* **Call shapes.** Upstream's constructors and functions keep upstream's
  parameters in upstream's order, so positional and keyword calls written for
  SOMA-X bind the same way here; SOMA-JAX's own options are keyword-only extras.
  The exception is `SOMALayer` itself, built by the classmethod
  `from_upstream_assets(...)`, which takes upstream's constructor parameters;
  its `prepare_identity()` returns the identity upstream caches, which `pose()`
  takes as `identity=` ([below](#differences-by-design)).
* **Forward parity (shipped).** The body layer matches upstream v0.3.6 at every
  LOD, on both rigs, with and without the v0.3.6 pose-corrective network, to
  float32 rounding: **≤ 2.9 µm** on vertices and **≤ 3.5 µm** on joints
  ([table](#the-body-forward)); `prepare_identity()` + `pose()` to ≤ 3.4 µm /
  7.6 µm, the hand layer to 0.25 µm.
* **Layer surface.** All 68 public attributes of upstream's `SOMALayer` exist
  here with upstream's meaning and values (torch state such as `device`
  excepted); on a procedural layer `bind_pose_world`, `t_pose_world`,
  `rig_data`, … describe the 110-joint skinning rig, exactly as upstream's do.
* **Fitting.** `PoseInversion` matches upstream through all three stages, and
  `MHRPoseInversion` through skeleton fit, DOF projection and refinement
  ([below](#pose-inversion)).
* **Differences** are listed in [By design](#differences-by-design),
  [Upstream defects](#upstream-defects-not-reproduced) and
  [SOMA-JAX additions](#soma-jax-additions).

## Measured against upstream

Each row runs upstream's torch implementation and the JAX one on identical
input. Figures are measured maxima; the tests assert looser bounds so they hold
across BLAS and driver versions.

### The body forward

**Shipped** (`test_forward_matches_upstream`): `SOMALayer.from_upstream_assets(lod=…,
procedural=…)` against upstream `SOMALayer(lod=…, enable_procedural_transforms=…)`
through `forward()`, poses `N(0, 0.45²)` rad, translations `N(0, 0.1²)` m,
random `N(0, 0.5²)` identities, `identity_model_type="soma"`, the v0.3.6
corrective checkpoint:

| LOD (vertices) | rig | correctives | max \|Δ vertex\| | max \|Δ joint\| |
|---|---|---|---|---|
| mid (18,056) | legacy (78 joints) | — | 2.9 µm | 3.4 µm |
| mid | procedural (110 joints) | off | 2.9 µm | 3.5 µm |
| mid | procedural | **on** | 2.9 µm | 3.5 µm |
| low (4,505) | legacy | — | 1.6 µm | 2.0 µm |
| low | procedural | off / on | 1.6 µm | 2.1 µm |
| xlo (612) | legacy | — | 1.7 µm | 3.2 µm |
| xlo | procedural | off | 1.8 µm | 3.3 µm |
| xlo | procedural | **on** | 1.9 µm | 3.3 µm |

Upstream's prepare-once / pose-per-frame workflow (**shipped**,
`test_prepare_identity_and_pose_match_upstream`, poses `N(0, 0.3²)` rad,
correctives on) agrees to ≤ 3.4 µm on vertices and ≤ 7.6 µm on joints and
transforms by default and with `absolute_pose=True`, `fk_only=True`, a stored
reference pose (`{"version": "v0.1.0"}`), SOMA bone scales and a global scale.
`load_motion` on the shipped `example_animation.usd` (body, both rigs, also
absolute and against a reference; hand, both sides) agrees to ≤ 4e-7 and its
reader `load_soma_animation` exactly; `public_joint_orient` to 1.2e-7. The
SOMA Hand layer's forward agrees to 0.25 µm with and without correctives,
including v0.3.6's identity-wrist corrective input.

The developer suite (v0.3.3) pins the reposed rest shape, the identity rest
shape and the fitted bind transforms of `prepare_identity(repose_to_bind_pose=True)`
to upstream's cached `_cached_rest_shape` / `_cached_identity_rest_shape` /
`_cached_bind_transforms_world`, and the faces and facial exclusion lists of
every LOD exactly (`tests/test_body_lod_parity.py`, `test_layer_parity.py`,
`test_procedural_parity.py`).

The other identity backends, built from the same assets as upstream builds
them (developer suite, v0.3.3: `tests/test_body_identity_backends.py`; poses
`N(0, 0.3²)` rad, random identities and body-part scales, no correctives):

| backend | max \|Δ vertex\| (mid / low / xlo) | max \|Δ joint\| |
|---|---|---|
| GarmentMeasurement | 7.9 / 1.8 / 1.3 µm | 4.8 µm |
| SMPL | 7.0 / 4.2 / 0.9 µm | 1.7 µm |
| SMPL-X | 13.8 / 3.9 / 0.8 µm | 2.3 µm |
| Anny | 59.7 / 59.0 / 30.4 µm | 3.4 µm |
| MHR | 152 / 2.4 / 1.5 µm | 19.3 µm |

The source→SOMA correspondence is bit-identical in every case; the larger mid-LOD
figures come from upstream's tetrahedral embedding, which is ill-conditioned on a
few near-degenerate source triangles and amplifies float32 noise in the source
forward. `tests/test_upstream_soma_layer.py` runs upstream's own layer cases on
every backend and LOD.

### Pose inversion

**Shipped** (`test_pose_inversion_matches_upstream`): upstream `PoseInversion`
and `SOMAPoseInversion` with their defaults (low LOD, analytical refit +
Lie-GN) on the same posed meshes (σ = 0.25 rad): rotation entries agree to
6.4e-3 (mean 3.6e-4) — neither side returns exactly orthonormal matrices, so
they are compared entry by entry — root translation to 9.8 µm, and the mean
per-vertex error is 2.416 vs 2.409 mm.

The developer suite (v0.3.3, `tests/test_pose_inversion_parity.py`) breaks
this down by stage, upstream `PoseInversion` against
`soma_jax.fitting.PoseInversion` (`SOMAPoseInversion`) on the same posed mesh:

| Stage | max \|ΔR\| | root translation | per-vertex error | mean error (upstream / JAX) |
|---|---|---|---|---|
| analytical refit | 1.0e-4 | 6.3e-7 m | 5.5e-6 m | 0.2827 / 0.2827 cm |
| + Lie-algebra Gauss–Newton (default) | 3.8e-3 | 3.9e-6 m | 1.3e-4 m | 0.1657 / 0.1661 cm |
| analytical + autograd (40 Adam steps) | 2.1e-4 | 1.6e-6 m | 7.8e-6 m | 0.2449 / 0.2449 cm |

Upstream runs the analytical refit through a fused Warp kernel; this runs the
per-joint torch path's algorithm in JAX. Lie-GN drifts more per joint because it
solves a dense `(3K × 3K)` normal equation each iteration and the damping ladder
can select differently (JAX has no `solve_ex` info flag, so solutions are
validated by finiteness); the reconstruction stays equivalent.

`MHRPoseInversion` (`tests/test_mhr_pose_inversion.py`): its two input files,
`MHR/MHR_base_rig.npz` and `MHR/parameter_transform.npz`, are not in SOMA-X's
public assets (git LFS or Hugging Face), and upstream's own tests skip without
them. Both implementations therefore run on stand-ins written from the shipped
`mhr_model_lod1.pt` (`tests/_mhr_stand_in.py`) — every array real, only joint
and parameter names synthetic. Pose parameters agree to
**6.5e-5** at rest and **5.3e-5** on a posed target through refit and Adam
refinement (losses to 2e-6), **6.4e-4** with the reduced-DOF refit; the
identity-reference, spine-bound, frozen-parameter, chunked and corrective
modes agree to the same order. Upstream's own six tests pass on the stand-ins.

### Components

| Component | Test | Agreement |
|---|---|---|
| `rotvec_to_matrix` / `matrix_to_rotvec` (v0.3.6), `align_vectors`, quaternions | **shipped** `test_transforms_match_upstream` | ≤2e-6 (rotation vectors, tiny angles and the zero rotation included), ≤1e-5 (`align_vectors`, all three methods); 5e-18 in float64 for the rotation vectors |
| corrective checkpoint, read without torch | **shipped** `test_correctives_checkpoint_reads_without_torch` | every tensor bit-identical to `torch.load` |
| NPZ export (`export_soma_npz`, v0.3.6 `resolve_joint_orient`) | **shipped** `test_npz_export_is_upstreams` | the fields of upstream's export, both rigs; each side reads the other's file |
| `align_vectors` (`auto`, `kabsch`, `newton-schulz`) | `test_soma_x_parity.py`, `test_rotation_alignment.py` | ≤3e-15 in float64; in float32 against upstream's float64, ≤4e-7 on generic correspondences and ≤3e-6 on near-planar (rank-deficient) ones |
| refit alignment `_align_vectors_auto` | `test_rotation_alignment.py` | ≤2.4e-7 forward; gradients equal torch's, including rank-deficient and zero covariances |
| RBF basis weights | `test_skeleton_transfer.py` | 2.3e-13 (float64); 6.0e-6 `linear`, 1.3e-6 `gaussian`, 9.1e-5 `thin_plate_spline` (float32 conditioning) |
| `SkeletonTransfer` | `test_upstream_skeleton_transfer.py`, `test_skeleton_transfer.py`, `test_mhr_pose_inversion.py` | joint by joint, to float32 rounding, on the SOMA and MHR rigs |
| procedural transforms | `test_upstream_procedural_transforms.py`, `test_upstream_procedural_layer.py` | upstream's cases, except those that patch torch internals or move devices |
| `BatchedSkinning` / `FKTopology` | `test_upstream_batched_skinning.py` | upstream's cases |
| `LaplacianMesh` | `test_laplacian.py`, `test_upstream_laplacian.py` | 6.9e-6 m |
| barycentric transfer | `test_upstream_barycentric_interp.py` | 1.2e-7 (`area`), 3.6e-7 (`edge`) |
| `ChamferLoss` | `test_upstream_chamfer.py` | against the Warp kernel, cache contract included |
| `CorrectivesMLP` | `test_soma_x_parity_modules.py`, `test_upstream_correctives_checkpoint.py` | forward 1e-4; checkpoints written here load upstream and vice versa |
| MHR TorchScript forward (`MHRNativeModel`) | `test_mhr_native.py` | 7.6e-5–1.8e-4 cm |
| SOMA Hand / MANO layers | `test_upstream_hand_layer.py`, `test_upstream_hand_tools.py` | upstream's hand-layer, template-regression and tool cases |
| reference poses, RTS smoothing, identity conversion | `test_upstream_reference_*`, `test_upstream_rts_smoothing.py`, `test_upstream_identity_conversion.py` | upstream's cases |
| SMPL-family transfer | `test_smpl_transfer.py` | all four stages of `transfer.py` |
| NPZ clips | `test_soma_x_parity_modules.py::TestIoRigKeys` | written here, read upstream and vice versa |
| template rig from USD | `test_rig_build.py` | bind transforms, joint names and parents exact; pruned weights 6e-8; derived local/world transforms 1 ulp (see below) |

## Upstream API coverage

**Module level.** `soma`, `soma.body`, `soma.fitting`, `soma.hand`, `soma.smpl`,
`soma.geometry` and every module under them map to `soma_jax` modules at the
same paths; all of upstream's public names exist here (often alongside a
SOMA-JAX-style name, e.g. `SE3_from_Rt` and `se3_from_rt`). The four that do
not are torch/Warp machinery — see [Not ported](#not-ported). As upstream's
`soma/__init__.py` does, `soma_jax/__init__.py` registers the pre-0.3 module
paths as the implementation modules themselves, so imports, private helpers
and pickled class references from those paths resolve to one implementation.

| SOMA-JAX | Upstream |
|---|---|
| `body/soma.py` (pre-0.3 path `soma_jax.soma`) | `body/soma.py` (`SOMALayer`, `SOMAPoseOutput`, `SOMAPublicRigView`) |
| `body/`, `body/identity_model.py` | `body/`, `body/identity_model.py` (SOMA, MHR, Anny, SMPL-family, GarmentMeasurement backends) |
| `identity_model.py` | `identity_model.py` (`BaseIdentityModel`, coordinate transforms, `create_identity_model`) |
| `fitting/pose_inversion.py`, `fitting/pose_inversion_mhr.py`, `fitting/rts_smoothing.py` (pre-0.3 paths `soma_jax.pose_inversion`, `…pose_inversion_mhr`, `…rts_smoothing`) | `fitting/` (`PoseInversion`, `MHRPoseInversion`, `smooth_pose`, …) |
| `hand/` | `hand/` (`SOMAHandLayer`, `MANOLayer`, identity model, loader) |
| `smpl/`, `smpl/layers.py`, `smpl/transfer.py` | `smpl/__init__.py` (`SMPLLayer`, `SMPLXLayer`, `create_smpl_family_layer`), `smpl/transfer.py` |
| `correctives_model.py` | `correctives_model.py` |
| `procedural_transforms.py` | `procedural_transforms.py` |
| `reference_poses.py` | `reference_poses.py` |
| `io.py`, `usd_io.py` | `io.py` (NPZ and USD halves) |
| `assets.py`, `units.py`, `_smpl_family_loader.py` | same names |
| `geometry/transforms.py`, `lbs.py`, `batched_skinning.py`, `rig_utils.py`, `skeleton_transfer.py`, `interpolate.py`, `laplacian.py`, `barycentric_interp.py` | same names |
| `geometry/chamfer.py` | `geometry/chamfer_warp.py` (`ChamferLoss`) |
| `body_models/mhr_native.py`, `body_models/anny_native.py` | the MHR TorchScript archive and the `anny` package upstream calls into |
| `rig_build.py` | the rig-assembly part of upstream's `SOMALayer` constructor |

**Class level.** Upstream's classes keep their constructor signatures and
members; the members that cannot exist on an immutable JAX object are listed in
[By design](#differences-by-design). Upstream's `SOMALayer` attributes all
exist with upstream's meaning:

* `bind_pose_world`, `bind_pose_local`, `t_pose_world`, `t_pose_local`,
  `bind_shape` and `rig_data` describe the **skinning** rig — the expanded
  110-joint twist rig on a procedural layer — in the asset's native
  centimetres, read from a `rig_data` assembled as upstream's constructor
  assembles it. The public 78-joint rig's are in `public_rig_view()`, as
  upstream's are.
* `rig_data` is built on first access from the same `SOMA_neutral.npz` and
  `SOMA_template_rig.usda`, so it exists on layers built by
  `from_upstream_assets`; layers loaded from a SOMA-JAX archive or built from a
  `soma_data` dict report their (public) rig instead.
* The rig's derived transforms (`t_pose_world` from `t_pose_local`,
  `bind_pose_local` from `bind_pose_world`) are computed in float64 here and in
  float32 by upstream's torch, and agree to 1 ulp (1.5e-5 cm). Everything read
  straight from the files is bit-identical.

## Differences by design

**Immutable layers.** SOMA-JAX layers are `equinox` modules. Upstream's
`prepare_identity()` caches the identity on the layer (`_cached_rest_shape`,
`_cached_bind_transforms_world`, …) for later `pose()` / `forward()` calls; here
`prepare_identity()` takes upstream's parameters and *returns* the identity, a
`SOMAIdentity` (a pytree, so it passes through `jit` and `vmap`), and `pose()`
takes upstream's parameters plus that identity as a keyword, `identity=`.
`forward()` prepares and poses in one call, as upstream's does. The hand and
SMPL-family layers do the same with their own identity objects. Upstream
methods that default to "the cached identity" (`public_rig_view()`,
`public_bind_transforms_world()`, …) default to the rig's bind pose instead, or
take the fitted binds as an argument. `SOMAPoseInversion.prepare_identity`
keeps upstream's stateful contract, since the inversion object is mutable on
both sides.

**No torch state.** `device`, `dtype`, `.to()`, `training`, registered buffers
and the stateful `batched_skinning` / `public_batched_skinning` objects do not
exist. Arrays live on JAX's default device (`jax.default_device` chooses it).
Every `device=` parameter upstream's signatures take is accepted in upstream's
position and ignored (identity models record it as `.device`).

**Construction.** `SOMALayer` is built by classmethods.
`SOMALayer.from_upstream_assets(...)` is upstream's constructor — upstream's
parameters in upstream's order, reading upstream's own two files — and
`SOMALayer.load(...)` reads a single-file SOMA-JAX archive. The default identity
backend is SOMA's own PCA (`identity_model_type="soma"`), where upstream's is
`"mhr"`: SOMA-JAX's MHR backend reads the MHR TorchScript archive and so needs
`torch`, which a default layer should not. Everything else defaults as
upstream: mid LOD, the procedural rig, the packaged corrective checkpoint on a
procedural layer, `mode="warp"` (top-8 sparse skinning; any other mode keeps
every influence, as upstream's dense fallback does), metres. SOMA-JAX's own
options (`npz_path`, `identity_model_path`, `sparse_k`, `fit_joint_regressor`)
are keyword-only, and its earlier keyword names (`procedural=`,
`correctives_path=`, `usd_path=`) remain as aliases. A missing or explicitly
requested but absent asset raises upstream's errors in upstream's order.

**Strict keyword arguments.** Upstream's identity-model constructors take
`**kwargs`, pop the keys they know (`nv_lod_mid_to_low`, `soma_low_lod_faces`,
`vertex_ids_to_exclude`) and drop the rest silently. SOMA-JAX names those keys
and rejects unknown ones, so a misspelled option raises instead of being
ignored.

**Optional where upstream requires a value.** `device` (ignored), the hand
identity models' `low_lod` (`False`), `CorrectivesMLP`'s `bindpose` /
`cors_per_joint` / `num_verts`, `ReferencePoseHistory.get_reference_pose`'s
`dtype` (float32), `single_axis_rotation_matrices`' `axis_signs` (1.0),
`batch_rodrigues`' `dtype` (the input's), and the MANO and SMPL-family layers'
`prepare_identity` `identity_coeffs` (the zero identity, as upstream's own
`_identity_coeffs` treats `None`). Calls that pass them behave as upstream's.

**Output joints.** `SOMALayer.__call__` returns all 78 joints with the virtual
Root at index 0; `SOMALayer.forward(...)` has upstream's signature and returns
upstream's 77.

**Gradients at degenerate inputs.** `jnp.linalg.norm` differentiates `‖x‖` at
`x = 0` as NaN; torch defines that gradient as zero. Where upstream's formulas
take such norms (`compute_covariance`'s virtual normal, `quaternion_log_xyzw`,
`quaternion_exp_xyzw`, `rotvec_to_matrix`) SOMA-JAX uses a norm with torch's
gradient, and its `jnp.where` branches feed unused SVDs well-conditioned
placeholders, so collinear or zero covariances backpropagate the same zeros
upstream's do. Forward values are unchanged.

**Host-side precompute.** The skeleton transfer's per-joint RBF systems are
assembled and LU-factored with NumPy/SciPy when their inputs are concrete
(bit-identical to `jax.scipy.linalg.lu_factor` on CPU, which calls the same
LAPACK routine); traced inputs keep the JAX path.

## Upstream defects not reproduced

Each was confirmed against the v0.3.6 code; SOMA-JAX implements what the code
evidently intends.

* **`PoseInversion(low_lod=True)` on a hand, MANO or SMPL-family layer** would
  build a *body* `SOMALayer`; SOMA-JAX raises instead.
* **Upstream's `mhr2soma` and `convert_amass_to_soma` on the default
  (procedural) layer** render through `batched_skinning.pose` with the 78
  public rotations, which that 110-joint rig rejects, and `mhr2soma`'s drift
  summary labels joints with `rig_data["joint_names"]`, the 110 skinning joints
  (every public joint from `RightShoulder` on mislabelled). The tool ports
  render the layer's own pose and use the public names.
* **Negative-index roots.** Upstream's torch indexing treats a parent id of `-1`
  as "the last joint" in several helpers; SOMA-JAX treats it as a root, as the
  self-parented convention upstream's own assets use.

Fixed upstream since v0.3.3, and done right here before: `PoseInversion`'s
default low-LOD layer import (`from .body import SOMALayer` inside
`soma/fitting`, v0.3.6); the conversion tools' NPZ export on the procedural rig
(the 110-joint orient and names applied to 78 public rotations, v0.3.6's
`resolve_joint_orient`); `mhr2soma --autograd-iters` ≥ 2 ("backward through the
graph a second time", v0.3.6); and `matrix_to_rotvec` / `rotvec_to_matrix`
(twice the rotation vector below 1e-3 rad, and non-rotations from the unit-axis
`K`, v0.3.6). `matrix_to_rotvec` keeps one float32 property of upstream's
formula: between 1e-3 and 1e-2 rad its angle comes from `arccos` of the trace,
which loses up to ~1e-4 rad there, on both sides alike.

Reproduced on purpose, because they are part of a public contract:
`ChamferLoss`'s mesh cache (a target mesh is captured on first use and reused,
even when later calls pass different vertices, until `refit=True` or
`clear_cache()`), and `CorrectivesMLP.save_checkpoint(native_unit=…)` accepting
and not recording the unit.

## SOMA-JAX additions

Everything below is absent upstream. None of it changes what the upstream-named
API computes.

**Modules.**

* `types.py` — `SOMAParams` / `SOMAOutput`, the pytree input and output of
  `SOMALayer.__call__`.
* `identity_packs.py` — identity backends built from precomputed packs
  (`tools/pipeline/build_identity_packs.py`) instead of the source model files.
* `rig_build.py` — the torch-free rig assembly behind `from_upstream_assets`
  (template merge, procedural-joint pruning, an optional linear joint
  regressor).
* `pose_inversion_lite.py` — a lightweight inverter (Kabsch + Newton–Schulz
  init, one Adam refinement, explicit DOF constraints); exported at top level as
  `soma_jax.PoseInversion`, which is therefore **not** upstream's
  `PoseInversion` (that is `soma_jax.fitting.PoseInversion`, also reachable as
  `soma_jax.pose_inversion.PoseInversion` and `soma_jax.SOMAPoseInversion`;
  upstream has no top-level `PoseInversion`).
* `body_models/` — standalone JAX SMPL, SMPL-H, SMPL-X, MHR and Anny models
  with their own parameter types.
* `geometry/warp_kabsch.py` — an optional Warp `svd3` covariance→rotation kernel
  for the benchmarks' hybrid pipeline. It is plain Kabsch, **not** upstream's
  default `auto` method; the two agree on well-conditioned covariances and
  differ on ill-conditioned ones (on the posed mesh: 0.66 mm max / 2.9 µm mean
  against the pure-JAX pipeline at the benchmark operating point, 0.69 mm max
  against SOMA-X's own meshes; `benchmarks/verify_fairness.py`).

**On upstream classes.**

* `SOMALayer`: `load`, `from_upstream_assets`, `rebind`, `attach_procedural_rig`,
  `build_skinning_rig`, `extend_rig_with_procedural_transforms`,
  `downsample_to_low_lod` (raises; kept to explain why), and public names for
  helpers upstream keeps private (`normalize_bone_scales`, `full_bone_scales`,
  `num_bone_scale_params`, `bone_scale_param_names` / `_segments`); its fields
  `v_template`, `weights`, `weight_values` / `weight_indices`,
  `J_regressor`, `joint_names`, `skeleton_levels`, `correctives` (the loaded
  `CorrectivesMLP`, or `None` without a checkpoint, as `correctives_model`).
  `prepare_identity(skeleton_fit="linear")` places the skeleton with the fitted
  `J_regressor` instead of the skeleton transfer, and `pose_arrays()` applies
  translation as an additive shift when no fitted binds are given;
  `prepare_identity_arrays()` / `pose_arrays()` are the array-level forms of
  `prepare_identity()` / `pose()` that the tools use, and `__call__(SOMAParams)`
  is SOMA-JAX's one-call form (78 joints in and out).
* `SkeletonTransfer(rotation_backend=…)` selects the optional Warp kernel;
  `SkeletonTransfer.fit_joint_rotations(vectorized=False)` runs the per-joint
  loop the vectorized path is checked against; upstream's `fit_rotations_warp`
  runs the vectorized JAX fit (upstream's runs the same fit in a Warp kernel).
* `CorrectivesMLP.offsets()` returns just the vertex offsets; `n_joints`,
  `n_vertices`, `cors_per_joint` name upstream's `J`, `V`, `K`.
* `SOMAHandLayer.prepare_identity` returns a `SOMAHandIdentity`; the SMPL family
  an `SMPLFamilyIdentity` (immutability, as above).
* `PoseMirrorSOMA` / `PoseMirrorMHR` (aliased to upstream's `PoseMirror_SOMA` /
  `PoseMirror_MHR`) expose their permutation tables.
* Keyword-only extras on upstream signatures: `SOMALayer.prepare_identity`'s
  `skeleton_fit` and `pose`'s `identity`;
  `PoseInversion(root_joint_idx=…)` and its `prepare_identity(skeleton_fit=…)`;
  `save_soma_npz`'s `rotation_repr` / `absolute_pose` (overriding what upstream
  infers); `export_soma_usd`'s `bind_transforms_world` / `rest_shape` (the
  identity a stateful layer would have cached); the identity backends'
  `mhr_model` / `anny_model` (share a loaded model), the hand backends'
  `output_unit` / `model_path`; `CorrectivesMLP(n_joints, n_vertices, W1, W2,
  key)` and `forward(training, key)` for dropout;
  `SMPLFamilyTopologyBridge(scale, asset_dir)`;
  `SOMAProceduralTransformDefinition(template_joint_count)`.

**Module-level names** (beside upstream's own, per module):

* `soma_jax`: the standalone body models and their parameter types,
  `SOMAParams`, `SOMAOutput`, `SOMAPoseInversion`, `PoseInversion` (the
  lightweight inverter), `PoseInversionResult`, `BatchedSkinning`,
  `CorrectivesMLP`, `PoseMirror*`, `chamfer_distance`, `apply_dof_constraints`,
  joint-orient helpers, `export_soma_usd`, `save_vertex_animation_usd`,
  `load_soma_npz`, `load_smpl_data`, `BodyModelOutput`.
* `assets`: local resolution of the asset set (`data_root`, `resolve`,
  `missing_assets` and the file-set constants), alongside upstream's Hugging
  Face `get_assets_dir`.
* `usd_io`: the LOD skin-mesh discovery helpers (`load_lod_rig`,
  `load_template_rig`, `LOD_*` constants).
* `correctives_model`: `load_correctives_pt`, `resolve_correctives_model_path`,
  and `read_torch_checkpoint`, which reads upstream's `.pt` checkpoint with
  NumPy (tensors and plain values only, as `torch.load(weights_only=True)`), so
  the default procedural layer applies upstream's correctives without PyTorch
  — bit-identical to reading it with torch.
* `procedural_transforms`: `ProceduralTransforms` (the layer's internal
  evaluator) and its helpers.
* `geometry`: SOMA-JAX names for upstream's transforms (`axis_angle_to_rotmat`,
  `rotmat_to_axis_angle`, `se3_from_rt`, `se3_inverse`, `rotmat_to_6d`, …),
  level-order FK and sparse LBS forms (`forward_kinematics`, `fk_levelorder`,
  `lbs_sparse`, `lbs_blend`, `lbs_transforms`), `RestJointSkinning`,
  `pose_from_bind`, `kabsch_points`, `rotation_from_covariance`,
  `laplacian_solve` (a zero-energy membrane fill, a different formulation from
  `LaplacianMesh`), the bidirectional vertex-set `chamfer_distance*`,
  `PoseMirror`, `infer_joint_orient_from_rest`, and skeleton utilities
  (`get_joint_subtree`, `compute_bone_lengths`, `group_body_part_vertex_ids`).
* `smpl`: `BarycentricBridge` (one-stage source → SOMA topology),
  `transfer_pose_between_layers`.

**Tools** with no upstream counterpart: `tools/download_assets.py`'s `--check`
(report what the local asset set lacks) and `--extras` (fetch
`GarmentMeasurements/point.npz`, which upstream does not ship) beside upstream's
own options; `tools/pipeline/` (rig and identity-pack builders, BVH parsing,
SMPL-X retargeting, the batch renderer),
`tools/compare_render/` (the SOMA-X vs SOMA-JAX comparison renders),
`tools/vis/`, `tools/audit_soma_features.py`,
`tools/convert/convert_correctives_pt_to_npz.py`,
`tools/convert/pack_gm_matrices_to_npz.py`,
`tools/convert/shape_space_convert.py`. Upstream's tools are ported under
their own names (see `tools/README.md`).

<a name="not-ported"></a>
## Not ported

* **torch / Warp machinery.** `setup_warp_for_ddp` (Warp under torch
  distributed), `NonPersistentModuleWrapper` (keeps a submodule out of a torch
  `state_dict`), `ChamferBatchedFunction` (a `torch.autograd.Function`) and
  `chamfer_distance_batched_kernel` (a Warp kernel), plus the Warp backends
  `lbs_warp`, `fused_refit_warp`, `align_vectors_warp`, `chamfer_warp`'s kernels
  and the `_warp_init` / `_warp_utils` / `_utils` helpers. Their behaviour is
  implemented in JAX and compared against them above.
* **Tests of that machinery**: `test_device.py` (torch device moves),
  `test_dataloader*.py` (torch `DataLoader` workers with Warp under fork/spawn),
  `test_warp_kernel_cache.py`, and these cases of otherwise-ported files:
  `test_skeleton_transfer.py`'s CPU↔GPU round trips;
  `test_soma_layer.py::test_soma_layer_pose_uses_explicit_fk_lbs_pipeline`
  (monkeypatches torch `BatchedSkinning.pose`); `test_reference_poses.py`'s
  float64 / TF32 dtype cases (`test_reference_cast_preserves_gradient`,
  `test_float32_reference_validation_with_tf32_enabled`), and the
  "before identity mutation" halves of four cases whose error checks are
  ported — an immutable layer holds no cached identity to protect.
* **Release infrastructure**: `tools/ci/` (Hugging Face asset packaging,
  publishing and verification), `tools/docker-entrypoint.sh`, and
  `test_hf_release_assets.py` (which tests those scripts; its one library-level
  case, the top-level hand-layer exports, is ported).
* **DCC plugins**: `tools/soma_procedural_blender` and `tools/soma_procedural_maya`
  evaluate the procedural definition inside Blender and Maya; they are not part
  of the Python library.
