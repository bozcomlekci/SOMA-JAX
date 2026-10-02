# Faithfulness to SOMA-X

How SOMA-JAX corresponds to upstream [SOMA-X](https://github.com/NVlabs/SOMA-X)
**v0.3.3** (the `third_party/SOMA-X` submodule, release commit `d6aa640`):
what is ported, what is measured against upstream and how closely, what
differs by design, which upstream defects are not reproduced, what SOMA-JAX
adds, and what has no JAX counterpart.

**On the test citations below.** The numbers come from modules under `tests/`,
named so each figure is attributable. The suite is developed and run locally
rather than distributed — `tests/` is git-ignored — so a clone will not contain
the file a citation names. See [`DESCRIPTION.md`](DESCRIPTION.md#testing) for
what the suite reports.

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
  The exceptions are `SOMALayer` itself, built by the classmethod
  `from_upstream_assets(...)`, which takes upstream's constructor parameters,
  and `SOMALayer.pose` ([below](#differences-by-design)).
* **Forward parity.** The body layer matches upstream at every LOD, on both
  rigs, with and without the pose-corrective network, to float32 rounding:
  **≤ 3.1 µm** on vertices and **≤ 5.1 µm** on joints
  ([table](#the-body-forward)).
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

`tests/test_body_lod_parity.py`: `SOMALayer.from_upstream_assets(lod=…,
procedural=…)` against upstream `SOMALayer(lod=…, enable_procedural_transforms=…)`,
poses `N(0, 0.35²)` rad, translations `N(0, 0.1²)` m, a zero and a random
(`N(0, 0.5²)`) identity, `identity_model_type="soma"`:

| LOD (vertices) | rig | correctives | max \|Δ vertex\| | max \|Δ joint\| |
|---|---|---|---|---|
| mid (18,056) | legacy (78 joints) | — | 3.1 µm | 5.0 µm |
| mid | procedural (110 joints) | off | 3.0 µm | 5.1 µm |
| mid | procedural | **on** | 3.0 µm | 5.1 µm |
| low (4,505) | legacy | — | 1.8 µm | 3.2 µm |
| low | procedural | off / on | 1.9 µm | 3.4 µm |
| xlo (612) | legacy | — | 1.6 µm | 2.5 µm |
| xlo | procedural | off | 1.7 µm | 2.4 µm |
| xlo | procedural | **on** | 1.9 µm | 2.4 µm |

The same test pins the reposed rest shape, the identity rest shape and the
fitted bind transforms of `prepare_identity(repose_to_bind_pose=True)` to
upstream's cached `_cached_rest_shape` / `_cached_identity_rest_shape` /
`_cached_bind_transforms_world`, and the faces and facial exclusion lists of
every LOD exactly. `tests/test_layer_parity.py` and
`tests/test_procedural_parity.py` repeat the comparison through the other entry
points (`pose()`, the explicit reposed path, bone scales).

The other identity backends, built from the same assets as upstream builds
them (`tests/test_body_identity_backends.py`; poses `N(0, 0.3²)` rad, random
identities and body-part scales, no correctives):

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

`tests/test_pose_inversion_parity.py`, upstream `PoseInversion` against
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
`prepare_identity()` takes upstream's parameters and *returns* the identity
(rest shape and joints; the fitted binds with `return_bind_transforms=True`),
and `forward()` / `__call__` fit the identity per call. The hand and
SMPL-family layers keep upstream's `pose(...)` signature plus one `identity=`
argument, the object their `prepare_identity` returns. Upstream methods that
default to "the cached identity" (`public_rig_view()`,
`public_bind_transforms_world()`, …) default to the rig's bind pose instead, or
take the fitted binds as an argument. `SOMAPoseInversion.prepare_identity` keeps
upstream's stateful contract, since the inversion object is mutable on both
sides.

**`SOMALayer.pose` is SOMA-JAX-shaped.** Upstream's
`pose(poses, transl=None, pose2rot=True, apply_correctives=True, absolute_pose=False, fk_only=False, return_transforms=None, *, reference_pose=None)`
poses the cached identity. SOMA-JAX's `pose(rotmats, transl, rest_verts,
rest_joints, ...)` is a lower-level entry point under the same name: it takes
78-joint local rotation matrices (Root included; no `pose2rot`), the prepared
rest shape and joints and, on the faithful path, `bind_transforms`; `transl` is
required; `apply_correctives` defaults to "apply when a checkpoint is loaded";
and it applies the T-pose orient only when given `joint_orient` or
`reference_pose` (upstream's orients unless `absolute_pose=True`). Code written
for upstream's `pose()` should call `forward()`, which has upstream's signature,
semantics and 77-joint output.

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

Each was confirmed against the v0.3.3 code; SOMA-JAX implements what the code
evidently intends.

* **`PoseInversion(soma, low_lod=True)` — the default — fails on a mid or xlo
  `SOMALayer`**: `soma/fitting/pose_inversion.py` builds its internal low-LOD
  layer after `from .body import SOMALayer`, which resolves to the nonexistent
  `soma.fitting.body` (`ModuleNotFoundError`). Upstream's `smpl2soma`,
  `mhr2soma` and `convert_amass_to_soma` construct exactly that. SOMA-JAX builds
  the low-LOD layer from the original's recorded construction.
* **`PoseInversion(low_lod=True)` on a hand, MANO or SMPL-family layer** would
  build a *body* `SOMALayer`; SOMA-JAX raises instead.
* **Upstream's conversion tools on the default (procedural) layer**: they remove
  `soma._t_pose_orient` — computed on the 110-joint skinning rig — from the 78
  public rotations `PoseInversion` returns (`RuntimeError` on the shape), label
  clips with `rig_data["joint_names"]`, the 110 skinning joints (every public
  joint from `RightShoulder` on mislabelled), and render through
  `batched_skinning.pose` with public rotations. The tool ports use the public
  rig's orient and names and the layer's own pose. `mhr2soma --autograd-iters`
  ≥ 2 also fails upstream ("backward through the graph a second time": the
  TorchScript parameters require grad, so the prepared identity carries graph
  history); the port's identity is a constant.
* **`matrix_to_rotvec`** returns twice the rotation vector below 1e-3 rad (its
  small-angle series applies the `w` factor to a vector that is `2w`).
* **`rotvec_to_matrix`** (unused upstream) does not return rotation matrices.
* **Negative-index roots.** Upstream's torch indexing treats a parent id of `-1`
  as "the last joint" in several helpers; SOMA-JAX treats it as a root, as the
  self-parented convention upstream's own assets use.

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
  `J_regressor` instead of the skeleton transfer, and `pose()` applies
  translation as an additive shift when no fitted binds are given.
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
  `skeleton_fit`, `return_bind_transforms`, `return_identity_rest_shape`;
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
* `correctives_model`: `load_correctives_pt`, `resolve_correctives_model_path`.
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
