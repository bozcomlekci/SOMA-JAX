# Faithfulness to SOMA-X

Status of the JAX port against upstream
[`third_party/SOMA-X`](https://github.com/NVlabs/SOMA-X), audited
module-by-module.

**On the test citations below.** Nearly every number here is produced by a
module under `tests/`, and those are named so the measurement is attributable.
The suite is developed and run locally rather than distributed — `tests/` is
git-ignored — so a clone will not contain the file a citation names. The figures
were all re-derived against upstream at the time of writing; see
[`DESCRIPTION.md`](DESCRIPTION.md#testing) for what the suite reports.

**Read this first — scope of the word "port."** The verified-faithful core is
the **LBS-only SOMA/PCA forward**: `identity_model_type="soma"`,
`apply_correctives=False`, mid or low LOD. That path is pinned against upstream
to 3.2e-6 m. Around it sit modules that are partial ports, approximations, or
JAX-only alternatives — the [correspondence map](#correspondence-map)
marks each one, and the differences are not all cosmetic.

**Constructor defaults differ from upstream.** `SOMALayer.load()` defaults to
the SOMA PCA backend with no correctives and no procedural transforms; upstream
`SOMALayer` defaults to the MHR backend, the Warp path, procedural transforms
*enabled*, and a real corrective checkpoint. A default-constructed layer on each
side is therefore not the same model. `soma_jax` also builds a **zero-valued**
corrective network when no checkpoint is given, so `apply_correctives=True`
silently applies nothing, where upstream would raise.

**One consequence is easy to mistake for a broken port.** Both sides tie the
repose to the correctives — upstream `forward` and `SOMALayer.__call__` alike set
`repose_to_bind_pose=apply_correctives`. But upstream defaults
`apply_correctives=True` (its constructor loads a checkpoint) and this layer
defaults it to `False` (it does not), so **at their respective defaults the two
sit in different repose modes**. Comparing `soma_jax` `layer(params)` against
upstream's default `prepare_identity` therefore shows ~**0.87 mm** at pose σ=0.2;
pointing upstream at `repose_to_bind_pose=False` drops the same comparison to
**0.0032 mm**. Both modes are parity-tested — see `tests/test_layer_parity.py`,
which drives each through the matching API.

## Verified exact (float32 precision)

End-to-end forward parity is enforced by `tests/test_layer_parity.py`: with the
template-merged asset, `SOMALayer.__call__` reproduces upstream
`SOMALayer.forward` to **3.2e-6 m max vertex difference** (0.00032 cm; mean
1.1e-6 m), for both `repose_to_bind_pose` settings — float32 round-off on an
exact-algorithm comparison. The test guards at 5e-5 m to stay robust across
BLAS/driver versions.

**Scope:** this parity run is **LBS-only** — both sides are called with
`apply_correctives=False`. Pose correctives are implemented and unit-tested on
the JAX side (`tests/test_soma_layer.py`), but there is **no test comparing a
real corrective checkpoint against upstream**, so correctives are outside the
"verified exact" claim below. The chain this covers:

| Stage | SOMA-JAX | Upstream match |
|---|---|---|
| Identity blend | `SOMAIdentityModel` (`coeffs · √eigenvalues @ pca`) | exact |
| Skeleton fit | `SkeletonTransfer.fit` (RBF + 2-stage Kabsch), default in `prepare_identity` | exact |
| Repose to bind | `_repose_full` (rebind + `bind_pose_local` absolute pose) | exact |
| Joint orient | `t_pose_world` remap applied by default in `__call__` | exact |
| Skinning | `pose_from_bind` (rebind + level-order FK + LBS); `transl` drives the hips FK slot | exact |
| Rig data | asset built from the **template-USD-merged** rig (see below) | exact |

**Low LOD.** `SOMALayer.load(path, lod="low")` mirrors upstream
`SOMALayer(lod="low")`: the rig, identity model and skeleton fit are all built
on the 4,505-vertex subset (not subsampled after the fact), the facial
inner-geometry exclusion lists are remapped into low-LOD indices, and
`triangles_low` becomes `faces`. Parity against upstream is enforced by
`test_low_lod_forward_matches_upstream` — **1.4e-6 m max / 4.7e-7 m mean**,
the same order as the mid-LOD result. `SOMAPoseInversion(layer, low_lod=True)`
accepts such a layer and subsamples full-resolution input itself.
`downsample_to_low_lod()` raises: subsetting an already-built layer leaves the
identity model and skeleton transfer at full resolution, which cannot be made
consistent.

Additionally parity-tested (`tests/test_soma_x_parity.py`, 54 tests): rotation
primitives, covariance/Kabsch/Newton–Schulz alignment, SE(3), quaternions,
LBS forms, top-K skinning, world↔local rig utils, skeleton-transfer stages.

### The template-rig merge (matters!)

Upstream SOMA-X does **not** use the `SOMA_neutral.npz` rig arrays as-is: it
merges `SOMA_template_rig.usda` over them (`soma/soma.py` — "Merge rig tensors
from the canonical template USD"). This is a **different skinning solve, not a
refinement of the same one**: the raw npz carries 60,735 nonzero weights and the
merged 78-joint rig carries 39,283, with 45,672 entries differing (mostly the
legs) and `bind_pose_world` moving up to 0.1275 cm. An asset built from the npz
alone diverges ~0.1–0.2 cm at leg/shoulder joints.

`SOMALayer.from_upstream_assets()` performs this merge directly from the two
upstream files, with no torch and no upstream package in the loop.
`docs/INSTALL.md` §4.2 builds the same archive **through the upstream layer**,
which remains useful as an independent check of the merge.

<a name="correspondence-map"></a>
## Correspondence map — what reimplements what

Every module in `soma_jax/` and its upstream counterpart. Each module's
docstring carries the same pointer, so the mapping is visible from the code as
well as from here.

| SOMA-JAX | Upstream `soma/…` | Status |
|---|---|---|
| `soma.py` | `soma.py` (`SOMALayer`) | forward is a port — pinned at 3.2e-6 m (mid LOD), 1.4e-6 m (low LOD), both LBS-only with `identity_model_type="soma"`. **Constructor defaults differ** from upstream; `rebind()` updates only `v_template` |
| `identity_model.py` | `identity_model.py` | **mixed.** `SOMAIdentityModel` (PCA) is a port. **MHR is now a real port** — `attach_native_archive()` runs the full `mhr_model_lod1.pt` forward in JAX (127-joint Momentum rig, parameter transform, pose correctives, LBS) to 7.6e-5–1.8e-4 cm depending on pose amplitude; without it the backend stays a linear-PCA stand-in. **Anny is a port too** — `anny_native.py` evaluates the rest shape (`template + B·coeffs`) exactly in JAX and delegates the phenotype→coefficient step to the `anny` package, which is what upstream itself does, so the delegation is the faithful behaviour rather than a gap. SMPL / Garment are transfer substitutes. Laplacian blend conditions + free/anchor set match upstream |
| `correctives_model.py` | `correctives_model.py` | **partial.** The loaded, inference-only masked MLP equation corresponds at dropout 0 / scale 1. Constructor and checkpoint defaults differ (`use_tanh` False here, True upstream), training dropout and diagnostics are absent, and the forward adds offsets **unscaled** where upstream multiplies by the cached global scale and requires the procedural rig |
| `rig_build.py` | `soma.py` (the rig-load path) | port of the parts of upstream's constructor that assemble the rig: the `SOMA_template_rig.usda` merge over `SOMA_neutral.npz`, world↔local conversion, the affine joint-regressor fit, and `derive_soma_rig_without_procedural_joints`. Torch-free, so `SOMALayer.from_upstream_assets()` needs neither PyTorch nor the upstream package. Checked against upstream's own `rig_data`: pruned weights 6e-8, `bind_pose_world` 0.0, parents and joint names exactly equal (`tests/test_rig_build.py`) |
| `procedural_transforms.py` | `procedural_transforms.py` | port. Definition parse, all three channel extractors, bind-alignment quaternions, `twist_rotations_from_source` and `expand_world_transforms_from_source_fk`. `SOMALayer.from_upstream_assets()` drives the expanded rig at **0.34–1.16 mm** vs upstream's default, parity-tested by `tests/test_procedural_parity.py`. Per-procedural-joint mode dispatch is ported (see [Per-joint extraction modes](#the-procedural-rig-parity-achieved) below) |
| `pose_inversion_soma.py` | `pose_inversion.py` (`PoseInversion`) | analytical + Lie-GN are parity-tested ports; autograd stage unverified. `transfer_to_soma` now covers all of upstream's routes: SOMA topology, full→low subsampling, the dedicated full-res-MHR→low-SOMA interpolator (`_setup_pose_transfer`, gated on low-LOD + MHR as upstream gates it), and the active identity model's correspondence |
| `usd_io.py` | `io.py` (USD half) | port, including the LOD-discovery chain (`find_lod_skin_mesh_name` / `load_lod_rig`, verified at 18056/4505/612 verts) and `load_template_rig` for the 122-joint skeleton. UV primvar filtering now checks type **and** interpolation, matching upstream |
| `io.py` | `io.py` (NPZ half) | port — `keep_root=False` default that actually strips Root from `poses`/`joint_names`, `absolute_pose` inferred from `joint_orient`, pose-shape `ValueError`, `global_scale`/`hand_type`. Verified interchangeable in both directions against upstream |
| `units.py` | `units.py` | port |
| `assets.py` | `assets.py` | same role, **different mechanism.** Upstream downloads a HuggingFace snapshot; this resolves the same files from `assets/third_party/` and the vendored submodule, and materialises a `data_root` view over them. Deliberate — the assets already ship in `third_party/SOMA-X`, so re-downloading them would be the divergence. `tools/download_assets.py` covers the fetch case |
| `smpl/__init__.py` | `smpl/__init__.py`, `smpl/transfer.py` | `SMPLFamilyPoseTransferResult`, `SMPLFamilyTopologyBridge` (two-stage source→SOMA-wrap→target) and `transfer_pose_between_layers` all ported. `BarycentricBridge` is a SOMA-JAX-only one-stage helper that stops at SOMA topology. Upstream's `SMPLLayer`/`SMPLXLayer` rigs not ported — this package drives `soma_jax.body_models` |
| `geometry/transforms.py` | `geometry/transforms.py` | alignment core is a port — matches upstream `align_vectors` in float64 to 1e-15 (CPU) / 1e-13 (GPU) for all three methods (float32: ≤1.5e-6 `auto`/`newton-schulz`, ≤9e-6 `kabsch` — SVD round-off, not algorithm). `rotmat_to_axis_angle` ports the three-branch form including near-π, with one **deliberate** divergence: upstream's small-angle branch returns 2× the correct vector, so this returns the correct one. Euler/quaternion conversions ported |
| `geometry/lbs.py` | `geometry/lbs.py` | port |
| `geometry/batched_skinning.py` | `geometry/batched_skinning.py` | port. Class defaults match upstream (`sparse_k=8`, `hips_idx=1`) and `align_translation` anchors X and Z on the translation joint as upstream does, leaving Y from FK |
| `geometry/rig_utils.py` | `geometry/rig_utils.py` | port. World↔local conversion, upstream's `precompute_joint_orient`, and `PoseMirrorSOMA`/`PoseMirrorMHR` (bit-exact negate-parameter tables). The old heuristic is kept under its own name, `infer_joint_orient_from_rest` |
| `geometry/skeleton_transfer.py` | `geometry/skeleton_transfer.py` | port (pure-PyTorch path) |
| `geometry/barycentric_interp.py` | `geometry/barycentric_interp.py` | the 4-coord deformation path is a port, matching upstream to 1.2e-7 (`normal_scale="area"`) and 3.6e-7 (`"edge"`) in float32. The nearest-face search, singular-tetrahedron fallback, and the 3-coord compatibility mode are JAX-only; upstream's stateful `BarycentricInterpolator` correspondence API is not ported |
| `geometry/interpolate.py` | `geometry/interpolate.py` | port — RBF basis weights match upstream to **2.3e-13 in float64** across all three kernels, i.e. the same algorithm. In float32 the linear solve's conditioning bounds it: 6.0e-6 (`linear`, the kernel `SkeletonTransfer` uses on both sides), 1.3e-6 (`gaussian`), 9.1e-5 (`thin_plate_spline`, the class default but unused by SOMA) |
| `geometry/laplacian.py` → `LaplacianMesh` | `geometry/laplacian.py` (`LaplacianMesh`) | port — 6.9e-6 m |
| `geometry/laplacian.py` → `laplacian_solve` | — | **JAX-only, different formulation** (zero-energy membrane) |
| `geometry/chamfer.py` | — | **JAX-only point-cloud loss.** Bidirectional mean-squared vertex-to-vertex; upstream's `ChamferLoss` is a one-way point-to-*triangle* query. Different loss, not a port |
| `geometry/warp_kabsch.py` | `geometry/align_vectors_warp.py` | **diverges** — implements `"kabsch"`, not upstream's default `"auto"` |
| `body_models/model_io.py` | `_smpl_family_loader.py` | port |
| `types.py` | — | JAX-only (`SOMAParams` / `SOMAOutput`) |
| `pose_inversion.py` | — | JAX-only lightweight alternative inverter |
| `body_models/{smpl,smplx}.py` | `smpl/__init__.py` (`SMPLLayer`, `SMPLXLayer`) | corresponding core math, different API/feature surface |
| `body_models/{_base,smplh,mhr,anny}.py` | — | JAX-only standalone models |
| `body_models/mhr_native.py` | `mhr_model_lod1.pt` (the shipped TorchScript archive) | transcription, not of a Python module but of the archive upstream calls into: blend shapes, face expressions, the parameter transform, local/global skeleton state, pose correctives and skinning, all in JAX. Agrees with the TorchScript forward to **7.6e-5–1.8e-4 cm** (rest pose to σ=0.25); `MHRIdentityModel.attach_native_archive()` wires it into the SOMA backend (`tests/test_mhr_native.py`) |
| `body_models/anny_native.py` | `identity_model.py` (`AnnySimplified`) | port of the rest-shape evaluation (`template + B·coeffs`); the phenotype→coefficient step is delegated to the `anny` package exactly as upstream does |


**Reading the Status column.** "port" means the JAX code implements the same
algorithm as the named upstream symbol. Where a row says *parity-tested*, a
test in `tests/` runs both implementations on identical inputs and asserts
agreement; where it says *not parity-tested*, the correspondence rests on
reading both sources, which is weaker. The modules with no parity test are
`correctives_model` — it needs a trained checkpoint — and `chamfer`, which is a
different loss rather than a port, so there is nothing to compare against.
Everything else in the map is exercised against upstream; `procedural_transforms`
by `tests/test_procedural_parity.py` and the NPZ half of `io` by
`tests/test_soma_x_parity_modules.py::TestIoRigKeys`, which round-trips clips in
both directions.

**How this map was audited.** Both source trees were read side by side, by me
and independently by a second model (GPT-5.6-sol), and their findings
reconciled. Where a row cites a number, it came from running both
implementations on the same input. Rows marked *not parity-tested* rest on
source reading alone.

Upstream modules with no port: `pose_inversion_mhr.py`,
`geometry/lbs_warp.py`, `geometry/fused_refit_warp.py`,
`geometry/chamfer_warp.py` (behaviour ported to pure JAX; the kernel is not),
`geometry/align_vectors_warp.py` (partially — see above), `_warp_utils.py`,
`geometry/_warp_init.py`, `geometry/_utils.py`. Rationale for each in
[Not ported](#not-ported) below.

## The procedural rig (parity achieved)

Upstream's default is the expanded 122-joint twist skeleton. It is worth having:
against upstream's own non-procedural mode the twist joints move the surface by
a median **6.7 mm at pose σ=0.05** rising to **148 mm at σ=1.2** (5 seeds of
`standard_normal((1, 77, 3)) * σ`; per-seed range 4.7–9.2 mm and 120–189 mm
respectively), concentrated in bands on the limbs. Regenerate with
`python benchmarks/plot_procedural_gap.py`
(`benchmarks/figures/procedural_rig_gap.png`).

Driving it takes **two** rigs. Measured on upstream's `SOMALayer` with
`enable_procedural_transforms=True`:

```
rig_data['joint_names']                 -> 122
skeleton_transfer.skinning_weights      -> (18056, 78)
skeleton_transfer.bind_world_transforms -> (78, 4, 4)
```

Identity, the per-identity skeleton fit and the bind data stay on the **78-joint
public rig in both modes**; only FK and LBS use the expanded skeleton. The public
pose contract is 78 joints / 77 posable, and `pose()` returns those either way.

`SOMALayer.from_upstream_assets()` reproduces this, pinned by
`tests/test_procedural_parity.py` against a captured upstream reference:

| clip | max | mean |
|---|---|---|
| rest | **0.34 mm** | 0.009 mm |
| σ=0.15 rad | **0.74 mm** | 0.024 mm |
| σ=0.45 rad | **1.16 mm** | 0.062 mm |

Four things had to be right *together* — each was found by measurement, and any
one alone leaves 16–52 mm:

1. **Order.** Upstream does not expand rotations and run FK on 122 joints. It
   runs FK on the public 78 and expands the resulting *world transforms*
   (`expand_world_transforms_from_source_fk`, `procedural_transforms.py:1405`,
   reached from `soma.py:1580` via `transform_expander=`). A twist joint is a
   single local step off its public parent, not an FK-chain link — driving it
   through FK lets the bind absorb its rotation exactly.
2. **Frame.** `aligned_x_swing_twist` reads the **posed world** transforms
   (`_twist_angles_from_source(source_rotations, source_world_transforms)`).
   Feeding local rotmats put the emitted twist rotations 1.87 out; threading the
   world transforms through brought them to 2.4e-4.
3. **Bind.** `_apply_translation_parameters` rewrites only the **translation
   column** and leaves every rotation block alone. Zeroing the twist bind
   rotation to identity desynchronises the bind from the posed transform.
4. **Local step.** The base rotation and local translation each joint composes
   come from *that identity's* expanded bind — `rebind()` recomputes them as
   `joint_world_to_local(bind_world)` (`batched_skinning.py:312`) — not from the
   static template T-pose.

A unit bug fell out of (3): `rig_build` keeps the USD/npz transforms in native
centimetres while the fitted `bind_transforms` are metres. The rejected
intermediate variants and their numbers are in the branch history.

**Bone scales** work through the expanded rig: upstream's
`_apply_target_bone_scales` maps each public scale on via `target_to_public` and
overrides every twist joint to follow its segment's **end** joint, so a stretched
forearm carries its twist helpers. Verified both ways — varying the scales moves
the mesh by 71.443 mm on *both* implementations identically, and parity holds at
1.30 mm with and without scaling.

**Both upstream rigs build from upstream's two assets.**
`from_upstream_assets(procedural=False)` ports
`derive_soma_rig_without_procedural_joints`: it prunes the procedural and
auxiliary joints from the template and aggregates each dropped joint's skin
weights onto its nearest kept parent. Against upstream's own non-procedural
`rig_data` the pruned weights agree to 6e-8, `bind_pose_world` to 0.0 and the
parents exactly; the forward matches at 0.34/0.74/1.16 mm.

**Per-joint extraction modes** are dispatched as upstream does — each twist
joint's parameter-matrix row routed into its own mode and the per-mode
contributions summed. Every published asset is uniform, so this only runs on a
mixed definition; `tests/test_soma_x_parity_modules.py` exercises one
synthetically and checks that switching a single joint disturbs no other.

## Faithful defaults vs. alternatives

| Behaviour | Default (SOMA-X) | Alternative (SOMA-JAX-only) |
|---|---|---|
| Skeleton fit | `skeleton_fit="auto"` → full RBF+Kabsch | `skeleton_fit="linear"` (J_regressor; used by benchmarks/tools) |
| Repose | `repose_to_bind_pose=True` | `False` (T-pose rest) |
| Translation | hips FK slot (via `bind_transforms`) | additive post-LBS shift (when `bind_transforms=None`) |
| Sparse LBS | top-K=8 (matches Warp path) | any `sparse_k`, or dense |
| Correctives input | absolute (orient-remapped) rotations | — |

Legacy assets without `bind_shape`/`eigenvalues` degrade gracefully to the
linear/unscaled paths (a warning-free fallback used by the synthetic test
fixtures).

## Implemented with a different (verified-equivalent) structure

`laplacian`, `interpolate`, `units`, and the deformation paths of
`batched_skinning`, `barycentric_interp`, `transforms` and `rig_utils` — APIs
are reorganised for JAX/equinox, numerics parity-tested against upstream.

**This section does not cover** `chamfer`,
`correctives_model` or `io`; those are *not* verified-equivalent, and the
[correspondence map](#correspondence-map) states what
each actually is. Nor does it cover the non-deformation surfaces of the four
modules above (public class defaults, name-colliding helpers, search and
degeneracy handling) — the map row for each names the divergence.

Known deliberate divergence: upstream's sparse-RBF path zeroes the *virtual
root* row; both dense paths return the bind position
(`test_upstream_sparse_rbf_zeroes_the_virtual_root`).

Two gaps in this group, both currently unguarded by tests:

* **MHR Laplacian blend — now ported and called.** Upstream re-solves the SOMA
  inner-face vertices (eye bags + mouth bag, 691 on this rig) after topology
  transfer, because a source mesh like MHR has no geometry there
  (`soma/identity_model.py`: `self._laplacian_mesh.solve(...)`).
  `soma_jax.geometry.laplacian.LaplacianMesh` ports upstream's `LaplacianMesh`
  for the configuration SOMA uses (order 1, hard constraints), including the
  detail that matters: the right-hand side is the **template's own Laplacian
  coordinates** (`L_U @ V_ref`), not zero, so the filled region reproduces the
  template's local shape instead of collapsing to a membrane.
  `tests/test_laplacian.py` pins it to upstream at **6.9e-6 m** max.

  Backend note: upstream builds the system in `torch.sparse`. The port
  assembles and factorises once on the host with SciPy — the cotangent build
  needs ragged scatter-add over faces, and shapes depend on the constraint set
  — while **every per-call solve is pure JAX** (gather, segment-sum, dense
  Cholesky solve on the small `(691, 691)` block, scatter), so it is `jit`-able,
  batched and differentiable.
* **NPZ animation format.** Now semantically interchangeable with upstream, not
  merely field-name compatible. `save_soma_npz` defaults to `keep_root=False`
  and **strips Root from both `poses` and `joint_names`** (upstream
  `soma/io.py:1008`) rather than only recording the flag; `absolute_pose` is
  inferred as `joint_orient is None`; the unparseable-pose-shape `ValueError`
  and the optional `global_scale` / `hand_type` fields are ported. Verified in
  both directions — upstream-written clips load here and port-written clips load
  in SOMA-X — by `tests/test_soma_x_parity_modules.py::TestIoRigKeys`.

## Pose inversion (faithful multi-stage solver)

`SOMAPoseInversion` (`soma_jax/pose_inversion_soma.py`) is the faithful port of
upstream `soma.pose_inversion.PoseInversion`, covering all three stages:
SkeletonTransfer warm start → top-down per-joint inverse-LBS Procrustes refit
(body / finger / full passes with `_update_root_translation` between rounds) →
Lie-algebra Gauss–Newton with the Kinematic Lever Arm Jacobian and per-frame
backtracking line search → optional autograd FK refinement. Also ported:
`PoseInversionResult`, `roundtrip`, `transfer_to_soma`, `prepare_identity`,
1-DOF elbow/knee constraints, heel/extremity vertex weighting, and joint pose
priors.

`tests/test_pose_inversion_parity.py` pins it against upstream (which runs the
analytical refit through a **fused Warp kernel**):

| Stage | Agreement with upstream |
|---|---|
| Analytical refit | `max |ΔR| = 1.3e-4`, root translation `4.8e-7 m`, per-vertex error `8.6e-6 m` |
| + Lie-GN | mean per-vertex error: upstream `0.1618 cm`, SOMA-JAX `0.1626 cm`; root translation `5.6e-6 m` |

**What the parity test actually guards:** the analytical refit and the Lie-GN
stage are both compared against upstream. The **autograd stage is not** — 
`test_pose_inversion_parity.py` only checks that it improves on its own warm
start and returns the right shapes; upstream's autograd path is never run.

Lie-GN drifts slightly more per joint because it solves a dense `(3K x 3K)`
normal equation each iteration — torch uses LU via `solve_ex`, JAX its own
solver, and the damping ladder/line-search branch can select differently. JAX
has no `solve_ex` info flag, so candidate solutions are validated by
finiteness instead. The reconstruction callers consume stays equivalent.

`soma_jax.PoseInversion` remains as the lightweight SOMA-JAX **alternative**
(single Kabsch init + one autograd refine + 1-DOF constraints).

## SOMALayer surface

Ported: bone-scale posing (`scale_params`), `fk_only`, the `transforms` output
field, and the public rig view (`public_rig_view`, `to_public_rotations`,
`public_skinning_weights`, `public_joint_names`). The bone-scale control layout
derives independently to exactly upstream's **56** active joints
(`NUM_BONE_SCALE_PARAMS`), with `scale_param_names` / `scale_param_segments`
naming each `(parent, child)` edge.

Two deliberate API shape differences, both because the JAX layer is immutable
and has no `_cached_*` identity state:

* `bone_scales` are passed to `pose()` per call rather than cached by
  `prepare_identity()`.
* `public_rig_view()` returns a plain dict rather than a frozen dataclass, and
  takes the fitted binds as an argument.

The 78-vs-77 joint convention still differs: SOMA-JAX takes all 78 joints with
Root at index 0 (which must be identity for parity), where upstream takes 77
public joints and pads Root internally.

## USD I/O

`soma_jax/usd_io.py` ports upstream's USD half of `io.py`: `save_soma_usd`,
`save_vertex_animation_usd`, `export_soma_usd`, `write_usd_mesh`,
`load_usd_mesh`, `load_usd_skeleton`, `load_usd_animation`,
`load_usd_skinning`, `list_usd_meshes`, `fan_triangulate`. `pxr` is imported
lazily, so `usd-core` stays optional. Round-trips are covered by
`tests/test_usd_io.py`: bind transforms, weights and rest vertices round-trip to
`atol=1e-6`; the animation round-trip is currently shape-checked, not compared
frame-by-frame.

`export_soma_usd` takes the fitted rig explicitly (`bind_transforms_world`,
`rest_shape`) since there is no cached identity to read it from.

The LOD-discovery chain used to *build* the asset is ported:
`find_lod_skin_mesh_name` and `load_lod_rig` (`usd_io.py`), verified against the
template USD at 18056 / 4505 / 612 vertices for mid / low / xlo, plus
`load_template_rig` for the 122-joint skeleton. `SOMALayer.from_upstream_assets()`
merges the npz and the template USD directly, so building the runtime rig no
longer requires torch or the upstream package; `docs/INSTALL.md` §4.2 remains as
the route that builds the archive *through* the upstream layer, which is useful
as an independent check of the merge.

<a name="not-ported"></a>
## Not ported

* **`pose_inversion_mhr`** — deliberately not ported. Upstream documents it as
  *"Private native-MHR pose inversion utilities … so MHR-specific DOF handling,
  co-located ankle distribution, and parameter-matrix projection can be removed
  or hidden for public releases"*, and it requires `MHR/MHR_base_rig.npz` +
  `MHR/parameter_transform.npz`, neither of which ships in the public
  `nvidia/SOMA-X` asset dump. Porting it would mean shipping code that cannot
  be verified against upstream.
* **Warp/CUDA kernels** (`lbs_warp`, `fused_refit_warp`, `chamfer_warp`,
  `align_vectors_warp`) — these are backends, not behaviour. The JAX
  implementations are parity-tested against them.

  One caveat: `soma_jax/geometry/warp_kabsch.py` is an *optional* Warp
  covariance→rotation kernel of SOMA-JAX's own (the benchmarks' "SVD in Warp"
  pipeline). It implements plain SVD Procrustes, i.e. `method="kabsch"`, **not**
  upstream's default `method="auto"` (Newton–Schulz on a gauge-regularized
  covariance), for which upstream's `align_vectors_warp` ships a dedicated
  kernel. The two agree on well-conditioned covariances and diverge on
  rank-deficient ones — measured **209 µm max / 0.944 µm mean** on the posed mesh
  at the benchmark operating point (`benchmarks/verify_fairness.py`, which needs
  a GPU: the kernel registers an XLA FFI handler for CUDA only, and
  `tests/test_skeleton_transfer.py` skips its Warp case on CPU for the same
  reason). **The pure-JAX path is the faithful one**: across 4096 problems, half
  of them near-planar (rank-deficient covariance), it reproduces upstream
  `align_vectors` to **1e-15 (CPU) / 1e-13 (GPU) in float64** — either way the
  two are the same algorithm, the backends differing only in which SVD they call.
  In float32 the agreement is bounded by SVD round-off rather than by the
  algorithm: ≤1.5e-6 for `auto` and `newton-schulz`, ≤9e-6 for `kabsch`, on both
  backends (`python benchmarks/verify_fairness.py --align-only` runs just this
  probe and needs no GPU; the parity test `tests/test_soma_x_parity.py` asserts
  at `TOL = 1e-5`).
* **`soma.smpl.SMPLLayer` / `SMPLXLayer` / `create_smpl_family_layer`** — the
  SOMA-style `BatchedSkinning` rigs upstream wraps the SMPL assets in. This port
  drives `soma_jax.body_models` instead, so `transfer_pose_between_layers` takes
  `betas` where upstream takes `identity_coeffs`. The transfer itself *is*
  ported (all four upstream stages); see `tests/test_smpl_transfer.py`.
* **Maya / Blender procedural plugins**, docs tooling — out of scope.

These are missing *features*, not silent behavioural differences: calling
patterns that exercise them fail loudly (missing argument/attribute) rather
than returning wrong numbers.
