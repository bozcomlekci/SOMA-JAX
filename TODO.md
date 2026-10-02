# TODO

## Give `SOMALayer.pose` upstream's signature and semantics

**Upstream** (`third_party/SOMA-X/soma/body/soma.py`, `SOMALayer.pose`) poses the
identity that `prepare_identity()` cached:
`pose(poses, transl=None, pose2rot=True, apply_correctives=True, absolute_pose=False, fk_only=False, return_transforms=None, *, reference_pose=None)`.
It takes the 77 posable joints (axis-angle by default), applies the T-pose
reference unless `absolute_pose=True`, and returns 77 joints / 78 transforms.

**SOMA-JAX** (`soma_jax/body/soma.py`, `SOMALayer.pose`) is a lower-level entry
point under the same name: `pose(rotmats, transl, rest_verts, rest_joints, ...)`
takes 78 rotation matrices plus the prepared rest shape, joints and binds,
requires `transl`, and orients only when given `joint_orient` or
`reference_pose`. It is the last public method whose call shape differs from
upstream's (`docs/FAITHFULNESS.md`, "Differences by design").

**Why.** Upstream's prepare-once / pose-per-frame workflow has no faithful fast
path here:

- `forward()` matches upstream but refits the identity on every call. On the
  RTX 5080 that is 0.665 vs 0.116 ms per call for 1 frame, and 0.888 vs
  0.306 ms for 64 frames.
- The low-level `pose()` called with its defaults lands up to 1.4 m from
  upstream's result for the same input.

**Plan** (the pattern SOMA-JAX's hand and SMPL-family layers already follow):

1. `prepare_identity(...)` returns an identity object: rest shape, fitted binds,
   identity rest shape, global scale and bone scales.
2. `pose(poses, transl=None, pose2rot=True, apply_correctives=True, absolute_pose=False, fk_only=False, return_transforms=None, *, reference_pose=None, identity)`
   with upstream's semantics and a `SOMAPoseOutput`.
3. Move today's low-level `pose()` to its own SOMA-JAX name and update its
   callers: `soma_jax/fitting/pose_inversion.py`,
   `tools/convert/{smpl2soma,mhr2soma,convert_amass_to_soma}.py`,
   `tools/pipeline/{motion_pipeline,demo_soma_vis}.py`,
   `tools/audit_soma_features.py`, the linear row of
   `benchmarks/{bench_forward_pass,bench_memory}.py`, the tests and the docs.
4. Remove the `SOMALayer.pose` exception from `docs/FAITHFULNESS.md` and
   `docs/DESCRIPTION.md`.

**Results do not change.** With the right arguments the prepared path already
reproduces `forward()` (0.36 µm at 64 frames), and nothing published uses
`SOMALayer.pose`. This is a breaking change for SOMA-JAX code that calls the
current `pose()`.
