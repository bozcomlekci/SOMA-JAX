"""Convert pose parameters between SOMA, SMPL, and SMPL-X.

Mirrors the surface of third_party/SOMA-X/tools/pose_converter.py, scoped to the
two retargeting directions soma_jax already supports end-to-end:

* ``smplx-to-soma``: SMPL-X motion clip (.npz) → SOMA 78-joint
  axis-angle sequence (inverse-LBS via :func:`tools.motion_pipeline.
  smplx_motion_to_soma_poses`).

* ``smpl-to-smplx``: SMPL .pkl motion → SMPL-X axis-angle sequence (uses
  SMPL-X's same 21-joint body block; hand/face stay at zero).

* ``soma-to-smplx``: take SOMA-format axis-angles and rewrite them as SMPL-X
  parent-relative locals using the existing
  ``tools.soma_to_smplx.smplx_poses_from_soma_world`` (the inverse direction
  used inside ``demo_soma_vis.py``).

Usage::

    python tools/pose_converter.py smplx-to-soma \\
        --smplx-motion path/to/motion.npz \\
        --smplx-model  data/smplx/SMPLX_NEUTRAL.npz \\
        --soma-model   assets/SOMA_neutral_fixed.npz \\
        --output       out_soma.npz

    python tools/pose_converter.py soma-to-smplx \\
        --soma-motion  out_soma.npz \\
        --smplx-model  data/smplx/SMPLX_NEUTRAL.npz \\
        --soma-model   assets/SOMA_neutral_fixed.npz \\
        --output       out_smplx.npz
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.pop("LD_LIBRARY_PATH", None)


def _cmd_smplx_to_soma(args):
    """Retarget an SMPL-X motion clip onto the SOMA 78-joint skeleton via
    mesh-based inverse-LBS.

    Pipeline (delegates to ``tools.motion_pipeline.smplx_motion_to_soma_poses``):

      1. Load the SMPL-X model + the clip (axis-angle per frame).
      2. Forward SMPL-X to get posed verts; transfer to SOMA topology via the
         HF ``SMPLX/SOMA_wrap.obj`` barycentric correspondence.
      3. Run :class:`soma_jax.PoseInversion` to recover the SOMA-frame world
         rotations that best reproduce that posed mesh on the SOMA bind shape.
      4. Convert world → local axis-angle and save together with the per-frame
         root translation, joint names, and per-vertex fit error.
    """
    import motion_pipeline as mp
    from soma_jax import SOMALayer
    layer = SOMALayer.load(args.soma_model, identity_model_type="soma")
    smplx = mp.load_smplx_jax(args.smplx_model)
    motion = mp.load_smplx_motion(args.smplx_motion, num_frames=args.num_frames)
    poses, extras = mp.smplx_motion_to_soma_poses(
        smplx, layer, motion, hf_dir=args.hf_dir,
        refine_iters=args.refine_iters, return_extras=True,
    )
    out = {
        "poses": np.asarray(poses),                                     # (T, 78, 3) axis-angle
        "root_translation": np.asarray(extras["root_translation"]),      # (T, 3)
        "joint_names": np.asarray(extras["joint_names"]),
        "per_vertex_error": np.asarray(extras["per_vertex_error"]),
    }
    np.savez(args.output, **out)
    print(f"wrote {args.output}  ({out['poses'].shape[0]} frames, "
          f"err mean={out['per_vertex_error'].mean()*100:.2f} cm max={out['per_vertex_error'].max()*100:.2f} cm)")


def _cmd_soma_to_smplx(args):
    """Convert a SOMA 78-joint axis-angle motion into SMPL-X parent-relative
    locals (body_pose / global_orient / jaw_pose / eye / hand chains).

    Pipeline:

      1. Load the SOMA motion + the SOMA layer.
      2. Run forward kinematics on the SOMA poses to produce per-frame
         per-joint **world rotation matrices**.
      3. Re-express those world rotations as SMPL-X parent-relative locals
         via the existing :func:`tools.soma_to_smplx.smplx_poses_from_soma_world`
         adapter (name-based joint mapping; finger/face joints are copied
         through 1:1 since SOMA's hand + face chains match SMPL-X's).
      4. Save in SMPL-X's standard axis-angle parameter blocks.
    """
    import motion_pipeline as mp
    from soma_to_smplx import smplx_poses_from_soma_world
    from soma_jax import SOMALayer
    import jax, jax.numpy as jnp

    soma_motion = dict(np.load(args.soma_motion, allow_pickle=False))
    poses_aa = soma_motion["poses"]                    # (T, 78, 3) axis-angle
    transl   = soma_motion["root_translation"]         # (T, 3)
    soma_names = list(soma_motion["joint_names"])
    layer = SOMALayer.load(args.soma_model, identity_model_type="soma")
    smplx = mp.load_smplx_jax(args.smplx_model)

    # Build SOMA per-frame world rotations via FK; we need them to feed the
    # soma_to_smplx adapter, which works in world-rotation space.
    from soma_jax.geometry.transforms import axis_angle_to_rotmat
    from soma_jax.geometry.lbs import forward_kinematics
    T = poses_aa.shape[0]
    rotmats = jax.vmap(jax.vmap(axis_angle_to_rotmat))(jnp.asarray(poses_aa))
    # Neutral SOMA rest joints (identity-independent), broadcast to T frames.
    rest_joints_single = layer.J_regressor @ layer.v_template   # (J, 3)
    rest_joints = jnp.broadcast_to(rest_joints_single, (T,) + rest_joints_single.shape)
    parents_np = layer._parents_np
    G = jax.vmap(lambda R, j: forward_kinematics(R, j, parents_np))(rotmats, rest_joints)
    world_rot = np.asarray(G[..., :3, :3])             # (T, J, 3, 3)

    smplx_orient = np.asarray(layer.t_pose_world[..., :3, :3]) if layer.t_pose_world is not None else None
    poses_smplx = smplx_poses_from_soma_world(
        world_rot, soma_names, np.asarray(smplx._parents_np),
        soma_orient_world=smplx_orient,
    )

    out = {
        "global_orient":   np.asarray(poses_smplx["global_orient"]),    # (T, 3)
        "body_pose":       np.asarray(poses_smplx["body_pose"]),         # (T, 63)
        "jaw_pose":        np.asarray(poses_smplx["jaw_pose"]),
        "leye_pose":       np.asarray(poses_smplx["leye_pose"]),
        "reye_pose":       np.asarray(poses_smplx["reye_pose"]),
        "left_hand_pose":  np.asarray(poses_smplx["left_hand_pose"]),
        "right_hand_pose": np.asarray(poses_smplx["right_hand_pose"]),
        "transl":          np.asarray(transl),
    }
    np.savez(args.output, **out)
    print(f"wrote {args.output}  ({T} frames, SMPL-X format)")


def _cmd_smpl_to_smplx(args):
    """Copy SMPL's first 21 body joints into SMPL-X's ``body_pose`` block.

    SMPL has 24 joints; SMPL-X reuses the first 22 (root + 21 body), then adds
    finger / face / eye chains. So the body part transfers verbatim: SMPL's
    ``poses[0:3]`` becomes SMPL-X's ``global_orient`` and ``poses[3:66]``
    becomes ``body_pose``. Hand and face chains are filled with zeros — the
    source SMPL clip simply has no information about them.

    Useful for upgrading an SMPL motion clip to SMPL-X so the SMPL-X column in
    ``demo_soma_vis.py`` can be driven from it.
    """
    src = dict(np.load(args.smpl_motion, allow_pickle=False))
    poses = src["poses"]                                          # (T, 72) or (T, 24, 3)
    if poses.ndim == 3:
        poses = poses.reshape(poses.shape[0], -1)
    T = poses.shape[0]
    out = {
        "global_orient":   poses[:, :3].astype(np.float32),
        "body_pose":       poses[:, 3 : 3 + 63].astype(np.float32),
        "jaw_pose":        np.zeros((T, 3), np.float32),
        "leye_pose":       np.zeros((T, 3), np.float32),
        "reye_pose":       np.zeros((T, 3), np.float32),
        "left_hand_pose":  np.zeros((T, 45), np.float32),
        "right_hand_pose": np.zeros((T, 45), np.float32),
        "transl":          src.get("trans", np.zeros((T, 3), np.float32)).astype(np.float32),
    }
    np.savez(args.output, **out)
    print(f"wrote {args.output}  ({T} frames, SMPL-X format)")


def main():
    """CLI entry point: parses ``smplx-to-soma`` / ``soma-to-smplx`` /
    ``smpl-to-smplx`` subcommands and dispatches to the matching ``_cmd_*``
    function."""
    p = argparse.ArgumentParser(description="SOMA <-> SMPL/SMPL-X pose conversion")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("smplx-to-soma", help="SMPL-X motion -> SOMA poses")
    s1.add_argument("--smplx-motion", required=True)
    s1.add_argument("--smplx-model", required=True)
    s1.add_argument("--soma-model", required=True)
    s1.add_argument("--hf-dir", default=str(REPO / "assets" / "hf"))
    s1.add_argument("--refine-iters", type=int, default=0)
    s1.add_argument("--num-frames", type=int, default=None)
    s1.add_argument("--output", required=True)
    s1.set_defaults(func=_cmd_smplx_to_soma)

    s2 = sub.add_parser("soma-to-smplx", help="SOMA poses -> SMPL-X poses")
    s2.add_argument("--soma-motion", required=True)
    s2.add_argument("--smplx-model", required=True)
    s2.add_argument("--soma-model", required=True)
    s2.add_argument("--output", required=True)
    s2.set_defaults(func=_cmd_soma_to_smplx)

    s3 = sub.add_parser("smpl-to-smplx", help="SMPL motion -> SMPL-X format")
    s3.add_argument("--smpl-motion", required=True)
    s3.add_argument("--output", required=True)
    s3.set_defaults(func=_cmd_smpl_to_smplx)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
