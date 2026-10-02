"""Hand-only animation demo for SOMAHandLayer.

Runs FK on the full-body motion to extract the wrist world transform, then
feeds the 24 finger-joint rotations to SOMAHandLayer, and applies the wrist
world transform to bring the wrist-local hand mesh into world space for
rendering. Pass ``--remove-wrist-translation`` to retain the animated wrist
orientation while keeping the hand centered at the world origin.

Upstream: ``tools/hand/demo_soma_hand_vis.py`` (SOMA-X v0.3.0), vendored
(Apache-2.0, NVIDIA) onto :class:`soma_jax.hand.SOMAHandLayer`. Differences:
FK and the forward run in JAX; ``get_smooth_noise`` draws its keyframes from
NumPy's generator (seeded by ``--seed``) instead of torch's, so a
``--random-shape`` clip is a different draw of the same process.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
for _p in (repo_root, repo_root / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from logging_utils import add_logging_args, configure_logging  # noqa: E402
from soma_rig_assets import load_public_mid_soma_rig  # noqa: E402
from vis_pyrender import (  # noqa: E402
    MeshRenderer,
    default_pyopengl_platform,
    look_at,
    set_pyopengl_platform,
)

logger = logging.getLogger(__name__)

# fmt: off
# Joint name tables for subsetting 94-joint nvskel93 motions to the 78-joint SOMA skeleton
_NVSKEL93_NAME = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "LeftForeArmTwist1", "LeftForeArmTwist2", "LeftArmTwist1", "LeftArmTwist2",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "RightForeArmTwist1", "RightForeArmTwist2", "RightArmTwist1", "RightArmTwist2",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "LeftShinTwist1", "LeftShinTwist2", "LeftLegTwist1", "LeftLegTwist2",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    "RightShinTwist1", "RightShinTwist2", "RightLegTwist1", "RightLegTwist2",
]
_NVSKEL77_NAME = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]
# fmt: on
_NVSKEL93TO77_IDX = [_NVSKEL93_NAME.index(name) for name in _NVSKEL77_NAME]

_BACKEND_COLORS = {
    "soma": (0.4, 0.8, 0.4, 1.0),
    "mhr": (0.98, 0.65, 0.15, 1.0),
    "mano": (0.55, 0.15, 0.85, 1.0),
}


def get_smooth_noise(T, dim, rng, num_keyframes=None):
    """Generate smoothly-interpolated random noise of shape (T, dim).

    Random keyframes are drawn from a unit-normal distribution and linearly
    interpolated to T frames (``align_corners=True``), producing temporally
    coherent variation.
    """
    if num_keyframes is None:
        num_keyframes = max(3, T // 30)
    keyframes = rng.standard_normal((dim, num_keyframes))
    src = np.linspace(0.0, 1.0, num_keyframes)
    dst = np.linspace(0.0, 1.0, T)
    return np.stack([np.interp(dst, src, k) for k in keyframes], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="SOMA hand-only animation demo")
    parser.add_argument("--data-root", default=None,
                        help="Path to SOMA assets (default: soma_jax.assets.data_root()).")
    parser.add_argument(
        "--motion-file",
        default=None,
        help="Path to motion file (.npy). If not found, uses T-pose. "
        "Default: the example_animation.npy that ships with the assets.",
    )
    parser.add_argument("--device", default=None,
                        help="Accepted for upstream CLI compatibility; JAX uses its default device.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for --random-shape / --shape-only noise (SOMA-JAX extra).")
    parser.add_argument("--output-dir", default="out/demo_hand")
    parser.add_argument(
        "--hand-type",
        default="left,right",
        help="Comma-separated list: 'left', 'right', or 'left,right'",
    )
    parser.add_argument("--image-size", type=int, default=2000)
    parser.add_argument(
        "--camera-framing-scale",
        type=float,
        default=1.4,
        help="Camera padding around the motion-wide hand bounds; larger values zoom out.",
    )
    parser.add_argument("--pyopengl-platform", default=default_pyopengl_platform())
    parser.add_argument(
        "--random-shape",
        action="store_true",
        default=False,
        help="Smoothly animate random PCA shape coefficients (default: neutral shape)",
    )
    parser.add_argument(
        "--identity-model-type",
        choices=("soma", "mano", "mhr"),
        default="soma",
        help="Identity backend to use for SOMAHandLayer (default: soma).",
    )
    parser.add_argument(
        "--mano-model-path",
        type=Path,
        default=None,
        help="Path to a user-supplied MANO model file when using the MANO backend.",
    )
    parser.add_argument(
        "--apply-correctives",
        action="store_true",
        default=False,
        help="Apply pose-dependent corrective offsets during the hand-layer forward pass.",
    )
    parser.add_argument(
        "--shape-only",
        action="store_true",
        default=False,
        help="Fixed T-pose, wrist at origin, animate shape only. Implies --random-shape. "
        "Ignores --motion-file.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=90,
        help="Number of frames to render in --shape-only mode (default: 90 = 3 s at 30 fps)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of motion frames rendered. Applies after motion loading/subsetting.",
    )
    parser.add_argument(
        "--remove-wrist-translation",
        action="store_true",
        help="Keep the animated wrist orientation but render with its world translation removed.",
    )
    parser.add_argument(
        "--skeleton-overlay",
        action="store_true",
        default=False,
        help="Render the 25-joint hand skeleton (octahedral bones) inside the mesh.",
    )
    parser.add_argument(
        "--mesh-alpha",
        type=float,
        default=None,
        help="Mesh opacity in [0, 1] (default: 1.0). With --skeleton-overlay the "
        "skeleton is composited over the mesh, so translucency is optional.",
    )
    parser.add_argument(
        "--skeleton-style",
        choices=["light", "skin"],
        default="light",
        help="Skeleton color style for --skeleton-overlay: 'light' = neutral "
        "light gray, 'skin' = darker tone (0.65x) of the mesh color.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    if args.mesh_alpha is None:
        args.mesh_alpha = 1.0

    set_pyopengl_platform(args.pyopengl_platform)
    import jax.numpy as jnp

    from soma_jax.assets import data_root as default_data_root
    from soma_jax.assets import resolve
    from soma_jax.geometry.rig_utils import joint_local_to_world
    from soma_jax.hand import SOMAHandLayer

    if args.data_root is None:
        args.data_root = str(default_data_root())
    if args.motion_file is None:
        args.motion_file = str(resolve("example_animation.npy", required=False) or "")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    hand_types = [h.strip().lower() for h in args.hand_type.split(",")]
    for ht in hand_types:
        if ht not in ("left", "right"):
            raise ValueError(f"Invalid hand type '{ht}'. Use 'left' or 'right'.")
    if args.mano_model_path is not None and args.identity_model_type != "mano":
        raise ValueError("--mano-model-path requires --identity-model-type mano")
    if args.mano_model_path is not None and len(hand_types) != 1:
        raise ValueError("--mano-model-path requires a single --hand-type")
    if args.camera_framing_scale <= 0:
        raise ValueError("--camera-framing-scale must be positive")

    data_root = Path(args.data_root)

    # ------------------------------------------------------------------ #
    # FK setup + motion loading (skipped in shape-only mode)
    # ------------------------------------------------------------------ #
    if not args.shape_only:
        rig_data = load_public_mid_soma_rig(data_root)
        full_parent_ids = np.asarray(rig_data["joint_parent_ids"])  # (78,)

        motion_path = Path(args.motion_file)
        if args.motion_file and motion_path.exists():
            logger.info(f"Loading motion from {motion_path}...")
            motion_full = np.load(motion_path).astype(np.float32)
        else:
            logger.info("Motion file not found. Using T-pose (identity rotations).")
            motion_full = np.broadcast_to(np.eye(4, dtype=np.float32), (30, 78, 4, 4)).copy()

        if motion_full.shape[1] == 94:
            subset_idx = [0] + [i + 1 for i in _NVSKEL93TO77_IDX]
            motion_full = motion_full[:, subset_idx]
        if args.max_frames is not None:
            motion_full = motion_full[: args.max_frames]

        T = motion_full.shape[0]
        joint_rot_mats = motion_full[..., :3, :3]  # (T, 78, 3, 3)
        world_4x4 = np.asarray(joint_local_to_world(jnp.asarray(motion_full),
                                                    full_parent_ids))  # (T, 78, 4, 4)

    # ------------------------------------------------------------------ #
    # Process each hand type
    # ------------------------------------------------------------------ #
    for hand_type in hand_types:
        logger.info(f"\n--- Processing {hand_type} hand ---")

        logger.info(f"Creating SOMAHandLayer ({hand_type})...")
        identity_model_kwargs = (
            {"model_path": args.mano_model_path} if args.mano_model_path is not None else None
        )
        hand_layer = SOMAHandLayer(
            data_root=str(data_root),
            hand_type=hand_type,
            identity_model_type=args.identity_model_type,
            identity_model_kwargs=identity_model_kwargs,
        )

        if args.shape_only:
            # Fixed T-pose with wrist at world origin; only shape varies.
            T_cur = (
                args.num_frames
                if args.max_frames is None
                else min(args.num_frames, args.max_frames)
            )
            finger_poses = np.broadcast_to(np.eye(3, dtype=np.float32), (T_cur, 25, 3, 3)).copy()
            wrist_world = np.broadcast_to(np.eye(4, dtype=np.float32), (T_cur, 4, 4)).copy()
        else:
            T_cur = T
            # Wrist world transform from FK
            wrist_world = world_4x4[:, hand_layer.wrist_global_id, :, :]  # (T, 4, 4)
            # Compute T-pose-relative finger poses for SOMAHandLayer (joint_orient = T-pose).
            finger_global_ids = np.asarray(hand_layer.hand_joint_ids_global)
            R_motion = joint_rot_mats[:, finger_global_ids]  # (T, 25, 3, 3)
            hand_t_pose_rots = np.asarray(hand_layer.t_pose_world)[:, :3, :3]  # (25, 3, 3)
            parent_local_ids = np.asarray(hand_layer.joint_parent_ids)
            T_parent = hand_t_pose_rots[parent_local_ids]
            T_self = hand_t_pose_rots
            finger_poses = T_parent[None] @ R_motion @ np.swapaxes(T_self[None], -2, -1)

        # Identity coefficients — smooth random or neutral
        if args.random_shape or args.shape_only:
            identity_coeffs = get_smooth_noise(T_cur, hand_layer.num_shape_components, rng)
        else:
            identity_coeffs = np.zeros((T_cur, hand_layer.num_shape_components), np.float32)

        # ------------------------------------------------------------------ #
        # Forward pass (all frames at once)
        # ------------------------------------------------------------------ #
        logger.info("Running forward pass...")
        out = hand_layer(
            jnp.asarray(finger_poses, jnp.float32),
            jnp.asarray(identity_coeffs),
            pose2rot=False,
            apply_correctives=args.apply_correctives,
        )
        out = {k: np.asarray(v) for k, v in out.items()}

        # out["vertices"]: (T_cur, Vh, 3) in wrist-local meters
        Vh = out["vertices"].shape[1]

        # ------------------------------------------------------------------ #
        # Apply wrist world transform for rendering
        # wrist_world is in cm (from FK on SOMA rig); hand verts are in meters.
        # Scale wrist translation to meters before applying.
        # ------------------------------------------------------------------ #
        wrist_world_m = np.array(wrist_world, np.float32)
        if args.remove_wrist_translation:
            wrist_world_m[:, :3, 3] = 0
        else:
            wrist_world_m[:, :3, 3] *= 0.01  # cm -> m
        ones = np.ones((T_cur, Vh, 1), np.float32)
        verts_h = np.concatenate([out["vertices"], ones], axis=-1)
        verts_world = (wrist_world_m[:, None] @ verts_h[..., None])[..., 0][..., :3]

        joints_world = None
        if args.skeleton_overlay:
            # out["joints"]: (T_cur, 25, 3) wrist-local meters -> world, like the vertices.
            joints_h = np.concatenate([out["joints"], np.ones((T_cur, 25, 1), np.float32)], axis=-1)
            joints_world = (wrist_world_m[:, None] @ joints_h[..., None])[..., 0][..., :3]

        if args.shape_only:
            # Remove per-frame centroid drift so only shape changes, not position.
            centroid = verts_world.mean(axis=1, keepdims=True)
            verts_world = verts_world - centroid
            if joints_world is not None:
                joints_world = joints_world - centroid

        # ------------------------------------------------------------------ #
        # Render
        # ------------------------------------------------------------------ #
        faces_np = np.asarray(hand_layer.faces)
        if args.shape_only:
            # Vertices are centroid-centered; camera looks at origin.
            all_verts_np = verts_world.reshape(-1, 3)
            bbox_extent = (all_verts_np.max(axis=0) - all_verts_np.min(axis=0)).max()
            cam_dist = bbox_extent * 2.0
            cam_pose = look_at(
                eye=np.array([0.0, 0.0, cam_dist]),
                target=np.array([0.0, 0.0, 0.0]),
                up=np.array([0.0, 1.0, 0.0]),
            )
            light_dir = np.array([0.0, -0.3, -1.0])
        else:
            # Camera from bounding box of all verts across all frames (meters)
            all_verts_np = verts_world.reshape(-1, 3)
            bbox_center = (all_verts_np.min(axis=0) + all_verts_np.max(axis=0)) / 2
            bbox_extent = (all_verts_np.max(axis=0) - all_verts_np.min(axis=0)).max()
            cam_dist = bbox_extent * 1.5
            cam_pose = look_at(
                eye=bbox_center + np.array([0.0, 0.0, cam_dist]),
                target=bbox_center,
                up=np.array([0.0, 1.0, 0.0]),
            )
            light_dir = np.array([0.0, -0.5, -1.0])
        # Focal length: project the motion-wide bounds with configurable padding.
        fl = args.image_size * cam_dist / (bbox_extent * args.camera_framing_scale)
        renderer = MeshRenderer(image_size=args.image_size, light_intensity=5, focal_length=fl)
        renderer.setup_mesh(
            faces=faces_np,
            mesh_color=(*_BACKEND_COLORS[args.identity_model_type][:3], args.mesh_alpha),
            cam_pose=cam_pose,
            light_dir=light_dir,
            metallic=0.0,
            roughness=0.5,
            base_color_factor=[0.9, 0.9, 0.9, 1.0],
        )
        if args.skeleton_overlay:
            hand_parents = np.asarray(hand_layer.joint_parent_ids)
            if args.skeleton_style == "skin":
                skel_color = tuple(0.65 * c for c in _BACKEND_COLORS[args.identity_model_type][:3])
                renderer.setup_skeleton(hand_parents, color=skel_color)
            else:
                renderer.setup_skeleton(hand_parents)

        if args.shape_only:
            suffix = "shape_only"
        elif args.random_shape:
            suffix = "rand_shape"
        else:
            suffix = "fixed_shape"
        if args.skeleton_overlay:
            suffix = f"{suffix}_skel"
        corrective_suffix = "correctives" if args.apply_correctives else "no_correctives"
        out_path = str(
            Path(args.output_dir)
            / f"hand_{hand_type}_{args.identity_model_type}_{corrective_suffix}_{suffix}.mp4"
        )
        writer = imageio.get_writer(out_path, fps=30)
        for t in tqdm(range(T_cur), desc=f"Rendering {hand_type}"):
            verts = verts_world[t]
            joints = joints_world[t] if joints_world is not None else None
            img = renderer.render_frame(verts, joints=joints)
            writer.append_data(img[..., ::-1])
        writer.close()
        renderer.delete()
        logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
