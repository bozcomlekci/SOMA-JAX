"""SOMA pyrender demo: every identity backend posed with the same motion.

Upstream: ``tools/demo_soma_vis.py`` (SOMA-X v0.3.3). Builds one SOMA layer per
identity backend (``soma``, ``mhr``, ``anny``, ``smpl``, ``smplx``, ``garment``)
— optionally with and without the procedural twist rig — poses them all with
``example_animation.npy`` (or a spinning T-pose) and renders one video per
model, with optional random shapes, animated SOMA bone scales, LOD choice,
correctives and a skeleton overlay.

JAX port: SOMA-JAX layers return the prepared identity instead of caching it,
so each pose batch runs the layer's forward with the same identity arguments
and ``repose_to_bind_pose`` upstream's ``prepare_identity`` call uses.
``--device`` is accepted and ignored and ``--mode`` has no effect (sparse
top-8 LBS stands in for Warp); ``--seed`` (SOMA-JAX extra) seeds
``--random-shape``, which upstream draws unseeded. The ``tools/pipeline/
demo_soma_vis.py`` pack-based demo is a separate SOMA-JAX tool.

Usage::

    python tools/demo_soma_vis.py --identity-model-type soma,smpl --max-frames 60
    python tools/demo_soma_vis.py --procedural-transforms both --skeleton-overlay
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Joint Names & Mapping
# --------------------------------------------------------------------------------
# fmt: off
nvskel93_name = [
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

nvskel77_name = [
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
nvskel93to77_idx = [nvskel93_name.index(name) for name in nvskel77_name]

color_map = {
    "soma": (0.4, 0.8, 0.4, 1.0),  # light green
    "soma_procedural": (0.2, 0.55, 1.0, 1.0),
    "soma_no_procedural": (0.4, 0.8, 0.4, 1.0),
    "mhr": (0.98, 0.65, 0.15, 1.0),  # blue
    "mhr_procedural": (1.0, 0.45, 0.1, 1.0),
    "mhr_no_procedural": (0.98, 0.65, 0.15, 1.0),
    "anny": (0.25, 0.75, 1.0, 1.0),  # yellow
    "anny_procedural": (0.05, 0.45, 1.0, 1.0),
    "anny_no_procedural": (0.25, 0.75, 1.0, 1.0),
    "smpl": (0.55, 0.15, 0.85, 1.0),  # pink
    "smpl_procedural": (0.75, 0.25, 1.0, 1.0),
    "smpl_no_procedural": (0.55, 0.15, 0.85, 1.0),
    "smplx": (0.55, 0.15, 0.85, 1.0),  # pink
    "smplx_procedural": (0.75, 0.25, 1.0, 1.0),
    "smplx_no_procedural": (0.55, 0.15, 0.85, 1.0),
    "garment": (0.15, 0.15, 1.0, 1.0),  # orange
    "garment_procedural": (0.35, 0.35, 1.0, 1.0),
    "garment_no_procedural": (0.15, 0.15, 1.0, 1.0),
}


def get_smooth_noise(T, dim, rng=None, num_keyframes=None, mode="normal"):
    """(T, dim) noise linearly interpolated between random keyframes
    (``F.interpolate(mode="linear", align_corners=True)``)."""
    rng = np.random.default_rng() if rng is None else rng
    if num_keyframes is None:
        num_keyframes = max(3, T // 30)
    if mode == "normal":
        keyframes = rng.standard_normal((dim, num_keyframes))
    elif mode == "uniform":
        keyframes = rng.random((dim, num_keyframes))
    else:
        raise ValueError(f"Unknown noise mode {mode!r}")
    x = np.linspace(0.0, num_keyframes - 1, T)
    xp = np.arange(num_keyframes)
    return np.stack([np.interp(x, xp, k) for k in keyframes], axis=1).astype(np.float32)


def get_soma_bone_scale_demo(T, model, amplitude=0.35):
    """Animated SOMA limb/finger scale params for visual validation."""
    scales = np.ones((T, model.num_scale_params), np.float32)
    phase = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    limb_scale = 1.0 + amplitude * np.sin(phase)
    finger_scale = 1.0 + amplitude * np.sin(phase + np.pi)
    finger_prefixes = model.FINGER_BONE_SCALE_JOINT_PREFIXES
    for scale_idx, name in enumerate(model.scale_param_names):
        value = (finger_scale if any(name.startswith(prefix) for prefix in finger_prefixes)
                 else limb_scale)
        scales[:, scale_idx] = value
    return scales


def save_video(frames, path, fps=30):
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps)
    logger.info(f"Saved {path}")


def _rows(value, start, end):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: v[start:end] for k, v in value.items()}
    return value[start:end]


def main():
    from vis_pyrender import default_pyopengl_platform

    parser = argparse.ArgumentParser(description="SOMA pyrender demo")
    parser.add_argument("--data-root", default=None,
                        help="Path to SOMA assets (default: soma_jax.assets)")
    parser.add_argument("--motion-file", default=None,
                        help="Path to motion file (.npy); default <data-root>/example_animation.npy. "
                             "If missing, uses a dummy motion.")
    parser.add_argument("--device", default="cuda:0",
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--output-dir", default="out/vis_identity_model")
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--pyopengl-platform", default=default_pyopengl_platform())
    parser.add_argument("--video-extension", choices=["mp4", "gif"], default="mp4",
                        help="Rendered animation format. Use gif when MP4/ffmpeg is unavailable.")
    parser.add_argument("--mode", choices=["warp", "dense"], default="warp",
                        help="Skinning backend (accepted for upstream CLI compatibility).")
    parser.add_argument("--random-shape", action="store_true", default=False)
    parser.add_argument("--soma-bone-scale-demo", action="store_true", default=False,
                        help="Animate SOMA limb/finger scale_params for visual bone-scale "
                             "validation.")
    parser.add_argument("--soma-bone-scale-amplitude", type=float, default=0.35,
                        help="Sinusoidal scale amplitude used by --soma-bone-scale-demo.")
    parser.add_argument("--identity-model-type", default="soma,mhr,anny,smpl,smplx,garment",
                        help="Comma-separated list of identity models to use. Options: soma, mhr, "
                             "anny, smpl, smplx garment (default: soma,mhr,anny,smpl,smplx,garment)")
    parser.add_argument("--pose-batch-size", type=int, default=0,
                        help="Run forward pass in batches of this many poses to reduce memory. "
                             "0 = process all frames at once (default). Try 32 or 64 if OOM.")
    parser.add_argument("--low-lod", action="store_true", default=False,
                        help="Use low level-of-detail mesh (deprecated alias for --lod low)")
    parser.add_argument("--lod", choices=["mid", "low", "xlo"], default=None,
                        help="Body mesh LOD to render. Defaults to mid, or low when --low-lod "
                             "is set.")
    parser.add_argument("--apply-correctives", action="store_true", default=False,
                        help="Apply pose corrective offsets (default: False)")
    parser.add_argument("--procedural-transforms", choices=["off", "on", "both"], default="off",
                        help="Enable SOMA procedural twist-joint rig evaluation. 'both' renders "
                             "paired videos with and without procedural joints.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit the number of rendered motion frames. 0 = render all frames.")
    parser.add_argument("--skeleton-overlay", action="store_true", default=False,
                        help="Render the public SOMA skeleton (octahedral bones) inside the mesh.")
    parser.add_argument("--mesh-alpha", type=float, default=None,
                        help="Mesh opacity in [0, 1] (default: 1.0). With --skeleton-overlay the "
                             "skeleton is composited over the mesh, so translucency is optional.")
    parser.add_argument("--skeleton-style", choices=["light", "skin"], default="light",
                        help="Skeleton color style for --skeleton-overlay: 'light' = neutral "
                             "light gray, 'skin' = darker tone (0.65x) of the mesh color.")
    parser.add_argument("--gender", default="neutral",
                        help="Gender of the model (default: neutral). Only used for smpl and "
                             "smplx models.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for --random-shape (SOMA-JAX extra; upstream is unseeded).")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    identity_models = [m.strip().lower() for m in args.identity_model_type.split(",")]
    valid_models = {"soma", "mhr", "anny", "smpl", "smplx", "garment"}
    invalid_models = set(identity_models) - valid_models
    if invalid_models:
        raise ValueError(
            f"Invalid identity model type(s): {invalid_models}. Valid options: {valid_models}")
    if args.soma_bone_scale_demo and "soma" not in identity_models:
        raise ValueError("--soma-bone-scale-demo requires identity-model-type to include soma")
    args.identity_models = identity_models
    if args.mesh_alpha is None:
        args.mesh_alpha = 1.0
    if args.lod is None:
        args.lod = "low" if args.low_lod else "mid"
    elif args.low_lod and args.lod != "low":
        raise ValueError("--low-lod is only compatible with --lod low")
    args.low_lod = args.lod == "low"

    import imageio.v2 as imageio
    import jax.numpy as jnp
    from tqdm import tqdm

    from soma_jax import SOMALayer
    from soma_jax.assets import data_root as default_data_root
    from soma_jax.geometry.rig_utils import joint_local_to_world, joint_world_to_local
    from soma_jax.types import SOMAParams
    from vis_pyrender import MeshRenderer, look_at, set_pyopengl_platform

    set_pyopengl_platform(args.pyopengl_platform)
    os.makedirs(args.output_dir, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    motion_file = (args.motion_file if args.motion_file is not None
                   else str(data_root / "example_animation.npy"))
    rng = np.random.default_rng(args.seed)

    procedural_variants = ([False, True] if args.procedural_transforms == "both"
                           else [args.procedural_transforms == "on"])
    model_specs = []
    for identity_model_type in args.identity_models:
        for enable_procedural_transforms in procedural_variants:
            if args.procedural_transforms == "both":
                proc_label = "procedural" if enable_procedural_transforms else "no_procedural"
                model_key = f"{identity_model_type}_{proc_label}"
            else:
                model_key = identity_model_type
            model_specs.append((model_key, identity_model_type, enable_procedural_transforms))

    logger.info(f"Initializing models: {', '.join(key for key, _, _ in model_specs)}...")
    models, model_identity_types, model_procedural = {}, {}, {}
    for model_key, identity_model_type, enable_procedural_transforms in model_specs:
        identity_model_kwargs = {"gender": args.gender} if identity_model_type == "smpl" else {}
        models[model_key] = SOMALayer.from_upstream_assets(
            identity_model_type=identity_model_type, lod=args.lod,
            procedural=enable_procedural_transforms,
            identity_model_kwargs=identity_model_kwargs, data_root=str(data_root))
        model_identity_types[model_key] = identity_model_type
        model_procedural[model_key] = enable_procedural_transforms

    reference_model = models[model_specs[0][0]]

    if motion_file and os.path.exists(motion_file):
        logger.info(f"Loading motion from {motion_file}...")
        motion_full = np.load(motion_file).astype(np.float32)
        joint_rot_mats_local = motion_full[..., :3, :3]
        root_trans = motion_full[..., 1, :3, 3]
    else:
        logger.info("No motion file provided or file not found. Using dummy motion "
                    "(T-pose rotation).")
        T = 30
        joint_rot_mats_local = np.tile(np.eye(3, dtype=np.float32), (T, 78, 1, 1))
        angle = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        zeros, ones = np.zeros_like(angle), np.ones_like(angle)
        rot_y = np.stack([np.stack([cos, zeros, sin], axis=-1),
                          np.stack([zeros, ones, zeros], axis=-1),
                          np.stack([-sin, zeros, cos], axis=-1)], axis=-2)   # (T, 3, 3)
        joint_rot_mats_local[:, 1] = rot_y  # Rotate Hips
        root_trans = np.zeros((T, 3), np.float32)

    if joint_rot_mats_local.shape[1] == 94:
        subset_idx = [0] + [i + 1 for i in nvskel93to77_idx]
        joint_rot_mats_local = joint_rot_mats_local[:, subset_idx]
    if args.max_frames > 0:
        joint_rot_mats_local = joint_rot_mats_local[: args.max_frames]
        root_trans = root_trans[: args.max_frames]

    # Absolute local rotations -> relative to the public T-pose orient.
    reference_transform_joint_indices = getattr(
        reference_model, "public_transform_joint_indices",
        np.arange(np.asarray(reference_model.t_pose_world).shape[0]))
    reference_parent_ids = np.asarray(getattr(
        reference_model, "public_joint_parent_ids", reference_model.joint_parent_ids))
    reference_t_pose_world = np.asarray(reference_model.t_pose_world)[
        np.asarray(reference_transform_joint_indices)]
    correction = np.swapaxes(reference_t_pose_world[:, :3, :3], -2, -1)
    joint_rot_mats_world = joint_local_to_world(jnp.asarray(joint_rot_mats_local),
                                                reference_parent_ids)
    joint_rot_mats_world = joint_rot_mats_world @ jnp.asarray(correction)
    joint_rot_mats_local = joint_world_to_local(joint_rot_mats_world, reference_parent_ids)

    T = joint_rot_mats_local.shape[0]
    pose = joint_rot_mats_local[:T, 1:]  # (T, 77, 3, 3): Hips (global orient) + body

    # Identity parameters.
    identity_coeffs_map = {}
    for model_type, model in models.items():
        identity_model_type = model_identity_types[model_type]
        n = model.identity_model.num_identity_coeffs
        if identity_model_type == "anny":
            anny_im = model.identity_model.identity_model
            if args.random_shape:
                phenotypes = {k: get_smooth_noise(T, 1, rng, mode="uniform")[:, 0]
                              for k in anny_im.phenotype_labels}
            else:
                phenotypes = {k: np.full((T,), 0.5, np.float32) for k in anny_im.phenotype_labels}
            local_changes = {k: np.zeros((T,), np.float32) for k in anny_im.local_change_labels}
            identity_coeffs_map[model_type] = (phenotypes, local_changes)
        elif identity_model_type == "mhr":
            n_scale = model.identity_model.num_scale_params
            if args.random_shape:
                coeffs = get_smooth_noise(T, n, rng)
                scale = get_smooth_noise(T, n_scale, rng, mode="normal") * 0.2
            else:
                coeffs = np.zeros((T, n), np.float32)
                scale = np.zeros((T, n_scale), np.float32)
            identity_coeffs_map[model_type] = (coeffs, scale)
        else:
            coeffs = (get_smooth_noise(T, n, rng) if args.random_shape
                      else np.zeros((T, n), np.float32))
            scale = (get_soma_bone_scale_demo(T, model, amplitude=args.soma_bone_scale_amplitude)
                     if identity_model_type == "soma" and args.soma_bone_scale_demo else None)
            identity_coeffs_map[model_type] = (coeffs, scale)

    transl = jnp.asarray(root_trans[:T])

    # Forward pass: upstream's prepare_identity() + pose(). With a constant
    # identity the first row is prepared once; per-frame identities are
    # prepared per batch.
    pose_batch_size = args.pose_batch_size if args.pose_batch_size > 0 else T
    logger.info(f"Running forward pass (pose_batch_size={pose_batch_size})...")
    per_frame_identity = args.random_shape or args.soma_bone_scale_demo

    outputs = {}
    for start in range(0, T, pose_batch_size):
        end = min(start + pose_batch_size, T)
        for model_type, model in models.items():
            coeffs, scale = identity_coeffs_map[model_type]
            if per_frame_identity:
                coeffs_b, scale_b = _rows(coeffs, start, end), _rows(scale, start, end)
            else:
                coeffs_b, scale_b = _rows(coeffs, 0, 1), _rows(scale, 0, 1)
            if not isinstance(coeffs_b, dict):
                coeffs_b = jnp.asarray(coeffs_b)
                scale_b = None if scale_b is None else jnp.asarray(scale_b)
            out_b = model(
                SOMAParams(poses=pose[start:end], transl=transl[start:end],
                           identity_coeffs=coeffs_b, scale_params=scale_b),
                apply_correctives=args.apply_correctives,
                repose_to_bind_pose=args.apply_correctives or model_procedural[model_type])
            if model_type not in outputs:
                outputs[model_type] = {"vertices": [], "joints": []}
                if args.skeleton_overlay:
                    outputs[model_type]["transforms"] = []
            outputs[model_type]["vertices"].append(np.asarray(out_b.vertices))
            outputs[model_type]["joints"].append(np.asarray(out_b.joints))
            if args.skeleton_overlay:
                outputs[model_type]["transforms"].append(np.asarray(out_b.transforms))
    for model_type in list(outputs.keys()):
        for key in outputs[model_type]:
            outputs[model_type][key] = np.concatenate(outputs[model_type][key], axis=0)

    # Render (model-first loop with a streaming video writer).
    logger.info("Rendering videos...")
    shape_suffix = "rand_shape" if args.random_shape else "fixed_shape"
    suffix = shape_suffix if args.lod == "mid" else f"{args.lod}_{shape_suffix}"
    if args.soma_bone_scale_demo:
        suffix = f"{suffix}_bone_scale"
    if args.procedural_transforms == "on":
        suffix = f"{suffix}_procedural"
    if args.skeleton_overlay:
        suffix = f"{suffix}_skel"
    faces = {model_type: np.asarray(models[model_type].faces) for model_type in models}
    cam_pose = look_at(eye=np.array([0.0, 1.0, 6.0]), target=np.array([0.0, 1.0, 0.0]),
                       up=np.array([0.0, 1.0, 0.0]))
    light_dir = np.array([0.0, -0.5, -1.0])
    renderer = MeshRenderer(image_size=args.image_size, light_intensity=5)

    for model_type in models:
        out_path = f"{args.output_dir}/{model_type}_{suffix}.{args.video_extension}"
        renderer.setup_mesh(
            faces=faces[model_type], mesh_color=(*color_map[model_type][:3], args.mesh_alpha),
            cam_pose=cam_pose, light_dir=light_dir, metallic=0.0, roughness=0.5,
            base_color_factor=[0.9, 0.9, 0.9, 1.0])
        if args.skeleton_overlay:
            parents = np.asarray(models[model_type].output_joint_parent_ids)
            # Drop the virtual Root (index 0) so no bone spikes to the origin;
            # joints parented to Root become skeleton roots (-1).
            skel_parents = parents[1:] - 1
            if args.skeleton_style == "skin":
                skel_color = tuple(0.65 * c for c in color_map[model_type][:3])
                renderer.setup_skeleton(skel_parents, color=skel_color)
            else:
                renderer.setup_skeleton(skel_parents)
        writer = imageio.get_writer(out_path, fps=30)
        for t in tqdm(range(T), desc=model_type):
            verts = outputs[model_type]["vertices"][t]
            joints = None
            if args.skeleton_overlay:
                joints = outputs[model_type]["transforms"][t, 1:, :3, 3]
            img = renderer.render_frame(verts, joints=joints)
            writer.append_data(img[..., ::-1])
        writer.close()
        logger.info(f"Saved {out_path}")
    renderer.delete()


if __name__ == "__main__":
    main()
