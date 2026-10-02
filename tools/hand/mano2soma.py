"""MANO to SOMA hand pose converter.

Upstream: ``tools/hand/mano2soma.py`` (SOMA-X v0.3.0). Same pipeline, arguments
and outputs, over SOMA-JAX layers:

  1. Load MANO test data -> posed MANO verts (778v)
  2. Transfer to SOMAHand topology (2859v) via the MANO identity backend's
     barycentric correspondence
  3. PoseInversion.fit() -> recovered rotations + root translation
  4. Forward pass SOMAHandLayer -> reconstructed verts
  5. Numerical comparison + renders with skeleton overlays

Input ``.npz`` files carry, per hand (``lh`` / ``rh``): ``mano_{lh,rh}`` (a dict
with ``betas``), ``verts_{lh,rh}`` (T, 778, 3) and ``joints_{lh,rh}`` (T, 21, 3)
— upstream's MANO test-data layout (MANO itself is licensed separately).

Usage::

    python tools/hand/mano2soma.py --input path/to/mano_test_data
    python tools/hand/mano2soma.py --input seq_0.npz --hand-type left --no-render
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from conversion_utils import add_hand_inversion_args  # noqa: E402
from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def render_overlay_frame(renderer, mano_verts, mano_faces, mano_joints, mano_parent_ids,
                         soma_verts, soma_faces, soma_joints, soma_parent_ids, cam_pose,
                         light_dir, joint_radius=0.15, bone_radius=0.06):
    """Render 3-panel: MANO mesh+skel | SOMA-Hand mesh+skel | overlay+both skels."""
    from vis_pyrender import overlay_skeleton, render_mesh_panel
    skel = dict(cam_pose=cam_pose, light_dir=light_dir, joint_radius=joint_radius,
                bone_radius=bone_radius, metallic=0.3, roughness=0.3)
    img_mano = render_mesh_panel(renderer, mano_verts, mano_faces,
                                 mesh_color=(0.65, 0.65, 0.65, 1.0),
                                 cam_pose=cam_pose, light_dir=light_dir)
    panel1 = overlay_skeleton(renderer, img_mano, mano_joints, mano_parent_ids,
                              color=(0.75, 0.2, 0.15, 1.0), **skel)
    img_soma = render_mesh_panel(renderer, soma_verts, soma_faces,
                                 mesh_color=(0.3, 0.78, 0.2, 1.0),
                                 cam_pose=cam_pose, light_dir=light_dir)
    panel2 = overlay_skeleton(renderer, img_soma, soma_joints, soma_parent_ids,
                              color=(0.15, 0.55, 0.15, 1.0), **skel)
    panel3 = (0.5 * img_mano.astype(np.float32) + 0.5 * img_soma.astype(np.float32)).astype(
        np.uint8)
    panel3 = overlay_skeleton(renderer, panel3, mano_joints, mano_parent_ids,
                              color=(0.75, 0.2, 0.15, 1.0), **skel)
    panel3 = overlay_skeleton(renderer, panel3, soma_joints, soma_parent_ids,
                              color=(0.15, 0.55, 0.15, 1.0), **skel)
    return np.concatenate([panel1, panel2, panel3], axis=1)


def load_mano_data(path):
    """Load MANO test sequences from a .npz file or directory of .npz files."""
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {path}")
    else:
        raise ValueError(f"Expected .npz file or directory, got: {path}")
    return [{"path": f, "data": np.load(f, allow_pickle=True)} for f in files]


def _export_usd(out_path, hand_layer, identity, rotations, root_translation):
    """Export skeletal USD from SOMAHandLayer results (the identity is shared)."""
    from soma_jax.usd_io import export_soma_usd
    export_soma_usd(out_path, hand_layer, rotations, root_translation,
                    bind_transforms_world=identity.bind_transforms_world[0],
                    rest_shape=identity.rest_shape[0])


def main():
    parser = argparse.ArgumentParser(description="MANO to SOMA hand pose converter.")
    parser.add_argument("--input", required=True,
                        help="Path to MANO .npz file or directory of .npz files.")
    parser.add_argument("--output-dir", default="out/mano2soma",
                        help="Output directory for renders, USDs, and stats (default: out/mano2soma).")
    parser.add_argument("--hand-type", default="both", choices=["left", "right", "both"],
                        help="Which hand(s) to process (default: both).")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    add_hand_inversion_args(parser, bcd_iters=1, lie_iters=3, batch_size=32)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30).")
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF FPS (default: 15).")
    parser.add_argument("--pyopengl-platform", default=None)
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    import jax.numpy as jnp

    from soma_jax.assets import data_root as default_data_root
    from soma_jax.geometry.rig_utils import joint_world_to_local
    from soma_jax.hand import MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS, SOMAHandLayer
    from soma_jax.fitting.pose_inversion import PoseInversion

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    os.makedirs(args.output_dir, exist_ok=True)

    sequences = load_mano_data(args.input)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    logger.info(f"Loaded {len(sequences)} test sequence(s) from {args.input}")

    if not args.no_render:
        from vis_pyrender import default_pyopengl_platform, set_pyopengl_platform
        set_pyopengl_platform(args.pyopengl_platform or default_pyopengl_platform())

    hand_types = ["left", "right"] if args.hand_type == "both" else [args.hand_type]
    all_stats = []

    for hand_type in hand_types:
        mano_key = f"mano_{hand_type[0]}h"
        verts_key = f"verts_{hand_type[0]}h"
        joints_key = f"joints_{hand_type[0]}h"
        logger.info(f"\n{'=' * 60}\n  {hand_type.upper()} HAND\n{'=' * 60}")

        hand_layer = SOMAHandLayer(data_root=str(data_root), hand_type=hand_type,
                                   identity_model_type="mano")
        pose_inv = PoseInversion(hand_layer, low_lod=False)
        faces_np = np.asarray(hand_layer.faces).astype(np.int32)
        parent_ids = np.asarray(hand_layer.joint_parent_ids).tolist()

        for seq_idx, seq in enumerate(sequences):
            seq_name = seq["path"].stem[:40]
            d = seq["data"]
            mano_params = d[mano_key].item()
            gt_mano_verts = np.asarray(d[verts_key], np.float32)
            gt_joints = np.asarray(d[joints_key], np.float32)
            betas = np.asarray(mano_params["betas"], np.float32)
            if betas.ndim == 1:
                betas = betas[None]

            T = gt_mano_verts.shape[0]
            if args.max_frames and T > args.max_frames:
                T = args.max_frames
                gt_mano_verts, gt_joints, betas = gt_mano_verts[:T], gt_joints[:T], betas[:T]
            logger.info(f"\n--- Seq {seq_idx}: {seq_name} ({T} frames) ---")

            # Step 1: transfer posed MANO verts to SOMAHand topology.
            wrist_pos = gt_joints[:, 0:1, :]
            gt_mano_centered = gt_mano_verts - wrist_pos
            soma_target_verts = hand_layer.identity_model.identity_model_to_soma(
                jnp.asarray(gt_mano_centered))
            logger.info(f"  MANO verts: {gt_mano_verts.shape}")
            logger.info(f"  SOMA target: {tuple(soma_target_verts.shape)} "
                        f"({hand_layer.output_unit.unit_name})")

            # Step 2: prepare identity and run pose inversion.
            betas_single = jnp.asarray(betas[:1])
            out_prefix = f"{args.output_dir}/{hand_type}_{seq_idx}"
            pose_inv.prepare_identity(betas_single)
            identity = hand_layer.prepare_identity(betas_single)

            # Skeleton-transfer-only USD.
            skel_world = pose_inv._skel_transfer.fit(soma_target_verts)
            skel_local = joint_world_to_local(skel_world, np.asarray(hand_layer.joint_parent_ids))
            _export_usd(f"{out_prefix}_skeltransfer.usda", hand_layer, identity,
                        skel_local[:, :, :3, :3], skel_local[:, 0, :3, 3])
            logger.info(f"  Skeleton transfer USD: {out_prefix}_skeltransfer.usda")

            # BCD + Lie-GN.
            t0 = time.perf_counter()
            result = pose_inv.fit(soma_target_verts, body_iters=0, finger_iters=0,
                                  full_iters=args.bcd_iters, lie_iters=args.lie_iters,
                                  lie_lambda=args.lie_lambda, batch_size=args.batch_size)
            result["rotations"].block_until_ready()
            dt = time.perf_counter() - t0
            lie_err = float(np.asarray(result["per_vertex_error"]).mean()) * 1000
            logger.info(f"  Inversion: {dt:.2f}s ({T / dt:.0f} FPS)")
            logger.info(f"  Pose inversion error (mean per-vert): {lie_err:.3f} mm")

            recovered_rotations = result["rotations"]
            root_translation = result["root_translation"]

            # Step 3: forward pass with the recovered parameters.
            recon = hand_layer.pose(recovered_rotations, identity, pose2rot=False,
                                    absolute_pose=True, global_translation=root_translation)
            recon_verts_m = np.asarray(recon["vertices"])
            recon_joints_m = np.asarray(recon["joints"])

            # Step 4: numerical comparison.
            per_vert_err = np.linalg.norm(np.asarray(soma_target_verts) - recon_verts_m, axis=-1)
            excl = hand_layer.excluded_vert_ids
            valid_mask = np.ones(per_vert_err.shape[1], dtype=bool)
            if excl is not None and len(excl) > 0:
                valid_mask[np.asarray(excl)] = False
            valid = per_vert_err[:, valid_mask]
            mean_err, max_err = valid.mean() * 1000, valid.max() * 1000
            median_err = float(np.median(valid)) * 1000
            per_frame_mean = valid.mean(axis=1) * 1000
            logger.info(f"\n  Round-trip vertex error (mm) [excl {(~valid_mask).sum()} "
                        f"boundary verts]:")
            logger.info(f"    mean:   {mean_err:.3f}")
            logger.info(f"    median: {median_err:.3f}")
            logger.info(f"    max:    {max_err:.3f}")
            logger.info(f"    per-frame mean: min={per_frame_mean.min():.3f}, "
                        f"max={per_frame_mean.max():.3f}")
            all_stats.append({
                "hand": hand_type, "seq": seq_idx, "frames": T,
                "mean_mm": float(mean_err), "median_mm": median_err, "max_mm": float(max_err),
                "per_frame_min_mm": float(per_frame_mean.min()),
                "per_frame_max_mm": float(per_frame_mean.max()),
                "inversion_time_s": dt, "ms_per_frame": dt / T * 1000, "fps": T / dt,
            })

            # Step 5: reconstruction USD.
            _export_usd(f"{out_prefix}_recon.usda", hand_layer, identity,
                        recovered_rotations, root_translation)
            logger.info(f"  Reconstruction USD: {out_prefix}_recon.usda")

            # Step 6: comparison video + GIF with skeleton overlays.
            if args.no_render:
                continue
            import imageio.v2 as imageio
            from tqdm import tqdm
            from vis_pyrender import MeshRenderer, compute_camera_pose, save_image

            mano_verts = np.asarray(soma_target_verts)
            mano_joints = gt_joints - gt_joints[:, 0:1, :]
            all_v = np.concatenate([mano_verts[0], recon_verts_m[0]], axis=0)
            cam_pose = compute_camera_pose(all_v, cam_dist_scale=4.5)
            light_dir = np.array([0.0, -0.3, -1.0])
            renderer = MeshRenderer(image_size=args.image_size, light_intensity=5.0)
            renderer.camera.zfar = 500.0

            mp4_path, gif_path = f"{out_prefix}_comparison.mp4", f"{out_prefix}_comparison.gif"
            writer = imageio.get_writer(mp4_path, fps=args.fps)
            gif_frames = []
            for t in tqdm(range(T), desc=f"Rendering {hand_type} seq {seq_idx}"):
                frame = render_overlay_frame(
                    renderer, mano_verts[t], faces_np, mano_joints[t],
                    MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS, recon_verts_m[t], faces_np,
                    recon_joints_m[t], parent_ids, cam_pose, light_dir,
                    joint_radius=0.0012, bone_radius=0.0005)
                writer.append_data(frame[..., ::-1])
                gif_frames.append(frame)
            writer.close()
            renderer.delete()
            imageio.mimsave(gif_path, gif_frames, fps=args.gif_fps, loop=0)
            still_path = f"{out_prefix}_frame0.png"
            save_image(still_path, gif_frames[0])
            logger.info(f"  Video: {mp4_path} ({T} frames)")
            logger.info(f"  GIF:   {gif_path} ({len(gif_frames)} frames @ {args.gif_fps} fps)")
            logger.info(f"  Still: {still_path}")

    logger.info(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    for s in all_stats:
        logger.info(
            f"  {s['hand']:5s} seq{s['seq']}: {s['frames']:3d}f  "
            f"mean={s['mean_mm']:.2f}mm  median={s['median_mm']:.2f}mm  "
            f"max={s['max_mm']:.2f}mm  "
            f"per-frame=[{s['per_frame_min_mm']:.2f}, {s['per_frame_max_mm']:.2f}]mm  "
            f"{s['ms_per_frame']:.1f}ms/f ({s['fps']:.0f} FPS)")
    np.savez(f"{args.output_dir}/stats.npz", stats=all_stats)
    logger.info(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
