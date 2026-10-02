"""Prototype SOMAHand-to-MANO round trip.

Upstream: ``tools/hand/soma2mano.py`` (SOMA-X v0.3.0). Same pipeline, arguments,
statistics and outputs, over SOMA-JAX layers:

  MANO test data -> SOMAHand pose inversion -> SOMAHand mesh ->
  MANO topology transfer -> MANO rig pose inversion.

Input ``.npz`` files carry, per hand (``lh`` / ``rh``): ``mano_{lh,rh}`` (a dict
with ``betas`` (T, 10), ``global_orient`` (T, 1, 3, 3), ``hand_pose``
(T, 15, 3, 3)) and ``cam_t_{lh,rh}`` (T, 3).

Usage::

    python tools/hand/soma2mano.py --input path/to/mano_test_data --no-render
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from conversion_utils import add_hand_inversion_args  # noqa: E402
from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_mano_data(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {path}")
    else:
        raise ValueError(f"Expected .npz file or directory, got {path}")
    return [{"path": f, "data": np.load(f, allow_pickle=True)} for f in files]


def build_soma_to_mano_interpolator(data_root: str | Path, hand_type: str):
    """``BarycentricInterpolator(SOMA_wrap, faces, base_hand)``: SOMAHand -> MANO topology."""
    import jax.numpy as jnp
    import trimesh

    from soma_jax.geometry.barycentric_interp import (
        barycentric_interpolate,
        compute_barycentric_coords,
    )
    mano_dir = Path(data_root) / "MANO"
    mesh_soma = trimesh.load(mano_dir / f"SOMA_wrap_{hand_type}.obj", maintain_order=True,
                             process=False)
    mesh_mano = trimesh.load(mano_dir / f"base_hand_{hand_type}.obj", maintain_order=True,
                             process=False)
    v_soma = np.asarray(mesh_soma.vertices, dtype=np.float32)
    f_soma = np.asarray(mesh_soma.faces, dtype=np.int64)
    v_mano = np.asarray(mesh_mano.vertices, dtype=np.float32)
    face_ids, bary = compute_barycentric_coords(v_mano, v_soma, f_soma)
    faces_j, face_ids_j, bary_j = (jnp.asarray(f_soma.astype(np.int32)), jnp.asarray(face_ids),
                                   jnp.asarray(bary, jnp.float32))
    return lambda verts: barycentric_interpolate(jnp.asarray(verts), faces_j, face_ids_j, bary_j)


def error_stats(error_m) -> dict[str, float]:
    error_mm = np.asarray(error_m) * 1000.0
    per_frame = error_mm.mean(axis=1)
    return {
        "mean_mm": float(error_mm.mean()),
        # torch.median of an even count returns the lower middle value.
        "median_mm": float(np.sort(error_mm.ravel())[(error_mm.size - 1) // 2]),
        "max_mm": float(error_mm.max()),
        "per_frame_min_mm": float(per_frame.min()),
        "per_frame_max_mm": float(per_frame.max()),
    }


def rotation_error_stats(recovered, target) -> dict[str, float]:
    """Angular error stats between two rotation-matrix animation arrays."""
    rel = np.asarray(recovered) @ np.swapaxes(np.asarray(target), -1, -2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    error_deg = np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    per_frame = error_deg.mean(axis=1)
    return {
        "mean_deg": float(error_deg.mean()),
        "median_deg": float(np.sort(error_deg.ravel())[(error_deg.size - 1) // 2]),
        "max_deg": float(error_deg.max()),
        "per_frame_min_deg": float(per_frame.min()),
        "per_frame_max_deg": float(per_frame.max()),
    }


def evaluate_source_mano_params(data_root, hand_type, mano_params, root_translation):
    """Evaluate stored MANO rotation-matrix params with the official ``smplx`` layer.

    Upstream's validation helper; needs ``torch`` and ``smplx`` (and chumpy-era
    pickles load through ``soma_jax.hand._smpl_family_loader``'s compat shim).
    """
    import torch
    from smplx import MANOLayer as SMPLXMANOLayer

    model_path = Path(data_root) / "MANO" / f"MANO_{hand_type.upper()}.pkl"
    layer = SMPLXMANOLayer(model_path=str(model_path), use_pca=False,
                           is_rhand=hand_type == "right", num_betas=10)
    root_translation = torch.as_tensor(np.asarray(root_translation), dtype=torch.float32)
    T = root_translation.shape[0]
    with torch.no_grad():
        out = layer(
            betas=torch.as_tensor(mano_params["betas"][:T]).float(),
            global_orient=torch.as_tensor(mano_params["global_orient"][:T]).float().reshape(T, 1, 3, 3),
            hand_pose=torch.as_tensor(mano_params["hand_pose"][:T]).float().reshape(T, 15, 3, 3),
            transl=root_translation, return_verts=True)
    return out.vertices.numpy(), out.joints.numpy()


def _export_usd(out_path, layer, identity, rotations, root_translation, fps, unit=None):
    from soma_jax.usd_io import export_soma_usd
    export_soma_usd(out_path, layer, rotations, root_translation,
                    bind_transforms_world=identity.bind_transforms_world[0],
                    rest_shape=identity.rest_shape[0], fps=fps, unit=unit, root_joint_idx=0,
                    skin_mesh_name=layer.default_skin_mesh_name)


def render_mesh_comparison(out_prefix, original, soma_to_mano, mano_recon, faces, image_size,
                           fps, max_frames) -> None:
    import imageio.v2 as imageio
    from vis_pyrender import (
        MeshRenderer,
        compute_camera_pose,
        default_pyopengl_platform,
        save_image,
        set_pyopengl_platform,
    )

    set_pyopengl_platform(default_pyopengl_platform())
    frame_count = original.shape[0] if max_frames is None else min(original.shape[0], max_frames)
    faces_np = np.asarray(faces).astype(np.int32)
    panels = [
        (np.asarray(original[:frame_count]), (0.65, 0.65, 0.65, 1.0)),
        (np.asarray(soma_to_mano[:frame_count]), (0.2, 0.65, 0.95, 1.0)),
        (np.asarray(mano_recon[:frame_count]), (0.3, 0.78, 0.2, 1.0)),
    ]
    cam_seed = np.concatenate([panel[0][0] for panel in panels], axis=0)
    cam_pose = compute_camera_pose(cam_seed, cam_dist_scale=4.5)
    light_dir = np.array([0.0, -0.3, -1.0])

    renderer = MeshRenderer(image_size=image_size, light_intensity=5.0)
    renderer.camera.zfar = 500.0
    comparison_writer = imageio.get_writer(f"{out_prefix}_comparison.mp4", fps=fps)
    soma_overlay_writer = imageio.get_writer(f"{out_prefix}_overlay_soma_to_mano.mp4", fps=fps)
    mano_overlay_writer = imageio.get_writer(f"{out_prefix}_overlay_mano_recon.mp4", fps=fps)
    first = None
    for frame_idx in range(frame_count):
        images = []
        for verts, color in panels:
            renderer.setup_mesh(faces=faces_np, mesh_color=color, metallic=0.0, roughness=0.5,
                                cam_pose=cam_pose, light_dir=light_dir,
                                base_color_factor=[0.9, 0.9, 0.9, 1.0])
            images.append(renderer.render_frame(verts[frame_idx]))
        frame = np.concatenate(images, axis=1)
        soma_overlay = np.clip(0.55 * images[0] + 0.45 * images[1], 0, 255).astype(np.uint8)
        mano_overlay = np.clip(0.55 * images[0] + 0.45 * images[2], 0, 255).astype(np.uint8)
        if first is None:
            first = (frame, soma_overlay, mano_overlay)
        comparison_writer.append_data(frame[..., ::-1])
        soma_overlay_writer.append_data(soma_overlay[..., ::-1])
        mano_overlay_writer.append_data(mano_overlay[..., ::-1])
    comparison_writer.close()
    soma_overlay_writer.close()
    mano_overlay_writer.close()
    renderer.delete()
    if first is not None:
        save_image(f"{out_prefix}_comparison_frame0.png", first[0])
        save_image(f"{out_prefix}_overlay_soma_to_mano_frame0.png", first[1])
        save_image(f"{out_prefix}_overlay_mano_recon_frame0.png", first[2])


def process_sequence(*, seq, seq_idx, hand_type, data_root, output_dir, mode, bcd_iters,
                     lie_iters, lie_lambda, batch_size, max_frames, export_usd, render,
                     image_size, fps, max_render_frames, device=None) -> dict[str, Any]:
    import jax.numpy as jnp

    del device  # upstream's torch device; JAX uses its default device

    from soma_jax.hand import MANOLayer, SOMAHandLayer
    from soma_jax.fitting.pose_inversion import PoseInversion
    from soma_jax.usd_io import save_vertex_animation_usd

    side = f"{hand_type[0]}h"
    data = seq["data"]
    mano_params = data[f"mano_{side}"].item()
    source_transl = np.asarray(data[f"cam_t_{side}"], np.float32)
    source_global_orient = np.asarray(mano_params["global_orient"], np.float32)
    source_hand_pose = np.asarray(mano_params["hand_pose"], np.float32)
    betas = np.asarray(mano_params["betas"], np.float32)
    if max_frames is not None:
        source_transl = source_transl[:max_frames]
        source_global_orient = source_global_orient[:max_frames]
        source_hand_pose = source_hand_pose[:max_frames]
        betas = betas[:max_frames]

    frame_count = source_transl.shape[0]
    betas_single = jnp.asarray(betas[:1])
    source_rotations = jnp.asarray(np.concatenate([
        source_global_orient.reshape(frame_count, 1, 3, 3),
        source_hand_pose.reshape(frame_count, 15, 3, 3)], axis=1))
    pose_kw = dict(pose2rot=False, absolute_pose=True)

    mano_layer = MANOLayer(data_root, hand_type=hand_type, mode=mode)
    mano_identity = mano_layer.prepare_identity(betas_single)
    source_mano = mano_layer.pose(source_rotations, mano_identity, **pose_kw,
                                  global_translation=jnp.asarray(source_transl))
    source_mano_verts = source_mano["vertices"]
    source_wrist = source_mano["joints"][:, 0]
    mano_centered = source_mano_verts - source_wrist[:, None, :]
    source_mano_centered = mano_layer.pose(
        source_rotations, mano_identity, **pose_kw,
        global_translation=jnp.zeros_like(jnp.asarray(source_transl)))["vertices"]

    hand_layer = SOMAHandLayer(data_root=str(data_root), hand_type=hand_type,
                               identity_model_type="mano", mode=mode)
    soma_inv = PoseInversion(hand_layer, low_lod=False)
    soma_inv.prepare_identity(betas_single)
    soma_target_m = soma_inv.transfer_to_soma(mano_centered)

    fit_kw = dict(body_iters=0, finger_iters=0, full_iters=bcd_iters, lie_iters=lie_iters,
                  lie_lambda=lie_lambda, batch_size=batch_size)
    t0 = time.perf_counter()
    soma_result = soma_inv.fit(soma_target_m, **fit_kw)
    soma_result["rotations"].block_until_ready()
    soma_time = time.perf_counter() - t0

    hand_identity = hand_layer.prepare_identity(betas_single)
    soma_recon = hand_layer.pose(soma_result["rotations"], hand_identity, **pose_kw,
                                 global_translation=soma_result["root_translation"])["vertices"]
    mano_from_soma = build_soma_to_mano_interpolator(data_root, hand_type)(soma_recon)

    mano_inv = PoseInversion(mano_layer, low_lod=False)
    mano_inv.prepare_identity(betas_single)
    t_direct = time.perf_counter()
    direct_mano_result = mano_inv.fit(source_mano_centered, **fit_kw)
    direct_mano_result["rotations"].block_until_ready()
    direct_mano_time = time.perf_counter() - t_direct
    t1 = time.perf_counter()
    mano_result = mano_inv.fit(mano_from_soma, **fit_kw)
    mano_result["rotations"].block_until_ready()
    mano_time = time.perf_counter() - t1

    mano_recon = mano_layer.pose(mano_result["rotations"], mano_identity, **pose_kw,
                                 global_translation=mano_result["root_translation"])["vertices"]
    direct_mano_recon = mano_layer.pose(
        direct_mano_result["rotations"], mano_identity, **pose_kw,
        global_translation=direct_mano_result["root_translation"])["vertices"]

    norm = lambda a, b: np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)  # noqa: E731
    soma_error = norm(soma_recon, soma_target_m)
    direct_mano_error = norm(direct_mano_recon, source_mano_centered)
    mano_error = norm(mano_recon, mano_from_soma)
    full_error = norm(mano_recon, mano_centered)
    source_mesh_roundtrip_error = norm(mano_recon, source_mano_centered)

    transl = np.asarray(source_wrist + mano_result["root_translation"])
    rotations = np.asarray(mano_result["rotations"])
    translation_error = np.linalg.norm(transl - source_transl, axis=-1)[:, None]
    direct_translation_error = np.linalg.norm(
        np.asarray(direct_mano_result["root_translation"]), axis=-1)[:, None]
    out_prefix = Path(output_dir) / f"{hand_type}_{seq_idx}_{seq['path'].stem[:32]}"
    stats = {
        "hand": hand_type,
        "sequence": str(seq["path"]),
        "frames": int(frame_count),
        "soma_fit": error_stats(soma_error),
        "direct_mano_fit": error_stats(direct_mano_error),
        "mano_fit": error_stats(mano_error),
        "full_roundtrip": error_stats(full_error),
        "source_mesh_roundtrip": error_stats(source_mesh_roundtrip_error),
        "direct_mano_rotation_roundtrip": rotation_error_stats(
            direct_mano_result["rotations"], source_rotations),
        "direct_mano_translation_roundtrip": error_stats(direct_translation_error),
        "source_mano_rotation_roundtrip": rotation_error_stats(rotations, source_rotations),
        "source_mano_translation_roundtrip": error_stats(translation_error),
        "soma_time_s": soma_time,
        "direct_mano_time_s": direct_mano_time,
        "mano_time_s": mano_time,
        "soma_ms_per_frame": soma_time / frame_count * 1000.0,
        "direct_mano_ms_per_frame": direct_mano_time / frame_count * 1000.0,
        "mano_ms_per_frame": mano_time / frame_count * 1000.0,
    }

    np.savez_compressed(
        f"{out_prefix}_mano_params.npz",
        betas=np.repeat(np.asarray(betas_single), frame_count, axis=0),
        global_orient=rotations[:, 0:1],
        hand_pose=rotations[:, 1:],
        transl=transl,
        root_translation=np.asarray(mano_result["root_translation"]),
        source_global_orient=np.asarray(source_rotations[:, 0:1]),
        source_hand_pose=np.asarray(source_rotations[:, 1:]),
        source_transl=source_transl,
        direct_global_orient=np.asarray(direct_mano_result["rotations"][:, 0:1]),
        direct_hand_pose=np.asarray(direct_mano_result["rotations"][:, 1:]),
        direct_root_translation=np.asarray(direct_mano_result["root_translation"]),
        stats_json=np.array(json.dumps(stats)),
    )
    with open(f"{out_prefix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if export_usd:
        save_vertex_animation_usd(f"{out_prefix}_mano_source.usda", np.asarray(source_mano_verts),
                                  np.asarray(mano_layer.faces), fps=float(fps), unit="meters",
                                  prim_path="/MANO_Source")
        _export_usd(f"{out_prefix}_mano_source_skel.usda", mano_layer, mano_identity,
                    source_rotations, source_transl, fps=float(fps), unit="meters")
        _export_usd(f"{out_prefix}_mano_direct.usda", mano_layer, mano_identity,
                    direct_mano_result["rotations"],
                    source_wrist + direct_mano_result["root_translation"], fps=float(fps),
                    unit="meters")
        _export_usd(f"{out_prefix}_soma_recon.usda", hand_layer, hand_identity,
                    soma_result["rotations"], soma_result["root_translation"] + source_wrist,
                    fps=float(fps))
        _export_usd(f"{out_prefix}_mano_recon.usda", mano_layer, mano_identity, rotations,
                    transl, fps=float(fps), unit="meters")

    if render:
        render_mesh_comparison(str(out_prefix), mano_centered, mano_from_soma, mano_recon,
                               mano_layer.faces, image_size, fps, max_render_frames)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prototype SOMAHand -> MANO roundtrip.")
    parser.add_argument("--input", required=True, help="MANO test .npz file or directory.")
    parser.add_argument("--output-dir", default="out/soma2mano")
    parser.add_argument("--hand-type", choices=["left", "right", "both"], default="both")
    parser.add_argument("--mode", choices=["warp"], default="warp",
                        help="Skinning mode for pose inversion. Only warp is supported by this "
                             "prototype.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-usd", action="store_true")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-render-frames", type=int, default=60)
    add_hand_inversion_args(parser, bcd_iters=3, lie_iters=10, batch_size=32)
    add_logging_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args)

    from soma_jax.assets import data_root as default_data_root
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_mano_data(args.input)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    hand_types = ["left", "right"] if args.hand_type == "both" else [args.hand_type]

    all_stats = []
    for hand_type in hand_types:
        for seq_idx, seq in enumerate(sequences):
            stats = process_sequence(
                seq=seq, seq_idx=seq_idx, hand_type=hand_type, data_root=data_root,
                output_dir=output_dir, mode=args.mode, bcd_iters=args.bcd_iters,
                lie_iters=args.lie_iters, lie_lambda=args.lie_lambda,
                batch_size=args.batch_size, max_frames=args.max_frames,
                export_usd=not args.no_usd, render=not args.no_render,
                image_size=args.image_size, fps=args.fps,
                max_render_frames=args.max_render_frames)
            all_stats.append(stats)
            logger.info(
                f"{hand_type} seq{seq_idx}: "
                f"SOMA mean={stats['soma_fit']['mean_mm']:.3f}mm, "
                f"direct MANO mean={stats['direct_mano_fit']['mean_mm']:.3f}mm, "
                f"MANO mean={stats['mano_fit']['mean_mm']:.3f}mm, "
                f"full mean={stats['full_roundtrip']['mean_mm']:.3f}mm, "
                f"rot mean={stats['source_mano_rotation_roundtrip']['mean_deg']:.3f}deg")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
