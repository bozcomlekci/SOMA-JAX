"""Convert pose parameters into native SMPL-family pose parameters.

Upstream: ``tools/pose_converter.py`` (SOMA-X v0.3.3). The source is a SOMA
``.npz`` (``--source soma`` / ``soma-procedural``) or an AMASS-style SMPL /
SMPL-X ``.npz``; the target is an SMPL or SMPL-X rig. The source rig is posed,
its mesh bridged onto the target topology, and pose inversion recovers the
target's absolute local rotations and root translation
(:func:`soma_jax.smpl.transfer_smpl_family_pose_parameters`). Optional
inspection outputs: source/target skeleton USDs and source / target / overlay
videos with skeletons.

JAX port: SOMA-JAX layers return their prepared identity instead of caching
it, so the source identity is re-prepared, with the same arguments, where
upstream reuses the cached one; ``--device`` and ``--mode`` are accepted and
ignored (JAX uses its default device; sparse top-8 LBS stands in for Warp).
The SMPL-family model files are licensed separately and resolved from
``--data-root`` as upstream (``<MODEL>/<MODEL>_<GENDER>.{npz,pkl}``).

Usage::

    python tools/convert/pose_converter.py --source soma --target smpl \\
        --input soma_motion.npz --output out/smpl_motion.npz
    python tools/convert/pose_converter.py --source smpl --target smplx \\
        --input amass_clip.npz --output out/smplx.npz --export-usd --render
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from logging_utils import add_logging_args, configure_logging  # noqa: E402

BODY_FIT_DEFAULTS = {
    "body_iters": 2,
    "full_iters": 1,
    "lie_iters": 3,
    "lie_lambda": 1e-1,
    "batch_size": 64,
}


def _is_soma_spec(spec: str) -> bool:
    normalized = spec.lower().replace("_", "-")
    return normalized in {
        "soma", "somalayer", "soma-body", "soma-proc", "soma-procedural",
        "soma-body-proc", "soma-body-procedural",
    }


def _is_procedural_soma_body_spec(spec: str) -> bool:
    normalized = spec.lower().replace("_", "-")
    return normalized in {"soma-proc", "soma-procedural", "soma-body-proc",
                          "soma-body-procedural"}


def _ensure_smpl_body_spec(spec: str) -> str:
    normalized = spec.lower().replace("_", "-")
    if normalized not in {"smpl", "smplx"}:
        raise ValueError(
            f"Public pose converter supports only SMPL/SMPL-X body specs, got {spec!r}.")
    return normalized


def _string_from_npz(data, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if hasattr(value, "shape") and value.shape == ():
        return str(value.item())
    return str(value)


def _load_soma_source(spec, input_path, data_root, device, mode, apply_correctives):
    """A SOMA ``.npz`` and the SOMA layer to pose it (identity backend and unit
    from the file)."""
    import jax.numpy as jnp

    from soma_jax import SOMALayer
    from soma_jax.io import load_soma_npz
    from soma_jax.units import Unit

    del device, mode
    data = load_soma_npz(input_path)
    output_unit = Unit.from_name(str(data["unit"]))
    identity_model_type = str(data["identity_model_type"])
    # Upstream's default checkpoint handling: `correctives_model.pt` on the
    # procedural rig, none on the legacy one.
    layer = SOMALayer.from_upstream_assets(
        identity_model_type=identity_model_type, lod="mid",
        procedural=_is_procedural_soma_body_spec(spec), output_unit=output_unit,
        data_root=str(data_root))
    source_pose_kwargs = {"apply_correctives": apply_correctives}

    poses = jnp.asarray(np.asarray(data["poses"]), jnp.float32)
    pose2rot = str(data["rotation_repr"]) == "rotvec"
    absolute_pose = bool(data["absolute_pose"])
    root_translation = jnp.asarray(np.asarray(data["transl"]), jnp.float32)
    identity = jnp.asarray(np.asarray(data["identity_coeffs"]), jnp.float32)

    prepare_kwargs = {}
    if "scale_params" in data:
        prepare_kwargs["scale_params"] = jnp.asarray(np.asarray(data["scale_params"]),
                                                     jnp.float32)
    if "global_scale" in data:
        prepare_kwargs["global_scale"] = data["global_scale"]
    identity_kwargs = {}
    if "bone_length_flexibles" in data:
        identity_kwargs["bone_length_flexibles"] = jnp.asarray(
            np.asarray(data["bone_length_flexibles"]), jnp.float32)
    if identity_kwargs:
        prepare_kwargs["kwargs"] = identity_kwargs

    return (layer, poses, pose2rot, absolute_pose, root_translation, identity,
            prepare_kwargs, source_pose_kwargs)


def _amass_pose_array(poses, source_num_joints: int, source_model_spec: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None]
    if poses.ndim == 3 and poses.shape[-2:] == (source_num_joints, 3):
        return poses
    if poses.ndim != 2:
        raise ValueError(f"Expected AMASS-style poses with shape (T, P), got {poses.shape}.")
    target_width = source_num_joints * 3
    if poses.shape[1] == target_width:
        return poses.reshape(poses.shape[0], source_num_joints, 3)
    out = np.zeros((poses.shape[0], target_width), dtype=np.float32)
    if source_model_spec == "smpl" and poses.shape[1] >= 66:
        out[:, :66] = poses[:, :66]
    elif source_model_spec == "smplx" and poses.shape[1] >= 156:
        out[:, :156] = poses[:, :156]
    else:
        width = min(poses.shape[1], target_width)
        out[:, :width] = poses[:, :width]
    return out.reshape(poses.shape[0], source_num_joints, 3)


def _load_amass_style_source(spec, input_path, data_root, device, mode, apply_correctives):
    """An AMASS-style ``.npz`` (``poses``/``trans``/``betas``/``gender``) and the
    SMPL-family layer to pose it."""
    import jax.numpy as jnp

    from soma_jax.smpl import create_smpl_family_layer
    from soma_jax.units import Unit

    del device
    data = np.load(input_path, allow_pickle=True)
    gender = _string_from_npz(data, "gender", "neutral").lower()
    spec = _ensure_smpl_body_spec(spec)
    layer = create_smpl_family_layer(spec, data_root, mode=mode, output_unit=Unit.METERS,
                                     gender=gender)
    if "poses" not in data:
        raise ValueError("SMPL-family input must contain AMASS-style key 'poses'.")
    poses = jnp.asarray(_amass_pose_array(data["poses"], layer.num_joints, layer.model_spec))
    num_frames = poses.shape[0]

    trans = data["trans"] if "trans" in data else np.zeros((num_frames, 3), dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim == 1:
        trans = trans[None]
    root_translation = jnp.asarray(trans)
    betas = (data["betas"] if "betas" in data
             else np.zeros(layer.num_identity_coeffs, dtype=np.float32))
    identity = jnp.asarray(np.asarray(betas, dtype=np.float32))
    return (layer, poses, True, True, root_translation, identity, {},
            {"apply_correctives": apply_correctives})


def _load_source(spec, input_path, data_root, device, mode, apply_correctives):
    if _is_soma_spec(spec):
        return _load_soma_source(spec, input_path, data_root, device, mode, apply_correctives)
    return _load_amass_style_source(spec, input_path, data_root, device, mode,
                                    apply_correctives)


def _stats(error) -> dict[str, float]:
    flat = np.sort(np.asarray(error).ravel())
    return {
        "mean": float(flat.mean()),
        # torch.median of an even count is the lower middle value.
        "median": float(flat[(flat.size - 1) // 2]),
        "max": float(flat[-1]),
    }


def _fit_kwargs_for_target(target_layer, *, body_iters=None, full_iters=None,
                           lie_iters=None, lie_lambda=None, batch_size=None) -> dict:
    kwargs = dict(BODY_FIT_DEFAULTS)
    overrides = {"body_iters": body_iters, "full_iters": full_iters, "lie_iters": lie_iters,
                 "lie_lambda": lie_lambda, "batch_size": batch_size}
    for key, value in overrides.items():
        if value is not None:
            kwargs[key] = value
    return kwargs


def _root_joint_idx(layer) -> int:
    return int(getattr(layer, "root_joint_idx", 0))


def _output_parent_ids(layer, layer_out: dict) -> np.ndarray:
    transforms = layer_out.get("transforms")
    output_parent_ids = getattr(layer, "output_joint_parent_ids", None)
    if output_parent_ids is not None:
        if transforms is None or transforms.shape[-3] == len(output_parent_ids):
            return np.asarray(output_parent_ids)
    public_parent_ids = getattr(layer, "public_joint_parent_ids", None)
    if (transforms is not None and public_parent_ids is not None
            and transforms.shape[-3] == len(public_parent_ids)):
        return np.asarray(public_parent_ids)
    return np.asarray(layer.joint_parent_ids)


def _skeleton_positions_for_render(layer_out: dict, parent_ids: np.ndarray) -> np.ndarray:
    joints = np.asarray(layer_out["joints"])
    if joints.shape[1] == len(parent_ids):
        return joints
    transforms = layer_out.get("transforms")
    if transforms is None:
        return joints
    transform_joints = np.asarray(transforms[..., :3, 3])
    if transform_joints.shape[1] == len(parent_ids):
        return transform_joints
    return joints


def _fitted_rig(layer, prepared):
    """``(rest_shape, bind_transforms_world)`` of a prepared identity — what
    upstream's ``export_soma_usd`` reads from the layer's cache."""
    from soma_jax.smpl.transfer import _SOMABodyIdentity
    if isinstance(prepared, _SOMABodyIdentity):
        rest, _, binds = layer.prepare_identity(
            prepared.identity_coeffs, prepared.scale_params,
            repose_to_bind_pose=prepared.repose_to_bind_pose, return_bind_transforms=True,
            global_scale=prepared.global_scale, kwargs=prepared.kwargs)
        return rest, binds
    return prepared.rest_shape, prepared.bind_transforms_world


def _export_inspection_usds(inspect_dir: Path, *, source_layer, target_layer, source_prepared,
                            target_prepared, source_rotations, source_root_translation,
                            target_rotations, target_root_translation, fps: float) -> dict:
    from soma_jax.usd_io import export_soma_usd

    inspect_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_skeleton_usd": inspect_dir / "source_skeleton.usda",
        "target_reconstruction_skeleton_usd": inspect_dir / "target_reconstruction_skeleton.usda",
    }
    for key, layer, prepared, rotations, root in (
            ("source_skeleton_usd", source_layer, source_prepared, source_rotations,
             source_root_translation),
            ("target_reconstruction_skeleton_usd", target_layer, target_prepared,
             target_rotations, target_root_translation)):
        rest, binds = _fitted_rig(layer, prepared)
        default_name = "source_mesh" if key.startswith("source") else "target_mesh"
        export_soma_usd(paths[key], layer, rotations, root, bind_transforms_world=binds,
                        rest_shape=rest, fps=fps, root_joint_idx=_root_joint_idx(layer),
                        skin_mesh_name=getattr(layer, "default_skin_mesh_name", default_name))
    return {key: str(path) for key, path in paths.items()}


def _render_inspection(inspect_dir: Path, *, source_layer, target_layer, source_out: dict,
                       target_out: dict, target_fit_vertices, fps: float, image_size: int,
                       max_render_frames) -> dict:
    import imageio.v2 as imageio

    from vis_pyrender import (MeshRenderer, compute_camera_pose, default_pyopengl_platform,
                              overlay_skeleton, render_mesh_panel, save_image,
                              set_pyopengl_platform)

    set_pyopengl_platform(default_pyopengl_platform())
    inspect_dir.mkdir(parents=True, exist_ok=True)

    source_vertices = np.asarray(source_out["vertices"])
    target_vertices = np.asarray(target_out["vertices"])
    fit_vertices = np.asarray(target_fit_vertices)
    source_faces = np.asarray(source_layer.faces).astype(np.int32)
    target_faces = np.asarray(target_layer.faces).astype(np.int32)
    source_parents = _output_parent_ids(source_layer, source_out).astype(np.int32)
    target_parents = _output_parent_ids(target_layer, target_out).astype(np.int32)
    source_joints = _skeleton_positions_for_render(source_out, source_parents)
    target_joints = _skeleton_positions_for_render(target_out, target_parents)

    frame_count = source_vertices.shape[0]
    if max_render_frames is not None:
        frame_count = min(frame_count, max_render_frames)

    cam_seed = np.concatenate([source_vertices[0], fit_vertices[0], target_vertices[0]], axis=0)
    source_cam_pose = compute_camera_pose(source_vertices[0], cam_dist_scale=4.5)
    target_cam_pose = compute_camera_pose(
        np.concatenate([fit_vertices[0], target_vertices[0]], axis=0), cam_dist_scale=4.5)
    light_dir = np.array([0.0, -0.3, -1.0])
    extent = np.linalg.norm(cam_seed.max(axis=0) - cam_seed.min(axis=0))
    joint_radius = max(float(extent) * 0.008, 0.001)
    bone_radius = joint_radius * 0.35

    comparison_path = inspect_dir / "render_comparison.mp4"
    overlay_path = inspect_dir / "render_overlay.mp4"
    comparison_writer = imageio.get_writer(str(comparison_path), fps=fps)
    overlay_writer = imageio.get_writer(str(overlay_path), fps=fps)
    first_comparison = None
    first_overlay = None
    renderer = MeshRenderer(image_size=image_size, light_intensity=5.0)
    renderer.camera.zfar = 500.0
    skel_kw = dict(light_dir=light_dir, joint_radius=joint_radius, bone_radius=bone_radius)
    try:
        for frame_idx in range(frame_count):
            source_panel = render_mesh_panel(
                renderer, source_vertices[frame_idx], source_faces,
                mesh_color=(0.65, 0.65, 0.65, 1.0), cam_pose=source_cam_pose,
                light_dir=light_dir)
            source_panel = overlay_skeleton(
                renderer, source_panel, source_joints[frame_idx], source_parents,
                color=(0.9, 0.15, 0.12, 1.0), cam_pose=source_cam_pose, **skel_kw)

            target_panel = render_mesh_panel(
                renderer, target_vertices[frame_idx], target_faces,
                mesh_color=(0.25, 0.75, 0.35, 1.0), cam_pose=target_cam_pose,
                light_dir=light_dir)
            target_panel = overlay_skeleton(
                renderer, target_panel, target_joints[frame_idx], target_parents,
                color=(0.0, 0.35, 0.12, 1.0), cam_pose=target_cam_pose, **skel_kw)

            fit_panel = render_mesh_panel(
                renderer, fit_vertices[frame_idx], target_faces,
                mesh_color=(0.65, 0.65, 0.65, 1.0), cam_pose=target_cam_pose,
                light_dir=light_dir)
            recon_panel = render_mesh_panel(
                renderer, target_vertices[frame_idx], target_faces,
                mesh_color=(0.15, 0.65, 0.95, 1.0), cam_pose=target_cam_pose,
                light_dir=light_dir)
            overlay_panel = np.clip(0.55 * fit_panel + 0.45 * recon_panel, 0, 255).astype(np.uint8)
            overlay_panel = overlay_skeleton(
                renderer, overlay_panel, target_joints[frame_idx], target_parents,
                color=(0.0, 0.25, 0.7, 1.0), cam_pose=target_cam_pose, **skel_kw)

            comparison = np.concatenate([source_panel, target_panel, overlay_panel], axis=1)
            if first_comparison is None:
                first_comparison = comparison
                first_overlay = overlay_panel
            comparison_writer.append_data(comparison[..., ::-1])
            overlay_writer.append_data(overlay_panel[..., ::-1])
    finally:
        comparison_writer.close()
        overlay_writer.close()
        renderer.delete()

    frame_path = inspect_dir / "render_comparison_frame0.png"
    overlay_frame_path = inspect_dir / "render_overlay_frame0.png"
    if first_comparison is not None:
        save_image(str(frame_path), first_comparison)
    if first_overlay is not None:
        save_image(str(overlay_frame_path), first_overlay)
    return {
        "comparison_video": str(comparison_path),
        "overlay_video": str(overlay_path),
        "comparison_frame": str(frame_path),
        "overlay_frame": str(overlay_frame_path),
    }


def _write_inspection_outputs(inspect_dir: Path, *, source_layer, target_layer, source_poses,
                              source_root, source_identity, source_prepare_kwargs,
                              source_pose2rot: bool, source_absolute_pose: bool,
                              source_pose_kwargs: dict, target_identity, result,
                              export_usd: bool, render: bool, fps: float, image_size: int,
                              max_render_frames) -> dict:
    from soma_jax.geometry.rig_utils import joint_world_to_local
    from soma_jax.smpl.transfer import _pose_layer, _prepare_layer_identity

    # Upstream poses the identities the transfer left cached on each layer.
    source_prepared = _prepare_layer_identity(source_layer, source_identity,
                                              source_prepare_kwargs)
    target_prepared = _prepare_layer_identity(target_layer, target_identity, None)
    source_out = _pose_layer(source_layer, source_prepared, source_poses, source_root,
                             pose2rot=source_pose2rot, absolute_pose=source_absolute_pose,
                             extra_kwargs=source_pose_kwargs)
    target_out = _pose_layer(target_layer, target_prepared, result.rotations,
                             result.root_translation, pose2rot=False, absolute_pose=True,
                             extra_kwargs={"apply_correctives": False})
    source_parent_ids = _output_parent_ids(source_layer, source_out)
    source_local = joint_world_to_local(source_out["transforms"], source_parent_ids)
    source_rotations = source_local[..., :3, :3]
    source_root_translation = source_local[:, _root_joint_idx(source_layer), :3, 3]

    outputs = {}
    if export_usd:
        outputs.update(_export_inspection_usds(
            inspect_dir, source_layer=source_layer, target_layer=target_layer,
            source_prepared=source_prepared, target_prepared=target_prepared,
            source_rotations=source_rotations, source_root_translation=source_root_translation,
            target_rotations=result.rotations, target_root_translation=result.root_translation,
            fps=fps))
    if render:
        outputs.update(_render_inspection(
            inspect_dir, source_layer=source_layer, target_layer=target_layer,
            source_out=source_out, target_out=target_out,
            target_fit_vertices=result.fit_vertices, fps=fps, image_size=image_size,
            max_render_frames=max_render_frames))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Upstream-layout asset directory (default: soma_jax.assets).")
    parser.add_argument("--source", required=True, help="soma, soma-procedural, smpl, or smplx")
    parser.add_argument("--target", required=True, help="SMPL-family body target: smpl or smplx")
    parser.add_argument("--input", type=Path, required=True, help="Input .npz path")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--device", default="cpu",
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--mode", default="warp", choices=("warp",))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--body-iters", type=int, default=None,
                        help="Override model-specific PoseInversion body_iters.")
    parser.add_argument("--full-iters", type=int, default=None,
                        help="Override model-specific PoseInversion full_iters.")
    parser.add_argument("--lie-iters", type=int, default=None,
                        help="Override model-specific PoseInversion lie_iters.")
    parser.add_argument("--lie-lambda", type=float, default=None,
                        help="Override model-specific PoseInversion Lie-GN regularization.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override model-specific pose-inversion chunk size.")
    parser.add_argument("--apply-correctives", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Apply SOMA source pose correctives before fitting target pose "
                             "parameters.")
    parser.add_argument("--inspect-dir", type=Path, default=None,
                        help="Directory for optional USD and render inspection outputs.")
    parser.add_argument("--export-usd", action=argparse.BooleanOptionalAction, default=False,
                        help="Export source/fit/reconstruction mesh USDs and source/target "
                             "skeletal USDs.")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False,
                        help="Render source, reconstruction, and overlay videos with skeletons.")
    parser.add_argument("--fps", type=float, default=30.0, help="Inspection output FPS.")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Rendered panel size in pixels.")
    parser.add_argument("--max-render-frames", type=int, default=60,
                        help="Maximum frames to render.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    from soma_jax.assets import data_root as default_data_root
    from soma_jax.smpl import create_smpl_family_layer, transfer_smpl_family_pose_parameters
    from soma_jax.units import Unit

    data_root = args.data_root if args.data_root is not None else default_data_root()
    (source_layer, source_poses, source_pose2rot, source_absolute_pose, source_root,
     source_id, source_prepare_kwargs, source_pose_kwargs) = _load_source(
        args.source, args.input, data_root, args.device, args.mode, args.apply_correctives)
    target_layer = create_smpl_family_layer(
        _ensure_smpl_body_spec(args.target), data_root, mode=args.mode,
        output_unit=Unit.METERS)

    if args.max_frames is not None:
        source_poses = source_poses[: args.max_frames]
        source_root = source_root[: args.max_frames]
        if source_id.ndim > 1 and source_id.shape[0] > 1:
            source_id = source_id[: args.max_frames]
        if "scale_params" in source_prepare_kwargs:
            scale_params = source_prepare_kwargs["scale_params"]
            if scale_params.ndim > 1 and scale_params.shape[0] > 1:
                source_prepare_kwargs["scale_params"] = scale_params[: args.max_frames]
        if "kwargs" in source_prepare_kwargs:
            for key, value in list(source_prepare_kwargs["kwargs"].items()):
                if hasattr(value, "ndim") and value.ndim > 1 and value.shape[0] > 1:
                    source_prepare_kwargs["kwargs"][key] = value[: args.max_frames]

    fit_kwargs = _fit_kwargs_for_target(
        target_layer, body_iters=args.body_iters, full_iters=args.full_iters,
        lie_iters=args.lie_iters, lie_lambda=args.lie_lambda, batch_size=args.batch_size)

    result = transfer_smpl_family_pose_parameters(
        source_layer, target_layer, source_poses, source_identity_coeffs=source_id,
        source_root_translation=source_root, source_pose2rot=source_pose2rot,
        source_absolute_pose=source_absolute_pose, source_prepare_kwargs=source_prepare_kwargs,
        source_pose_kwargs=source_pose_kwargs, fit_kwargs=fit_kwargs)

    inspection_outputs = {}
    if args.export_usd or args.render:
        from soma_jax.smpl.transfer import _adapt_identity_coeffs, _layer_identity_coeffs
        inspect_dir = args.inspect_dir
        if inspect_dir is None:
            inspect_dir = args.output.with_suffix("").parent / f"{args.output.stem}_inspect"
        source_identity = _layer_identity_coeffs(source_layer, source_id)
        inspection_outputs = _write_inspection_outputs(
            inspect_dir, source_layer=source_layer, target_layer=target_layer,
            source_poses=source_poses, source_root=source_root,
            source_identity=source_identity, source_prepare_kwargs=source_prepare_kwargs,
            source_pose2rot=source_pose2rot, source_absolute_pose=source_absolute_pose,
            source_pose_kwargs=source_pose_kwargs,
            target_identity=_adapt_identity_coeffs(source_identity, target_layer),
            result=result, export_usd=args.export_usd, render=args.render, fps=args.fps,
            image_size=args.image_size, max_render_frames=args.max_render_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target_rotations=np.asarray(result.rotations),
        target_root_translation=np.asarray(result.root_translation),
        per_vertex_error=np.asarray(result.per_vertex_error),
        source_vertices=np.asarray(result.source_vertices),
        fit_vertices=np.asarray(result.fit_vertices),
        reconstructed_vertices=np.asarray(result.reconstructed_vertices),
        source=np.asarray(args.source),
        target=np.asarray(args.target),
        fit_kwargs_json=np.asarray(json.dumps(fit_kwargs)),
        inspection_outputs_json=np.asarray(json.dumps(inspection_outputs)),
    )
    print(json.dumps({
        "fit_kwargs": fit_kwargs,
        "inspection_outputs": inspection_outputs,
        "per_vertex_error": _stats(result.per_vertex_error),
    }, indent=2))


if __name__ == "__main__":
    main()
