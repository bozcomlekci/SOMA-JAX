"""SMPL to SOMA pose converter.

Upstream: ``tools/smpl2soma.py`` (SOMA-X). Converts SMPL posed meshes to SOMA
skeleton parameters with pose inversion — analytical iterative inverse-LBS
Newton-Schulz refinement, Lie-algebra Gauss-Newton, optionally followed by
autograd FK optimization — on the SMPL animation that ships with the assets
(``SMPL/smpl_anim.npy``).

JAX port. The SMPL forward is :class:`soma_jax.smpl.SMPLLayer` placed at
``J0(betas) + transl``, which reproduces ``smplx``'s SMPL output (0.5 µm on
this clip); SOMA-JAX refits on a low-LOD layer it builds itself, as upstream's
``PoseInversion(low_lod=True)`` does internally. The SMPL model file is
licensed separately: by default ``<data-root>/SMPL/SMPL_NEUTRAL.{pkl,npz}``
(upstream's location), or pass ``--smpl-model-path`` (SOMA-JAX extra).

Usage::

    python tools/convert/smpl2soma.py --smpl-model-path data/smpl/SMPL_NEUTRAL.pkl
    python tools/convert/smpl2soma.py --body-iters 3 --full-iters 1 --batch-size 64
    python tools/convert/smpl2soma.py --autograd-iters 10  # analytical + autograd
    python tools/convert/smpl2soma.py --no-render --output-npz out/smpl2soma.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from conversion_utils import add_inversion_args, export_soma_npz  # noqa: E402
from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _smpl_model_path(data_root: Path, override) -> Path:
    if override is not None:
        return Path(override)
    for ext in ("pkl", "npz"):
        candidate = data_root / "SMPL" / f"SMPL_NEUTRAL.{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No SMPL model at {data_root / 'SMPL'}/SMPL_NEUTRAL.{{pkl,npz}}; the SMPL model is "
        "licensed separately — pass --smpl-model-path.")


def smpl_vertices(smpl_layer, identity, global_orient, body_pose, transl):
    """SMPL vertices as ``smplx`` produces them: the root placed at ``J0 + transl``."""
    import jax.numpy as jnp
    poses = jnp.concatenate([global_orient.reshape(-1, 1, 3), body_pose.reshape(-1, 23, 3)], 1)
    root = identity.bind_transforms_world[:, 0, :3, 3]
    return smpl_layer.pose(poses, identity, pose2rot=True, apply_correctives=True,
                           global_translation=root + transl)["vertices"]


def main():
    parser = argparse.ArgumentParser(description="SMPL to SOMA pose converter.")
    add_inversion_args(parser, batch_size=None, autograd=True)
    parser.add_argument("--subsample", type=int, default=4,
                        help="Frame subsampling factor (default: 4).")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument("--output-usd", default=None, help="Output .usd/.usda/.usdc with UsdSkel.")
    parser.add_argument("--fps", type=int, default=30, help="Output video/USD FPS (default: 30).")
    parser.add_argument("--smpl-model-path", default=None,
                        help="Licensed SMPL model file (SOMA-JAX extra; default "
                             "<data-root>/SMPL/SMPL_NEUTRAL.{pkl,npz}).")
    add_logging_args(parser)
    from soma_jax.io import add_npz_args
    add_npz_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    import jax.numpy as jnp

    from soma_jax import SOMALayer
    from soma_jax.assets import data_root as default_data_root
    from soma_jax.geometry.transforms import rotation_6d_to_rotmat, rotmat_to_axis_angle
    from soma_jax.fitting.pose_inversion import PoseInversion
    from soma_jax.smpl.layers import SMPLLayer

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    smpl_path = _smpl_model_path(data_root, args.smpl_model_path)

    # --- Load SMPL animation ---
    smpl_rot_mats = np.load(data_root / "SMPL" / "smpl_anim.npy", allow_pickle=True).item()
    to_aa = lambda r6: rotmat_to_axis_angle(  # noqa: E731
        rotation_6d_to_rotmat(jnp.asarray(r6, jnp.float32).reshape(-1, 6))).reshape(
            r6.shape[:-1] + (3,))
    body_pose = to_aa(smpl_rot_mats["body_pose_6d"])
    global_orient = to_aa(smpl_rot_mats["global_orient_6d"])
    betas = jnp.asarray(smpl_rot_mats["betas"], jnp.float32)
    transl = jnp.asarray(smpl_rot_mats["transl"], jnp.float32)

    seq_len = body_pose.shape[0]
    idx = np.arange(0, seq_len, args.subsample)
    body_pose, global_orient, betas, transl = body_pose[idx], global_orient[idx], betas[idx], transl[idx]
    num_frames = len(idx)
    logger.info(f"Loaded {num_frames} frames (subsampled {args.subsample}x from {seq_len})")

    # --- SMPL model ---
    smpl_layer = SMPLLayer(data_root, model_path=smpl_path)
    smpl_identity = smpl_layer.prepare_identity(betas[:1])
    smpl_faces = np.asarray(smpl_layer.faces)

    # --- SOMA + pose inversion (refits on an internal low-LOD layer) ---
    soma = SOMALayer.from_upstream_assets(
        identity_model_type="smpl", identity_model_kwargs={"model_path": str(smpl_path)},
        data_root=args.data_root)
    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(betas[:1])

    batch_size = args.batch_size or num_frames
    parts = [f"body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters}"]
    if args.lie_iters > 0:
        parts.append(f"lie-gn={args.lie_iters}, lambda={args.lie_lambda}")
    if args.autograd_iters > 0:
        parts.append(f"autograd={args.autograd_iters}, lr={args.autograd_lr}")
    if args.batch_size:
        parts.append(f"batch_size={batch_size}")
    logger.info(f"\nInverting ({', '.join(parts)})...")
    fit_kw = dict(body_iters=args.body_iters, finger_iters=args.finger_iters,
                  full_iters=args.full_iters, lie_iters=args.lie_iters,
                  lie_lambda=args.lie_lambda, autograd_iters=args.autograd_iters,
                  autograd_lr=args.autograd_lr)

    # Warmup. Upstream warms up on one frame to compile its Warp kernels; XLA
    # compiles per shape, so the port warms up on the first chunk's shape.
    w = min(batch_size, num_frames)
    inv.fit(smpl_vertices(smpl_layer, smpl_identity, global_orient[:w], body_pose[:w],
                          transl[:w]), **fit_kw)["rotations"].block_until_ready()

    t0 = time.perf_counter()
    all_rotations, all_root_transl, all_errors = [], [], []
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        verts = smpl_vertices(smpl_layer, smpl_identity, global_orient[start:end],
                              body_pose[start:end], transl[start:end])
        result = inv.fit(verts, **fit_kw)
        # Host copies per chunk, as upstream's `.cpu()`.
        all_rotations.append(np.asarray(result["rotations"]))
        all_root_transl.append(np.asarray(result["root_translation"]))
        all_errors.append(np.asarray(result["per_vertex_error"]))
    dt = time.perf_counter() - t0

    rotations = jnp.asarray(np.concatenate(all_rotations, axis=0))
    root_transl = jnp.asarray(np.concatenate(all_root_transl, axis=0))
    err = np.concatenate(all_errors, axis=0)
    logger.info(f"  Time: {dt:.3f}s ({num_frames / dt:.0f} fps)")
    logger.info(f"  Mean vertex error: {float(err.mean()):.6f} m")
    logger.info(f"  Max vertex error:  {float(err.max()):.6f} m")

    if args.output_npz:
        export_soma_npz(args.output_npz, rotations, root_transl, inv.soma,
                        output_unit=args.output_unit, keep_root=args.keep_root,
                        identity_coeffs=np.asarray(betas[:1]))

    if args.output_usd:
        # The full-resolution layer with the fitted identity, as upstream exports.
        from soma_jax.usd_io import export_soma_usd
        rest, _, binds = soma.prepare_identity(betas[:1], return_bind_transforms=True)
        export_soma_usd(args.output_usd, soma, rotations, root_transl,
                        bind_transforms_world=binds, rest_shape=rest, fps=float(args.fps))

    if args.no_render:
        return

    # --- Render: SMPL forward + SOMA reconstruction on the refit layer ---
    from vis_pyrender import default_pyopengl_platform, render_comparison_video, set_pyopengl_platform
    set_pyopengl_platform(default_pyopengl_platform())
    rest, joints, binds = inv.soma.prepare_identity(betas[:1], return_bind_transforms=True)
    smpl_verts_all, soma_verts_all = [], []
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        smpl_verts_all.append(np.asarray(smpl_vertices(
            smpl_layer, smpl_identity, global_orient[start:end], body_pose[start:end],
            transl[start:end])))
        sv = inv.soma.pose(rotations[start:end], root_transl[start:end], rest, joints,
                           bind_transforms=binds, absolute_pose=True,
                           apply_correctives=False).vertices
        soma_verts_all.append(np.asarray(sv))

    tag_parts = [f"analytical_b{args.body_iters}f{args.full_iters}"]
    if args.lie_iters > 0:
        tag_parts.append(f"lie{args.lie_iters}")
    if args.autograd_iters > 0:
        tag_parts.append(f"ag{args.autograd_iters}")
    out_name = "out/smpl2soma_" + "_".join(tag_parts) + ".mp4"
    Path(out_name).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nRendering comparison video -> {out_name}")
    render_comparison_video(out_name, np.concatenate(smpl_verts_all, axis=0), smpl_faces,
                            np.concatenate(soma_verts_all, axis=0), np.asarray(inv.soma.faces),
                            label_source="SMPL", cam_dist_scale=3.0)


if __name__ == "__main__":
    main()
