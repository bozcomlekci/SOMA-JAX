"""Convert the AMASS dataset to SOMA format.

Upstream: ``tools/convert_amass_to_soma.py`` (SOMA-X v0.3.3). AMASS
(https://amass.is.tue.mpg.de/) holds motion capture in SMPL-H format; this
converts sequences to SOMA pose parameters with the same analytical
inverse-LBS pipeline as ``smpl2soma.py`` — single files, or a whole dataset
with the folder structure mirrored.

Typical AMASS npz file structure:

- ``poses``: (T, 156) full-body pose in axis-angle (52 joints x 3)
- ``trans``: (T, 3) root translation
- ``betas``: (10,) or (16,) shape parameters
- ``gender``: ``'male'``, ``'female'`` or ``'neutral'``
- ``mocap_framerate``: frame rate of the original capture
- ``dmpls``: (T, 8) optional soft-tissue parameters

As upstream, the first 22 joints drive the neutral SMPL model (hands zeroed)
with the first 10 betas.

JAX port: the SMPL forward is :class:`soma_jax.smpl.layers.SMPLLayer` placed at
``J0(betas) + trans`` (``smplx``'s SMPL output), and the refit runs on a
low-LOD layer SOMA-JAX builds itself (upstream's ``PoseInversion(low_lod=True)``
builds it internally). The SMPL model file is licensed separately: by default
``<data-root>/SMPL/SMPL_NEUTRAL.{pkl,npz}``, or ``--smpl-model-path``
(SOMA-JAX extra).

Upstream v0.3.3 cannot save or render with its procedural (twist-joint) layer —
``_save_conversion`` removes the 110-joint twist-rig orient from the 78 public
rotations and records the 110 twist-rig names, and the render calls
``batched_skinning.pose`` with public rotations — so every file of a batch run
fails. The port saves the public rig (as ``conversion_utils.export_soma_npz``)
and renders the layer's own pose (public FK, twist expansion, LBS, no
correctives).

Usage::

    # Single file
    python tools/convert/convert_amass_to_soma.py --input <amass.npz> --output-npz out/test.npz
    python tools/convert/convert_amass_to_soma.py --input <amass.npz> --output-npz out/test.npz --no-render

    # Batch convert an entire dataset (mirrors the folder structure)
    python tools/convert/convert_amass_to_soma.py --input-dir /path/to/amass/ --output-dir out/amass_soma/
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from conversion_utils import export_soma_npz  # noqa: E402
from logging_utils import add_logging_args, configure_logging  # noqa: E402
from smpl2soma import _smpl_model_path, smpl_vertices  # noqa: E402

logger = logging.getLogger(__name__)


def load_amass_npz(npz_path):
    """Load an AMASS npz file and extract the SMPL-relevant data.

    Returns:
        dict with ``poses`` (T, 72) full SMPL pose (24 joints x 3 axis-angle),
        ``betas`` (10,), ``trans`` (T, 3), ``gender`` and ``fps``.
    """
    data = np.load(npz_path, allow_pickle=True)

    logger.info(f"\nLoading AMASS data from {npz_path}")
    keys = list(data.keys())
    logger.info(f"  Available keys: {keys}")
    for key in keys:
        if hasattr(data[key], "shape"):
            logger.info(f"    {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            logger.info(f"    {key}: {type(data[key])} = {data[key]}")

    # AMASS poses[:, :66] = 22 joints x 3 (root + 21 body joints); padded to
    # 72 = 24 joints x 3 for SMPL.
    poses_amass = data["poses"][:, :66]
    T = poses_amass.shape[0]
    poses_smpl = np.zeros((T, 72))
    poses_smpl[:, :66] = poses_amass
    trans = data["trans"] if "trans" in data else np.zeros((T, 3))
    betas = data["betas"][:10] if "betas" in data else np.zeros(10)
    gender = str(data["gender"]) if "gender" in data else "neutral"
    fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 30.0

    logger.info(f"  Sequence length: {T} frames")
    logger.info(f"  Shape (betas): {betas.shape}")
    logger.info(f"  Gender: {gender}")
    logger.info(f"  FPS: {fps}")
    return {"poses": poses_smpl, "betas": betas, "trans": trans, "gender": gender, "fps": fps}


def _fit_kwargs(args):
    return dict(body_iters=args.body_iters, finger_iters=args.finger_iters,
                full_iters=args.full_iters, lie_iters=args.lie_iters,
                lie_lambda=args.lie_lambda, autograd_iters=args.autograd_iters,
                autograd_lr=args.autograd_lr)


def convert_amass_sequence(amass_data, args, smpl_layer, inv, device=None):
    """Convert one AMASS sequence to SOMA with pose inversion.

    Args:
        amass_data: dict from :func:`load_amass_npz`.
        args: parsed command-line arguments.
        smpl_layer: the SMPL forward (:class:`soma_jax.smpl.layers.SMPLLayer`).
        inv: the :class:`~soma_jax.fitting.pose_inversion.PoseInversion`.
        device: accepted for upstream signature compatibility; JAX uses its
            default device.

    Returns:
        dict with ``rotations`` (T, J, 3, 3) absolute rotations,
        ``root_transl`` (T, 3), ``betas`` (T, 10), ``body_pose`` (T, 69),
        ``global_orient`` (T, 3), ``transl`` (T, 3) and ``per_vertex_error``.
    """
    import jax.numpy as jnp

    del device
    poses_smpl = jnp.asarray(amass_data["poses"], jnp.float32)
    betas = jnp.asarray(amass_data["betas"], jnp.float32)
    trans = jnp.asarray(amass_data["trans"], jnp.float32)

    T = poses_smpl.shape[0]
    global_orient = poses_smpl[:, :3]
    body_pose = poses_smpl[:, 3:]
    betas_expanded = jnp.broadcast_to(betas[None], (T, betas.shape[0]))
    batch_size = args.batch_size or T

    # Identity for this sequence's betas (one shape per sequence).
    inv.prepare_identity(betas_expanded[:1])
    smpl_identity = smpl_layer.prepare_identity(betas_expanded[:1])
    fit_kw = _fit_kwargs(args)

    # Warmup. Upstream warms up on one frame to compile its Warp kernels; XLA
    # compiles per shape, so the port warms up on the first chunk's shape.
    w = min(batch_size, T)
    inv.fit(smpl_vertices(smpl_layer, smpl_identity, global_orient[:w], body_pose[:w],
                          trans[:w]), **fit_kw)["rotations"].block_until_ready()

    parts = [f"body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters}"]
    if args.lie_iters > 0:
        parts.append(f"lie-gn={args.lie_iters}, lambda={args.lie_lambda}")
    if args.autograd_iters > 0:
        parts.append(f"autograd={args.autograd_iters}, lr={args.autograd_lr}")
    if args.batch_size:
        parts.append(f"batch_size={batch_size}")
    logger.info(f"\nInverting {T} frames ({', '.join(parts)})...")

    t0 = time.perf_counter()
    all_rotations, all_root_transl, all_errors = [], [], []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        verts = smpl_vertices(smpl_layer, smpl_identity, global_orient[start:end],
                              body_pose[start:end], trans[start:end])
        result = inv.fit(verts, **fit_kw)
        # Host copies per chunk, as upstream's `.cpu()`.
        all_rotations.append(np.asarray(result["rotations"]))
        all_root_transl.append(np.asarray(result["root_translation"]))
        all_errors.append(np.asarray(result["per_vertex_error"]))
    dt = time.perf_counter() - t0

    err = np.concatenate(all_errors, axis=0)
    logger.info(f"  Time: {dt:.3f}s ({T / dt:.0f} fps)")
    logger.info(f"  Mean vertex error: {err.mean():.6f} m")
    logger.info(f"  Max vertex error:  {err.max():.6f} m")
    return {
        "rotations": jnp.asarray(np.concatenate(all_rotations, axis=0)),
        "root_transl": jnp.asarray(np.concatenate(all_root_transl, axis=0)),
        "betas": betas_expanded,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": trans,
        "per_vertex_error": err,
    }


def _save_conversion(conv, inv, args, output_path):
    """Save a conversion result to a SOMA ``.npz`` (relative rotvec, public rig)."""
    export_soma_npz(output_path, conv["rotations"], conv["root_transl"], inv.soma,
                    output_unit=args.output_unit, keep_root=args.keep_root,
                    identity_coeffs=np.asarray(conv["betas"][:1]),
                    extra_arrays={"per_vertex_error": conv["per_vertex_error"]})


def main():
    from tqdm import tqdm

    from soma_jax.io import add_npz_args

    parser = argparse.ArgumentParser(description="Convert AMASS dataset to SOMA format.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input",
                             help="Path to a single AMASS .npz file (e.g., 01_01_poses.npz)")
    input_group.add_argument("--input-dir",
                             help="Path to AMASS root directory for batch conversion")
    parser.add_argument("--output-dir", default=None,
                        help="Root output directory for batch mode (mirrors input folder "
                             "structure).")
    parser.add_argument("--data-root", default=None,
                        help="Path to SOMA assets (default: soma_jax.assets)")
    parser.add_argument("--device", default="cuda:0",
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Process frames in chunks of this size (default: all at once).")
    parser.add_argument("--body-iters", type=int, default=2,
                        help="Analytical body iterations (default: 2).")
    parser.add_argument("--finger-iters", type=int, default=0,
                        help="Analytical finger iterations (default: 0).")
    parser.add_argument("--full-iters", type=int, default=1,
                        help="Analytical full iterations (default: 1).")
    parser.add_argument("--lie-iters", type=int, default=3,
                        help="Lie algebra Gauss-Newton iterations (default: 3).")
    parser.add_argument("--lie-lambda", type=float, default=1e-1,
                        help="Tikhonov regularisation for Lie-GN (default: 1e-1).")
    parser.add_argument("--autograd-iters", type=int, default=0,
                        help="Autograd FK refinement iterations after analytical solve "
                             "(default: 0 = off).")
    parser.add_argument("--autograd-lr", type=float, default=5e-3,
                        help="Learning rate for autograd FK (default: 5e-3).")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle file order in batch mode (useful for multi-worker "
                             "parallelism).")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip files whose output already exists (default: on). Use "
                             "--no-skip-existing to force.")
    parser.add_argument("--smpl-model-path", default=None,
                        help="Licensed SMPL model file (SOMA-JAX extra; default "
                             "<data-root>/SMPL/SMPL_NEUTRAL.{pkl,npz}).")
    add_logging_args(parser)
    add_npz_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")

    from soma_jax import SOMALayer
    from soma_jax.assets import data_root as default_data_root
    from soma_jax.fitting.pose_inversion import PoseInversion
    from soma_jax.smpl.layers import SMPLLayer

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    smpl_path = _smpl_model_path(data_root, args.smpl_model_path)

    # Models are created once and reused across sequences.
    smpl_layer = SMPLLayer(data_root, model_path=smpl_path)
    soma = SOMALayer.from_upstream_assets(
        identity_model_type="smpl", identity_model_kwargs={"model_path": str(smpl_path)},
        data_root=args.data_root)
    inv = PoseInversion(soma, low_lod=True)

    if args.input_dir:
        # --- Batch mode ---
        input_root = Path(args.input_dir)
        output_root = Path(args.output_dir)
        npz_files = sorted(input_root.rglob("*.npz"))
        if not npz_files:
            logger.warning(f"No .npz files found under {input_root}")
            return

        # Pre-filter for skip-existing so the progress bar reflects real work.
        num_skipped = 0
        if args.skip_existing:
            work_items = []
            for npz_path in npz_files:
                rel_path = npz_path.relative_to(input_root)
                out_path = output_root / rel_path
                if out_path.exists():
                    num_skipped += 1
                else:
                    work_items.append((npz_path, rel_path, out_path))
        else:
            work_items = [(p, p.relative_to(input_root), output_root / p.relative_to(input_root))
                          for p in npz_files]
        if args.shuffle:
            random.shuffle(work_items)

        logger.info(f"Found {len(npz_files)} .npz files under {input_root}"
                    f" ({num_skipped} already exist, {len(work_items)} to convert)")

        num_failed = 0
        num_converted = 0
        pbar = tqdm(work_items, unit="file", dynamic_ncols=True)
        for npz_path, rel_path, out_path in pbar:
            pbar.set_postfix_str(str(rel_path), refresh=False)
            # Re-check at processing time (another worker may have finished it).
            if args.skip_existing and out_path.exists():
                num_skipped += 1
                continue
            try:
                amass_data = load_amass_npz(npz_path)
                conv = convert_amass_sequence(amass_data, args, smpl_layer, inv)
                _save_conversion(conv, inv, args, str(out_path))
                num_converted += 1
            except Exception as e:
                logger.warning(f"FAILED {rel_path}: {e}")
                num_failed += 1

        logger.info(f"\nBatch complete: {num_converted} converted, {num_skipped} skipped, "
                    f"{num_failed} failed")
        return

    # --- Single file mode ---
    amass_data = load_amass_npz(args.input)
    conv = convert_amass_sequence(amass_data, args, smpl_layer, inv)
    if args.output_npz:
        _save_conversion(conv, inv, args, args.output_npz)
    if args.no_render:
        return

    # --- Render comparison video ---
    from vis_pyrender import (default_pyopengl_platform, look_at, render_comparison_video,
                              set_pyopengl_platform)
    set_pyopengl_platform(default_pyopengl_platform())

    rotations, root_transl = conv["rotations"], conv["root_transl"]
    betas, body_pose = conv["betas"], conv["body_pose"]
    global_orient, transl = conv["global_orient"], conv["transl"]
    rest, joints, binds = inv.soma.prepare_identity(betas[:1], return_bind_transforms=True)
    smpl_identity = smpl_layer.prepare_identity(betas[:1])

    num_frames = rotations.shape[0]
    batch_size = args.batch_size or num_frames
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

    output_video = (args.output_npz.replace(".npz", ".mp4") if args.output_npz
                    else "out/amass2soma.mp4")
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    logger.info("\nRendering comparison video...")
    # Per-frame centroid from the source mesh, applied to both so the overlay
    # stays aligned.
    smpl_verts_all = np.concatenate(smpl_verts_all, axis=0)
    soma_verts_all = np.concatenate(soma_verts_all, axis=0)
    centroids = smpl_verts_all.mean(axis=1, keepdims=True)
    smpl_verts_all = smpl_verts_all - centroids
    soma_verts_all = soma_verts_all - centroids

    vmin = smpl_verts_all[0].min(axis=0)
    vmax = smpl_verts_all[0].max(axis=0)
    center = (vmin + vmax) * 0.5
    extent = np.linalg.norm(vmax - vmin) + 1e-6
    eye = center - np.array([0.0, max(extent * 3.0, 1.0), 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_pose = look_at(eye, center, up)
    light_dir = np.array([0.0, 0.5, -0.5])

    render_comparison_video(
        output_video, smpl_verts_all, np.asarray(smpl_layer.faces), soma_verts_all,
        np.asarray(inv.soma.faces), label_source="SMPL (AMASS)", cam_pose=cam_pose,
        light_dir=light_dir, fps=int(amass_data["fps"]))


if __name__ == "__main__":
    main()
