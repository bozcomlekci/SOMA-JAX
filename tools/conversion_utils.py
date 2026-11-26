"""Shared utilities for the ``*2soma`` conversion tools.

Upstream: ``tools/conversion_utils.py`` (SOMA-X). Same argument sets and the
same layer-oriented NPZ export (absolute rotations -> joint orient removed ->
axis-angle -> ``save_soma_npz``), over SOMA-JAX layers.

Includes v0.3.6's :func:`resolve_joint_orient`, which sizes the joint orient
and names to the rotations being exported (the public rig on a procedural
layer), and its corrected ``matrix_to_rotvec``.
"""
from __future__ import annotations

import numpy as np


def add_inversion_args(parser, *, body_iters=2, finger_iters=0, full_iters=1, lie_iters=3,
                       lie_lambda=1e-1, batch_size=64, autograd=False):
    """Add standard PoseInversion parameters to an ArgumentParser.

    Defaults are model-specific — callers override via keyword arguments.
    ``autograd=True`` adds ``--autograd-iters`` / ``--autograd-lr`` /
    ``--autograd-translation-lr-scale``.
    """
    parser.add_argument("--body-iters", type=int, default=body_iters,
                        help=f"Analytical body iterations (default: {body_iters}).")
    parser.add_argument("--finger-iters", type=int, default=finger_iters,
                        help=f"Analytical finger iterations (default: {finger_iters}).")
    parser.add_argument("--full-iters", type=int, default=full_iters,
                        help=f"Analytical full iterations (default: {full_iters}).")
    parser.add_argument("--lie-iters", type=int, default=lie_iters,
                        help=f"Lie algebra Gauss-Newton iterations (default: {lie_iters}).")
    parser.add_argument("--lie-lambda", type=float, default=lie_lambda,
                        help=f"Tikhonov regularisation for Lie-GN (default: {lie_lambda}).")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help=f"Batch size for processing (default: {batch_size}).")
    if autograd:
        parser.add_argument("--autograd-iters", type=int, default=0,
                            help="Autograd FK optimization steps after analytical solve "
                                 "(default: 0 = off).")
        parser.add_argument("--autograd-lr", type=float, default=5e-3,
                            help="Autograd learning rate (default: 5e-3).")
        parser.add_argument("--autograd-translation-lr-scale", type=float, default=1.0,
                            help="Multiplier for the autograd root-translation learning rate "
                                 "(default: 1).")
    parser.add_argument("--device", default=None,
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--data-root", default=None,
                        help="Upstream-layout asset directory (default: soma_jax.assets).")


def add_hand_inversion_args(parser, *, bcd_iters=1, lie_iters=3, lie_lambda=1e-1, batch_size=64):
    """Add hand-specific PoseInversion parameters to an ArgumentParser.

    For hand-only pose inversion the body/finger/full split is unnecessary — a
    single ``--bcd-iters`` controls the analytical solve (mapped to
    ``full_iters``, with ``body_iters=0`` and ``finger_iters=0``).
    """
    parser.add_argument("--bcd-iters", type=int, default=bcd_iters,
                        help=f"BCD analytical iterations (default: {bcd_iters}).")
    parser.add_argument("--lie-iters", type=int, default=lie_iters,
                        help=f"Lie algebra Gauss-Newton iterations (default: {lie_iters}).")
    parser.add_argument("--lie-lambda", type=float, default=lie_lambda,
                        help=f"Tikhonov regularisation for Lie-GN (default: {lie_lambda}).")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help=f"Batch size for processing (default: {batch_size}).")
    parser.add_argument("--device", default=None,
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--data-root", default=None,
                        help="Upstream-layout asset directory (default: soma_jax.assets).")


def resolve_joint_orient(soma, num_joints):
    """Joint-orient arrays matching ``num_joints`` rotations from ``soma``.

    Upstream ``tools/conversion_utils.resolve_joint_orient`` (v0.3.6).
    Procedural layers skin an expanded target skeleton (110 joints) while
    ``PoseInversion.fit`` and ``SOMALayer.pose`` operate on the public skeleton
    (78 joints), so the skinning rig's orient cannot be applied to public
    rotations.

    Returns:
        ``(orient, orient_parent_T, joint_names)`` sized to ``num_joints``.
    """
    orient = getattr(soma, "_t_pose_orient", None)
    if orient is not None and orient.shape[0] == num_joints:
        return orient, soma._t_pose_orient_parent_T, [str(n) for n in soma.rig_data["joint_names"]]
    public_names = [str(n) for n in getattr(soma, "public_joint_names", ())]
    if len(public_names) == num_joints and hasattr(soma, "public_joint_orient"):
        orient, orient_parent_T = soma.public_joint_orient()
        return orient, orient_parent_T, public_names
    raise ValueError(
        "Cannot resolve joint orient for rotations with "
        f"{num_joints} joints: layer target skeleton has "
        f"{None if orient is None else orient.shape[0]} joints and public skeleton has "
        f"{len(public_names)}.")


def export_soma_npz(output_path, rotations, root_transl, soma, *, output_unit, keep_root=False,
                    identity_coeffs=None, scale_params=None, extra_arrays=None):
    """Layer-oriented NPZ export, parallel to :func:`soma_jax.usd_io.export_soma_usd`.

    Converts absolute rotation matrices -> relative (joint orient removed) ->
    axis-angle, applies the unit conversion, and delegates to
    :func:`soma_jax.io.save_soma_npz`.

    Args:
        output_path: destination ``.npz`` path.
        rotations: (N, J, 3, 3) absolute rotation matrices.
        root_transl: (N, 3) root translation in the layer's output unit
            (metres for :class:`soma_jax.SOMALayer`).
        soma: the layer (joint orient, names, identity backend).
        output_unit: target unit name (e.g. ``"meters"``).
        keep_root: include the virtual root joint in the output.
        identity_coeffs, scale_params: identity data to store.
        extra_arrays: additional arrays to store.
    """
    import jax.numpy as jnp

    from soma_jax.geometry.rig_utils import remove_joint_orient_local
    from soma_jax.geometry.transforms import matrix_to_rotvec
    from soma_jax.io import save_soma_npz
    from soma_jax.units import Unit

    rotations = jnp.asarray(rotations)
    orient, orient_parent_T, joint_names = resolve_joint_orient(soma, rotations.shape[1])
    orient = jnp.asarray(orient, rotations.dtype)
    orient_parent_T = jnp.asarray(orient_parent_T, rotations.dtype)
    rel = remove_joint_orient_local(rotations, orient, orient_parent_T)
    poses_rotvec = np.asarray(matrix_to_rotvec(rel.reshape(-1, 3, 3))).reshape(
        rotations.shape[0], rotations.shape[1], 3)

    layer_unit = getattr(soma, "output_unit", Unit.METERS)
    unit_scale = layer_unit.meters_per_unit / Unit.from_name(output_unit).meters_per_unit
    save_transl = np.asarray(root_transl, np.float32) * unit_scale

    save_soma_npz(
        str(output_path), poses_rotvec.astype(np.float32), save_transl,
        joint_names=joint_names,
        identity_model_type=soma.identity_model_type,
        identity_coeffs=identity_coeffs,
        scale_params=scale_params,
        joint_orient=np.asarray(orient),
        unit=output_unit,
        keep_root=keep_root,
        extra_arrays=extra_arrays,
    )
