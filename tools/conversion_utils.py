"""Shared utilities for the ``*2soma`` conversion tools.

Upstream: ``tools/conversion_utils.py`` (SOMA-X). Same argument sets and the
same layer-oriented NPZ export (absolute rotations -> joint orient removed ->
axis-angle -> ``save_soma_npz``), over SOMA-JAX layers.

One deliberate difference: rotations are converted with
:func:`soma_jax.geometry.transforms.rotmat_to_axis_angle`, whose small-angle
branch returns the correct vector; upstream's ``matrix_to_rotvec`` doubles
rotations below 1e-3 rad (see docs/FAITHFULNESS.md).
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


def _joint_orient_pair(layer):
    """Upstream ``layer._t_pose_orient`` / ``_t_pose_orient_parent_T``, on the public rig.

    Upstream's body layer computes that pair on its skinning rig — 110 joints
    on the default twist rig — which the 78 public rotations it is applied to
    cannot use; the SOMA body layer here supplies its public joints' orient.
    """
    from soma_jax.reference_poses import _orient_pair
    if hasattr(layer, "_t_pose_orient"):          # the hand layer keeps the pair
        return layer._t_pose_orient, layer._t_pose_orient_parent_T
    if hasattr(layer, "public_transform_joint_indices"):
        t_pose = np.asarray(layer.t_pose_world)[np.asarray(layer.public_transform_joint_indices)]
        return _orient_pair(t_pose, np.asarray(layer.public_joint_parent_ids))
    return _orient_pair(layer.t_pose_world, np.asarray(layer.joint_parent_ids))


def _joint_names(layer) -> list[str]:
    # The SOMA body layer's rotations cover its public joints; upstream labels
    # them with `rig_data["joint_names"]`, the 110 skinning joints on a twist rig.
    public = getattr(layer, "public_joint_names", None)
    if public is not None:
        return [str(n) for n in public]
    rig = getattr(layer, "rig_data", None)
    if rig is not None:
        return [str(n) for n in rig["joint_names"]]
    return [str(n) for n in layer.joint_names]


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

    from soma_jax.geometry.transforms import rotmat_to_axis_angle
    from soma_jax.io import save_soma_npz
    from soma_jax.reference_poses import _remove_orient
    from soma_jax.units import Unit

    rotations = jnp.asarray(rotations)
    orient, orient_parent_T = _joint_orient_pair(soma)
    rel = _remove_orient(rotations, jnp.asarray(orient), jnp.asarray(orient_parent_T))
    poses_rotvec = np.asarray(rotmat_to_axis_angle(rel.reshape(-1, 3, 3))).reshape(
        rotations.shape[0], rotations.shape[1], 3)

    layer_unit = getattr(soma, "output_unit", Unit.METERS)
    unit_scale = layer_unit.meters_per_unit / Unit.from_name(output_unit).meters_per_unit
    save_transl = np.asarray(root_transl, np.float32) * unit_scale

    save_soma_npz(
        str(output_path), poses_rotvec.astype(np.float32), save_transl,
        joint_names=_joint_names(soma),
        identity_model_type=soma.identity_model_type,
        identity_coeffs=identity_coeffs,
        scale_params=scale_params,
        joint_orient=np.asarray(orient),
        unit=output_unit,
        keep_root=keep_root,
        extra_arrays=extra_arrays,
    )
