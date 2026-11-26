"""Native-MHR pose inversion: recover MHR pose/model parameters from MHR vertices.

Upstream: ``soma/fitting/pose_inversion_mhr.py`` (SOMA-X v0.3.3), exported as
``soma.fitting.MHRPoseInversion``. Upstream calls the module private: it keeps
the MHR-specific DOF handling, the co-located ankle split and the
parameter-matrix projection apart from the public SOMA conversion code.

The pipeline: :class:`~soma_jax.geometry.skeleton_transfer.SkeletonTransfer`
fits world transforms to the posed MHR mesh; the inverse-LBS refit of
:mod:`soma_jax.fitting.pose_inversion` refines them in MHR's named joint groups,
with each pass projected onto the joints' active Euler DOFs; the local
transforms are converted to the parameter-transform DOF vector and projected
onto the 136 pose parameters by a ridge pseudo-inverse; an optional Adam stage
refines those parameters through the MHR forward
(:class:`~soma_jax.body_models.mhr_native.MHRNativeModel`, a pure-JAX
evaluation of ``mhr_model_lod1.pt``).

Assets: ``<data_root>/MHR/MHR_base_rig.npz`` and
``<data_root>/MHR/parameter_transform.npz`` are not part of upstream's public
asset set (neither its git LFS files nor the Hugging Face release ship them),
so the constructor raises ``FileNotFoundError`` without them — exactly as
upstream does, whose own tests skip in that case.

Differences from upstream, none of which change results:

* Upstream mutates ``pose_local`` in place; every step here returns the
  updated array.
* ``device`` is accepted for signature compatibility; arrays live on JAX's
  default device (choose it with ``jax.default_device``).
* ``torch.optim.Adam`` is ``optax.adam`` with the same defaults.
"""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..geometry.rig_utils import joint_world_to_local
from ..geometry.skeleton_transfer import SkeletonTransfer
from ..geometry.transforms import (
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    quaternion_xyzw_to_matrix,
    se3_from_rt,
    se3_inverse,
)
from .pose_inversion import (
    PoseInversionResult,
    _bexpand,
    _bexpand4,
    _build_world_transforms,
    _precompute_refit_cache,
    _refit_joint,
    _run_refit_passes,
    _skin,
    _to_sparse_weights,
    _update_root_translation,
)

logger = logging.getLogger(__name__)

__all__ = ["MHRPoseInversion", "MHRPoseInversionResult"]

_BODY_PASS_GROUPS = [
    ["root"],
    ["c_spine0", "l_upleg", "r_upleg"],
    ["c_spine1", "l_lowleg", "r_lowleg"],
    ["c_spine2", "l_foot", "r_foot"],
    ["c_spine3", "l_talocrural", "r_talocrural"],
    ["l_clavicle", "r_clavicle", "c_neck", "l_subtalar", "r_subtalar"],
    ["l_uparm", "r_uparm", "c_head", "l_transversetarsal", "r_transversetarsal"],
    ["l_lowarm", "r_lowarm", "l_ball", "r_ball"],
    ["l_wrist_twist", "r_wrist_twist"],
    ["l_wrist", "r_wrist"],
]

_FINGER_PASS_GROUPS = [
    ["l_thumb0", "r_thumb0", "l_pinky0", "r_pinky0"],
    ["l_index1", "r_index1", "l_middle1", "r_middle1", "l_ring1", "r_ring1",
     "l_pinky1", "r_pinky1", "l_thumb1", "r_thumb1"],
    ["l_index2", "r_index2", "l_middle2", "r_middle2", "l_ring2", "r_ring2",
     "l_pinky2", "r_pinky2", "l_thumb2", "r_thumb2"],
    ["l_index3", "r_index3", "l_middle3", "r_middle3", "l_ring3", "r_ring3",
     "l_pinky3", "r_pinky3", "l_thumb3", "r_thumb3"],
]

_COLOCATED_ANKLE_PAIRS = (("l_foot", "l_talocrural"), ("r_foot", "r_talocrural"))
_FLEXIBLE_SLICE = slice(130, 136)
_DISABLED_POSE_PARAM_IDS = (
    6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
)
_ACTIVE_SPINE_PARAM_BOUNDS = {
    7: (-0.9, 0.9),
    9: (-0.7, 0.7),
    11: (-0.5, 1.5),
    13: (-0.9, 0.9),
    15: (-0.7, 0.7),
    17: (-0.5, 1.5),
}
_REDUCED_DOF_REFIT_ITERS = 8
_REDUCED_DOF_REFIT_LR = 5e-2


class MHRPoseInversionResult(dict):
    """Dictionary result with attribute access for native-MHR inversion."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _joint_names_from_dof_names(joint_dof_names):
    n_joints = len(joint_dof_names) // 7
    return [str(joint_dof_names[j * 7]).split(".", 1)[0] for j in range(n_joints)]


def _mhr_skeleton_state_to_transforms(skel_state):
    """MHR skeleton-state rows ``(t, q_xyzw, s)`` to world transforms (scale in R)."""
    R = quaternion_xyzw_to_matrix(skel_state[..., 3:7])
    R = R * skel_state[..., 7, None, None]
    return se3_from_rt(R, skel_state[..., :3])


def _load_parameter_transform(dtype, npz_path, ridge_lambda=1e-7):
    """Read ``parameter_transform.npz`` and build the pose-DOF projection.

    The ridge pseudo-inverse is solved in float64 on the host, as upstream
    solves it in float64 before casting.
    """
    with np.load(npz_path, allow_pickle=False) as full_data:
        parameter_transform_np = full_data["parameter_transform"]
        parameter_names = [str(x) for x in full_data["parameter_names"]]
        joint_dof_names = [str(x) for x in full_data["joint_dof_names"]]
        pre_rotations = full_data["pre_rotations"]

    joint_names = _joint_names_from_dof_names(joint_dof_names)
    n_joints = len(joint_names)
    root_idx = joint_names.index("root")
    main_joint_mask = np.asarray([not name.endswith("_proc") for name in joint_names])
    num_main = int(main_joint_mask.sum())
    joint_orients = jnp.asarray(pre_rotations, dtype=dtype)

    pose_skip = np.asarray([name.startswith("scale_") for name in parameter_names[:204]])
    pose_indices = np.where(~pose_skip)[0]

    root_dofs = root_idx * 7 + np.arange(6)
    other_j = np.concatenate([np.arange(root_idx), np.arange(root_idx + 1, n_joints)])
    other_dofs = (other_j[:, None] * 7 + np.asarray([3, 4, 5])).reshape(-1)
    dof_indices = np.concatenate([root_dofs, other_dofs])

    non_root_main = np.concatenate([main_joint_mask[:root_idx], main_joint_mask[root_idx + 1:]])
    dof_mask = np.concatenate([np.ones(6, dtype=bool), np.repeat(non_root_main, 3)])

    P_full = np.asarray(parameter_transform_np, dtype=np.float64)
    P_pose_main = P_full[dof_indices][:, pose_indices][dof_mask]
    n_pose = P_pose_main.shape[1]
    AtA_lam = P_pose_main.T @ P_pose_main + ridge_lambda * np.eye(n_pose)
    P_inv_pose_main = np.linalg.solve(AtA_lam, P_pose_main.T).T

    pose_names = [name for name in parameter_names[:204] if not name.startswith("scale_")]
    return {
        "joint_names": joint_names,
        "joint_orients": joint_orients,
        "main_joint_mask": main_joint_mask,
        "num_main": num_main,
        "pose_indices": pose_indices,
        "pose_names": pose_names,
        "P_pose_main": jnp.asarray(P_pose_main, dtype=dtype),
        "P_inv_pose_main": jnp.asarray(P_inv_pose_main, dtype=dtype),
        "dof_mask": dof_mask,
    }


def _get_dof_masks(pt_data):
    """Local Euler-axis masks for the MHR joints with fewer than 3 active DOFs."""
    joint_names = pt_data["joint_names"]
    main_joint_mask = pt_data["main_joint_mask"]
    root_idx = joint_names.index("root")
    non_root_main = [
        j for j in range(len(joint_names)) if j != root_idx and bool(main_joint_mask[j])
    ]

    masks = {}
    P = np.asarray(pt_data["P_pose_main"])
    for k, j_idx in enumerate(non_root_main):
        row_start = 6 + 3 * k
        active = tuple(bool(np.abs(P[row_start + d]).max() > 1e-10) for d in range(3))
        if sum(active) < 3:
            masks[joint_names[j_idx]] = active
    return masks


def _groups_to_indices(groups, joint_names):
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    out = []
    for group in groups:
        idxs = [name_to_idx[name] for name in group if name in name_to_idx]
        if idxs:
            out.append(idxs)
    return out


def _as_batched_tensor(value, shape, batch_size, dtype, name):
    if value is None:
        return jnp.zeros((batch_size, *shape), dtype=dtype)
    tensor = jnp.asarray(value, dtype=dtype)
    if tensor.shape == shape:
        tensor = tensor[None]
    if tensor.ndim == len(shape) + 1 and tensor.shape[0] == 1 and batch_size > 1:
        tensor = jnp.broadcast_to(tensor, (batch_size, *shape))
    if tensor.shape != (batch_size, *shape):
        raise ValueError(f"{name} must have shape {(batch_size, *shape)} or {shape}.")
    return tensor


def _constrain_dof_local(pose_local, joint_names, pt_data, dof_masks):
    """Zero the inactive local Euler axes of the reduced-DOF joints.

    Each joint only reads and writes its own rotation, so upstream's per-joint
    loop runs here as one batched update.
    """
    pt_joint_names = pt_data["joint_names"]
    data_ids, pt_ids, active = [], [], []
    for jname, axes in dof_masks.items():
        if jname not in joint_names or jname not in pt_joint_names:
            continue
        data_ids.append(joint_names.index(jname))
        pt_ids.append(pt_joint_names.index(jname))
        active.append(axes)
    if not data_ids:
        return pose_local

    data_ids = np.asarray(data_ids)
    R_orient = pt_data["joint_orients"][np.asarray(pt_ids)]                 # (K, 3, 3)
    R_local = jnp.swapaxes(R_orient, -2, -1)[None] @ pose_local[:, data_ids, :3, :3]
    euler = matrix_to_euler_xyz(R_local)
    constrained = jnp.where(jnp.asarray(active)[None], euler, 0.0)
    return pose_local.at[:, data_ids, :3, :3].set(
        R_orient[None] @ euler_xyz_to_matrix(constrained))


def _torch_style_clamp(x, lo, hi):
    """``torch.clamp``: its gradient passes wherever ``lo <= x <= hi``.

    ``jnp.clip`` blocks the gradient on the bounds themselves, where a clamped
    parameter sits after every optimizer step.
    """
    return jnp.where((x >= lo) & (x <= hi), x, jnp.clip(x, lo, hi))


def _refit_reduced_dof_joint(
    pose_local,
    target,
    j_idx,
    W,
    D,
    cache,
    jcache,
    pt_data,
    active_axes,
    *,
    iters,
    lr,
):
    """Re-fit one MHR joint directly in its active local Euler axes (Adam)."""
    joint_name = cache["joint_names"][j_idx]
    pt_joint_names = pt_data["joint_names"]
    if joint_name not in pt_joint_names:
        return pose_local

    pt_idx = pt_joint_names.index(joint_name)
    R_orient = pt_data["joint_orients"][pt_idx]
    B = pose_local.shape[0]
    active_ids = np.asarray([axis for axis, is_active in enumerate(active_axes) if is_active])

    if len(active_ids) == 0:
        return pose_local.at[:, j_idx, :3, :3].set(jnp.broadcast_to(R_orient, (B, 3, 3)))

    arm_vids = jcache["arm_vids"]
    sw = jcache["sub_weight_sum"].reshape(1, -1, 1)
    bv = _bexpand(jcache["bind_verts_arm"], B)
    q_world = _skin(bv, jcache["sub_bone_weights"], jcache["sub_bone_indices"], D)
    c_xyz = _skin(bv, jcache["non_bone_weights"], jcache["non_bone_indices"], D)

    W_p_inv = se3_inverse(W[:, j_idx])
    R_inv = W_p_inv[:, :3, :3]
    t_inv = W_p_inv[:, :3, 3]
    src = q_world @ jnp.swapaxes(R_inv, -2, -1) + t_inv[:, None, :] * sw
    tgt = target[:, arm_vids, :] - c_xyz - W[:, j_idx, :3, 3][:, None, :] * sw

    parent_idx = int(cache["parents"][j_idx])
    parent_world = None if parent_idx == j_idx or parent_idx < 0 else W[:, parent_idx, :3, :3]

    current_euler = matrix_to_euler_xyz(R_orient.T[None] @ pose_local[:, j_idx, :3, :3])

    def to_rotation(params):
        euler = jnp.zeros_like(current_euler).at[:, active_ids].set(params)
        return R_orient[None] @ euler_xyz_to_matrix(euler)

    def loss_fn(params):
        R_world = to_rotation(params)
        if parent_world is not None:
            R_world = parent_world @ R_world
        pred = src @ jnp.swapaxes(R_world, -2, -1)
        return jnp.mean(jnp.square(pred - tgt))

    optimizer = optax.adam(lr)
    params = current_euler[:, active_ids]
    state = optimizer.init(params)
    best_loss = float("inf")
    best_params = params
    value_and_grad = jax.value_and_grad(loss_fn)
    for _ in range(iters):
        loss, grad = value_and_grad(params)
        cur_loss = float(loss)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_params = params
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)

    return pose_local.at[:, j_idx, :3, :3].set(to_rotation(best_params))


def _redistribute_colocated_transforms(pose_local, joint_names, pt_data):
    """Split the co-located MHR ankle rotations between foot and talocrural joints."""
    joint_orients = pt_data["joint_orients"]
    pt_joint_names = pt_data["joint_names"]
    B = pose_local.shape[0]
    zeros = jnp.zeros((B,), dtype=pose_local.dtype)

    for foot_name, talo_name in _COLOCATED_ANKLE_PAIRS:
        if foot_name not in joint_names or talo_name not in joint_names:
            continue
        foot_idx = joint_names.index(foot_name)
        talo_idx = joint_names.index(talo_name)
        R_orient_foot = joint_orients[pt_joint_names.index(foot_name)][None]
        R_orient_talo = joint_orients[pt_joint_names.index(talo_name)][None]

        R_combined = pose_local[:, foot_idx, :3, :3] @ pose_local[:, talo_idx, :3, :3]
        R_c = jnp.swapaxes(R_orient_foot, -2, -1) @ R_combined
        rz_t = jnp.arctan2(R_c[:, 1, 0], R_c[:, 1, 1])
        B_mat = euler_xyz_to_matrix(jnp.stack([zeros, zeros, rz_t], axis=-1))
        M = R_orient_talo @ B_mat
        A = R_c @ jnp.swapaxes(M, -2, -1)
        ry_f = jnp.arcsin(jnp.clip(-A[:, 2, 0], -1.0, 1.0))
        rx_f = jnp.arctan2(A[:, 2, 1], A[:, 2, 2])

        pose_local = pose_local.at[:, foot_idx, :3, :3].set(
            R_orient_foot @ euler_xyz_to_matrix(jnp.stack([rx_f, ry_f, zeros], axis=-1)))
        pose_local = pose_local.at[:, talo_idx, :3, :3].set(
            R_orient_talo @ euler_xyz_to_matrix(jnp.stack([zeros, zeros, rz_t], axis=-1)))
    return pose_local


def _main_joint_layout(pt_data, data_joint_names):
    """``(pt_ids, data_ids, dof_columns)`` of the non-root main joints present in the data."""
    pt_joint_names = pt_data["joint_names"]
    root_idx_pt = pt_joint_names.index("root")
    non_root_main = [j for j, is_main in enumerate(pt_data["main_joint_mask"])
                     if is_main and j != root_idx_pt]
    pt_ids, data_ids, columns = [], [], []
    for k, pt_idx in enumerate(non_root_main):
        name = pt_joint_names[pt_idx]
        if name not in data_joint_names:
            continue
        pt_ids.append(pt_idx)
        data_ids.append(data_joint_names.index(name))
        columns.append(6 + 3 * k + np.arange(3))
    return (np.asarray(pt_ids, dtype=np.int64), np.asarray(data_ids, dtype=np.int64),
            np.asarray(columns, dtype=np.int64).reshape(-1, 3))


def _transforms_to_dofs(transforms, data_joint_names, pt_data, root_bind_translation=None):
    """Local MHR transforms to the parameter-transform DOF vector."""
    pt_joint_names = pt_data["joint_names"]
    joint_orients = pt_data["joint_orients"]
    root_idx_pt = pt_joint_names.index("root")
    num_main = int(np.asarray(pt_data["main_joint_mask"]).sum())

    B = transforms.shape[0]
    dofs = jnp.zeros((B, 3 + num_main * 3), dtype=transforms.dtype)

    root_data_idx = data_joint_names.index("root")
    root_trans = transforms[:, root_data_idx, :3, 3]
    if root_bind_translation is not None:
        root_trans = root_trans - root_bind_translation
    dofs = dofs.at[:, :3].set(root_trans)
    dofs = dofs.at[:, 3:6].set(matrix_to_euler_xyz(
        joint_orients[root_idx_pt].T[None] @ transforms[:, root_data_idx, :3, :3]))

    pt_ids, data_ids, columns = _main_joint_layout(pt_data, data_joint_names)
    if len(pt_ids):
        euler = matrix_to_euler_xyz(
            jnp.swapaxes(joint_orients[pt_ids], -2, -1)[None] @ transforms[:, data_ids, :3, :3])
        dofs = dofs.at[:, columns].set(euler)
    return dofs


def _disabled_pose_param_ids():
    return np.asarray(_DISABLED_POSE_PARAM_IDS, dtype=np.int64)


def _clamp_active_spine(pose_params, clamp=jnp.clip):
    for param_idx, (lo, hi) in _ACTIVE_SPINE_PARAM_BOUNDS.items():
        pose_params = pose_params.at[:, param_idx].set(clamp(pose_params[:, param_idx], lo, hi))
    return pose_params


def _constrain_refined_pose(pose, fixed_flex, fixed_disabled, clamp, *, optimize_flexibles,
                            freeze_disabled_pose_params, bound_active_spine):
    """Upstream's refinement constraints: re-impose the frozen params and spine bounds."""
    if not optimize_flexibles:
        if freeze_disabled_pose_params:
            pose = pose.at[:, _disabled_pose_param_ids()].set(fixed_disabled)
        else:
            pose = pose.at[:, _FLEXIBLE_SLICE].set(fixed_flex)
    if bound_active_spine:
        pose = _clamp_active_spine(pose, clamp)
    return pose


@partial(jax.jit, static_argnames=("model", "flags", "apply_correctives"))
def _refine_step(pose, state, lr, target, identity_coeffs, scale_params, face_expr_coeffs,
                 fixed_flex, fixed_disabled, *, model, flags, apply_correctives):
    """One Adam step of the MHR refinement — compiled once per model and flag set.

    Returns the updated pose and optimizer state, the loss before the step and
    the pose the model saw (what upstream keeps as the best pose).
    """
    optimize_flexibles, freeze_disabled_pose_params, bound_active_spine = flags
    constraints = dict(optimize_flexibles=optimize_flexibles,
                       freeze_disabled_pose_params=freeze_disabled_pose_params,
                       bound_active_spine=bound_active_spine)

    def loss_fn(pose):
        # What upstream feeds the model: the frozen entries carry no gradient,
        # and the spine bounds clamp like `torch.clamp`.
        pose_for_model = _constrain_refined_pose(
            pose, fixed_flex, fixed_disabled, _torch_style_clamp, **constraints)
        pred, _ = model(identity_coeffs,
                        jnp.concatenate([pose_for_model, scale_params], axis=1),
                        face_expr_coeffs, apply_correctives)
        return jnp.mean(jnp.square(pred - target)), pose_for_model

    optimizer = optax.adam(lr)
    (loss, pose_for_model), grad = jax.value_and_grad(loss_fn, has_aux=True)(pose)
    updates, state = optimizer.update(grad, state, pose)
    # Upstream re-imposes the constraints on the optimized tensor too.
    pose = _constrain_refined_pose(optax.apply_updates(pose, updates), fixed_flex,
                                   fixed_disabled, jnp.clip, **constraints)
    return pose, state, loss, pose_for_model


def _apply_pose_param_constraints(
    pose_params,
    *,
    freeze_disabled_pose_params,
    bound_active_spine,
):
    if freeze_disabled_pose_params:
        pose_params = pose_params.at[:, _disabled_pose_param_ids()].set(0.0)
    else:
        pose_params = pose_params.at[:, _FLEXIBLE_SLICE].set(0.0)
    if bound_active_spine:
        pose_params = _clamp_active_spine(pose_params)
    return pose_params


def _pose_local_from_result(result, B, J, root_idx, dtype):
    """Rotations and root translation in otherwise-zero 4x4 blocks, as upstream."""
    pose_local = jnp.zeros((B, J, 4, 4), dtype=dtype)
    pose_local = pose_local.at[:, :, :3, :3].set(result["rotations"])
    return pose_local.at[:, root_idx, :3, 3].set(result["root_translation"])


def _reconstruct_from_local_pose(pose_local, cache, bind_shape, bone_weights, bone_indices):
    B = pose_local.shape[0]
    W = _build_world_transforms(pose_local, cache)
    D = W @ _bexpand4(cache["W_bind_inv"], B)
    return _skin(_bexpand(bind_shape, B), bone_weights, bone_indices, D)


def _result_from_local_pose(
    pose_local, target, cache, bind_shape, bone_weights, bone_indices, root_idx
):
    vertices = _reconstruct_from_local_pose(
        pose_local, cache, bind_shape, bone_weights, bone_indices)
    return PoseInversionResult(
        rotations=pose_local[:, :, :3, :3],
        root_translation=pose_local[:, root_idx, :3, 3],
        per_vertex_error=jnp.linalg.norm(vertices - target, axis=-1),
    )


def _concat_results(chunks):
    """Upstream's merge of chunked :meth:`MHRPoseInversion.fit` results."""
    cat = lambda key: jnp.concatenate([c[key] for c in chunks], axis=0)  # noqa: E731
    return MHRPoseInversionResult({
        "pose_params": cat("pose_params"),
        "model_params": cat("model_params"),
        "init_pose_params": cat("init_pose_params"),
        "per_vertex_error": cat("per_vertex_error"),
        "pre_refine_per_vertex_error": cat("pre_refine_per_vertex_error"),
        "skeletal_per_vertex_error": cat("skeletal_per_vertex_error"),
        "pose_local": cat("pose_local"),
        "skeletal_pose_local": cat("skeletal_pose_local"),
        "loss_history": [c["loss_history"] for c in chunks],
        "iters_run": sum(c["iters_run"] for c in chunks),
    })


class MHRPoseInversion:
    """Native-MHR pose inversion helper (upstream ``MHRPoseInversion``).

    Args:
        data_root: asset root holding ``MHR/MHR_base_rig.npz``,
            ``MHR/parameter_transform.npz`` and ``MHR/mhr_model_lod1.pt``.
        device: accepted for upstream signature compatibility; JAX places
            arrays on its default device.
        dtype: floating dtype of the solve.
        use_warp_for_rotations: forwarded to :class:`SkeletonTransfer`
            (recorded there; the JAX rotation fit has no Warp switch).
        skeleton_transfer_rotation_method: rotation extraction of the
            skeleton transfer (``"auto"``, ``"kabsch"``, ``"newton-schulz"``).
        refit_rotation_method: rotation extraction of the inverse-LBS refit.
        use_reduced_dof_refit: refit the reduced-DOF joints with Adam directly
            in their active Euler axes instead of Procrustes + projection.
        reduced_dof_refit_iters, reduced_dof_refit_lr: that Adam's settings.
        use_identity_reference_for_skeleton: fit the skeleton against the
            identity's own reference state (one frame at a time).
    """

    def __init__(
        self,
        data_root: str | Path,
        device: Any = None,
        dtype=jnp.float32,
        use_warp_for_rotations: bool = True,
        skeleton_transfer_rotation_method: str = "auto",
        refit_rotation_method: str = "auto",
        use_reduced_dof_refit: bool = False,
        reduced_dof_refit_iters: int = _REDUCED_DOF_REFIT_ITERS,
        reduced_dof_refit_lr: float = _REDUCED_DOF_REFIT_LR,
        use_identity_reference_for_skeleton: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.device = device
        self.dtype = jnp.dtype(dtype)
        self.use_warp_for_rotations = use_warp_for_rotations
        self.skeleton_transfer_rotation_method = skeleton_transfer_rotation_method
        self.refit_rotation_method = refit_rotation_method
        self.use_reduced_dof_refit = use_reduced_dof_refit
        self.reduced_dof_refit_iters = int(reduced_dof_refit_iters)
        self.reduced_dof_refit_lr = float(reduced_dof_refit_lr)
        self.use_identity_reference_for_skeleton = use_identity_reference_for_skeleton

        mhr_root = self.data_root / "MHR"
        rig_path = mhr_root / "MHR_base_rig.npz"
        pt_path = mhr_root / "parameter_transform.npz"
        if not rig_path.is_file():
            raise FileNotFoundError(f"Missing MHR rig asset: {rig_path}")
        if not pt_path.is_file():
            raise FileNotFoundError(f"Missing MHR parameter transform asset: {pt_path}")

        with np.load(rig_path, allow_pickle=False) as rig_data:
            self.joint_names = [str(x) for x in rig_data["joint_names"]]
            self.joint_parent_ids = np.asarray(rig_data["joint_parent_ids"])
            self.bind_world = jnp.asarray(rig_data["bind_pose_world"], dtype=self.dtype)
            self.bind_local = jnp.asarray(rig_data["bind_pose_local"], dtype=self.dtype)
            self.bind_shape = jnp.asarray(rig_data["bind_shape"], dtype=self.dtype)
            skinning_weights = np.asarray(rig_data["skinning_weights"], dtype=np.float32)
        self.skinning_weights = jnp.asarray(skinning_weights, dtype=self.dtype)

        self.root_joint_idx = self.joint_names.index("root")
        self.root_bind_translation = self.bind_local[self.root_joint_idx, :3, 3]
        self.pt_data = _load_parameter_transform(self.dtype, pt_path)
        self.dof_masks = _get_dof_masks(self.pt_data)

        self.skel_transfer = self._build_skeleton_transfer(self.bind_world, self.bind_shape)
        self.cache = self._build_refit_cache(self.bind_world, self.bind_shape)

        max_k = int((skinning_weights > 1e-6).sum(axis=1).max())
        bone_weights, bone_indices = _to_sparse_weights(skinning_weights, max_k)
        self.bone_weights = jnp.asarray(bone_weights, dtype=self.dtype)
        self.bone_indices = jnp.asarray(bone_indices)
        self.model_state_joint_indices = np.asarray(
            [self.pt_data["joint_names"].index(name) for name in self.joint_names],
            dtype=np.int64)
        self._mhr_model = None

    def _load_mhr_model(self):
        if self._mhr_model is None:
            from ..body_models.mhr_native import MHRNativeModel
            self._mhr_model = MHRNativeModel.from_torchscript(
                self.data_root / "MHR" / "mhr_model_lod1.pt")
        return self._mhr_model

    def _build_skeleton_transfer(self, bind_world, bind_shape):
        return SkeletonTransfer(
            self.joint_parent_ids,
            np.asarray(bind_world),
            np.asarray(bind_shape),
            np.asarray(self.skinning_weights),
            rotation_method=self.skeleton_transfer_rotation_method,
            root_joint_idx=self.root_joint_idx,
            use_warp_for_rotations=self.use_warp_for_rotations,
        )

    def _build_refit_cache(self, bind_world, bind_shape):
        cache = _precompute_refit_cache(
            self.joint_names,
            self.joint_parent_ids,
            np.asarray(bind_world),
            np.asarray(bind_shape),
            np.asarray(self.skinning_weights),
            np.asarray(bind_world),
            root_idx=self.root_joint_idx,
        )
        cache["root_idx"] = self.root_joint_idx
        cache["body_groups"] = _groups_to_indices(_BODY_PASS_GROUPS, self.joint_names)
        cache["finger_groups"] = _groups_to_indices(_FINGER_PASS_GROUPS, self.joint_names)
        cache["constrained_set"] = set()
        cache["constrained_data"] = None
        return cache

    def _set_bind_state(self, bind_world, bind_shape):
        self.bind_world = bind_world
        self.bind_local = joint_world_to_local(bind_world, self.joint_parent_ids)
        self.bind_shape = bind_shape
        self.root_bind_translation = self.bind_local[self.root_joint_idx, :3, 3]
        self.skel_transfer = self._build_skeleton_transfer(bind_world, bind_shape)
        self.cache = self._build_refit_cache(bind_world, bind_shape)

    def _bind_state(self):
        return {
            "bind_world": self.bind_world,
            "bind_local": self.bind_local,
            "bind_shape": self.bind_shape,
            "root_bind_translation": self.root_bind_translation,
            "skel_transfer": self.skel_transfer,
            "cache": self.cache,
        }

    def _restore_bind_state(self, state):
        self.bind_world = state["bind_world"]
        self.bind_local = state["bind_local"]
        self.bind_shape = state["bind_shape"]
        self.root_bind_translation = state["root_bind_translation"]
        self.skel_transfer = state["skel_transfer"]
        self.cache = state["cache"]

    def _identity_reference_state(
        self,
        identity_coeffs,
        scale_params,
        face_expr_coeffs,
        reference_pose_params,
        *,
        apply_correctives,
    ):
        model = self._load_mhr_model()
        model_params = jnp.concatenate([reference_pose_params, scale_params], axis=1)
        bind_shape, skel_state = model(
            identity_coeffs, model_params, face_expr_coeffs, apply_correctives)
        bind_world = _mhr_skeleton_state_to_transforms(skel_state)[
            :, self.model_state_joint_indices]
        return bind_world, bind_shape

    def _fit_skeletal_and_project(
        self,
        target,
        *,
        body_iters,
        finger_iters,
        full_iters,
        identity_coeffs,
        scale_params,
        face_expr_coeffs,
        identity_reference_pose_params,
        apply_correctives,
    ):
        if not self.use_identity_reference_for_skeleton:
            skeletal_result = self.fit_skeletal_transforms(
                target, body_iters=body_iters, finger_iters=finger_iters,
                full_iters=full_iters)
            init_pose_params, skeletal_pose_local = self._skeletal_to_pose_params(
                skeletal_result)
            return skeletal_result, init_pose_params, skeletal_pose_local

        if target.shape[0] != 1:
            raise ValueError("Identity-conditioned MHR skeleton fitting expects one frame.")

        bind_world, bind_shape = self._identity_reference_state(
            identity_coeffs,
            scale_params,
            face_expr_coeffs,
            identity_reference_pose_params,
            apply_correctives=apply_correctives,
        )
        saved_state = self._bind_state()
        try:
            self._set_bind_state(bind_world[0], bind_shape[0])
            skeletal_result = self.fit_skeletal_transforms(
                target, body_iters=body_iters, finger_iters=finger_iters,
                full_iters=full_iters)
            init_pose_params, skeletal_pose_local = self._skeletal_to_pose_params(
                skeletal_result)
        finally:
            self._restore_bind_state(saved_state)
        return skeletal_result, init_pose_params, skeletal_pose_local

    def _constrain_mhr_local(self, pose_local, group=None):
        if group is None:
            pose_local = _redistribute_colocated_transforms(
                pose_local, self.joint_names, self.pt_data)
            return _constrain_dof_local(pose_local, self.joint_names, self.pt_data,
                                        self.dof_masks)

        group_names = [self.joint_names[int(idx)] for idx in group]
        limited = {name: self.dof_masks[name] for name in group_names if name in self.dof_masks}
        if limited:
            pose_local = _constrain_dof_local(pose_local, self.joint_names, self.pt_data,
                                              limited)
        return pose_local

    def _finish_body_or_full_pass(self, pose_local):
        pose_local = _redistribute_colocated_transforms(
            pose_local, self.joint_names, self.pt_data)
        return _constrain_dof_local(pose_local, self.joint_names, self.pt_data, self.dof_masks)

    def _run_mhr_refit_groups(self, pose_local, target, groups):
        if self.use_reduced_dof_refit and self.reduced_dof_refit_iters > 0:
            joint_cache = self.cache["joint_cache"]
            B = pose_local.shape[0]
            for group in groups:
                W = _build_world_transforms(pose_local, self.cache)
                D = W @ _bexpand4(self.cache["W_bind_inv"], B)
                for j_idx in group:
                    jcache = joint_cache.get(j_idx)
                    if jcache is None:
                        continue
                    active_axes = self.dof_masks.get(self.joint_names[int(j_idx)])
                    if active_axes is None:
                        pose_local = _refit_joint(
                            pose_local, target, int(j_idx), W, D, self.cache, jcache,
                            None, self.refit_rotation_method)
                    else:
                        pose_local = _refit_reduced_dof_joint(
                            pose_local, target, int(j_idx), W, D, self.cache, jcache,
                            self.pt_data, active_axes,
                            iters=self.reduced_dof_refit_iters,
                            lr=self.reduced_dof_refit_lr)
                pose_local = self._constrain_mhr_local(pose_local, group)
            return pose_local

        for group in groups:
            pose_local = _run_refit_passes(
                pose_local, target, self.cache, [group], False, None,
                self.refit_rotation_method)
            pose_local = self._constrain_mhr_local(pose_local, group)
        return pose_local

    def _skeletal_to_pose_params(self, skeletal_result):
        B = skeletal_result["rotations"].shape[0]
        pose_local = _pose_local_from_result(
            skeletal_result, B, len(self.joint_names), self.root_joint_idx,
            skeletal_result["rotations"].dtype)
        pose_local = self._finish_body_or_full_pass(pose_local)
        dofs = _transforms_to_dofs(
            pose_local, self.joint_names, self.pt_data,
            root_bind_translation=self.root_bind_translation)
        pose_params = dofs @ self.pt_data["P_inv_pose_main"]
        params_204 = jnp.zeros((B, 204), dtype=pose_params.dtype)
        params_204 = params_204.at[:, self.pt_data["pose_indices"]].set(pose_params)
        return params_204[:, :136], pose_local

    def pose_params_to_local_transforms(self, pose_params) -> jnp.ndarray:
        """Native MHR pose/model params to diagnostic local transforms."""
        pose_params = jnp.asarray(pose_params, dtype=self.dtype)
        if pose_params.ndim == 1:
            pose_params = pose_params[None]
        if pose_params.ndim != 2 or pose_params.shape[1] not in (136, 204):
            raise ValueError("pose_params must have shape (B, 136) or (B, 204).")

        B = pose_params.shape[0]
        params_204 = jnp.zeros((B, 204), dtype=self.dtype)
        params_204 = params_204.at[:, :pose_params.shape[1]].set(pose_params)
        dofs = params_204[:, self.pt_data["pose_indices"]] @ self.pt_data["P_pose_main"].T

        joint_orients = self.pt_data["joint_orients"]
        root_idx_pt = self.pt_data["joint_names"].index("root")
        pose_local = jnp.broadcast_to(self.bind_local, (B,) + self.bind_local.shape)
        pose_local = pose_local.at[:, self.root_joint_idx, :3, 3].set(
            dofs[:, :3] + self.root_bind_translation)
        pose_local = pose_local.at[:, self.root_joint_idx, :3, :3].set(
            joint_orients[root_idx_pt][None] @ euler_xyz_to_matrix(dofs[:, 3:6]))
        pt_ids, data_ids, columns = _main_joint_layout(self.pt_data, self.joint_names)
        if len(pt_ids):
            pose_local = pose_local.at[:, data_ids, :3, :3].set(
                joint_orients[pt_ids][None] @ euler_xyz_to_matrix(dofs[:, columns]))
        return pose_local

    def model_skeleton_state_to_local_transforms(self, skel_state) -> jnp.ndarray:
        """Map MHR skeleton state to diagnostic local transforms by joint name."""
        skel_state = jnp.asarray(skel_state, dtype=self.dtype)
        if skel_state.ndim == 2:
            skel_state = skel_state[None]
        if skel_state.ndim != 3 or skel_state.shape[-1] != 8:
            raise ValueError("skel_state must have shape (B, J, 8).")
        world = _mhr_skeleton_state_to_transforms(skel_state)[:, self.model_state_joint_indices]
        return joint_world_to_local(world, self.joint_parent_ids)

    def model_params_to_local_transforms(
        self,
        identity_coeffs,
        model_params,
        face_expr_coeffs=None,
        *,
        apply_correctives: bool = False,
    ) -> jnp.ndarray:
        model_params = jnp.asarray(model_params, dtype=self.dtype)
        if model_params.ndim == 1:
            model_params = model_params[None]
        if model_params.ndim != 2 or model_params.shape[1] != 204:
            raise ValueError("model_params must have shape (B, 204).")

        B = model_params.shape[0]
        identity_coeffs = _as_batched_tensor(
            identity_coeffs, (45,), B, self.dtype, "identity_coeffs")
        face_expr_coeffs = _as_batched_tensor(
            face_expr_coeffs, (72,), B, self.dtype, "face_expr_coeffs")
        model = self._load_mhr_model()
        _, skel_state = model(identity_coeffs, model_params, face_expr_coeffs, apply_correctives)
        return self.model_skeleton_state_to_local_transforms(skel_state)

    def fit_skeletal_transforms(
        self,
        posed_vertices_mhr,
        *,
        body_iters: int = 10,
        finger_iters: int = 2,
        full_iters: int = 1,
    ) -> PoseInversionResult:
        target = jnp.asarray(posed_vertices_mhr, dtype=self.dtype)
        if target.ndim == 2:
            target = target[None]
        if target.shape[-2:] != self.bind_shape.shape:
            raise ValueError(
                f"Expected MHR vertices with shape (B, {self.bind_shape.shape[0]}, 3), "
                f"got {tuple(target.shape)}.")

        pose_world = self.skel_transfer.fit(target)
        pose_local = joint_world_to_local(pose_world, self.joint_parent_ids)
        pose_local = self._constrain_mhr_local(pose_local, None)

        for _ in range(body_iters):
            pose_local = self._run_mhr_refit_groups(pose_local, target, self.cache["body_groups"])
            pose_local = self._finish_body_or_full_pass(pose_local)
            pose_local = _update_root_translation(pose_local, target, self.cache)

        for _ in range(finger_iters):
            pose_local = self._run_mhr_refit_groups(
                pose_local, target, self.cache["finger_groups"])

        all_groups = self.cache["body_groups"] + self.cache["finger_groups"]
        for _ in range(full_iters):
            pose_local = self._run_mhr_refit_groups(pose_local, target, all_groups)
            pose_local = self._finish_body_or_full_pass(pose_local)
            pose_local = _update_root_translation(pose_local, target, self.cache)

        return _result_from_local_pose(
            pose_local, target, self.cache, self.bind_shape, self.bone_weights,
            self.bone_indices, self.root_joint_idx)

    def _refine_pose_params(
        self,
        init_pose_params,
        target,
        identity_coeffs,
        scale_params,
        face_expr_coeffs,
        *,
        refine_iters,
        lr,
        optimize_flexibles,
        freeze_disabled_pose_params,
        bound_active_spine,
        apply_correctives,
    ):
        if refine_iters <= 0:
            return init_pose_params, [], jnp.zeros((0,), dtype=target.dtype)

        model = self._load_mhr_model()
        fixed_flex = init_pose_params[:, _FLEXIBLE_SLICE]
        fixed_disabled = init_pose_params[:, _disabled_pose_param_ids()]
        flags = (bool(optimize_flexibles), bool(freeze_disabled_pose_params),
                 bool(bound_active_spine))
        data = (target, identity_coeffs, scale_params, face_expr_coeffs, fixed_flex,
                fixed_disabled)

        pose_opt = init_pose_params
        state = optax.adam(lr).init(pose_opt)
        lr = jnp.asarray(lr, dtype=init_pose_params.dtype)
        loss_history = []
        best_loss = float("inf")
        best_pose = init_pose_params
        for _ in range(refine_iters):
            next_pose, state, loss, pose_for_model = _refine_step(
                pose_opt, state, lr, *data, model=model, flags=flags,
                apply_correctives=bool(apply_correctives))
            cur_loss = float(loss)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_pose = pose_for_model
            pose_opt = next_pose
            loss_history.append(cur_loss)

        pose_final = _constrain_refined_pose(
            best_pose, fixed_flex, fixed_disabled, jnp.clip, optimize_flexibles=flags[0],
            freeze_disabled_pose_params=flags[1], bound_active_spine=flags[2])
        pred, _ = model(identity_coeffs, jnp.concatenate([pose_final, scale_params], axis=1),
                        face_expr_coeffs, apply_correctives)
        per_vertex_error = jnp.linalg.norm(pred - target, axis=-1)
        return pose_final, loss_history, per_vertex_error

    def fit(
        self,
        posed_vertices_mhr,
        *,
        identity_coeffs=None,
        scale_params=None,
        face_expr_coeffs=None,
        identity_reference_pose_params=None,
        body_iters: int = 10,
        finger_iters: int = 2,
        full_iters: int = 1,
        refine_iters: int = 0,
        lr: float = 1e-4,
        optimize_flexibles: bool = False,
        freeze_disabled_pose_params: bool = False,
        bound_active_spine: bool = False,
        apply_correctives: bool = False,
        batch_size: int | None = None,
    ) -> MHRPoseInversionResult:
        """Invert MHR-topology vertices to native MHR pose/model parameters.

        Args:
            posed_vertices_mhr: (B, V, 3) or (V, 3) MHR vertices.
            identity_coeffs, scale_params, face_expr_coeffs: (B, 45), (B, 68),
                (B, 72) or unbatched; zeros when omitted.
            identity_reference_pose_params: (B, 136) reference pose for
                ``use_identity_reference_for_skeleton``.
            body_iters, finger_iters, full_iters: refit passes.
            refine_iters, lr: Adam refinement through the MHR forward.
            optimize_flexibles: let the flexible/disabled pose params move.
            freeze_disabled_pose_params: freeze all disabled params, not only
                the flexible slice.
            bound_active_spine: clamp the active spine params to their bounds.
            apply_correctives: evaluate the MHR forward with pose correctives.
            batch_size: process the frames in chunks of this size.

        Returns:
            :class:`MHRPoseInversionResult` with ``pose_params``,
            ``model_params``, ``init_pose_params``, the error diagnostics,
            ``pose_local``, ``skeletal_pose_local``, ``skeletal_result``,
            ``loss_history`` and ``iters_run``.
        """
        target = jnp.asarray(posed_vertices_mhr, dtype=self.dtype)
        if target.ndim == 2:
            target = target[None]
        B = target.shape[0]
        settings = dict(
            body_iters=body_iters, finger_iters=finger_iters, full_iters=full_iters,
            refine_iters=refine_iters, lr=lr, optimize_flexibles=optimize_flexibles,
            freeze_disabled_pose_params=freeze_disabled_pose_params,
            bound_active_spine=bound_active_spine, apply_correctives=apply_correctives,
            batch_size=None,
        )

        if batch_size is not None and B > batch_size:
            chunks = []
            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                chunk_kwargs = {}
                for key, value in (
                    ("identity_coeffs", identity_coeffs),
                    ("scale_params", scale_params),
                    ("face_expr_coeffs", face_expr_coeffs),
                    ("identity_reference_pose_params", identity_reference_pose_params),
                ):
                    if value is not None:
                        value_t = jnp.asarray(value)
                        chunk_kwargs[key] = value_t[start:end] if value_t.ndim > 1 else value
                chunks.append(self.fit(target[start:end], **settings, **chunk_kwargs))
            return _concat_results(chunks)

        identity_coeffs = _as_batched_tensor(
            identity_coeffs, (45,), B, self.dtype, "identity_coeffs")
        scale_params = _as_batched_tensor(scale_params, (68,), B, self.dtype, "scale_params")
        face_expr_coeffs = _as_batched_tensor(
            face_expr_coeffs, (72,), B, self.dtype, "face_expr_coeffs")
        identity_reference_pose_params = _as_batched_tensor(
            identity_reference_pose_params, (136,), B, self.dtype,
            "identity_reference_pose_params")

        if self.use_identity_reference_for_skeleton and B > 1:
            chunks = [
                self.fit(
                    target[i:i + 1],
                    identity_coeffs=identity_coeffs[i:i + 1],
                    scale_params=scale_params[i:i + 1],
                    face_expr_coeffs=face_expr_coeffs[i:i + 1],
                    identity_reference_pose_params=identity_reference_pose_params[i:i + 1],
                    **settings,
                )
                for i in range(B)
            ]
            return _concat_results(chunks)

        skeletal_result, init_pose_params, skeletal_pose_local = self._fit_skeletal_and_project(
            target,
            body_iters=body_iters,
            finger_iters=finger_iters,
            full_iters=full_iters,
            identity_coeffs=identity_coeffs,
            scale_params=scale_params,
            face_expr_coeffs=face_expr_coeffs,
            identity_reference_pose_params=identity_reference_pose_params,
            apply_correctives=apply_correctives,
        )
        if not optimize_flexibles:
            init_pose_params = _apply_pose_param_constraints(
                init_pose_params,
                freeze_disabled_pose_params=freeze_disabled_pose_params,
                bound_active_spine=bound_active_spine,
            )

        model = self._load_mhr_model()
        pre_vertices, _ = model(
            identity_coeffs,
            jnp.concatenate([init_pose_params, scale_params], axis=1),
            face_expr_coeffs,
            apply_correctives,
        )
        pre_refine_error = jnp.linalg.norm(pre_vertices - target, axis=-1)

        pose_params, loss_history, refined_error = self._refine_pose_params(
            init_pose_params,
            target,
            identity_coeffs,
            scale_params,
            face_expr_coeffs,
            refine_iters=refine_iters,
            lr=lr,
            optimize_flexibles=optimize_flexibles,
            freeze_disabled_pose_params=freeze_disabled_pose_params,
            bound_active_spine=bound_active_spine,
            apply_correctives=apply_correctives,
        )
        model_params = jnp.concatenate([pose_params, scale_params], axis=1)
        pose_local = self.model_params_to_local_transforms(
            identity_coeffs, model_params, face_expr_coeffs,
            apply_correctives=apply_correctives)

        return MHRPoseInversionResult(
            pose_params=pose_params,
            model_params=model_params,
            init_pose_params=init_pose_params,
            per_vertex_error=refined_error if refine_iters > 0 else pre_refine_error,
            pre_refine_per_vertex_error=pre_refine_error,
            skeletal_per_vertex_error=skeletal_result["per_vertex_error"],
            pose_local=pose_local,
            skeletal_pose_local=skeletal_pose_local,
            skeletal_result=skeletal_result,
            loss_history=loss_history,
            iters_run=refine_iters,
        )
