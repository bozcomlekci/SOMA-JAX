"""Faithful JAX port of SOMA-X's :mod:`soma.pose_inversion`.

Recovers SOMA skeleton rotations from posed mesh vertices using the same
multi-stage solver as upstream ``soma.pose_inversion.PoseInversion``:

- **Analytical**: a :class:`~soma_jax.geometry.skeleton_transfer.SkeletonTransfer`
  warm start followed by top-down per-joint inverse-LBS Procrustes refits
  (body pass, finger pass, full pass), with root-translation updates between
  rounds.
- **Lie-GN**: FK-aware dense Gauss-Newton in SO(3). Each iteration solves one
  ``(3K x 3K)`` normal equation for all active joint twists simultaneously
  using the Kinematic Lever Arm Jacobian, then accepts the step per frame via
  backtracking line search.
- **Autograd FK**: Adam on 6D local rotations + root translation through
  FK + LBS, optionally with extremity vertex weighting and a pose prior.

``fit()`` returns a :class:`PoseInversionResult` carrying ``rotations``
(absolute local rotation matrices), ``root_translation`` and
``per_vertex_error``, matching upstream.

This module is the *faithful* inversion path.
:class:`soma_jax.pose_inversion.PoseInversion` remains available as the
lightweight SOMA-JAX alternative (single Kabsch init + one autograd refine).

Usage::

    from soma_jax import SOMALayer, SOMAPoseInversion

    layer = SOMALayer.load("assets/SOMA_neutral_fixed.npz")
    inv = SOMAPoseInversion(layer)
    inv.prepare_identity(identity_coeffs)

    result = inv.fit(posed_vertices)             # analytical + Lie-GN
    result = inv.fit(posed_vertices, lie_iters=0)               # analytical only
    result = inv.fit(posed_vertices, autograd_iters=10)         # + autograd FK

Upstream: ``soma/pose_inversion.py :: PoseInversion``
    Faithful port of that code. All three stages: SkeletonTransfer warm start -> inverse-LBS Procrustes refit -> Lie-algebra Gauss-Newton -> optional autograd FK.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .geometry.lbs import compute_skeleton_levels, fk_levelorder_transforms, lbs_sparse
from .geometry.rig_utils import (
    body_part_vertex_ids,
    get_joint_descendents,
    joint_world_to_local,
)
from .geometry.skeleton_transfer import SkeletonTransfer
from .geometry.transforms import (
    axis_angle_to_rotmat,
    compute_covariance,
    newton_schulz,
    regularize_covariance_with_reference,
    rotation_6d_to_rotmat,
    rotation_from_covariance,
    rotation_matrices_are_valid,
    se3_from_rt,
    se3_inverse,
)

# Joints constrained to Z-only rotation in the t-pose-relative frame.
_1DOF_Z_JOINTS = frozenset({"LeftForeArm", "RightForeArm", "LeftShin", "RightShin"})

_HIPS_IDX = 1  # SOMA Hips joint (child of the virtual Root at 0)
_ROTATION_METHODS = frozenset({"auto", "kabsch", "newton-schulz"})
_LIE_GN_DAMPING_FACTORS = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
_AUTO_REFIT_PRIOR_STRENGTH = 0.05
_MAX_LBS_K = 5  # Cap sparse LBS K to reduce work (K=5 loses < 0.01% weight)
_LINE_SEARCH_ALPHAS = (1.0, 0.5, 0.25, 0.125)


class PoseInversionResult(dict):
    """Result of :meth:`SOMAPoseInversion.fit`.

    Behaves like a ``dict`` (``result["rotations"]``) while also supporting
    attribute access (``result.rotations``), matching upstream.

    Keys:
        rotations: (B, J, 3, 3) absolute local rotation matrices.
        root_translation: (B, 3) hips translation in the layer's output unit.
        per_vertex_error: (B, V) L2 reconstruction error per vertex.
    """

    def __getattr__(self, name: str) -> jnp.ndarray:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _validate_rotation_method(method: str, parameter_name: str) -> str:
    if method not in _ROTATION_METHODS:
        choices = "', '".join(sorted(_ROTATION_METHODS))
        raise ValueError(f"Unknown {parameter_name}: {method!r}. Use '{choices}'.")
    return method


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _bexpand(t: jnp.ndarray, B: int) -> jnp.ndarray:
    """Broadcast an unbatched (N, 3) tensor to (B, N, 3)."""
    if t.ndim == 2:
        return jnp.broadcast_to(t[None], (B,) + t.shape)
    if t.ndim == 3 and t.shape[0] == 1 and B > 1:
        return jnp.broadcast_to(t, (B,) + t.shape[1:])
    return t


def _bexpand4(t: jnp.ndarray, B: int) -> jnp.ndarray:
    """Broadcast an unbatched (J, 4, 4) tensor to (B, J, 4, 4)."""
    if t.ndim == 3:
        return jnp.broadcast_to(t[None], (B,) + t.shape)
    if t.ndim == 4 and t.shape[0] == 1 and B > 1:
        return jnp.broadcast_to(t, (B,) + t.shape[1:])
    return t


def _skin(verts: jnp.ndarray, bone_weights, bone_indices, D: jnp.ndarray) -> jnp.ndarray:
    """Sparse LBS against 4x4 bone transforms — upstream ``linear_blend_skinning``.

    Weights are used **unnormalized** here: the refit relies on partial
    (subtree / non-subtree) weight sums, so the blend must stay linear.
    """
    return lbs_sparse(verts, jnp.zeros_like(verts), D[..., :3, :], bone_weights, bone_indices)


def _to_sparse_weights(dense_weights: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert (V, J) dense weights to top-K sparse ``(weights, indices)``.

    Unlike :func:`~soma_jax.geometry.batched_skinning.topk_skinning`, values are
    **not** renormalized — the caller needs the true partial weight mass.
    """
    V, J = dense_weights.shape
    actual_K = min(K, J)
    idx = np.argsort(dense_weights, axis=1)[:, -actual_K:][:, ::-1]
    vals = np.take_along_axis(dense_weights, idx, axis=1)
    if actual_K < K:
        pad = K - actual_K
        vals = np.concatenate([vals, np.zeros((V, pad), dtype=vals.dtype)], axis=1)
        idx = np.concatenate([idx, np.zeros((V, pad), dtype=idx.dtype)], axis=1)
    return vals.astype(np.float32), idx.astype(np.int32)


def _align_vectors_auto(
    target: jnp.ndarray,
    source: jnp.ndarray,
    reference_rotation: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Newton-Schulz-first Procrustes with a weak reference gauge for twist.

    Mirrors upstream ``_align_vectors_auto``: regularize the covariance toward
    the current world rotation so an under-constrained twist axis keeps its
    previous value instead of snapping to an arbitrary one, then fall back to
    SVD Kabsch only where Newton-Schulz did not land in SO(3).
    """
    covariance = compute_covariance(target, source, virtual_normal=True, eps=eps)
    regularized = regularize_covariance_with_reference(
        covariance,
        reference_rotation=reference_rotation,
        prior_strength=_AUTO_REFIT_PRIOR_STRENGTH,
        eps=eps,
    )
    R = newton_schulz(regularized)
    valid = rotation_matrices_are_valid(R, det_tol=1e-3, orthogonality_tol=1e-3)
    fallback = rotation_from_covariance(regularized, method="kabsch")
    return jnp.where(valid[..., None, None], R, fallback)


def _classify_joints(joint_names, parents: np.ndarray) -> tuple[set[int], set[int]]:
    """Split joints into body and finger index sets (fingers = below a Hand)."""
    hand_indices = [i for i, name in enumerate(joint_names) if str(name).endswith("Hand")]
    finger_set: set[int] = set()
    for hand_idx in hand_indices:
        finger_set.update(get_joint_descendents(parents, hand_idx))
    body_set = set(range(len(joint_names))) - finger_set
    return body_set, finger_set


# ---------------------------------------------------------------------------
# Vertex importance weighting
# ---------------------------------------------------------------------------

_GROUP_ROOTS = {
    "head": ["Head"],
    "hands": ["LeftHand", "RightHand"],
    "feet": ["LeftFoot", "RightFoot"],
}


def _heel_vertex_ids(joint_names, parents, skinning_weights, bind_shape, bind_joints) -> list[int]:
    """Rear foot vertices, i.e. those behind each Foot joint toward the heel."""
    if bind_shape is None or bind_joints is None:
        return []
    bind_shape = np.asarray(bind_shape)
    bind_joints = np.asarray(bind_joints)
    if bind_shape.ndim == 3:
        bind_shape = bind_shape[0]
    if bind_joints.ndim == 3:
        bind_joints = bind_joints[0]

    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    heel_ids: list[np.ndarray] = []
    for side in ("Left", "Right"):
        foot_idx = name_to_idx.get(f"{side}Foot")
        toe_idx = name_to_idx.get(f"{side}ToeBase")
        if foot_idx is None or toe_idx is None:
            continue
        foot_vids = body_part_vertex_ids(skinning_weights, parents, foot_idx, include_root=True)
        if not foot_vids:
            continue
        foot_vids_a = np.asarray(foot_vids, dtype=np.int64)
        foot_to_toe = bind_joints[toe_idx] - bind_joints[foot_idx]
        foot_to_toe = foot_to_toe / max(float(np.linalg.norm(foot_to_toe)), 1e-12)
        proj = (bind_shape[foot_vids_a] - bind_joints[foot_idx]) @ foot_to_toe
        heel_ids.append(foot_vids_a[proj <= np.quantile(proj, 0.35)])

    if not heel_ids:
        return []
    return np.unique(np.concatenate(heel_ids)).tolist()


def _compute_vertex_weights(
    joint_names,
    parents,
    skinning_weights,
    leaf_weight,
    bind_shape=None,
    bind_joints=None,
) -> Optional[jnp.ndarray]:
    """Per-vertex importance weights from body-part grouping.

    ``leaf_weight`` is either a scalar (uniform upweight of head/hands/feet) or
    a per-group mapping such as ``{"head": 2, "hands": 2, "feet": 5,
    "heels": 10}``. Returns ``None`` when nothing is upweighted.
    """
    if isinstance(leaf_weight, Mapping):
        group_weights = dict(leaf_weight)
    else:
        if leaf_weight <= 1.0:
            return None
        group_weights = {"head": leaf_weight, "hands": leaf_weight, "feet": leaf_weight}

    W = np.asarray(skinning_weights)
    weights = np.ones(W.shape[0], dtype=np.float32)
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}

    any_upweight = False
    for group_name, w in group_weights.items():
        if w <= 1.0:
            continue
        if group_name == "heels":
            vids = _heel_vertex_ids(joint_names, parents, W, bind_shape, bind_joints)
            if vids:
                weights[vids] = w
                any_upweight = True
            continue
        for root_name in _GROUP_ROOTS.get(group_name, []):
            j_idx = name_to_idx.get(root_name)
            if j_idx is None:
                continue
            weights[body_part_vertex_ids(W, parents, j_idx, include_root=True)] = w
            any_upweight = True

    return jnp.asarray(weights) if any_upweight else None


def _normalized_vertex_weights(*args, **kwargs) -> Optional[jnp.ndarray]:
    """Mean-one version of :func:`_compute_vertex_weights` (``None`` if uniform)."""
    weights = _compute_vertex_weights(*args, **kwargs)
    if weights is None:
        return None
    return weights / jnp.maximum(weights.mean(), jnp.finfo(weights.dtype).eps)


def _joint_pose_prior_weights(joint_names, joint_weights) -> Optional[jnp.ndarray]:
    """Per-joint pose-prior multipliers, or ``None`` for a uniform prior."""
    if not joint_weights:
        return None
    weights = np.ones(len(joint_names), dtype=np.float32)
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    for joint_name, weight in joint_weights.items():
        if joint_name not in name_to_idx:
            raise ValueError(f"Unknown joint in pose-prior weights: {joint_name!r}")
        weights[name_to_idx[joint_name]] = float(weight)
    return jnp.asarray(weights)


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------


def _precompute_refit_cache(
    joint_names,
    parents: np.ndarray,
    bind_world: np.ndarray,
    bind_shape: np.ndarray,
    skinning_weights: np.ndarray,
    t_pose_world: np.ndarray,
    root_idx: int = _HIPS_IDX,
) -> dict[str, Any]:
    """Precompute the sparse per-joint LBS decomposition used by the refit.

    For every non-leaf joint this splits its influenced vertices' skinning
    weights into a *subtree* part (moves when the joint rotates) and a
    *non-subtree* part (does not), which is what makes the per-joint
    inverse-LBS Procrustes solve exact.
    """
    from .geometry.batched_skinning import topk_skinning

    parents = np.asarray(parents).astype(np.int64)
    J = len(parents)
    W = np.asarray(skinning_weights, dtype=np.float32)
    bind_world = np.asarray(bind_world, dtype=np.float32)
    bind_shape_np = np.asarray(bind_shape, dtype=np.float32)

    bind_local = np.asarray(joint_world_to_local(jnp.asarray(bind_world), parents))
    bind_local_t = jnp.asarray(bind_local[:, :3, 3])
    W_bind_inv = se3_inverse(jnp.asarray(bind_world))
    levels = compute_skeleton_levels(parents)
    bone_indices, bone_weights = topk_skinning(W, 8)

    children_count = np.zeros(J, dtype=np.int64)
    for j in range(J):
        p = int(parents[j])
        if p != j and p >= 0:
            children_count[p] += 1
    end_joints = {j for j in range(J) if children_count[j] == 0}

    body_set, finger_set = _classify_joints(joint_names, parents)

    # Per-joint subtree decomposition. The virtual root (0) is skipped for
    # full-body rigs (root_idx=1); hand-only rigs (root_idx=0) keep it because
    # joint 0 is then a real wrist with geometry.
    first_joint = 0 if root_idx == 0 else 1
    joint_infos = []
    max_K = 1
    for j_idx in range(first_joint, J):
        if j_idx in end_joints:
            continue
        subtree = [j_idx] + get_joint_descendents(parents, j_idx)
        arm_mask = W[:, subtree].sum(axis=1) > 0.01
        arm_vids = np.where(arm_mask)[0]
        if len(arm_vids) == 0:
            continue

        subtree_mask = np.zeros(J, dtype=bool)
        subtree_mask[subtree] = True

        sw_arm = W[arm_vids]
        sw_arm = sw_arm * (sw_arm > 1e-6)
        sw_sub = sw_arm * subtree_mask.astype(np.float32)
        sw_non = sw_arm * (~subtree_mask).astype(np.float32)

        max_K = max(
            max_K,
            int((sw_sub > 0).sum(axis=1).max()),
            int((sw_non > 0).sum(axis=1).max()),
        )
        joint_infos.append((j_idx, arm_vids, sw_sub, sw_non))

    max_K = min(max_K, _MAX_LBS_K)

    joint_cache: dict[int, dict[str, Any]] = {}
    for j_idx, arm_vids, sw_sub, sw_non in joint_infos:
        sub_bw, sub_bi = _to_sparse_weights(sw_sub, max_K)
        non_bw, non_bi = _to_sparse_weights(sw_non, max_K)
        joint_cache[j_idx] = {
            "arm_vids": arm_vids,
            "bind_verts_arm": jnp.asarray(bind_shape_np[arm_vids]),
            "sub_bone_weights": jnp.asarray(sub_bw),
            "sub_bone_indices": jnp.asarray(sub_bi),
            "non_bone_weights": jnp.asarray(non_bw),
            "non_bone_indices": jnp.asarray(non_bi),
            "sub_weight_sum": jnp.asarray(sw_sub.sum(axis=1)),
        }

    body_groups, finger_groups = [], []
    for level in levels:
        bg = [j for j in level if j in joint_cache and j in body_set]
        fg = [j for j in level if j in joint_cache and j in finger_set]
        if bg:
            body_groups.append(bg)
        if fg:
            finger_groups.append(fg)

    # 1-DOF constraint data, vectorised over all constrained joints.
    t_orient = jnp.asarray(np.asarray(t_pose_world, dtype=np.float32)[:, :3, :3])
    constrained_indices = [
        j for j, name in enumerate(joint_names) if str(name) in _1DOF_Z_JOINTS
    ]
    constrained_data = None
    if constrained_indices:
        safe_parents = np.where(parents < 0, np.arange(J), parents)
        constrained_data = {
            "indices": np.asarray(constrained_indices, dtype=np.int64),
            "orient_j": t_orient[np.asarray(constrained_indices)],
            "orient_p": t_orient[safe_parents[np.asarray(constrained_indices)]],
        }

    return {
        "joint_names": list(joint_names),
        "parents": parents,
        "bind_local_t": bind_local_t,
        "W_bind_inv": W_bind_inv,
        "levels": levels,
        "joint_cache": joint_cache,
        "body_groups": body_groups,
        "finger_groups": finger_groups,
        "constrained_data": constrained_data,
        "constrained_set": set(constrained_indices),
        "t_pose_orient": t_orient,
        "skinning_weights": jnp.asarray(W),
        "bone_weights": jnp.asarray(bone_weights),
        "bone_indices": jnp.asarray(bone_indices),
        "root_idx": int(root_idx),
    }


# ---------------------------------------------------------------------------
# FK, refit, constraints, root translation
# ---------------------------------------------------------------------------


def _build_world_transforms(pose_local: jnp.ndarray, cache) -> jnp.ndarray:
    """World transforms from local rotations + bind translations (root overridden)."""
    root_idx = cache["root_idx"]
    B = pose_local.shape[0]
    local_t = _bexpand(cache["bind_local_t"], B)
    local_t = local_t.at[:, root_idx, :].set(pose_local[:, root_idx, :3, 3])
    T_local = se3_from_rt(pose_local[:, :, :3, :3], local_t)
    return fk_levelorder_transforms(T_local, cache["levels"], cache["parents"])


def _refit_joint(
    pose_local: jnp.ndarray,
    target: jnp.ndarray,
    j_idx: int,
    W: jnp.ndarray,
    D: jnp.ndarray,
    cache,
    jcache,
    vert_weights: Optional[jnp.ndarray] = None,
    rotation_method: str = "auto",
) -> jnp.ndarray:
    """Re-fit one joint by inverse-LBS Procrustes alignment.

    The subtree's contribution is mapped into the joint's parent frame and
    aligned against the residual target (the non-subtree contribution removed),
    so the recovered rotation is exactly the one LBS would need at this joint.
    """
    arm_vids = jcache["arm_vids"]
    B = pose_local.shape[0]
    bv = _bexpand(jcache["bind_verts_arm"], B)

    q_world = _skin(bv, jcache["sub_bone_weights"], jcache["sub_bone_indices"], D)
    c_xyz = _skin(bv, jcache["non_bone_weights"], jcache["non_bone_indices"], D)

    W_p_inv = se3_inverse(W[:, j_idx])
    R_inv = W_p_inv[:, :3, :3]
    t_inv = W_p_inv[:, :3, 3]

    sw = jcache["sub_weight_sum"].reshape(1, -1, 1)
    src = jnp.einsum("bnc,bdc->bnd", q_world, R_inv) + t_inv[:, None, :] * sw

    p_parent = W[:, j_idx, :3, 3]
    tgt = target[:, arm_vids, :] - c_xyz - p_parent[:, None, :] * sw

    # Weighted alignment: scaling both sides by sqrt(w) makes the covariance
    # tgt^T @ diag(w) @ src without materialising the diagonal.
    if vert_weights is not None:
        sqrt_w = jnp.sqrt(vert_weights[arm_vids])[None, :, None]
        tgt = tgt * sqrt_w
        src = src * sqrt_w

    if rotation_method == "auto":
        R_new = _align_vectors_auto(tgt, src, W[:, j_idx, :3, :3])
    else:
        H = compute_covariance(tgt, src, virtual_normal=True)
        R_new = rotation_from_covariance(H, method=rotation_method)

    grandparent_idx = int(cache["parents"][j_idx])
    if grandparent_idx == j_idx or grandparent_idx < 0:
        # Self-parented root: the world rotation IS the local rotation.
        return pose_local.at[:, j_idx, :3, :3].set(R_new)
    R_gp_world = W[:, grandparent_idx, :3, :3]
    return pose_local.at[:, j_idx, :3, :3].set(
        jnp.einsum("bnm,bnp->bmp", R_gp_world, R_new)
    )


def _constrain_1dof_z(pose_local: jnp.ndarray, cache) -> jnp.ndarray:
    """Constrain elbow/knee joints to Z-only rotation in the t-pose frame."""
    cd = cache["constrained_data"]
    if cd is None:
        return pose_local
    indices = cd["indices"]
    orient_j = cd["orient_j"]
    orient_p = cd["orient_p"]

    B = pose_local.shape[0]
    R_abs = pose_local[:, indices, :3, :3]

    R_tpose = orient_p[None] @ R_abs @ jnp.swapaxes(orient_j, -2, -1)[None]
    rz = jnp.arctan2(R_tpose[:, :, 1, 0], R_tpose[:, :, 0, 0])

    cos_rz, sin_rz = jnp.cos(rz), jnp.sin(rz)
    zeros = jnp.zeros_like(cos_rz)
    ones = jnp.ones_like(cos_rz)
    R_z = jnp.stack(
        [
            jnp.stack([cos_rz, -sin_rz, zeros], axis=-1),
            jnp.stack([sin_rz, cos_rz, zeros], axis=-1),
            jnp.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )

    R_constrained = jnp.swapaxes(orient_p, -2, -1)[None] @ R_z @ orient_j[None]
    return pose_local.at[:, indices, :3, :3].set(R_constrained)


def _update_root_translation(
    pose_local: jnp.ndarray,
    target: jnp.ndarray,
    cache,
    vert_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Shift the root translation to minimise the mean vertex residual."""
    root_idx = cache["root_idx"]
    jcache = cache["joint_cache"].get(root_idx)
    if jcache is None:
        return pose_local
    B = pose_local.shape[0]
    W = _build_world_transforms(pose_local, cache)
    D = jnp.einsum("bjmn,bjnp->bjmp", W, _bexpand4(cache["W_bind_inv"], B))

    arm_vids = jcache["arm_vids"]
    bv = _bexpand(jcache["bind_verts_arm"], B)
    current = _skin(bv, jcache["sub_bone_weights"], jcache["sub_bone_indices"], D)
    residual = target[:, arm_vids, :] - current
    if vert_weights is not None:
        w = vert_weights[arm_vids]
        delta_t = (residual * w[None, :, None]).sum(axis=1) / jnp.maximum(
            w.sum(), jnp.finfo(residual.dtype).eps
        )
    else:
        delta_t = residual.mean(axis=1)
    return pose_local.at[:, root_idx, :3, 3].add(delta_t)


def _run_refit_passes(
    pose_local: jnp.ndarray,
    target: jnp.ndarray,
    cache,
    groups,
    constrain_1dof: bool = True,
    vert_weights: Optional[jnp.ndarray] = None,
    rotation_method: str = "auto",
) -> jnp.ndarray:
    """One top-down refit round: per skeleton level, rebuild FK then refit joints.

    Joints at the same depth are independent, so FK only has to be rebuilt once
    per level; 1-DOF constraints are applied after each level so children see
    already-constrained parents.
    """
    joint_cache = cache["joint_cache"]
    constrained_set = cache["constrained_set"]
    B = pose_local.shape[0]

    for group in groups:
        W = _build_world_transforms(pose_local, cache)
        D = jnp.einsum("bjmn,bjnp->bjmp", W, _bexpand4(cache["W_bind_inv"], B))

        for j_idx in group:
            jcache = joint_cache.get(j_idx)
            if jcache is None:
                continue
            pose_local = _refit_joint(
                pose_local, target, j_idx, W, D, cache, jcache,
                vert_weights, rotation_method,
            )

        if constrain_1dof and constrained_set.intersection(group):
            pose_local = _constrain_1dof_z(pose_local, cache)

    return pose_local


def _solve_lie_gn_normal_equations(JtJ: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve the batched Lie-GN normal equations with a damping ladder.

    Mirrors upstream's deterministic fallback: try progressively stronger
    Tikhonov damping and accept the first finite solution per batch element,
    then fall back to a diagonal (Jacobi) solve. JAX has no ``solve_ex`` info
    flag, so solutions are validated by finiteness instead.
    """
    B, N, _ = JtJ.shape
    dtype = JtJ.dtype
    eps = jnp.finfo(dtype).eps

    eye = jnp.broadcast_to(jnp.eye(N, dtype=dtype), (B, N, N))
    diag_scale_raw = jnp.abs(JtJ.diagonal(axis1=-2, axis2=-1)).mean(axis=-1)
    scaled_system = diag_scale_raw > eps
    diag_scale = jnp.maximum(diag_scale_raw, eps)

    solution = jnp.zeros_like(rhs)
    solved = jnp.zeros(B, dtype=bool)

    for damping in _LIE_GN_DAMPING_FACTORS:
        system = JtJ if damping == 0.0 else JtJ + eye * (diag_scale * damping)[:, None, None]
        candidate = jnp.linalg.solve(system, rhs[..., None])[..., 0]
        ok = jnp.all(jnp.isfinite(candidate), axis=-1)
        if damping != 0.0:
            ok = ok & scaled_system
        update = ok & ~solved
        solution = jnp.where(update[:, None], candidate, solution)
        solved = solved | update

    diag = JtJ.diagonal(axis1=-2, axis2=-1)
    fallback_den = jnp.where(jnp.abs(diag) > eps, diag, diag_scale[:, None])
    fallback = rhs / fallback_den
    use_fallback = (
        ~solved & jnp.all(jnp.isfinite(fallback), axis=-1) & scaled_system
    )
    return jnp.where(use_fallback[:, None], fallback, solution)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SOMAPoseInversion:
    """Invert posed vertices to SOMA skeleton rotations — faithful SOMA-X solver.

    Args:
        soma_layer: a :class:`~soma_jax.SOMALayer`.
        low_lod: assert that the layer handed in is low-LOD. Upstream builds a
            *separate* internal low-LOD layer for the refit; SOMA-JAX refits on
            whichever layer you pass, so build it with
            ``SOMALayer.load(path, lod="low")`` and hand that in. Full-
            resolution inputs are subsampled automatically when the layer is
            low-LOD, so ``fit()`` still accepts mid-LOD vertices.
        skeleton_transfer_rotation_method: rotation extraction for the initial
            :class:`SkeletonTransfer` estimate (``"auto"``, ``"kabsch"``,
            ``"newton-schulz"``).
        refit_rotation_method: rotation extraction for the analytical
            inverse-LBS refit.
        root_joint_idx: index of the root joint (1 = Hips for full-body SOMA).
    """

    def __init__(
        self,
        soma_layer,
        low_lod: bool = False,
        skeleton_transfer_rotation_method: str = "auto",
        refit_rotation_method: str = "auto",
        root_joint_idx: int = _HIPS_IDX,
    ) -> None:
        self._soma_orig = soma_layer
        self._root_joint_idx = int(root_joint_idx)
        self.skeleton_transfer_rotation_method = _validate_rotation_method(
            skeleton_transfer_rotation_method, "skeleton_transfer_rotation_method"
        )
        self.refit_rotation_method = _validate_rotation_method(
            refit_rotation_method, "refit_rotation_method"
        )

        self.soma = soma_layer
        self._num_verts = int(soma_layer.v_template.shape[0])

        # When the layer is already low-LOD, keep the mid->low index map so
        # fit() can still accept full-resolution SOMA vertices (upstream's
        # `_soma_full_num_verts` path).
        self._mid_to_low = None
        self._full_num_verts = None
        lod_map = getattr(soma_layer, "_lod_mid_to_low_np", None)
        if lod_map is not None and self._num_verts == len(lod_map):
            self._mid_to_low = np.asarray(lod_map, dtype=np.int64)
            # Prefer the recorded source-mesh size. `max() + 1` is not a valid
            # substitute: the shipped rig's map is exactly `arange(V_low)`, so
            # it would report the low count and full-resolution input would be
            # rejected.
            recorded = getattr(soma_layer, "_lod_mid_num_verts", None)
            self._full_num_verts = (
                int(recorded) if recorded is not None
                else int(self._mid_to_low.max()) + 1
            )

        if low_lod and self._mid_to_low is None:
            raise ValueError(
                "low_lod=True but this layer is not low-LOD. SOMA-JAX refits on the "
                "layer you pass rather than building an internal one, so load it "
                "with SOMALayer.load(path, lod='low') and hand that layer in instead."
            )
        self._cache = None
        self._skel_transfer = None
        self._rest_shape = None
        self._bind_world = None

        # Upstream gates the dedicated MHR interpolator on exactly these three
        # conditions (`soma/pose_inversion.py`): a low-LOD layer, the MHR
        # backend, and a mid->low map to subsample the wrap with. Built eagerly
        # like upstream, but a missing MHR asset degrades to the identity-model
        # fallback rather than breaking construction of a usable solver.
        self._pose_transfer = None
        self._pose_transfer_num_verts = None
        if (self._mid_to_low is not None
                and getattr(soma_layer, "identity_model_type", None) == "mhr"):
            try:
                self._setup_pose_transfer()
            except (FileNotFoundError, ImportError):
                pass

    @property
    def joint_names(self) -> list[str]:
        return list(self.soma.joint_names)

    def _setup_pose_transfer(self) -> None:
        """Build the direct full-res-MHR -> low-SOMA interpolator.

        Port of upstream ``PoseInversion._setup_pose_transfer``. Upstream builds
        this **only** for a low-LOD layer on the MHR backend, and the reason is
        specific: the identity model's own interpolator is built for the *low-res*
        MHR mesh (``base_body_lod6.obj``), but pose inversion is handed
        *full-res* lod1 vertices (18439). Routing those through the identity
        model would apply a correspondence built for a different mesh.

        So this embeds the 4,505 low-LOD SOMA wrap points directly in the
        full-res MHR mesh:

        * source: ``MHR/base_body_lod1.obj``  (18,439 verts)
        * target: ``MHR/SOMA_wrap_lod1.obj``  (18,056) subsampled by
          ``lod_mid_to_low`` -> 4,505

        giving a one-hop 18,439 -> 4,505 transfer.

        Note this is **not** an accuracy improvement over transferring to
        mid-SOMA and then subsampling: ``compute_barycentric_coords`` is
        per-query-point, so embedding ``v_soma[mid_to_low]`` yields the same
        (face, barycentric) pair as embedding ``v_soma`` and then selecting —
        the two routes agree to 0.0 (``tests/test_pose_inversion.py``). The
        reason upstream builds it is availability: for a low-LOD MHR layer the
        identity model's interpolator has the wrong *source* mesh, so without
        this there is no valid route for lod1 input at all.
        """
        import trimesh

        from .assets import resolve
        from .geometry.barycentric_interp import compute_barycentric_coords

        mhr = trimesh.load(resolve("MHR/base_body_lod1.obj"),
                           maintain_order=True, process=False)
        v_mhr = np.asarray(mhr.vertices, np.float32)
        f_mhr = np.asarray(mhr.faces, np.int32)

        wrap = trimesh.load(resolve("MHR/SOMA_wrap_lod1.obj"),
                            maintain_order=True, process=False)
        v_soma = np.asarray(wrap.vertices, np.float32)
        v_soma_low = v_soma[self._mid_to_low]

        face_ids, bary = compute_barycentric_coords(v_soma_low, v_mhr, f_mhr)
        self._pose_transfer = (f_mhr, np.asarray(face_ids, np.int32),
                              np.asarray(bary, np.float32))
        self._pose_transfer_num_verts = int(v_mhr.shape[0])

    def transfer_to_soma(self, vertices: jnp.ndarray) -> jnp.ndarray:
        """Bring vertices onto the topology used for the refit.

        Full-resolution SOMA input is subsampled through ``lod_mid_to_low``
        when running low-LOD; anything already on the refit topology passes
        through unchanged. Full-res MHR input goes through the dedicated
        interpolator built by :meth:`_setup_pose_transfer`, and anything else
        falls back to the active identity model's correspondence.
        """
        squeezed = vertices.ndim == 2
        if squeezed:
            vertices = vertices[None]
        V = vertices.shape[-2]
        if V == self._num_verts:
            out = vertices
        elif self._mid_to_low is not None and V == self._full_num_verts:
            out = vertices[:, self._mid_to_low, :]
        elif (self._pose_transfer is not None
              and V == self._pose_transfer_num_verts):
            # Upstream prefers this over the identity model whenever it exists.
            from .geometry.barycentric_interp import barycentric_interpolate
            faces, face_ids, bary = self._pose_transfer
            out = barycentric_interpolate(
                vertices, jnp.asarray(faces), jnp.asarray(face_ids), jnp.asarray(bary))
        else:
            # Neither SOMA topology nor a full-res SOMA mesh: fall back to the
            # active identity model's topology transfer, as upstream does
            # (`soma/pose_inversion.py` -> `identity_model._to_soma_interp`).
            out = self._transfer_via_identity_model(vertices, V)
        return out[0] if squeezed else out

    def _transfer_via_identity_model(self, vertices: jnp.ndarray, V: int) -> jnp.ndarray:
        """Map source-topology vertices onto SOMA topology via the identity model.

        Mirrors upstream's fallback: when the input is on the identity
        backend's own mesh (e.g. MHR), reuse that backend's barycentric
        correspondence instead of refusing the input.
        """
        model = getattr(self.soma, "identity_model", None)
        face_ids = getattr(model, "_face_ids", None)
        bary = getattr(model, "_bary_coords", None)
        src_faces = getattr(model, "src_faces", None)
        src_template = getattr(model, "v_template", None)
        if model is None or face_ids is None or bary is None or src_faces is None:
            raise ValueError(
                f"Vertex count {V} matches neither the refit topology "
                f"({self._num_verts}) nor the full-resolution SOMA mesh "
                f"({self._full_num_verts}), and identity model "
                f"{type(model).__name__ if model else 'None'} carries no "
                f"topology transfer."
            )
        if src_template is not None and int(np.asarray(src_template).shape[0]) != V:
            raise ValueError(
                f"Vertex count {V} does not match the identity model's source "
                f"topology ({int(np.asarray(src_template).shape[0])})."
            )
        from .geometry.barycentric_interp import barycentric_interpolate
        out = barycentric_interpolate(
            vertices, jnp.asarray(src_faces), jnp.asarray(face_ids), jnp.asarray(bary))
        if out.shape[-2] != self._num_verts:
            # Low-LOD refit against a mid-LOD correspondence.
            if self._mid_to_low is not None and out.shape[-2] == self._full_num_verts:
                out = out[:, self._mid_to_low, :]
            else:
                raise ValueError(
                    f"Identity transfer produced {out.shape[-2]} vertices, "
                    f"expected {self._num_verts}.")
        return out

    def prepare_identity(
        self,
        identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
        repose_to_bind_pose: bool = True,
        skeleton_fit: str = "auto",
    ) -> None:
        """Fit the rig for an identity, then build the refit caches.

        Structural caches (sparse per-joint weights, level groups) depend only
        on topology, but the bind data they hold is identity-specific, so the
        cache is rebuilt per identity exactly as upstream does.

        Args:
            identity_coeffs: (B, C) or (C,) identity coefficients.
            scale_params: optional (B, S) body-part scales.
            repose_to_bind_pose: repose the rest shape into SOMA's bind pose.
            skeleton_fit: forwarded to ``SOMALayer.prepare_identity``.
        """
        coeffs = identity_coeffs[None] if identity_coeffs.ndim == 1 else identity_coeffs
        if scale_params is not None and scale_params.ndim == 1:
            scale_params = scale_params[None]

        rest_shape, _, bind_world = self.soma.prepare_identity(
            coeffs,
            scale_params,
            repose_to_bind_pose=repose_to_bind_pose,
            skeleton_fit=skeleton_fit,
            return_bind_transforms=True,
        )
        if bind_world is None:
            raise RuntimeError(
                "SOMAPoseInversion needs the full skeleton fit: load an asset with "
                "bind_shape + bind_pose_world (see docs/INSTALL.md §4.2)."
            )
        self._rest_shape = rest_shape
        self._bind_world = bind_world

        bind_world_0 = np.asarray(bind_world[0])
        bind_shape_0 = np.asarray(rest_shape[0])
        parents = np.asarray(self.soma._parents_np).astype(np.int64)

        src = self.soma.skeleton_transfer
        if self._skel_transfer is None:
            self._skel_transfer = SkeletonTransfer(
                parents,
                bind_world_0,
                bind_shape_0,
                np.asarray(self.soma.weights),
                rotation_method=self.skeleton_transfer_rotation_method,
                vertex_ids_to_exclude=(
                    src.vertex_ids_to_exclude if src is not None else None
                ),
            )
        else:
            self._skel_transfer.update_bind(bind_world_0, bind_shape_0)

        t_pose_world = self.soma.t_pose_world
        if t_pose_world is None:
            t_pose_world = np.broadcast_to(np.eye(4, dtype=np.float32), (len(parents), 4, 4))

        self._cache = _precompute_refit_cache(
            self.joint_names,
            parents,
            bind_world_0,
            bind_shape_0,
            np.asarray(self.soma.weights),
            np.asarray(t_pose_world),
            root_idx=self._root_joint_idx,
        )
        # Batched identities keep per-identity bind data.
        if bind_world.shape[0] > 1:
            bind_local = joint_world_to_local(bind_world, parents)
            self._cache["bind_local_t"] = bind_local[:, :, :3, 3]
            self._cache["W_bind_inv"] = se3_inverse(bind_world)
            for jc in self._cache["joint_cache"].values():
                jc["bind_verts_arm"] = rest_shape[:, jc["arm_vids"]]

    def fit(
        self,
        posed_vertices: jnp.ndarray,
        body_iters: int = 2,
        finger_iters: int = 0,
        full_iters: int = 1,
        lie_iters: int = 3,
        lie_lambda: float = 1e-1,
        autograd_iters: int = 0,
        autograd_lr: float = 5e-3,
        autograd_translation_lr_scale: float = 1.0,
        autograd_pose_prior: float = 0.0,
        autograd_leaf_weight=None,
        autograd_pose_prior_weights=None,
        constrain_1dof: bool = False,
        leaf_weight=1.0,
        batch_size: Optional[int] = None,
    ) -> PoseInversionResult:
        """Fit SOMA skeleton rotations to posed vertices.

        Stage selection follows upstream:

        - default (``body_iters=2, full_iters=1, lie_iters=3``): analytical
          warm start then Lie-algebra Gauss-Newton.
        - ``lie_iters=0``: analytical only.
        - ``body_iters=0, full_iters=0, lie_iters=0, autograd_iters=N``:
          pure autograd FK.
        - ``autograd_iters=N`` on top of the default: analytical + Lie-GN warm
          start feeding autograd refinement.

        Args:
            posed_vertices: (B, V, 3) vertices on the refit (or full-res SOMA)
                topology.
            body_iters: analytical rounds over the body chain.
            finger_iters: analytical rounds over the finger chains.
            full_iters: analytical rounds over all joints.
            lie_iters: Lie-GN iterations; each solves one (3K x 3K) system.
            lie_lambda: Marquardt damping applied to active diagonal blocks.
            autograd_iters: Adam steps through FK + LBS.
            autograd_lr: Adam learning rate.
            autograd_translation_lr_scale: root-translation LR multiplier.
            autograd_pose_prior: weight of the prior pulling local rotations
                back toward the warm start (0 disables).
            autograd_leaf_weight: vertex weighting used only by the autograd
                stage; defaults to ``leaf_weight``.
            autograd_pose_prior_weights: per-joint pose-prior multipliers.
            constrain_1dof: project elbows/knees onto a Z hinge (analytical).
            leaf_weight: extremity importance — a float, or a per-group dict
                such as ``{"head": 2, "hands": 2, "feet": 5, "heels": 10}``.
            batch_size: process the batch in chunks of this size.

        Returns:
            :class:`PoseInversionResult` with ``rotations`` (B, J, 3, 3),
            ``root_translation`` (B, 3) and ``per_vertex_error`` (B, V).
        """
        if self._cache is None:
            raise RuntimeError("Call prepare_identity() first.")

        if posed_vertices.ndim == 2:
            posed_vertices = posed_vertices[None]
        B = posed_vertices.shape[0]

        if batch_size is not None and B > batch_size:
            kwargs = dict(
                body_iters=body_iters, finger_iters=finger_iters,
                full_iters=full_iters, lie_iters=lie_iters,
                lie_lambda=lie_lambda, autograd_iters=autograd_iters,
                autograd_lr=autograd_lr,
                autograd_translation_lr_scale=autograd_translation_lr_scale,
                autograd_pose_prior=autograd_pose_prior,
                autograd_leaf_weight=autograd_leaf_weight,
                autograd_pose_prior_weights=autograd_pose_prior_weights,
                constrain_1dof=constrain_1dof, leaf_weight=leaf_weight,
            )
            # Batched identities carry per-identity bind data in the cache, so
            # each chunk needs its own slice of it — mirrors upstream's
            # _save_bind_cache / _slice_bind_cache / _restore_bind_cache.
            saved = self._save_bind_cache()
            try:
                chunks = []
                for s in range(0, B, batch_size):
                    end = min(s + batch_size, B)
                    if saved is not None:
                        self._slice_bind_cache(saved, s, end)
                    chunks.append(self.fit(posed_vertices[s:end], **kwargs))
            finally:
                if saved is not None:
                    self._restore_bind_cache(saved)
            return PoseInversionResult(
                {k: jnp.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
            )

        cache = self._cache
        target = self.transfer_to_soma(posed_vertices)

        result = None
        if body_iters > 0 or finger_iters > 0 or full_iters > 0:
            result = self._fit_analytical(
                target, cache, body_iters, finger_iters, full_iters,
                constrain_1dof, leaf_weight,
            )

        if lie_iters > 0:
            result = self._fit_lie_algebra_gn(
                target, cache, lie_iters, lie_lambda, leaf_weight, init_result=result,
            )

        if autograd_iters > 0:
            result = self._fit_autograd_fk(
                target, cache, autograd_iters, autograd_lr,
                autograd_translation_lr_scale,
                leaf_weight if autograd_leaf_weight is None else autograd_leaf_weight,
                autograd_pose_prior, autograd_pose_prior_weights,
                init_result=result,
            )

        if result is None:
            raise ValueError(
                "At least one of body_iters, finger_iters, full_iters, "
                "lie_iters, or autograd_iters must be > 0."
            )
        return result

    # -- stages ----------------------------------------------------------

    def _bind_joint_positions(self, cache) -> jnp.ndarray:
        bind_world = se3_inverse(cache["W_bind_inv"])
        if bind_world.ndim == 4:
            bind_world = bind_world[0]
        return bind_world[:, :3, 3]

    # ------------------------------------------------------------------
    # Chunked fitting over batched identities
    # ------------------------------------------------------------------
    def _save_bind_cache(self) -> Optional[dict]:
        """Snapshot the per-identity cache entries before chunked slicing.

        Returns ``None`` for a single (broadcast) identity, where nothing in
        the cache is batched and chunks need no slicing.
        """
        cache = self._cache
        rest = self._rest_shape
        if rest is None or rest.ndim < 3 or rest.shape[0] <= 1:
            return None
        return {
            "rest_shape": rest,
            "bind_local_t": cache.get("bind_local_t"),
            "W_bind_inv": cache.get("W_bind_inv"),
            "bind_verts_arm": {j: jc["bind_verts_arm"]
                               for j, jc in cache["joint_cache"].items()
                               if "bind_verts_arm" in jc},
        }

    def _slice_bind_cache(self, saved: dict, start: int, end: int) -> None:
        """Narrow the per-identity cache entries to identities ``[start:end)``."""
        cache = self._cache
        self._rest_shape = saved["rest_shape"][start:end]
        for key in ("bind_local_t", "W_bind_inv"):
            if saved[key] is not None:
                cache[key] = saved[key][start:end]
        for j, v in saved["bind_verts_arm"].items():
            cache["joint_cache"][j]["bind_verts_arm"] = v[start:end]

    def _restore_bind_cache(self, saved: dict) -> None:
        """Put the full-batch per-identity cache entries back."""
        cache = self._cache
        self._rest_shape = saved["rest_shape"]
        for key in ("bind_local_t", "W_bind_inv"):
            if saved[key] is not None:
                cache[key] = saved[key]
        for j, v in saved["bind_verts_arm"].items():
            cache["joint_cache"][j]["bind_verts_arm"] = v

    def _rest_shape_b(self, B: int) -> jnp.ndarray:
        rest = self._rest_shape
        if rest.ndim == 2:
            rest = rest[None]
        if rest.shape[0] == 1 and B > 1:
            rest = jnp.broadcast_to(rest, (B,) + rest.shape[1:])
        return rest

    def _per_vertex_error(self, pose_local, target, cache) -> jnp.ndarray:
        B = pose_local.shape[0]
        W = _build_world_transforms(pose_local, cache)
        D = jnp.einsum("bjmn,bjnp->bjmp", W, _bexpand4(cache["W_bind_inv"], B))
        recon = _skin(self._rest_shape_b(B), cache["bone_weights"], cache["bone_indices"], D)
        return jnp.linalg.norm(recon - target, axis=-1)

    def _result(self, pose_local, target, cache) -> PoseInversionResult:
        return PoseInversionResult(
            rotations=pose_local[:, :, :3, :3],
            root_translation=pose_local[:, cache["root_idx"], :3, 3],
            per_vertex_error=self._per_vertex_error(pose_local, target, cache),
        )

    def _fit_analytical(
        self, target, cache, body_iters, finger_iters, full_iters,
        constrain_1dof, leaf_weight,
    ) -> PoseInversionResult:
        """Analytical iterative inverse-LBS refinement."""
        body_groups = cache["body_groups"]
        finger_groups = cache["finger_groups"]
        all_groups = body_groups + finger_groups

        vert_weights = _compute_vertex_weights(
            cache["joint_names"], cache["parents"], cache["skinning_weights"],
            leaf_weight,
            bind_shape=self._rest_shape, bind_joints=self._bind_joint_positions(cache),
        )

        pose_world = self._skel_transfer.fit(target)
        pose_local = joint_world_to_local(pose_world, cache["parents"])
        if constrain_1dof:
            pose_local = _constrain_1dof_z(pose_local, cache)

        for _ in range(body_iters):
            pose_local = _run_refit_passes(
                pose_local, target, cache, body_groups, constrain_1dof,
                vert_weights, self.refit_rotation_method,
            )
            pose_local = _update_root_translation(pose_local, target, cache, vert_weights)

        for _ in range(finger_iters):
            pose_local = _run_refit_passes(
                pose_local, target, cache, finger_groups, constrain_1dof,
                vert_weights, self.refit_rotation_method,
            )

        for _ in range(full_iters):
            pose_local = _run_refit_passes(
                pose_local, target, cache, all_groups, constrain_1dof,
                vert_weights, self.refit_rotation_method,
            )
            pose_local = _update_root_translation(pose_local, target, cache, vert_weights)

        return self._result(pose_local, target, cache)

    def _init_pose_local(self, target, cache, init_result):
        """Warm start: reuse a previous stage's result, else skeleton transfer."""
        B = target.shape[0]
        J = len(cache["parents"])
        root_idx = cache["root_idx"]
        if init_result is not None:
            pose_local = jnp.zeros((B, J, 4, 4), dtype=target.dtype)
            pose_local = pose_local.at[:, :, :3, :3].set(init_result["rotations"])
            pose_local = pose_local.at[:, root_idx, :3, 3].set(init_result["root_translation"])
            return pose_local
        return joint_world_to_local(self._skel_transfer.fit(target), cache["parents"])

    def _fit_lie_algebra_gn(
        self, target, cache, n_iters, lambda_reg, leaf_weight, init_result=None,
    ) -> PoseInversionResult:
        """FK-aware dense Lie-algebra Gauss-Newton refinement.

        Solves all joint rotations at once through the Kinematic Lever Arm
        Jacobian ``q_{i,j} = sum_{k in D(j)} w_{i,k} (p_{i,k} - c_j)``, which
        captures that rotating a joint moves its whole subtree — not just its
        directly attached vertices, as an independent-joint approximation
        would assume.
        """
        B = target.shape[0]
        dtype = target.dtype
        W_weights = cache["skinning_weights"]
        W_bind_inv = cache["W_bind_inv"]
        parents = cache["parents"]
        J = len(parents)
        root_idx = cache["root_idx"]

        bind_shape = self._rest_shape_b(B)
        vert_weights = _normalized_vertex_weights(
            cache["joint_names"], parents, W_weights, leaf_weight,
            bind_shape=bind_shape, bind_joints=self._bind_joint_positions(cache),
        )
        W_bind_inv_b = _bexpand4(W_bind_inv, B)
        pose_local = self._init_pose_local(target, cache, init_result)

        # A[j, k] = 1 when k lies in the subtree of j.
        A = np.zeros((J, J), dtype=np.float32)
        for k in range(J):
            cur = k
            while True:
                A[cur, k] = 1.0
                par = int(parents[cur])
                if par == cur or par < 0:
                    break
                cur = par
        A = jnp.asarray(A)

        AW = A @ W_weights.T  # (J, V) total descendant weight per vertex
        # Only joints whose subtree influences geometry are solvable; the rest
        # give structurally singular blocks and are factored out.
        active_np = np.asarray(jnp.any(AW > 0, axis=1))
        active_idx = np.where(active_np)[0]
        K_act = len(active_idx)
        eye3 = jnp.eye(3, dtype=dtype)
        V_verts = bind_shape.shape[-2]
        safe_parents = np.where(parents < 0, np.arange(J), parents)
        is_self_parent = safe_parents == np.arange(J)

        for _ in range(n_iters):
            W_world = _build_world_transforms(pose_local, cache)
            D = jnp.einsum("bjmn,bjnp->bjmp", W_world, W_bind_inv_b)
            R_D = D[:, :, :3, :3]
            t_D = D[:, :, :3, 3]
            c = W_world[:, :, :3, 3]

            p_world = jnp.einsum("bjmn,bvn->bjvm", R_D, bind_shape) + t_D[:, :, None, :]
            v_curr = jnp.einsum("vj,bjvm->bvm", W_weights, p_world)
            residual = target - v_curr

            weighted_p = p_world * W_weights.T[None, :, :, None]
            AWP = (A @ weighted_p.reshape(B, J, V_verts * 3)).reshape(B, J, V_verts, 3)
            q_full = AWP - AW[None, :, :, None] * c[:, :, None, :]
            q = q_full[:, active_idx]

            e_exp = jnp.broadcast_to(residual[:, None, :, :], (B, K_act, V_verts, 3))
            if vert_weights is not None:
                wv = vert_weights.reshape(1, 1, V_verts, 1)
                Jte_act = (jnp.cross(q, e_exp) * wv).sum(axis=2)
                q_weighted = q * wv
            else:
                Jte_act = jnp.cross(q, e_exp).sum(axis=2)
                q_weighted = q

            dot_sum = jnp.einsum("bjvm,bkvm->bjk", q_weighted, q)
            outer_sum = jnp.einsum("bjvn,bkvm->bjkmn", q_weighted, q)
            JtJ_blocks = dot_sum[..., None, None] * eye3 - outer_sum

            JtJ = JtJ_blocks.transpose(0, 1, 3, 2, 4).reshape(B, K_act * 3, K_act * 3)
            diag_idx = jnp.arange(K_act * 3)
            JtJ = JtJ.at[:, diag_idx, diag_idx].multiply(1.0 + lambda_reg)

            delta_act = _solve_lie_gn_normal_equations(
                JtJ, Jte_act.reshape(B, K_act * 3)
            ).reshape(B, K_act, 3)
            delta_omega = jnp.zeros((B, J, 3), dtype=dtype).at[:, active_idx].set(delta_act)

            # Per-frame backtracking line search: near the model-mismatch floor
            # the linearization stops being trustworthy, so a step is kept only
            # where it actually reduces that frame's error.
            residual_norm = jnp.linalg.norm(residual, axis=-1)
            if vert_weights is not None:
                pre_err = (residual_norm * vert_weights[None]).sum(axis=-1) / jnp.maximum(
                    vert_weights.sum(), jnp.finfo(dtype).eps
                )
            else:
                pre_err = residual_norm.mean(axis=-1)

            R_world = W_world[:, :, :3, :3]
            R_world_parents = R_world[:, safe_parents]
            pose_local_accepted = pose_local
            accepted_err = pre_err

            for alpha in _LINE_SEARCH_ALPHAS:
                dR = jax.vmap(jax.vmap(axis_angle_to_rotmat))(alpha * delta_omega)
                R_world_new = dR @ R_world
                R_local_try = jnp.einsum("bjnm,bjnp->bjmp", R_world_parents, R_world_new)
                R_local_try = jnp.where(
                    jnp.asarray(is_self_parent)[None, :, None, None], R_world_new, R_local_try
                )
                pose_local_try = pose_local.at[:, :, :3, :3].set(R_local_try)
                err_try = self._per_vertex_error(pose_local_try, target, cache)
                if vert_weights is not None:
                    err_try = (err_try * vert_weights[None]).sum(axis=-1) / jnp.maximum(
                        vert_weights.sum(), jnp.finfo(dtype).eps
                    )
                else:
                    err_try = err_try.mean(axis=-1)
                improved = err_try < accepted_err
                pose_local_accepted = jnp.where(
                    improved[:, None, None, None], pose_local_try, pose_local_accepted
                )
                accepted_err = jnp.where(improved, err_try, accepted_err)

            pose_local = pose_local_accepted
            pose_local = _update_root_translation(pose_local, target, cache, vert_weights)

        return self._result(pose_local, target, cache)

    def _fit_autograd_fk(
        self, target, cache, n_iters, lr, translation_lr_scale, leaf_weight,
        pose_prior=0.0, pose_prior_weights=None, init_result=None,
    ) -> PoseInversionResult:
        """Adam refinement of local 6D rotations + root translation through FK + LBS."""
        B = target.shape[0]
        dtype = target.dtype
        parents = cache["parents"]
        J = len(parents)
        root_idx = cache["root_idx"]
        has_virtual_root = root_idx > 0

        bind_shape = self._rest_shape_b(B)
        bone_weights = cache["bone_weights"]
        bone_indices = cache["bone_indices"]
        W_bind_inv_b = _bexpand4(cache["W_bind_inv"], B)
        bind_local_t = _bexpand(cache["bind_local_t"], B)

        pose_local_init = self._init_pose_local(target, cache, init_result)
        R_local_init = pose_local_init[:, :, :3, :3]
        root_t_init = pose_local_init[:, root_idx, :3, 3]
        if has_virtual_root:
            R_local_init = R_local_init.at[:, 0].set(jnp.eye(3, dtype=dtype))

        vert_weights = _normalized_vertex_weights(
            cache["joint_names"], parents, cache["skinning_weights"], leaf_weight,
            bind_shape=bind_shape, bind_joints=self._bind_joint_positions(cache),
        )
        joint_prior_weights = _joint_pose_prior_weights(cache["joint_names"], pose_prior_weights)
        prior_slice = slice(1, None) if has_virtual_root else slice(None)

        first = 1 if has_virtual_root else 0
        root_6d = jnp.broadcast_to(
            jnp.eye(3, dtype=dtype)[:2, :].reshape(1, 1, 6), (B, 1, 6)
        )
        params = {
            "rot6d": R_local_init[:, first:, :2, :].reshape(B, J - first, 6),
            "transl": root_t_init,
        }

        def all_rotations(rot6d):
            r6 = rot6d.reshape(B, J - first, 6)
            if has_virtual_root:
                r6 = jnp.concatenate([root_6d, r6], axis=1)
            return jax.vmap(jax.vmap(rotation_6d_to_rotmat))(r6)

        def vertices_of(rot6d, transl):
            R_local = all_rotations(rot6d)
            local_t = bind_local_t.at[:, root_idx].set(transl)
            T_local = se3_from_rt(R_local, local_t)
            W = fk_levelorder_transforms(T_local, cache["levels"], parents)
            D = jnp.einsum("bjmn,bjnp->bjmp", W, W_bind_inv_b)
            return _skin(bind_shape, bone_weights, bone_indices, D), R_local

        def loss_fn(p):
            verts, R_local = vertices_of(p["rot6d"], p["transl"])
            if vert_weights is not None:
                loss = (vert_weights[None, :, None] * (verts - target) ** 2).mean()
            else:
                loss = ((verts - target) ** 2).mean()
            if pose_prior > 0.0:
                R_delta = R_local[:, prior_slice] - R_local_init[:, prior_slice]
                if joint_prior_weights is not None:
                    w = joint_prior_weights[prior_slice].reshape(1, -1, 1, 1)
                    loss = loss + pose_prior * (w * R_delta ** 2).mean()
                else:
                    loss = loss + pose_prior * (R_delta ** 2).mean()
            return loss

        optimizer = optax.multi_transform(
            {"rot": optax.adam(lr), "transl": optax.adam(lr * translation_lr_scale)},
            {"rot6d": "rot", "transl": "transl"},
        )

        def step(carry, _):
            p, state = carry
            grads = jax.grad(loss_fn)(p)
            updates, state = optimizer.update(grads, state)
            return (optax.apply_updates(p, updates), state), None

        (params, _), _ = jax.lax.scan(
            step, (params, optimizer.init(params)), None, length=n_iters
        )

        verts, R_local = vertices_of(params["rot6d"], params["transl"])
        per_vertex_error = jnp.linalg.norm(verts - target, axis=-1)

        R_rel = jnp.einsum(
            "bjmn,bjpn->bjmp", R_local[:, prior_slice], R_local_init[:, prior_slice]
        )
        cos_angle = (jnp.trace(R_rel, axis1=-2, axis2=-1) - 1.0) * 0.5
        drift = jnp.zeros((B, J), dtype=dtype).at[:, prior_slice].set(
            jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        )

        return PoseInversionResult(
            rotations=R_local,
            root_translation=params["transl"],
            per_vertex_error=per_vertex_error,
            local_rotation_drift=drift,
            root_translation_drift=jnp.linalg.norm(params["transl"] - root_t_init, axis=-1),
        )

    def roundtrip(self, posed_vertices, **kwargs) -> tuple[jnp.ndarray, PoseInversionResult]:
        """Invert then re-pose, for verification.

        Returns:
            ``(vertices, result)`` — the reconstruction on the refit topology
            and the :class:`PoseInversionResult` that produced it.
        """
        result = self.fit(posed_vertices, **kwargs)
        cache = self._cache
        B = result["rotations"].shape[0]
        J = len(cache["parents"])
        pose_local = jnp.zeros((B, J, 4, 4), dtype=result["rotations"].dtype)
        pose_local = pose_local.at[:, :, :3, :3].set(result["rotations"])
        pose_local = pose_local.at[:, cache["root_idx"], :3, 3].set(result["root_translation"])
        W = _build_world_transforms(pose_local, cache)
        D = jnp.einsum("bjmn,bjnp->bjmp", W, _bexpand4(cache["W_bind_inv"], B))
        vertices = _skin(
            self._rest_shape_b(B), cache["bone_weights"], cache["bone_indices"], D
        )
        return vertices, result
