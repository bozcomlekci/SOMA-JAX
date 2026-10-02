"""Batched forward kinematics and linear blend skinning for SOMA-JAX.

Pipeline (``third_party/SOMA-X/soma/geometry/batched_skinning.py``):

    R_oriented = orient_parent_T @ R_in @ orient   (if not absolute_pose)
    local_t    = bind_local_translations           (the translation joint's
                                                    slot replaced by the
                                                    global translation, so the
                                                    root motion is injected
                                                    into FK, not added as a
                                                    rigid post-LBS shift)
    T_local    = SE3(R_oriented, local_t)
    T_world    = level-order FK of T_local
    [optional align_translation: anchor the translation joint's X and Z, then
     Y-shift T_world so the lowest joint lands at the requested floor height]
    T_bone     = T_world @ inverse_bind
    verts      = sum_j W[v,j] * (T_bone_j applied to bind_shape[v])

Upstream: ``soma/geometry/batched_skinning.py``
    :class:`BatchedSkinning`, :class:`FKTopology` and :func:`topk_skinning` are
    ports of upstream's, with the same constructor, attributes and methods
    (``mode="warp"`` evaluates upstream's sparse top-8 kernel in JAX).
    :func:`pose_from_bind` is the functional form the SOMA layers use, and
    :class:`RestJointSkinning` — rest-joint bind derivation, ``(vertices,
    joints)`` output — is a SOMA-JAX extra kept for the legacy repose of assets
    without bind data.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .lbs import (
    lbs,
    lbs_blend,
    lbs_sparse,
    compute_skeleton_levels,
)
from .rig_utils import (
    apply_joint_orient_local,
    compute_skeleton_levels as rig_compute_skeleton_levels,
    joint_local_to_world_levelorder,
    joint_world_to_local,
    precompute_joint_orient,
)
from .transforms import se3_from_rt, se3_inverse


def pose_from_bind(
    bind_world: jnp.ndarray,
    rest_verts: jnp.ndarray,
    weights: jnp.ndarray,
    skeleton_levels: list,
    parents: np.ndarray,
    local_rotmats: jnp.ndarray,
    hips_translation: jnp.ndarray,
    hips_idx: int = 1,
    weight_values: Optional[jnp.ndarray] = None,
    weight_indices: Optional[jnp.ndarray] = None,
    local_translation_scales: Optional[jnp.ndarray] = None,
    skip_lbs: bool = False,
    align_translation: Optional[jnp.ndarray] = None,
) -> tuple[Optional[jnp.ndarray], jnp.ndarray]:
    """Functional rebind + pose: skin ``rest_verts`` against per-batch bind
    world transforms.

    This is the jit-able equivalent of SOMA-X's per-identity flow
    ``BatchedSkinning.rebind(bind_transforms, rest_shape)`` followed by
    ``BatchedSkinning.pose(...)`` — used when the bind transforms come from a
    per-identity ``SkeletonTransfer.fit`` and therefore differ across the
    batch (the eqx :class:`RestJointSkinning` module precomputes its bind state
    from a single rest skeleton at construction time, so it cannot express
    batched binds inside ``jax.jit``).

    Math (identical to the module's ``pose``):

        bind_local_t[j] = R_bind[parent(j)]^T @ (t_bind[j] - t_bind[parent(j)])
        T_local[j]      = SE3(R_in[j], bind_local_t[j])   (hips slot replaced
                                                           by hips_translation)
        T_world         = level-order FK of T_local
        T_bone          = T_world @ inverse(bind_world)
        verts           = LBS(rest_verts, T_bone, weights)

    Args:
        bind_world: (B, J, 4, 4) per-identity bind world transforms
            (e.g. ``SkeletonTransfer.fit`` output).
        rest_verts: (B, V, 3) per-identity rest mesh (same identities).
        weights: (V, J) dense skinning weights.
        skeleton_levels: output of :func:`compute_skeleton_levels` for
            ``parents`` (precomputed once — static under jit).
        parents: (J,) parent indices (numpy, static under jit).
        local_rotmats: (B, J, 3, 3) local joint rotations in the absolute
            skinning frame (i.e. post-joint-orient, or identity for the bind
            pose itself).
        hips_translation: (B, 3) world position injected into the hips
            joint's local translation slot (SOMA-X semantic — the body root
            MOVES TO this position).
        hips_idx: which joint receives ``hips_translation`` (SOMA rig: 1 =
            Hips, child of the virtual Root at 0).
        weight_values: optional (V, K) sparse top-K skinning weights. When
            given together with ``weight_indices``, LBS runs the sparse
            kernel — matching SOMA-X's Warp path, which skins with top-8
            sparse weights (``topk_skinning(W, K=8)``), not the dense matrix.
        weight_indices: optional (V, K) joint indices for ``weight_values``.
        local_translation_scales: optional (B, J) per-joint bone-length
            multipliers applied to the parent-relative bind translations
            before FK — SOMA-X's ``local_translations`` override, which is how
            ``scale_params`` stretch individual bones. The hips slot is
            unaffected since it carries ``hips_translation`` instead.
        skip_lbs: run forward kinematics only and return ``None`` for the
            vertices (SOMA-X's ``fk_only``).
        align_translation: optional (3,) or (B, 3) anchor, upstream's
            ``BatchedSkinning.pose(align_translation=...)``: its X and Z
            replace the hips joint's local translation (``hips_translation`` is
            then ignored, as upstream ignores ``global_translation``), and after
            FK the whole skeleton is Y-shifted so its lowest joint lands at the
            anchor's Y. Upstream reposes identities to the bind pose this way.

    Returns:
        ``(posed_verts, T_world)`` — (B, V, 3) skinned vertices (``None`` when
        ``skip_lbs``) and (B, J, 4, 4) world joint transforms.

    Batch dims broadcast: a singleton pose drives every identity in a batch of
    binds (and vice versa), as upstream's ``BatchedSkinning`` FK does. Sizes
    other than 1 must agree.
    """
    B = max(local_rotmats.shape[0], bind_world.shape[0], rest_verts.shape[0],
            hips_translation.shape[0])

    def _batch(x):
        return x if x is None or x.shape[0] == B else jnp.broadcast_to(x, (B,) + x.shape[1:])

    local_rotmats, bind_world, rest_verts, hips_translation, local_translation_scales = (
        _batch(local_rotmats), _batch(bind_world), _batch(rest_verts),
        _batch(hips_translation), _batch(local_translation_scales))
    # XLA in jaxlib <= 0.6.2 (the newest wheel for Python 3.10) constant-folds
    # a gather along a non-leading axis incorrectly: under `jax.jit`, the
    # `t - t[:, parents]` below evaluates wrongly when a prepared identity's
    # binds are closed over as constants (a clip posed with them came out
    # metres off). jaxlib 0.11 folds it correctly. The barrier stops the fold.
    bind_world, rest_verts = jax.lax.optimization_barrier((bind_world, rest_verts))
    J = local_rotmats.shape[1]
    parents_np = np.asarray(parents).astype(np.int64)

    # Bind-local translations from the per-batch bind world transforms:
    # root keeps its world position; child j gets parent-frame offset.
    # NOTE: slice into R / t blocks FIRST (basic indexing), then gather
    # parents with a single advanced index. Combining `[:, parents, :3, 3]`
    # in one step triggers numpy's advanced-indexing reordering (the slice
    # between the two advanced indices moves the gathered dim to the front),
    # silently producing (J, B, 3) instead of (B, J, 3).
    safe_parents = np.maximum(parents_np, 0)
    R_all = bind_world[..., :3, :3]                                  # (B, J, 3, 3)
    t_all = bind_world[..., :3, 3]                                   # (B, J, 3)
    R_parent = R_all[:, safe_parents]                                # (B, J, 3, 3)
    t_self = t_all
    t_parent = t_all[:, safe_parents]                                # (B, J, 3)
    delta = t_self - t_parent
    local_t = jnp.einsum("bjnm,bjn->bjm", R_parent, delta)           # R^T @ delta
    is_root = jnp.asarray(parents_np < 0)[None, :, None]
    local_t = jnp.where(is_root, t_self, local_t)

    # Bone-length scaling stretches each parent-to-child offset before FK, so
    # the change propagates down the chain exactly as SOMA-X's
    # `local_translations` override does.
    if local_translation_scales is not None:
        local_t = local_t * local_translation_scales[..., None]

    # Replace the hips slot with the requested world position (one-hot mask
    # keeps this autograd-safe and jit-friendly).
    j_mask = jax.nn.one_hot(hips_idx, J, dtype=local_t.dtype)[None, :, None]
    if align_translation is not None:
        # Upstream anchors only X and Z of the translation joint (mask [0, 2])
        # and leaves Y to the floor shift below.
        anchor = jnp.broadcast_to(
            jnp.asarray(align_translation, local_t.dtype).reshape(-1, 3), (B, 3))
        m = j_mask * jnp.asarray([1.0, 0.0, 1.0], dtype=local_t.dtype)[None, None, :]
        local_t = local_t * (1.0 - m) + anchor[:, None, :] * m
    else:
        local_t = local_t * (1.0 - j_mask) + hips_translation[:, None, :] * j_mask

    # FK in level order.
    T_local = se3_from_rt(local_rotmats, local_t)                    # (B, J, 4, 4)
    T_world = T_local
    for level in skeleton_levels[1:]:
        joint_ids = np.asarray(level, dtype=np.int64)
        parent_ids = parents_np[joint_ids]
        T_world = T_world.at[:, joint_ids].set(
            jnp.einsum("bjmn,bjnp->bjmp",
                       T_world[:, parent_ids], T_local[:, joint_ids])
        )

    if align_translation is not None:
        # Floor lock: the lowest joint (over the whole skeleton) lands on the
        # anchor's Y.
        shift = T_world[..., 1, 3].min(axis=1, keepdims=True) + anchor[:, 1:2]
        T_world = T_world.at[..., 1, 3].add(-shift)

    if skip_lbs:
        return None, T_world

    # Bone transforms against the per-batch inverse bind, then LBS
    # (sparse top-K when the caller provides precomputed sparse weights —
    # matching SOMA-X's Warp kernels — dense otherwise).
    inv_bind = se3_inverse(bind_world)                               # (B, J, 4, 4)
    bone_T = jnp.einsum("bjmn,bjnp->bjmp", T_world, inv_bind)        # (B, J, 4, 4)
    bone_Rt = bone_T[..., :3, :]                                     # (B, J, 3, 4)
    zeros = jnp.zeros_like(rest_verts)
    if weight_values is not None and weight_indices is not None:
        posed = lbs_sparse(rest_verts, zeros, bone_Rt, weight_values, weight_indices)
    else:
        posed = lbs_blend(rest_verts, zeros, bone_Rt, weights)
    return posed, T_world


def topk_skinning(
    W: np.ndarray,
    K: int = 8,
    weight_eps: float = 1e-12,
    sort_desc: bool = True,
    pad_index: int = -1,
    dtype_idx=np.int32,
    dtype_w=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense (V, J) skinning weights -> sparse top-K indices/values.

    Port of upstream ``topk_skinning``, including the behaviours that matter at
    the edges: weights at or below ``weight_eps`` are pruned before the
    selection, fewer-than-K influences are **padded** with ``pad_index`` and
    zero weight, and rows that sum to zero stay zero instead of dividing by a
    fudge factor. ``k`` defaults to 8, matching upstream and SOMA-X's Warp path.

    Args:
        W: (V, J) dense skinning weights.
        K: influences to keep per vertex.
        weight_eps: prune weights at or below this before selecting.
        sort_desc: return the K influences in descending weight order.
        pad_index: joint index used to pad when J < K.
        dtype_idx: dtype of the index output.
        dtype_w: dtype of the weight output.

    Returns:
        (indices (V, K), weights (V, K)).
    """
    W = np.asarray(W, dtype=np.float32)
    V, J = W.shape
    k = int(K)
    k_eff = min(k, J)

    W_masked = np.where(W > weight_eps, W, 0.0)

    # Top-k_eff by weight; np.argpartition then sort the slice for stability.
    idx = np.argpartition(-W_masked, kth=k_eff - 1, axis=1)[:, :k_eff]
    vals = np.take_along_axis(W_masked, idx, axis=1)
    if sort_desc:
        order = np.argsort(-vals, axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)

    if k_eff < k:
        pad = k - k_eff
        idx = np.concatenate([idx, np.full((V, pad), pad_index, idx.dtype)], axis=1)
        vals = np.concatenate([vals, np.zeros((V, pad), vals.dtype)], axis=1)

    total = vals.sum(axis=1, keepdims=True)
    vals = np.where(total > 0, vals / np.clip(total, 1e-20, None), 0.0)
    return idx.astype(dtype_idx), vals.astype(dtype_w)


def _bind_world_from_rest(rest_joints: np.ndarray,
                          joint_orient: Optional[np.ndarray]) -> np.ndarray:
    """Build per-joint world bind transforms.

    rest_joints: (J, 3)
    joint_orient: (J, 3, 3) — world bind rotation per joint, identity if None.
    Returns: (J, 4, 4)
    """
    J = rest_joints.shape[0]
    R = np.eye(3, dtype=np.float32)[None].repeat(J, axis=0) if joint_orient is None \
        else np.asarray(joint_orient, dtype=np.float32)
    T = np.zeros((J, 4, 4), dtype=np.float32)
    T[:, :3, :3] = R
    T[:, :3, 3] = np.asarray(rest_joints, dtype=np.float32)
    T[:, 3, 3] = 1.0
    return T


def _bind_local_t_from_world(bind_world: np.ndarray,
                             parents: np.ndarray) -> np.ndarray:
    """Per-joint translation in the parent's bind frame (used to drive FK).

    For root, this is the joint's own world position (the chain anchor).
    For non-root joints j with parent p:
        bind_local_t[j] = bind_world[p].rotation.T @ (bind_world[j].t - bind_world[p].t)
    """
    J = bind_world.shape[0]
    parents_np = np.asarray(parents).astype(int)
    out = np.zeros((J, 3), dtype=np.float32)
    for j in range(J):
        p = parents_np[j]
        if p < 0 or p == j:
            out[j] = bind_world[j, :3, 3]
            continue
        R_p_T = bind_world[p, :3, :3].T
        delta = bind_world[j, :3, 3] - bind_world[p, :3, 3]
        out[j] = R_p_T @ delta
    return out


def _se3_inverse_np(T: np.ndarray) -> np.ndarray:
    """Numpy SE(3) inverse for the (J, 4, 4) precompute."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_T = np.swapaxes(R, -2, -1)
    out = np.zeros_like(T)
    out[..., :3, :3] = R_T
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R_T, t)
    out[..., 3, 3] = 1.0
    return out


class RestJointSkinning(eqx.Module):
    """SMPL-style batched skinning built from rest joints (SOMA-JAX extra).

    The same FK + LBS pipeline as :class:`BatchedSkinning`, but the bind
    transforms are derived from rest joint positions and an optional per-joint
    orientation instead of being passed in, and :meth:`pose` returns
    ``(vertices, joints)``. It backs the legacy repose of assets without bind
    data; :class:`BatchedSkinning` is upstream's class.

    Attributes:
        rest_verts: (V, 3) rest-pose template vertices.
        rest_joints: (J, 3) rest joint positions.
        weights: (V, J) dense skinning weights.
        weight_indices: (V, K) sparse top-K joint indices (if use_sparse).
        weight_values: (V, K) sparse top-K weights (if use_sparse).
        joint_orient: optional (J, 3, 3) world bind orientation per joint.
        bind_local_t: (J, 3) per-joint translation in parent's bind frame.
        inverse_bind: (J, 4, 4) inverse of bind world transform per joint.
        hips_idx: which joint receives the per-frame `hips_translation`.
    """

    rest_verts: jnp.ndarray                             # (V, 3)
    rest_joints: jnp.ndarray                            # (J, 3)
    weights: jnp.ndarray                                # (V, J)
    weight_indices: Optional[jnp.ndarray]               # (V, K)
    weight_values: Optional[jnp.ndarray]                # (V, K)
    joint_orient: Optional[jnp.ndarray]                 # (J, 3, 3)
    bind_local_t: jnp.ndarray                           # (J, 3)
    inverse_bind: jnp.ndarray                           # (J, 4, 4)
    _parents_np: np.ndarray = eqx.field(static=True)
    skeleton_levels: list = eqx.field(static=True)
    use_sparse: bool = eqx.field(static=True)
    hips_idx: int = eqx.field(static=True)

    def __init__(
        self,
        rest_verts: np.ndarray,
        rest_joints: np.ndarray,
        weights: np.ndarray,
        parents: np.ndarray,
        joint_orient: Optional[np.ndarray] = None,
        sparse_k: int = 8,
        hips_idx: int = 1,
    ):
        """Defaults follow upstream ``BatchedSkinning``: ``K=8`` influences
        per vertex, and joint 1 (Hips) receives the global translation —
        joint 0 is SOMA's *virtual* Root, which must stay identity."""
        self.rest_verts = jnp.array(rest_verts, dtype=jnp.float32)
        self.rest_joints = jnp.array(rest_joints, dtype=jnp.float32)
        self.weights = jnp.array(weights, dtype=jnp.float32)
        self._parents_np = np.asarray(parents, dtype=np.int32)
        self.skeleton_levels = compute_skeleton_levels(self._parents_np)
        self.hips_idx = int(hips_idx)

        if sparse_k < weights.shape[1]:
            idx, val = topk_skinning(np.asarray(weights), sparse_k)
            self.weight_indices = jnp.array(idx)
            self.weight_values = jnp.array(val)
            self.use_sparse = True
        else:
            self.weight_indices = None
            self.weight_values = None
            self.use_sparse = False

        self.joint_orient = (
            jnp.array(joint_orient, dtype=jnp.float32)
            if joint_orient is not None else None
        )

        # Precompute bind world + inverse + parent-local bind translations
        # so pose() can inject `hips_translation` directly into the FK chain.
        bind_world = _bind_world_from_rest(np.asarray(rest_joints),
                                           np.asarray(joint_orient) if joint_orient is not None else None)
        self.bind_local_t = jnp.asarray(
            _bind_local_t_from_world(bind_world, self._parents_np))
        self.inverse_bind = jnp.asarray(_se3_inverse_np(bind_world))

    def pose(
        self,
        local_rotmats: jnp.ndarray,
        hips_translation: jnp.ndarray,
        absolute_pose: bool = False,
        return_transforms: bool = False,
        align_translation: Optional[jnp.ndarray] = None,
    ):
        """Pose the rest mesh — faithful SOMA-X pipeline.

        Args:
            local_rotmats: (B, J, 3, 3) local rotation matrices.
            hips_translation: (B, 3) world position for the hips joint. Replaces
                `bind_local_t[hips_idx]`, so the body root MOVES TO this position
                (rather than being shifted by it). For the SMPL family where the
                root IS the hips, this is the standard semantic.
            absolute_pose: if True, ``local_rotmats`` are already in the absolute
                skinning frame (e.g. absolute rotations) — skip the
                joint-orient remap. If False (default), they are T-pose-relative
                and get conjugated by `orient_parent_T @ R @ orient`.
            return_transforms: if True, also return the per-joint world
                transforms (B, J, 4, 4).
            align_translation: optional (B, 3) anchor. Its X and Z replace the
                translation joint's local translation (upstream masks
                components [0, 2]); its Y is the floor height the posed mesh is
                shifted onto. The entire
                skeleton is Y-shifted so the lowest joint lands at this height.

        Returns:
            (verts, joints) or (verts, joints, T_world) when return_transforms.
        """
        if not absolute_pose and self.joint_orient is not None:
            local_rotmats = apply_joint_orient_local(
                local_rotmats, self.joint_orient, self._parents_np)

        B, J = local_rotmats.shape[:2]

        # Build per-frame local translations: copy bind_local_t and replace the
        # hips slot with the per-frame `hips_translation`. One-hot mask avoids
        # in-place writes (autograd-safe, jit-friendly).
        local_t = jnp.broadcast_to(self.bind_local_t[None], (B, J, 3))
        j_mask = jax.nn.one_hot(self.hips_idx, J, dtype=local_t.dtype)[None, :, None]
        if align_translation is not None:
            # Upstream anchors the translation joint's X and Z to the requested
            # position and leaves Y to the floor shift below, rather than
            # replacing the whole vector (mask [0, 2] in
            # `soma/geometry/batched_skinning.py`). Anchoring all three would
            # also override the height the floor alignment is about to set.
            anchor = jnp.broadcast_to(align_translation[:, None, :], (B, J, 3))
            xz = jnp.asarray([1.0, 0.0, 1.0], dtype=local_t.dtype)[None, None, :]
            m = j_mask * xz
            local_t = local_t * (1.0 - m) + anchor * m
        else:
            hips_t_b = jnp.broadcast_to(hips_translation[:, None, :], (B, J, 3))
            local_t = local_t * (1.0 - j_mask) + hips_t_b * j_mask

        # Build T_local and FK in level order (parallel across joints at same depth).
        T_local = se3_from_rt(local_rotmats, local_t)                  # (B, J, 4, 4)
        T_world = self._fk_levelorder(T_local)                          # (B, J, 4, 4)

        if align_translation is not None:
            y_world = T_world[..., 1, 3]                                # (B, J)
            y_floor = y_world.min(axis=1, keepdims=True)                # (B, 1)
            shift = y_floor + align_translation[:, 1:2]
            T_world = T_world.at[..., 1, 3].add(-shift)

        # Bone transform = T_world @ inverse_bind  (full 4x4 product, then slice).
        bone_T = jnp.einsum("bjmn,jnp->bjmp", T_world, self.inverse_bind)  # (B, J, 4, 4)

        # LBS with the standard (R, t) form
        bone_Rt = bone_T[..., :3, :]                                    # (B, J, 3, 4)
        rest_verts_b = jnp.broadcast_to(self.rest_verts[None], (B,) + self.rest_verts.shape)
        zeros = jnp.zeros_like(rest_verts_b)
        if self.use_sparse:
            posed = lbs_sparse(
                rest_verts_b, zeros, bone_Rt,
                self.weight_values, self.weight_indices,
            )
        else:
            posed = lbs_blend(rest_verts_b, zeros, bone_Rt, self.weights)

        posed_joints = T_world[..., :3, 3]

        if return_transforms:
            return posed, posed_joints, T_world
        return posed, posed_joints

    def _fk_levelorder(self, T_local: jnp.ndarray) -> jnp.ndarray:
        """Level-order forward kinematics on full SE(3) transforms.

        Args:
            T_local: (B, J, 4, 4) local transforms.

        Returns:
            (B, J, 4, 4) world transforms.
        """
        parents = self._parents_np
        T_world = T_local
        # Levels[0] = root(s); their world transform = local transform.
        for level in self.skeleton_levels[1:]:
            joint_ids = np.asarray(level, dtype=np.int64)
            parent_ids = parents[joint_ids].astype(np.int64)
            T_world = T_world.at[:, joint_ids].set(
                jnp.einsum("bjmn,bjnp->bjmp",
                           T_world[:, parent_ids],
                           T_local[:, joint_ids])
            )
        return T_world

    def rebind(self, new_rest_verts: jnp.ndarray, new_rest_joints: jnp.ndarray) -> "RestJointSkinning":
        """Return a new RestJointSkinning with updated rest shape, sharing weights.

        Recomputes bind_local_t / inverse_bind for the new rest positions —
        without this rebuild the FK chain would still use the old skeleton.

        Args:
            new_rest_verts: (V, 3) new rest vertices.
            new_rest_joints: (J, 3) new rest joint positions.

        Returns:
            New RestJointSkinning instance.
        """
        new_joints_np = np.asarray(new_rest_joints)
        new_jorient_np = np.asarray(self.joint_orient) if self.joint_orient is not None else None
        bind_world = _bind_world_from_rest(new_joints_np, new_jorient_np)
        new = eqx.tree_at(lambda m: m.rest_verts, self, new_rest_verts)
        new = eqx.tree_at(lambda m: m.rest_joints, new, new_rest_joints)
        new = eqx.tree_at(lambda m: m.bind_local_t, new,
                          jnp.asarray(_bind_local_t_from_world(bind_world, self._parents_np)))
        new = eqx.tree_at(lambda m: m.inverse_bind, new, jnp.asarray(_se3_inverse_np(bind_world)))
        return new


# ---------------------------------------------------------------------------
# Upstream ``BatchedSkinning`` (soma/geometry/batched_skinning.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FKTopology:
    """Optional FK-only source topology for a target LBS rig.

    Upstream ``FKTopology``: :class:`BatchedSkinning` always owns one target
    topology for LBS. A source topology describes a related FK rig — the public
    skeleton of a procedural layer — whose world transforms caller-owned logic
    expands into the target topology.
    """

    parent_ids: Sequence[int]
    target_joint_indices: Optional[Sequence[int]] = None
    joint_orient: Optional[jnp.ndarray] = None
    global_translation_joint_idx: Optional[int] = None
    bind_world_transforms: Optional[jnp.ndarray] = None


class BatchedSkinning:
    """Cache-friendly FK + LBS for repeated posing against a fixed rig topology.

    Port of upstream ``BatchedSkinning``: constructor, attributes and methods,
    including the optional FK-only source topology (:class:`FKTopology`).
    ``mode="warp"`` skins with the top-8 sparse weights of
    :func:`topk_skinning`, as upstream's Warp kernel does; ``"dense"`` uses the
    full weights through :func:`~soma_jax.geometry.lbs.lbs`. The object holds
    JAX arrays and its methods are traceable; :meth:`rebind` updates it in
    place, as upstream's does. Upstream's ``device`` attribute has no JAX
    counterpart.

    Args:
        joint_parent_ids: (J,) parent indices.
        skinning_weights: (V, J) skinning weights.
        bind_world_transforms: (B, J, 4, 4) or (J, 4, 4) bind poses in world space.
        bind_shapes: (B, V, 3) or (V, 3) bind-pose vertices.
        joint_orient: optional (J, M, M), M >= 3, initial world orientation of
            each joint; poses are relative to it unless ``absolute_pose``.
        mode: ``"warp"`` (sparse top-8, default) or ``"dense"``.
        global_translation_joint_idx: joint receiving ``global_translation``;
            default 1 (Hips under the full-body virtual Root).
        root_joint_idx: deprecated alias for ``global_translation_joint_idx``.
        source_fk: optional FK-only source topology.
    """

    def __init__(
        self,
        joint_parent_ids,
        skinning_weights,
        bind_world_transforms,
        bind_shapes,
        joint_orient=None,
        mode: str = "warp",
        global_translation_joint_idx: Optional[int] = None,
        root_joint_idx: Optional[int] = None,
        *,
        source_fk: Optional[FKTopology] = None,
    ) -> None:
        if global_translation_joint_idx is None:
            global_translation_joint_idx = root_joint_idx if root_joint_idx is not None else 1
        skinning_weights = jnp.asarray(skinning_weights)
        bind_world_transforms = jnp.asarray(bind_world_transforms)
        bind_shapes = jnp.asarray(bind_shapes)
        dtypes = {x.dtype for x in (skinning_weights, bind_world_transforms, bind_shapes)}
        if len(dtypes) != 1:
            raise TypeError(f"All BatchedSkinning inputs must share dtype; got {dtypes}.")
        self.dtype = dtypes.pop()
        self.mode = mode

        batched = bind_shapes.ndim == 3
        num_joints = len(joint_parent_ids)
        if num_joints != bind_world_transforms.shape[1 if batched else 0]:
            raise ValueError(
                "joint_parent_ids and bind_world_transforms must have the same number of joints.")
        if batched and bind_world_transforms.shape[0] != bind_shapes.shape[0]:
            raise ValueError("bind_world_transforms and bind_shapes must have the same batch size.")
        self.bind_batched = batched
        self.num_joints = num_joints
        self.global_translation_joint_idx = global_translation_joint_idx
        self.joint_parent_ids = self._as_parent_list(joint_parent_ids)
        self.skinning_weights = skinning_weights
        self.bind_world_transforms = bind_world_transforms
        bind_local_transforms, self.inverse_bind_transform = joint_world_to_local(
            bind_world_transforms, self.joint_parent_ids, return_inverse=True)
        self.local_rotations = bind_local_transforms[..., :3, :3]
        self.local_translations = bind_local_transforms[..., :3, 3]
        self.bind_shapes = bind_shapes
        self.joint_orient = None
        self._orient_parent_T = None
        if joint_orient is not None:
            joint_orient = jnp.asarray(joint_orient, self.dtype)
            if num_joints != joint_orient.shape[0]:
                raise ValueError(
                    "joint_orient must have the same number of joints as joint_parent_ids.")
            self.joint_orient, self._orient_parent_T = precompute_joint_orient(
                joint_orient, self.joint_parent_ids)

        self._levels = rig_compute_skeleton_levels(self.joint_parent_ids)

        self._bone_weights = None
        self._bone_indices = None
        if self.mode == "warp":
            self._prepare_warp_data()
        self.source_joint_parent_ids = None
        self.source_target_joint_indices = None
        self.source_joint_indices = None
        self.source_num_joints = 0
        self.source_global_translation_joint_idx = 0
        self.source_bind_world_transforms = None
        self.source_bind_batched = False
        self.source_local_rotations = None
        self.source_local_translations = None
        self.source_joint_orient = None
        self._source_orient_parent_T = None
        self._source_levels = None
        self._configure_source_fk(source_fk)

    @staticmethod
    def _as_parent_list(joint_parent_ids) -> list[int]:
        return [int(p) for p in np.asarray(joint_parent_ids).reshape(-1)]

    @staticmethod
    def _bind_joint_count(bind_world_transforms) -> int:
        return (bind_world_transforms.shape[1] if bind_world_transforms.ndim == 4
                else bind_world_transforms.shape[0])

    @staticmethod
    def _is_batched_bind(bind_world_transforms) -> bool:
        return bind_world_transforms.ndim == 4

    def _target_bind_subset(self, bind_world_transforms):
        if self.source_target_joint_indices is None:
            raise RuntimeError(
                "source_fk.target_joint_indices are required to derive source bind transforms")
        return bind_world_transforms[..., self.source_target_joint_indices, :, :]

    def _set_source_bind_world_transforms(self, source_bind_world_transforms) -> None:
        if self.source_joint_parent_ids is None:
            raise RuntimeError("Cannot set source bind transforms without source topology")
        source_bind_world_transforms = jnp.asarray(source_bind_world_transforms, self.dtype)
        if self._bind_joint_count(source_bind_world_transforms) != self.source_num_joints:
            raise ValueError(
                "source_bind_world_transforms must have the same number of joints as "
                "source_joint_parent_ids.")
        self.source_bind_world_transforms = source_bind_world_transforms
        self.source_bind_batched = self._is_batched_bind(source_bind_world_transforms)
        source_local_transforms = joint_world_to_local(
            source_bind_world_transforms, self.source_joint_parent_ids)
        self.source_local_rotations = source_local_transforms[..., :3, :3]
        self.source_local_translations = source_local_transforms[..., :3, 3]

    def _configure_source_fk(self, source_fk: Optional[FKTopology]) -> None:
        if source_fk is None:
            return
        if source_fk.parent_ids is None:
            raise ValueError("source_fk.parent_ids are required for source FK config.")
        if source_fk.target_joint_indices is None and source_fk.bind_world_transforms is None:
            raise ValueError(
                "source FK config requires source_fk.target_joint_indices or "
                "source_fk.bind_world_transforms.")

        self.source_joint_parent_ids = self._as_parent_list(source_fk.parent_ids)
        self.source_num_joints = len(self.source_joint_parent_ids)
        if source_fk.target_joint_indices is not None:
            self.source_target_joint_indices = np.asarray(
                source_fk.target_joint_indices, np.int64).reshape(-1)
            # Backward-compatible alias, as upstream keeps for older callers.
            self.source_joint_indices = self.source_target_joint_indices
            if self.source_target_joint_indices.size != self.source_num_joints:
                raise ValueError(
                    "source_fk.target_joint_indices must have the same length as "
                    "source_fk.parent_ids.")
        source_global_translation_joint_idx = source_fk.global_translation_joint_idx
        if source_global_translation_joint_idx is None:
            source_global_translation_joint_idx = (
                self.global_translation_joint_idx
                if self.global_translation_joint_idx < self.source_num_joints else 0)
        self.source_global_translation_joint_idx = source_global_translation_joint_idx

        source_bind_world_transforms = source_fk.bind_world_transforms
        if source_bind_world_transforms is None:
            source_bind_world_transforms = self._target_bind_subset(self.bind_world_transforms)
        self._set_source_bind_world_transforms(source_bind_world_transforms)

        if source_fk.joint_orient is not None:
            joint_orient = jnp.asarray(source_fk.joint_orient, self.dtype)
            if self.source_num_joints != joint_orient.shape[0]:
                raise ValueError(
                    "source_fk.joint_orient must have the same number of joints as "
                    "source_fk.parent_ids.")
            self.source_joint_orient, self._source_orient_parent_T = precompute_joint_orient(
                joint_orient, self.source_joint_parent_ids)
        self._source_levels = rig_compute_skeleton_levels(self.source_joint_parent_ids)

    def rebind(self, bind_world_transforms, bind_shapes) -> None:
        """Rebind the skeleton to new bind poses and shapes (in place).

        Args:
            bind_world_transforms: (B, J, 4, 4) new joint bind poses in world space.
            bind_shapes: (B, V, 3) new bind-pose vertices.
        """
        bind_world_transforms = jnp.asarray(bind_world_transforms)
        bind_shapes = jnp.asarray(bind_shapes)
        self.bind_batched = bind_shapes.ndim == 3
        self.bind_world_transforms = bind_world_transforms
        bind_local_transforms, self.inverse_bind_transform = joint_world_to_local(
            bind_world_transforms, self.joint_parent_ids, return_inverse=True)
        self.local_rotations = bind_local_transforms[..., :3, :3]
        self.local_translations = bind_local_transforms[..., :3, 3]
        self.bind_shapes = bind_shapes
        if (self.source_joint_parent_ids is not None
                and self.source_target_joint_indices is not None):
            self._set_source_bind_world_transforms(
                self._target_bind_subset(bind_world_transforms))

    def _prepare_warp_data(self) -> None:
        """Sparse top-8 bone weights and indices for the sparse LBS path."""
        bone_indices, bone_weights = topk_skinning(np.asarray(self.skinning_weights))
        self._bone_indices = jnp.asarray(bone_indices)
        self._bone_weights = jnp.asarray(bone_weights, self.dtype)

    def get_bone_weights(self):
        """Sparse (V, K) weights in ``"warp"`` mode, the dense (V, J) ones otherwise."""
        if self.mode == "warp":
            if self._bone_weights is None:
                self._prepare_warp_data()
            return self._bone_weights
        return self.skinning_weights

    def get_bone_indices(self):
        """Sparse (V, K) indices in ``"warp"`` mode, ``None`` otherwise."""
        if self.mode == "warp":
            if self._bone_indices is None:
                self._prepare_warp_data()
            return self._bone_indices
        return None

    def forward_kinematics(
        self,
        local_rotations,
        global_translation=None,
        align_translation=None,
        absolute_pose: bool = False,
        *,
        hips_translations=None,
        local_translations=None,
    ):
        """Batched forward kinematics: per-joint world transforms (B, J, 4, 4).

        Supports many characters x one pose, one character x many poses, N x N
        (i-th character with i-th pose), and one x one.

        Args:
            local_rotations: (J, 3, 3) or (B, J, 3, 3).
            global_translation: (3,) or (B, 3) translation of the joint
                ``global_translation_joint_idx``.
            align_translation: optional (3,) anchor: its X and Z replace that
                joint's translation and the skeleton is shifted so its lowest
                joint sits at its Y.
            absolute_pose: rotations are absolute (True) or relative to the
                joint orient (False, default).
            hips_translations: deprecated alias for ``global_translation``.
            local_translations: (J, 3) or (B, J, 3) per-call override of the
                bind-pose local translations (bone scaling, custom offsets).
        """
        return self._forward_kinematics_impl(
            local_rotations=local_rotations, global_translation=global_translation,
            align_translation=align_translation, absolute_pose=absolute_pose,
            hips_translations=hips_translations, local_translations=local_translations,
            num_joints=self.num_joints, bind_world_transforms=self.bind_world_transforms,
            bind_batched=self.bind_batched,
            default_local_translations=self.local_translations,
            global_translation_joint_idx=self.global_translation_joint_idx,
            levels=self._levels, joint_orient=self.joint_orient,
            orient_parent_T=self._orient_parent_T)

    def forward_source_kinematics(
        self,
        local_rotations,
        global_translation=None,
        align_translation=None,
        absolute_pose: bool = False,
        *,
        hips_translations=None,
        local_translations=None,
    ):
        """Run FK over the optional source (public) topology."""
        if self.source_joint_parent_ids is None:
            raise RuntimeError("forward_source_kinematics requires source FK config.")
        return self._forward_kinematics_impl(
            local_rotations=local_rotations, global_translation=global_translation,
            align_translation=align_translation, absolute_pose=absolute_pose,
            hips_translations=hips_translations, local_translations=local_translations,
            num_joints=self.source_num_joints,
            bind_world_transforms=self.source_bind_world_transforms,
            bind_batched=self.source_bind_batched,
            default_local_translations=self.source_local_translations,
            global_translation_joint_idx=self.source_global_translation_joint_idx,
            levels=self._source_levels, joint_orient=self.source_joint_orient,
            orient_parent_T=self._source_orient_parent_T)

    def expand_source_world_transforms(
        self,
        source_rotations,
        source_world_transforms,
        transform_expander: Callable[..., jnp.ndarray],
        *,
        target_local_translations=None,
    ):
        """Expand source FK output to target world transforms with a caller-supplied hook."""
        if self.source_joint_parent_ids is None:
            raise RuntimeError("expand_source_world_transforms requires source FK config.")
        if target_local_translations is None:
            target_local_translations = self.local_translations
        return transform_expander(
            source_rotations=source_rotations,
            source_world_transforms=source_world_transforms,
            target_local_rotations=self.local_rotations,
            target_local_translations=target_local_translations,
            target_joint_count=self.num_joints,
        )

    def _forward_kinematics_impl(
        self, *, local_rotations, global_translation, align_translation, absolute_pose,
        hips_translations, local_translations, num_joints, bind_world_transforms,
        bind_batched, default_local_translations, global_translation_joint_idx, levels,
        joint_orient, orient_parent_T,
    ):
        if global_translation is None and hips_translations is not None:
            global_translation = hips_translations
        local_rotations = jnp.asarray(local_rotations)
        if tuple(local_rotations.shape[-3:]) != (num_joints, 3, 3):
            raise ValueError(
                f"Expected local_rotations to have shape (...,{num_joints},3,3); "
                f"got {local_rotations.shape}")
        if local_rotations.ndim == 3:
            local_rotations = local_rotations[None]

        rot_batch = local_rotations.shape[0]
        bind_batch = bind_world_transforms.shape[0] if bind_batched else 1

        if global_translation is None:
            global_translation = jnp.zeros((rot_batch, 3), local_rotations.dtype)
        global_translation = jnp.asarray(global_translation)
        if tuple(global_translation.shape) not in [(3,), (rot_batch, 3)]:
            raise ValueError(
                f"Expected global_translation to have shape (3,) or ({rot_batch},3); "
                f"got {global_translation.shape}")

        if rot_batch == 1 and bind_batch > 1:
            batch_size = bind_batch
            local_rotations = jnp.broadcast_to(local_rotations.astype(self.dtype),
                                               (batch_size, num_joints, 3, 3))
        elif rot_batch >= 1 and bind_batch == 1:
            batch_size = rot_batch
            local_rotations = local_rotations.astype(self.dtype)
        elif rot_batch == bind_batch:
            batch_size = rot_batch
            local_rotations = local_rotations.astype(self.dtype)
        else:
            raise ValueError(
                f"Incompatible batches: rotations={rot_batch}, bind={bind_batch}. "
                "Provide (1xB), (Bx1), (1x1), or (BxB) with equal B.")

        if align_translation is not None:
            align_translation = jnp.asarray(align_translation, self.dtype)

        local_t = jnp.asarray(
            default_local_translations if local_translations is None else local_translations,
            self.dtype)
        local_t = jnp.broadcast_to(local_t, (batch_size, num_joints, 3))

        j_mask = jax.nn.one_hot(global_translation_joint_idx, num_joints,
                                dtype=self.dtype)[None, :, None]
        if align_translation is not None:
            comp_m = jnp.asarray([1.0, 0.0, 1.0], self.dtype)[None, None, :]
            M = j_mask * comp_m
            anchor = (align_translation[:, None, :] if align_translation.ndim == 2
                      else align_translation[None, None, :])
            local_t = local_t * (1 - M) + anchor * M
        else:
            gt = global_translation.astype(self.dtype)
            if gt.ndim == 1:
                gt = gt[None, :]
            local_t = local_t * (1 - j_mask) + gt[:, None, :] * j_mask

        if joint_orient is not None and not absolute_pose:
            local_rotations = apply_joint_orient_local(local_rotations, joint_orient,
                                                       orient_parent_T)
        T_local = se3_from_rt(local_rotations, local_t)
        T_world = joint_local_to_world_levelorder(T_local, levels)

        if align_translation is not None:
            y_world = T_world[..., 1, 3]
            y_offset = y_world.min(axis=1, keepdims=True)
            floor = (align_translation[:, 1:2] if align_translation.ndim == 2
                     else align_translation[1])
            shift = y_offset + floor
            T_world = T_world.at[..., 1, 3].add((y_world - shift) - y_world)
        return T_world

    def linear_blend_skinning(self, world_transforms, *, bind_shapes=None,
                              inverse_bind_transform=None):
        """Linear blend skinning from already-computed FK world transforms.

        Args:
            world_transforms: (J, 4, 4) or (B, J, 4, 4).
            bind_shapes: optional override of the bind-pose vertices.
            inverse_bind_transform: optional override of the inverse binds.

        Returns:
            (B, V, 3) posed vertices.
        """
        world_transforms = jnp.asarray(world_transforms)
        if tuple(world_transforms.shape[-3:]) != (self.num_joints, 4, 4):
            raise ValueError(
                f"Expected world_transforms to have shape (...,{self.num_joints},4,4); "
                f"got {world_transforms.shape}")
        if world_transforms.ndim == 3:
            world_transforms = world_transforms[None]
        world_transforms = world_transforms.astype(self.dtype)
        batch_size = world_transforms.shape[0]

        bind_shapes = jnp.asarray(self.bind_shapes if bind_shapes is None else bind_shapes,
                                  self.dtype)
        inverse_bind_transform = jnp.asarray(
            self.inverse_bind_transform if inverse_bind_transform is None
            else inverse_bind_transform, self.dtype)

        bind_batch = bind_shapes.shape[0] if bind_shapes.ndim == 3 else 1
        if bind_batch > 1 and batch_size == 1:
            batch_size = bind_batch
            world_transforms = jnp.broadcast_to(world_transforms,
                                                (batch_size, self.num_joints, 4, 4))
        elif bind_batch not in (1, batch_size):
            raise ValueError(
                f"Incompatible batches: world_transforms={batch_size}, bind_shapes={bind_batch}.")
        if bind_shapes.ndim == 2:
            bind_shapes = jnp.broadcast_to(bind_shapes[None], (batch_size,) + bind_shapes.shape)
        elif bind_shapes.shape[0] == 1 and batch_size > 1:
            bind_shapes = jnp.broadcast_to(bind_shapes, (batch_size,) + bind_shapes.shape[1:])

        inverse_bind_batch = (inverse_bind_transform.shape[0]
                              if inverse_bind_transform.ndim == 4 else 1)
        if inverse_bind_batch > 1 and batch_size == 1:
            batch_size = inverse_bind_batch
            world_transforms = jnp.broadcast_to(world_transforms,
                                                (batch_size, self.num_joints, 4, 4))
            if bind_shapes.shape[0] == 1:
                bind_shapes = jnp.broadcast_to(bind_shapes, (batch_size,) + bind_shapes.shape[1:])
        elif inverse_bind_batch not in (1, batch_size):
            raise ValueError(
                "Incompatible batches: "
                f"world_transforms={batch_size}, inverse_bind_transform={inverse_bind_batch}.")
        if inverse_bind_transform.ndim == 3:
            inverse_bind_transform = inverse_bind_transform[None]
        inverse_bind_transform = jnp.broadcast_to(inverse_bind_transform,
                                                  (batch_size, self.num_joints, 4, 4))

        bone_transforms = world_transforms @ inverse_bind_transform
        if self.mode == "warp":
            return self._warp_skinning(bind_shapes, bone_transforms)
        return lbs(bind_shapes, self.skinning_weights, bone_transforms)

    def pose(
        self,
        local_rotations,
        global_translation=None,
        align_translation=None,
        return_transforms: bool = False,
        absolute_pose: bool = False,
        fk_only: bool = False,
        *,
        hips_translations=None,
        local_translations=None,
    ):
        """Pose the meshes with forward kinematics + LBS.

        Arguments as for :meth:`forward_kinematics`; ``return_transforms`` also
        returns the world transforms, and ``fk_only`` returns only them (B, J, 4, 4).

        Returns:
            Posed vertices (B, V, 3), ``(vertices, world_transforms)``, or the
            world transforms alone with ``fk_only``.
        """
        T_world = self.forward_kinematics(
            local_rotations=local_rotations, global_translation=global_translation,
            align_translation=align_translation, absolute_pose=absolute_pose,
            hips_translations=hips_translations, local_translations=local_translations)
        if fk_only:
            return T_world
        posed_shapes = self.linear_blend_skinning(T_world)
        if return_transforms:
            return posed_shapes, T_world
        return posed_shapes

    def _warp_skinning(self, bind_verts, bone_transforms):
        """Sparse LBS: ``sum_k w[v, k] * transform_point(T[idx[v, k]], v)``.

        What upstream's Warp kernel computes; padded influences (index -1,
        weight 0) contribute nothing.
        """
        idx = jnp.maximum(self._bone_indices, 0)
        T = bone_transforms[:, idx]                                  # (B, V, K, 4, 4)
        points = (jnp.einsum("bvkij,bvj->bvki", T[..., :3, :3], bind_verts)
                  + T[..., :3, 3])
        return jnp.einsum("vk,bvki->bvi", self._bone_weights, points)
