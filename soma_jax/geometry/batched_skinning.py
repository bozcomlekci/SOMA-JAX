"""Batched skinning wrapper for SOMA-JAX — faithful JAX port of SOMA-X's
``soma.geometry.batched_skinning.BatchedSkinning``.

Pipeline (matches `third_party/SOMA-X/soma/geometry/batched_skinning.py`):

    R_oriented = orient_parent_T @ R_in @ orient   (if not absolute_pose)
    local_t    = bind_local_translations           (hips slot replaced by
                                                    `hips_translation` so the
                                                    root motion is
                                                    *injected into FK*, not
                                                    added as a rigid post-LBS
                                                    shift)
    T_local    = SE3(R_oriented, local_t)
    T_world    = level-order FK of T_local
    [optional align_translation: anchor the translation joint's X and Z, then
     Y-shift T_world so the lowest joint lands at the requested floor height]
    T_bone     = T_world @ inverse_bind
    verts      = Σ_j W[v,j] · (T_bone_j ⊙ bind_shape[v])

The class accepts the SMPL-style (rest_verts, rest_joints) interface for
backward compatibility — when `joint_orient` is None the per-joint bind world
rotation is identity, so `bind_world[j] = SE3(I, rest_joints[j])` and the LBS
collapses to the legacy `lbs_transforms` formula `t_rel = t - R @ j_rest`.
With `joint_orient` provided, the full bind world transforms (rotation +
translation) are used, exactly as in SOMA-X.

Upstream: ``soma/geometry/batched_skinning.py``
    Port for the stock path the SOMA forward uses. Class defaults match
    upstream: `sparse_k=8` (upstream `K: int = 8`) and `hips_idx=1`
    (upstream `global_translation_joint_idx`, which also defaults to 1), and
    `align_translation` anchors the translation joint's X and Z exactly as
    upstream does, leaving Y to the floor shift. One divergence remains:
    `topk_skinning` does not prune tiny weights or pad when K > J.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .lbs import (
    forward_kinematics,
    lbs_transforms,
    lbs,
    lbs_sparse,
    compute_skeleton_levels,
)
from .rig_utils import apply_joint_orient_local
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
) -> tuple[Optional[jnp.ndarray], jnp.ndarray]:
    """Functional rebind + pose: skin ``rest_verts`` against per-batch bind
    world transforms.

    This is the jit-able equivalent of SOMA-X's per-identity flow
    ``BatchedSkinning.rebind(bind_transforms, rest_shape)`` followed by
    ``BatchedSkinning.pose(...)`` — used when the bind transforms come from a
    per-identity ``SkeletonTransfer.fit`` and therefore differ across the
    batch (the eqx :class:`BatchedSkinning` module precomputes its bind state
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

    Returns:
        ``(posed_verts, T_world)`` — (B, V, 3) skinned vertices (``None`` when
        ``skip_lbs``) and (B, J, 4, 4) world joint transforms.
    """
    B, J = local_rotmats.shape[:2]
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
        posed = lbs(rest_verts, zeros, bone_Rt, weights)
    return posed, T_world


def topk_skinning(
    weights: np.ndarray,
    k: int = 8,
    weight_eps: float = 1e-12,
    sort_desc: bool = True,
    pad_index: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense (V, J) skinning weights -> sparse top-K indices/values.

    Port of upstream ``topk_skinning``, including the behaviours that matter at
    the edges: weights at or below ``weight_eps`` are pruned before the
    selection, fewer-than-K influences are **padded** with ``pad_index`` and
    zero weight, and rows that sum to zero stay zero instead of dividing by a
    fudge factor. ``k`` defaults to 8, matching upstream and SOMA-X's Warp path.

    Args:
        weights: (V, J) dense skinning weights.
        k: influences to keep per vertex.
        weight_eps: prune weights at or below this before selecting.
        sort_desc: return the K influences in descending weight order.
        pad_index: joint index used to pad when J < k.

    Returns:
        (indices (V, k) int32, values (V, k) float32).
    """
    W = np.asarray(weights, dtype=np.float32)
    V, J = W.shape
    k_eff = min(int(k), J)

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
    return idx.astype(np.int32), vals.astype(np.float32)


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


class BatchedSkinning(eqx.Module):
    """Stand-alone batched skinning module — JAX port of SOMA-X's BatchedSkinning.

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
            posed = lbs(rest_verts_b, zeros, bone_Rt, self.weights)

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

    def rebind(self, new_rest_verts: jnp.ndarray, new_rest_joints: jnp.ndarray) -> "BatchedSkinning":
        """Return a new BatchedSkinning with updated rest shape, sharing weights.

        Recomputes bind_local_t / inverse_bind for the new rest positions —
        without this rebuild the FK chain would still use the old skeleton.

        Args:
            new_rest_verts: (V, 3) new rest vertices.
            new_rest_joints: (J, 3) new rest joint positions.

        Returns:
            New BatchedSkinning instance.
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
