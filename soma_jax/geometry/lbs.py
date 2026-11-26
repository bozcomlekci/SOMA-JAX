"""Linear Blend Skinning and Forward Kinematics for SOMA-JAX.

All operations are differentiable and compatible with jit/vmap/grad.

Upstream: ``soma/geometry/lbs.py``
    Faithful port of ``batch_rodrigues`` and ``lbs``. Level-order forward
    kinematics (upstream ``rig_utils``), the compact-transform ``lbs_blend``
    and the sparse top-K ``lbs_sparse`` are SOMA-JAX building blocks.
"""
from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def compute_skeleton_levels(parents: np.ndarray) -> list[list[int]]:
    """Group joints by tree depth (BFS level order) for efficient FK.

    Args:
        parents: (J,) integer parent indices. A root is marked either by a
            negative parent (``parents[i] < 0``) or by being self-parented
            (``parents[i] == i``, the SOMA rig convention).

    Returns:
        List of lists, each containing joint indices at the same depth.
    """
    J = len(parents)
    levels: list[list[int]] = []
    assigned = [-1] * J
    # Roots: parent < 0 OR self-parented (SOMA's root has parents[0] == 0).
    queue = [i for i in range(J) if parents[i] < 0 or parents[i] == i]
    for j in queue:
        assigned[j] = 0
    level = 0
    while queue:
        levels.append(queue)
        next_queue = []
        for j in queue:
            for c in range(J):
                if parents[c] == j and c != j:      # skip the self-parented root
                    assigned[c] = level + 1
                    next_queue.append(c)
        queue = next_queue
        level += 1
    return levels


def forward_kinematics(
    local_rotmats: jnp.ndarray,
    rest_joints: jnp.ndarray,
    parents,
) -> jnp.ndarray:
    """Compute global joint transforms via sequential FK.

    Args:
        local_rotmats: (J, 3, 3) local rotation matrices.
        rest_joints: (J, 3) joint positions in rest pose.
        parents: (J,) parent indices; root has parent < 0.
                 Accepts both numpy and JAX arrays; internally converted to
                 a JAX constant so that vmap closures and lax.scan indexing
                 both work correctly.

    Returns:
        (J, 4, 4) global SE(3) transform per joint.
    """
    J = local_rotmats.shape[0]
    # Convert to JAX constant (works for both numpy and jax inputs).
    # Using numpy closure → convert inside so scan can index with a tracer.
    parents_jax = jnp.asarray(parents)
    safe_parents = jnp.maximum(parents_jax, 0)

    # Root may be encoded as parent < 0 OR self-parented (parents[i] == i, as in
    # the SOMA skeleton where parents[0] == 0). Treat both as root so the root's
    # local translation is its absolute rest position (not a zero self-offset).
    is_root_mask = (parents_jax < 0) | (parents_jax == jnp.arange(parents_jax.shape[0]))

    # Local translations (bone offset from parent)
    local_t = jnp.where(
        is_root_mask[:, None],
        rest_joints,
        rest_joints - rest_joints[safe_parents],
    )  # (J, 3)

    # Start with identity transforms
    G = jnp.zeros((J, 4, 4), dtype=local_rotmats.dtype)
    G = G.at[:, 3, 3].set(1.0)

    def step(G, i):
        R_l = local_rotmats[i]          # (3, 3)
        t_l = local_t[i]               # (3,)
        p = safe_parents[i]
        R_p = G[p, :3, :3]
        t_p = G[p, :3, 3]

        R_g = R_p @ R_l
        t_g = R_p @ t_l + t_p

        # parents_jax is a JAX constant — safe to index with scan tracer i
        is_root = (parents_jax[i] < 0) | (parents_jax[i] == i)
        R_g = jnp.where(is_root, R_l, R_g)
        t_g = jnp.where(is_root, t_l, t_g)

        G_new = jnp.zeros((4, 4), dtype=local_rotmats.dtype)
        G_new = G_new.at[:3, :3].set(R_g)
        G_new = G_new.at[:3, 3].set(t_g)
        G_new = G_new.at[3, 3].set(1.0)
        return G.at[i].set(G_new), None

    G, _ = jax.lax.scan(step, G, jnp.arange(J))
    return G  # (J, 4, 4)


def fk_levelorder(
    local_rotmats: jnp.ndarray,
    rest_joints: jnp.ndarray,
    parents: jnp.ndarray,
    levels: list[list[int]],
) -> jnp.ndarray:
    """Level-order FK: process all joints at the same depth in parallel.

    More efficient than sequential scan when the tree is wide.

    Args:
        local_rotmats: (J, 3, 3) local rotation matrices.
        rest_joints: (J, 3) joint rest positions.
        parents: (J,) parent indices.
        levels: joint indices grouped by skeleton depth (from compute_skeleton_levels).

    Returns:
        (J, 4, 4) global SE(3) transforms.
    """
    J = local_rotmats.shape[0]
    safe_parents = jnp.maximum(parents, 0)

    # Root may be parent < 0 OR self-parented (parents[i] == i, as in SOMA).
    root_mask = (parents < 0) | (parents == jnp.arange(J))
    local_t = jnp.where(
        root_mask[:, None],
        rest_joints,
        rest_joints - rest_joints[safe_parents],
    )

    G = jnp.zeros((J, 4, 4), dtype=local_rotmats.dtype)
    G = G.at[:, 3, 3].set(1.0)

    for level_joints in levels:
        idx = jnp.array(level_joints)
        par = safe_parents[idx]

        R_l = local_rotmats[idx]           # (Lj, 3, 3)
        t_l = local_t[idx]                 # (Lj, 3)
        R_p = G[par, :3, :3]              # (Lj, 3, 3)
        t_p = G[par, :3, 3]              # (Lj, 3)

        R_g = jnp.einsum("bij,bjk->bik", R_p, R_l)
        t_g = jnp.einsum("bij,bj->bi", R_p, t_l) + t_p

        is_root = root_mask[idx][:, None]
        R_g = jnp.where(is_root[:, :, None], R_l, R_g)
        t_g = jnp.where(is_root, t_l, t_g)

        G_level = jnp.zeros((len(level_joints), 4, 4), dtype=local_rotmats.dtype)
        G_level = G_level.at[:, :3, :3].set(R_g)
        G_level = G_level.at[:, :3, 3].set(t_g)
        G_level = G_level.at[:, 3, 3].set(1.0)
        G = G.at[idx].set(G_level)

    return G


def fk_levelorder_transforms(
    T_local: jnp.ndarray,
    skeleton_levels: list,
    parents: np.ndarray,
) -> jnp.ndarray:
    """Batched level-order FK on full SE(3) transforms.

    JAX equivalent of SOMA-X's
    ``rig_utils.joint_local_to_world_levelorder``: all joints at the same tree
    depth compose against their parents in one batched matmul, so the chain
    costs one op per depth level instead of one per joint.

    Args:
        T_local: (B, J, 4, 4) parent-relative transforms.
        skeleton_levels: output of :func:`compute_skeleton_levels`.
        parents: (J,) parent indices (numpy — static under jit).

    Returns:
        (B, J, 4, 4) world transforms.
    """
    parents_np = np.asarray(parents).astype(np.int64)
    plan = tuple(
        (tuple(int(j) for j in level), tuple(int(parents_np[j]) for j in level))
        for level in skeleton_levels[1:]
    )
    return _fk_levelorder_transforms(T_local, plan)


# Jitted with the level plan static: eager callers (the pose-inversion refit
# rebuilds FK once per skeleton level) then run one executable instead of a
# gather/einsum/scatter dispatch per level.
@partial(jax.jit, static_argnums=(1,))
def _fk_levelorder_transforms(T_local: jnp.ndarray, plan: tuple) -> jnp.ndarray:
    T_world = T_local
    for joint_ids, parent_ids in plan:
        joint_ids = np.asarray(joint_ids, dtype=np.int64)
        parent_ids = np.asarray(parent_ids, dtype=np.int64)
        T_world = T_world.at[:, joint_ids].set(
            jnp.einsum("bjmn,bjnp->bjmp", T_world[:, parent_ids], T_local[:, joint_ids])
        )
    return T_world


def lbs_transforms(G: jnp.ndarray, rest_joints: jnp.ndarray) -> jnp.ndarray:
    """Compute per-joint bone transform (global minus rest contribution).

    The bone transform converts a point from rest-pose space to posed space:
        p_posed = (R_bone @ p_rest) + t_bone

    Args:
        G: (..., J, 4, 4) global SE(3) transforms from FK.
        rest_joints: (J, 3) joint rest positions.

    Returns:
        (..., J, 3, 4) bone transforms [R | t_rel].
    """
    R = G[..., :3, :3]   # (..., J, 3, 3)
    t = G[..., :3, 3]    # (..., J, 3)
    # Subtract influence of rest joint: t_rel = t - R @ j_rest
    t_rel = t - jnp.einsum("...jrc,...jc->...jr", R, rest_joints)
    return jnp.concatenate([R, t_rel[..., None]], axis=-1)  # (..., J, 3, 4)


def batch_rodrigues(
    rot_vecs: jnp.ndarray,
    epsilon: float = 1e-8,
    dtype=None,
) -> jnp.ndarray:
    """Rotation vectors -> rotation matrices — upstream ``lbs.batch_rodrigues``.

    Rodrigues on the unnormalized vectors, ``R = I + c1 K + c2 K^2`` with
    ``c1 = sin(theta)/theta`` and ``c2 = (1 - cos(theta))/theta^2``, falling
    back to the zeroth-order ``c1 = 1``, ``c2 = 1/2`` for ``theta <= 1e-4``,
    exactly as upstream. ``epsilon`` is accepted for signature compatibility
    (upstream does not use it). Takes (..., 3) where upstream takes (N, 3).

    Args:
        rot_vecs: (..., 3) axis-angle vectors.
        epsilon: unused, as upstream.
        dtype: optional dtype to compute in; defaults to ``rot_vecs``'s.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    rot_vecs = jnp.asarray(rot_vecs)
    if dtype is not None:
        rot_vecs = rot_vecs.astype(dtype)
    sq = jnp.sum(rot_vecs * rot_vecs, axis=-1, keepdims=True)
    # `theta` with a gradient that stays finite at 0; it only feeds c1 and c2,
    # which are constant there.
    theta = jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.0)), 0.0)
    mask = theta > 1e-4
    t = jnp.where(mask, theta, 1.0)
    c1 = jnp.where(mask, jnp.sin(t) / t, 1.0)
    c2 = jnp.where(mask, (1.0 - jnp.cos(t)) / (t * t), 0.5)

    rx, ry, rz = rot_vecs[..., 0], rot_vecs[..., 1], rot_vecs[..., 2]
    zeros = jnp.zeros_like(rx)
    K = jnp.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                  axis=-1).reshape(rot_vecs.shape[:-1] + (3, 3))
    ident = jnp.eye(3, dtype=rot_vecs.dtype)
    return ident + c1[..., None] * K + c2[..., None] * (K @ K)


def lbs(
    bind_shape: jnp.ndarray,
    skinning_weights: jnp.ndarray,
    target_pose_world: jnp.ndarray,
) -> jnp.ndarray:
    """Linear blend skinning with (J, 4, 4) skinning transforms — upstream ``lbs``.

    ``target_pose_world`` holds each joint's skinning transform (posed world
    transform times inverse bind); every joint's transform is applied to the
    bind shape and the results are blended with ``skinning_weights``.

    Supported, as upstream:

    * target batched only: ``bind_shape`` (V, 3), ``target_pose_world`` (..., J, 4, 4)
    * all batched alike: ``bind_shape`` (..., V, 3), ``target_pose_world`` (..., J, 4, 4)
    * unbatched: ``bind_shape`` (V, 3), ``target_pose_world`` (J, 4, 4)

    ``skinning_weights`` is always (V, J).

    Returns:
        (..., V, 3) blended vertices, with the active batch dims.
    """
    if bind_shape.ndim < 2 or bind_shape.shape[-1] != 3:
        raise ValueError(f"`bind_shape` must end with (V,3); got {bind_shape.shape}.")
    V = bind_shape.shape[-2]
    bind_batched = bind_shape.ndim > 2
    bind_batch = bind_shape.shape[:-2] if bind_batched else ()

    if target_pose_world.ndim < 3 or target_pose_world.shape[-2:] != (4, 4):
        raise ValueError(
            f"`target_pose_world` must end with (J,4,4); got {target_pose_world.shape}.")
    J = target_pose_world.shape[-3]
    tgt_batched = target_pose_world.ndim > 3
    tgt_batch = target_pose_world.shape[:-3] if tgt_batched else ()

    if tuple(skinning_weights.shape) != (V, J):
        raise ValueError(
            f"`skinning_weights` must be (V,J) with V={V}, J={J}; got {skinning_weights.shape}.")
    if bind_batched:
        if not tgt_batched:
            raise ValueError(
                "When bind_shape/inverse_bind_pose are batched, target_pose_world must also "
                "be batched.")
        if tgt_batch != bind_batch:
            raise ValueError(
                "Batch dims must match across bind/inverse/target; got "
                f"{bind_batch} vs {tgt_batch}.")

    R = target_pose_world[..., :3, :3]
    t = target_pose_world[..., :3, 3]
    tv = jnp.einsum("...jmk,...vk->...jvm", R, bind_shape) + t[..., None, :]
    return jnp.einsum("vj,...jvm->...vm", skinning_weights, tv)


def lbs_blend(
    v_rest: jnp.ndarray,
    pose_correctives: jnp.ndarray,
    bone_transforms: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Linear Blend Skinning forward pass with compact bone transforms.

    SOMA-JAX extra: SMPL-style skinning of ``v_rest + pose_correctives`` with
    the (B, J, 3, 4) transforms of :func:`lbs_transforms`. Upstream's
    :func:`lbs` takes (J, 4, 4) skinning transforms instead.

    Args:
        v_rest: (B, V, 3) rest-pose vertices (after shape blend shapes).
        pose_correctives: (B, V, 3) pose-dependent vertex displacements.
        bone_transforms: (B, J, 3, 4) bone transforms from lbs_transforms().
        weights: (V, J) skinning weights.

    Returns:
        (B, V, 3) posed vertices.
    """
    v = v_rest + pose_correctives
    # Blend rotation: (B, V, 3, 3)
    R_blend = jnp.einsum("vj,bjrc->bvrc", weights, bone_transforms[..., :3])
    # Blend translation: (B, V, 3)
    t_blend = jnp.einsum("vj,bjr->bvr", weights, bone_transforms[..., 3])
    return jnp.einsum("bvrc,bvc->bvr", R_blend, v) + t_blend


def lbs_sparse(
    v_rest: jnp.ndarray,
    pose_correctives: jnp.ndarray,
    bone_transforms: jnp.ndarray,
    weight_values: jnp.ndarray,
    weight_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Sparse top-K LBS for efficiency when K << J.

    Args:
        v_rest: (B, V, 3) rest-pose vertices.
        pose_correctives: (B, V, 3) pose correctives.
        bone_transforms: (B, J, 3, 4) bone transforms.
        weight_values: (V, K) top-K skinning weights.
        weight_indices: (V, K) top-K joint indices (int32).

    Returns:
        (B, V, 3) posed vertices.
    """
    B, V = v_rest.shape[:2]
    K = weight_values.shape[1]
    v = v_rest + pose_correctives  # (B, V, 3)

    # Gather bone transforms for each vertex's K influences: (B, V, K, 3, 4)
    # bone_transforms: (B, J, 3, 4)
    bone_trans_vk = bone_transforms[:, weight_indices, :, :]  # (B, V, K, 3, 4)

    # Blend: weighted sum over K influences
    # R_blend: (B, V, 3, 3)
    R_blend = jnp.einsum("vk,bvkrc->bvrc", weight_values, bone_trans_vk[..., :3])
    t_blend = jnp.einsum("vk,bvkr->bvr", weight_values, bone_trans_vk[..., 3])

    return jnp.einsum("bvrc,bvc->bvr", R_blend, v) + t_blend
