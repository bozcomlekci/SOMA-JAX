"""Pose inversion for SOMA-JAX.

Recovers per-joint rotation matrices from posed mesh vertices via two stages:
  1. Analytical init: weighted Kabsch SVD per joint + Newton-Schulz orthogonalization.
  2. Autograd refinement: Adam optimization in 6D rotation space via jax.lax.scan.

For SOMA→X model conversion the analytical path alone achieves ~1,200 FPS.
Adding refinement gives higher accuracy at the cost of runtime.

References:
  - Kabsch (1976): optimal rotation via SVD
  - Newton-Schulz iteration for nearest orthogonal matrix
  - Zhou et al. (2019): continuous 6D rotation representation

Upstream: none — SOMA-JAX-only.
    Lightweight alternative inverter (single Kabsch init + one autograd refine). The faithful solver is pose_inversion_soma.py.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

from .geometry.transforms import (
    kabsch,
    newton_schulz,
    rotation_6d_to_rotmat,
    rotmat_to_6d,
    axis_angle_to_rotmat,
)
from .geometry.lbs import (
    forward_kinematics,
    lbs_transforms,
    lbs,
    lbs_sparse,
    compute_skeleton_levels,
    fk_levelorder,
)
from .types import SOMAOutput


def _weighted_kabsch_per_joint(
    posed_verts: jnp.ndarray,
    rest_verts: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate per-joint rotation via weighted Kabsch alignment.

    Each joint is aligned using vertices weighted by that joint's LBS weight,
    giving an estimate of the joint's world (absolute) rotation relative to
    the rest pose.

    Args:
        posed_verts: (V, 3) posed vertices.
        rest_verts: (V, 3) rest-pose vertices.
        weights: (V, J) skinning weights.

    Returns:
        (J, 3, 3) initial rotation matrix estimates.
    """
    J = weights.shape[1]

    def per_joint(j):
        w = weights[:, j]
        R = kabsch(rest_verts, posed_verts, weights=w)
        return R

    return jax.vmap(per_joint)(jnp.arange(J))


def _analytical_init(
    posed_verts: jnp.ndarray,
    rest_verts: jnp.ndarray,
    weights: jnp.ndarray,
    ns_iters: int = 10,
) -> jnp.ndarray:
    """Compute initial rotation estimate via Kabsch + Newton-Schulz.

    Args:
        posed_verts: (V, 3) posed mesh vertices.
        rest_verts: (V, 3) rest-pose vertices.
        weights: (V, J) skinning weights.
        ns_iters: Newton-Schulz iterations for orthogonalization.

    Returns:
        (J, 3, 3) orthogonal rotation matrices (initial estimate).
    """
    R_init = _weighted_kabsch_per_joint(posed_verts, rest_verts, weights)
    return jax.vmap(lambda R: newton_schulz(R, num_iter=ns_iters))(R_init)


def _build_fk_lbs_fn(rest_verts, weights, rest_joints, parents, skeleton_levels):
    """Build a function that maps (J, 3, 3) rotmats → (V, 3) posed verts."""

    def fk_lbs(rotmats):
        G = fk_levelorder(rotmats, rest_joints, parents, skeleton_levels)
        bone_T = lbs_transforms(G[None], rest_joints[None])  # (1, J, 3, 4)
        posed = lbs(
            rest_verts[None],
            jnp.zeros_like(rest_verts[None]),
            bone_T,
            weights,
        )[0]  # (V, 3)
        return posed

    return fk_lbs


def _autograd_refine(
    R_init: jnp.ndarray,
    posed_verts: jnp.ndarray,
    rest_verts: jnp.ndarray,
    weights: jnp.ndarray,
    rest_joints: jnp.ndarray,
    parents: jnp.ndarray,
    skeleton_levels: list,
    num_iters: int = 50,
    lr: float = 1e-3,
    vertex_weights: Optional[jnp.ndarray] = None,
    dof_constraints: Optional[dict] = None,
) -> jnp.ndarray:
    """Refine rotation estimates via Adam in 6D rotation space.

    Uses jax.lax.scan so the entire optimization loop is JIT-compiled.

    Args:
        R_init: (J, 3, 3) initial rotation matrices.
        posed_verts: (V, 3) target posed vertices.
        rest_verts: (V, 3) rest-pose vertices.
        weights: (V, J) skinning weights.
        rest_joints: (J, 3) rest joint positions.
        parents: (J,) parent indices.
        skeleton_levels: joint depth ordering for level-order FK.
        num_iters: Adam optimization steps.
        lr: Adam learning rate.
        vertex_weights: optional (V,) per-vertex importance weights.
        dof_constraints: optional ``{joint: hinge_axis}``. The hinge holds on
            the joint's parent-relative rotation throughout the optimisation,
            so the rest of the chain compensates for what it removes.

    Returns:
        (J, 3, 3) refined world rotation matrices.
    """
    # ``R_init`` is WORLD (absolute) rotations from the analytical Kabsch init, but
    # ``fk_lbs_fn`` below reconstructs from LOCAL (parent-relative) rotations
    # (fk_levelorder composes parent -> child). Optimise in local space, then map
    # back to world on return so the recovered rotations keep the same (world)
    # convention as the analytical init and the caller expects.
    parents_j = jnp.asarray(parents)
    J = R_init.shape[0]
    safe_par = jnp.maximum(parents_j, 0)
    is_root = (parents_j < 0) | (parents_j == jnp.arange(J))

    def _world_to_local(Rw):                       # local[j] = world[parent]^T @ world[j]
        Rl = jnp.einsum("jab,jbc->jac", jnp.swapaxes(Rw[safe_par], -1, -2), Rw)
        return jnp.where(is_root[:, None, None], Rw, Rl)

    r6d_init = jax.vmap(rotmat_to_6d)(_world_to_local(R_init))  # (J, 6), LOCAL
    fk_lbs_fn = _build_fk_lbs_fn(rest_verts, weights, rest_joints, parents, skeleton_levels)

    def to_local(r6d):
        rotmats = jax.vmap(rotation_6d_to_rotmat)(r6d)   # LOCAL
        if dof_constraints:
            rotmats = apply_dof_constraints(rotmats, dof_constraints)
        return rotmats

    def loss_fn(r6d):
        pred_verts = fk_lbs_fn(to_local(r6d))
        diff = pred_verts - posed_verts
        if vertex_weights is not None:
            return jnp.sum(vertex_weights[:, None] * diff ** 2)
        return jnp.mean(diff ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(r6d_init)

    def opt_step(carry, _):
        r6d, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(r6d)
        updates, new_state = optimizer.update(grads, state)
        return (optax.apply_updates(r6d, updates), new_state), loss

    (r6d_final, _), losses = jax.lax.scan(
        opt_step, (r6d_init, opt_state), None, length=num_iters
    )
    R_local = to_local(r6d_final)
    # local -> world (fk_levelorder's rotation block) to match R_init's convention.
    return fk_levelorder(R_local, rest_joints, parents, skeleton_levels)[:, :3, :3]


def _project_to_1dof(R: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    """Project a 3D rotation onto a 1-DOF (hinge) rotation around a fixed axis.

    Used for knees, elbows, and other hinge joints that physically rotate
    around only one axis.

    Args:
        R: (3, 3) rotation matrix.
        axis: (3,) unit axis vector (e.g. [1, 0, 0] for X-axis hinge).

    Returns:
        (3, 3) rotation matrix constrained to rotate only around axis.
    """
    # Recover the rotation angle around the axis by projecting log(R) onto axis
    # cos(theta) ≈ (trace(R) - 1) / 2 — but we want the signed angle around axis
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_t = jnp.clip((trace - 1) / 2, -1.0, 1.0)
    sin_t_vec = jnp.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) * 0.5
    # Signed angle around `axis`
    signed_sin = jnp.dot(sin_t_vec, axis)
    angle = jnp.arctan2(signed_sin, cos_t)
    return axis_angle_to_rotmat(axis * angle)


def apply_dof_constraints(
    rotmats: jnp.ndarray,
    constraints: dict[int, jnp.ndarray],
) -> jnp.ndarray:
    """Apply 1-DOF hinge constraints to specified joints.

    Args:
        rotmats: (J, 3, 3) parent-relative (local) joint rotations — a hinge
            constrains a joint relative to its parent, so world rotations go
            through :func:`apply_dof_constraints_world` instead.
        constraints: dict {joint_index: axis_vector_3D}. Axes should be unit vectors.

    Returns:
        (J, 3, 3) rotations with constrained joints projected to single-axis rotation.
    """
    for joint_idx, axis in constraints.items():
        rotmats = rotmats.at[joint_idx].set(_project_to_1dof(rotmats[joint_idx], axis))
    return rotmats


def apply_dof_constraints_world(
    world_rotmats: jnp.ndarray,
    constraints: dict[int, jnp.ndarray],
    parents: np.ndarray,
) -> jnp.ndarray:
    """:func:`apply_dof_constraints` for world (absolute) rotations.

    Each constrained joint's rotation relative to its parent is projected onto
    its hinge; every other joint keeps its world rotation. Parents are handled
    before children, so a hinge under another hinge is relative to the
    parent's constrained rotation.

    Args:
        world_rotmats: (J, 3, 3) world rotations.
        constraints: dict {joint_index: axis_vector_3D}, axes in the parent's
            frame (for a world-aligned rest pose: world axes).
        parents: (J,) parent indices; a root has a negative index or itself.

    Returns:
        (J, 3, 3) world rotations with the hinges enforced.
    """
    parents = np.asarray(parents)

    def depth(j):
        d = 0
        while 0 <= parents[j] != j:
            j, d = int(parents[j]), d + 1
        return d

    for j in sorted(constraints, key=depth):
        axis, p = constraints[j], int(parents[j])
        if p < 0 or p == j:
            new = _project_to_1dof(world_rotmats[j], axis)
        else:
            local = jnp.swapaxes(world_rotmats[p], -1, -2) @ world_rotmats[j]
            new = world_rotmats[p] @ _project_to_1dof(local, axis)
        world_rotmats = world_rotmats.at[j].set(new)
    return world_rotmats


class PoseInversion:
    """Pose inversion: recover joint rotations from posed mesh vertices.

    Supports three modes:
    - 'analytical': fast Kabsch + Newton-Schulz (~1200 FPS)
    - 'autograd': Adam refinement in 6D space
    - 'combined': analytical init + autograd refinement

    Args:
        rest_verts: (V, 3) rest-pose vertices.
        weights: (V, J) skinning weights.
        rest_joints: (J, 3) rest joint positions.
        parents: (J,) parent indices.
        ns_iters: Newton-Schulz iterations.
        dof_constraints: optional dict {joint_idx: hinge_axis} for 1-DOF joints
            (knees, elbows). The hinge constrains the joint's rotation relative
            to its parent; the axis is in the rest pose's (world-aligned) frame.

    :meth:`fit` returns world (absolute) rotations: compose a joint's parent
    out (``R[parent].T @ R[j]``) for parent-relative ones.
    """

    def __init__(
        self,
        rest_verts: jnp.ndarray,
        weights: jnp.ndarray,
        rest_joints: jnp.ndarray,
        parents: np.ndarray,
        ns_iters: int = 10,
        dof_constraints: dict[int, jnp.ndarray] | None = None,
    ):
        self.rest_verts = rest_verts
        self.weights = weights
        self.rest_joints = rest_joints
        self.parents = jnp.array(parents)
        self._parents_np = np.asarray(parents)
        self.skeleton_levels = compute_skeleton_levels(self._parents_np)
        self.ns_iters = ns_iters
        self.dof_constraints = dof_constraints or {}

        # Pre-compile the JIT-able analytical init
        self._jit_analytical = jax.jit(self._analytical_single)

    def _analytical_single(self, posed_verts: jnp.ndarray) -> jnp.ndarray:
        return _analytical_init(
            posed_verts, self.rest_verts, self.weights, self.ns_iters
        )

    def fit(
        self,
        posed_verts: jnp.ndarray,
        mode: str = "combined",
        num_refine_iters: int = 50,
        lr: float = 1e-3,
        vertex_weights: Optional[jnp.ndarray] = None,
        chunked: bool = False,
        chunk_size: int = 32,
    ) -> jnp.ndarray:
        """Recover joint rotations from posed vertices.

        Args:
            posed_verts: (B, V, 3) or (V, 3) posed mesh vertices.
            mode: 'analytical', 'autograd', or 'combined'.
            num_refine_iters: Adam steps (used in 'autograd' and 'combined').
            lr: Adam learning rate.
            vertex_weights: optional (V,) or (B, V) per-vertex importance.
            chunked: if True, process in chunks to reduce peak memory.
            chunk_size: batch size per chunk.

        Returns:
            (B, J, 3, 3) or (J, 3, 3) recovered world (absolute) rotation matrices.
        """
        unbatched = posed_verts.ndim == 2
        if unbatched:
            posed_verts = posed_verts[None]
        B = posed_verts.shape[0]

        if chunked and B > chunk_size:
            chunks = [
                self.fit(
                    posed_verts[i : i + chunk_size],
                    mode=mode,
                    num_refine_iters=num_refine_iters,
                    lr=lr,
                    vertex_weights=vertex_weights,
                    chunked=False,
                )
                for i in range(0, B, chunk_size)
            ]
            rotmats = jnp.concatenate(chunks, axis=0)
            return rotmats[0] if unbatched else rotmats

        def fit_single(verts, v_weights=None):
            if mode in ("analytical", "combined"):
                R = _analytical_init(verts, self.rest_verts, self.weights, self.ns_iters)
            else:
                # Start from identity for pure autograd
                R = jnp.broadcast_to(jnp.eye(3, dtype=verts.dtype)[None], (self.weights.shape[1], 3, 3))

            if mode in ("autograd", "combined"):
                # 1-DOF constraints (knee/elbow hinges) hold inside the
                # refinement, so the rest of the chain compensates.
                R = _autograd_refine(
                    R, verts, self.rest_verts, self.weights, self.rest_joints,
                    self.parents, self.skeleton_levels,
                    num_iters=num_refine_iters, lr=lr,
                    vertex_weights=v_weights,
                    dof_constraints=self.dof_constraints,
                )
            elif self.dof_constraints:
                R = apply_dof_constraints_world(R, self.dof_constraints, self._parents_np)
            return R

        if vertex_weights is not None and vertex_weights.ndim == 2:
            rotmats = jax.vmap(fit_single)(posed_verts, vertex_weights)
        else:
            rotmats = jax.vmap(lambda v: fit_single(v, vertex_weights))(posed_verts)

        return rotmats[0] if unbatched else rotmats

    def fit_hierarchical(
        self,
        posed_verts: jnp.ndarray,
        num_refine_iters: int = 20,
        lr: float = 5e-3,
    ) -> jnp.ndarray:
        """Hierarchical pose inversion: refine joint by joint, top-down.

        Starts from the analytical init and walks the skeleton from root to
        leaves. Each joint's parent-relative rotation is refined against the
        vertices it skins (weighted by its skinning weight) with its ancestors
        already refined, so errors do not accumulate down kinematic chains.
        ``dof_constraints`` hold throughout.

        Args:
            posed_verts: (B, V, 3) or (V, 3) posed mesh vertices.
            num_refine_iters: Adam steps per joint.
            lr: Adam learning rate.

        Returns:
            (B, J, 3, 3) or (J, 3, 3) recovered world (absolute) rotation
            matrices, as :meth:`fit` returns.
        """
        unbatched = posed_verts.ndim == 2
        if unbatched:
            posed_verts = posed_verts[None]

        J = self.weights.shape[1]
        safe_par = jnp.maximum(self.parents, 0)
        is_root = (self.parents < 0) | (self.parents == jnp.arange(J))
        fk_lbs_fn = _build_fk_lbs_fn(self.rest_verts, self.weights, self.rest_joints,
                                     self.parents, self.skeleton_levels)
        constraints = self.dof_constraints
        opt = optax.adam(lr)

        def constrain(R_local):
            return apply_dof_constraints(R_local, constraints) if constraints else R_local

        def init_local(verts):
            # The analytical init is world rotations; the refinement (like
            # fk_levelorder) works on parent-relative ones.
            Rw = _analytical_init(verts, self.rest_verts, self.weights, self.ns_iters)
            Rl = jnp.einsum("jab,jbc->jac", jnp.swapaxes(Rw[safe_par], -1, -2), Rw)
            return constrain(jnp.where(is_root[:, None, None], Rw, Rl))

        def refine_joint(R_local, verts, j):
            v_w = self.weights[:, j]

            def loss_fn(r6):
                R_all = constrain(R_local.at[j].set(rotation_6d_to_rotmat(r6)))
                return jnp.sum(v_w[:, None] * (fk_lbs_fn(R_all) - verts) ** 2)

            def step(carry, _):
                r6, state = carry
                updates, state = opt.update(jax.grad(loss_fn)(r6), state)
                return (optax.apply_updates(r6, updates), state), None

            r6 = rotmat_to_6d(R_local[j])
            (r6, _), _ = jax.lax.scan(step, (r6, opt.init(r6)), None,
                                      length=num_refine_iters)
            return constrain(R_local.at[j].set(rotation_6d_to_rotmat(r6)))

        # One compiled step serves every joint: the joint index is traced.
        refine = jax.jit(jax.vmap(refine_joint, in_axes=(0, 0, None)))
        R_local = jax.jit(jax.vmap(init_local))(posed_verts)
        skinned = np.asarray(self.weights).sum(axis=0) > 0
        for level_joints in self.skeleton_levels:
            for j in level_joints:
                if skinned[j]:           # a joint that skins nothing has no signal
                    R_local = refine(R_local, posed_verts, jnp.int32(j))

        rotmats = jax.vmap(lambda R: fk_levelorder(
            R, self.rest_joints, self.parents, self.skeleton_levels)[:, :3, :3])(R_local)
        return rotmats[0] if unbatched else rotmats
