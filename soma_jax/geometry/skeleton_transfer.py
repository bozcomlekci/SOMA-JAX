"""Skeleton fitting + transfer utilities for SOMA-JAX.

Provides:

* ``RBFJointRegressor``-backed joint position fitting (RBF regressor per
  joint, sparse precomputed basis-weight matrix when possible).
* Full ``SkeletonTransfer`` JAX port of
  ``third_party/SOMA-X/soma/geometry/skeleton_transfer.py`` — matches the
  pure-PyTorch path of that module (no Warp), implementing
  ``fit_joint_positions`` + ``fit_joint_rotations`` + ``fit``.
* Mirror utility ``PoseMirror`` (existing) — kept as-is.

Upstream: ``soma/geometry/skeleton_transfer.py``
    Faithful port of that code. Per-joint RBF position regression + two-stage Kabsch rotation fit; PoseMirror.
"""
from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import jax
import jax.numpy as jnp

from .transforms import (
    align_vectors,
    compute_covariance,
    kabsch,
    rodrigues_rotation,
    rotation_from_covariance,
    se3_from_rt,
)
from .rig_utils import get_joint_children_ids
from .interpolate import RadialBasisFunction


def _build_rbf_regressors(
    rest_vertices: np.ndarray,
    skinning_weights: np.ndarray,
    joints: np.ndarray,
) -> list[np.ndarray]:
    """(Legacy) per-joint linear regressor proxy from skinning weights.

    Retained for backward compatibility with callers that imported it; the
    full SOMA-X-faithful path lives in :class:`SkeletonTransfer` now.
    """
    J = skinning_weights.shape[1]
    regressors = []
    for j in range(J):
        w = skinning_weights[:, j]
        w_norm = w / (w.sum() + 1e-8)
        regressors.append(w_norm[None, :] * np.eye(3)[:, None])
    return regressors


def fit_joint_positions(
    vertices: jnp.ndarray,
    J_regressor: jnp.ndarray,
) -> jnp.ndarray:
    """Regress joint positions via a precomputed linear regressor.

    Args:
        vertices: (..., V, 3) vertex positions.
        J_regressor: (J, V) regressor matrix (e.g. SOMA's J_regressor).

    Returns:
        (..., J, 3) joint positions.
    """
    return jnp.einsum("jv,...vd->...jd", J_regressor, vertices)


# ---------------------------------------------------------------------------
# Full SkeletonTransfer port (matches third_party/SOMA-X/soma/geometry/skeleton_transfer.py
# pure-PyTorch path; Warp variants are CUDA-only and intentionally omitted).
# ---------------------------------------------------------------------------
def _joint_world_to_local_np(bind_world: np.ndarray,
                              parents: np.ndarray) -> np.ndarray:
    """Numpy joint world → parent-local (used in precompute, runs once)."""
    parents = np.asarray(parents).astype(int)
    inv = np.linalg.inv(bind_world)
    safe_parents = np.where(parents < 0, 0, parents)
    local = inv[safe_parents] @ bind_world
    return local


class SkeletonTransfer:
    """JAX port of SOMA-X ``SkeletonTransfer``.

    Adapts the SOMA skeleton (joint positions + per-joint orientations) to a
    new identity's mesh. Two stages:

    1. ``fit_joint_positions`` — per-joint RBF regressor maps the deformed
       mesh's vertices to the new joint position. Each joint's regressor is
       trained on its skinning-weight support (and that of its parent), so
       face/eye/leaf joints without skinning weight inherit the closest
       supported chain.
    2. ``fit_joint_rotations`` — for each joint, recover a rotation that
       (a) aligns its skinned vertex cloud (inverse-LBS, Kabsch) and (b)
       further rotates child joint offsets to land on the new joint
       positions, again via Kabsch. Composed with the bind orientation.

    Args mirror SOMA-X's constructor — see ``__init__`` docstring.
    """

    def __init__(
        self,
        joint_parent_ids: np.ndarray,
        bind_world_transforms: np.ndarray,
        bind_shape: np.ndarray,
        skinning_weights: np.ndarray,
        rbf_kernel: str = "linear",
        vertex_ids_to_exclude: Optional[Iterable[int]] = None,
        freeze_rotations: Optional[Iterable[int]] = None,
        skip_endjoints: bool = True,
        use_sparse_rbf_matrix: bool = True,
        rotation_method: str = "auto",
        skip_inverse_lbs: bool = False,
        rotation_backend: str = "jax",
    ):
        """
        Args:
            joint_parent_ids: (J,) parent indices.
            bind_world_transforms: (J, 4, 4) canonical bind world transforms.
            bind_shape: (V, 3) canonical mesh vertices.
            skinning_weights: (V, J).
            rbf_kernel: kernel for RBF regressor (default 'linear' matches
                SOMA-X's default).
            vertex_ids_to_exclude: optional vertex ids dropped from every
                regressor's support (e.g. UV-seam duplicates).
            freeze_rotations: joints whose rotation should stay at bind
                (typically the root or fixed accessories).
            skip_endjoints: leaf joints (no children) inherit parent rotation
                rather than fitting their own. Matches SOMA-X default.
            use_sparse_rbf_matrix: precompute a sparse (J, V) basis-weight
                matrix so the position fit is one matmul per identity. Disable
                only if memory matters more than speed.
            rotation_method: 'kabsch' (SVD) or 'newton-schulz' (iterative).
            skip_inverse_lbs: skip the per-joint vertex Kabsch and use the
                identity initial rotation — useful when the skinning support
                is too noisy to fit reliably.
            rotation_backend: where the covariance→rotation SVD runs in the
                vectorized fit. ``"jax"`` (default) uses ``jnp.linalg.svd``;
                ``"warp"`` runs a hand-written Warp ``svd3`` kernel on-device
                via XLA FFI (Warp+JAX hybrid — JAX builds covariances + does
                FK/LBS, Warp extracts rotations). The Warp kernel implements
                plain SVD Procrustes, so it is exact for
                ``rotation_method="kabsch"`` but only an *approximation* of the
                default ``"auto"`` (Newton–Schulz on a gauge-regularized
                covariance), diverging on ill-conditioned covariances. Requires
                Warp; accepts ``rotation_method`` ``"auto"`` or ``"kabsch"``.
        """
        self.joint_parent_ids = (
            list(joint_parent_ids.tolist())
            if hasattr(joint_parent_ids, "tolist")
            else list(joint_parent_ids)
        )
        self.num_joints = len(self.joint_parent_ids)
        self.joint_children_ids = get_joint_children_ids(np.asarray(self.joint_parent_ids))

        self.bind_world_transforms = np.asarray(bind_world_transforms, dtype=np.float32)
        self.bind_local_transforms = _joint_world_to_local_np(
            self.bind_world_transforms, np.asarray(self.joint_parent_ids))
        self.bind_shape = np.asarray(bind_shape, dtype=np.float32)
        self.skinning_weights = np.asarray(skinning_weights, dtype=np.float32)
        self.rbf_kernel = rbf_kernel
        self.vertex_ids_to_exclude = (
            list(vertex_ids_to_exclude) if vertex_ids_to_exclude is not None else None)
        self.freeze_rotations = set(freeze_rotations) if freeze_rotations else set()
        self.skip_endjoints = bool(skip_endjoints)
        self.use_sparse_rbf_matrix = bool(use_sparse_rbf_matrix)
        self.rotation_method = rotation_method
        self.skip_inverse_lbs = bool(skip_inverse_lbs)
        self.rotation_backend = rotation_backend
        if rotation_backend not in ("jax", "warp"):
            raise ValueError(
                f"rotation_backend must be 'jax' or 'warp', got {rotation_backend!r}")
        if rotation_backend == "warp" and rotation_method not in ("auto", "kabsch"):
            raise ValueError(
                "rotation_backend='warp' supports rotation_method 'auto' or "
                "'kabsch'; the Warp kernel implements SVD Procrustes. With "
                "'auto' it is a fast approximation, not the same algorithm — "
                "see soma_jax/geometry/warp_kabsch.py.")
        self._warp_kabsch = None
        if rotation_backend == "warp":
            from .warp_kabsch import kabsch_rotation_warp, is_available
            if not is_available():
                raise RuntimeError(
                    "rotation_backend='warp' but Warp is unavailable; "
                    "pip install warp-lang.")
            self._warp_kabsch = kabsch_rotation_warp

        self.regressor_mask: Optional[np.ndarray] = None
        self.joint_pos_regressors: list[Optional[RadialBasisFunction]] = []
        self._sparse_rbf_matrix: Optional[jnp.ndarray] = None
        self._precompute_regressors()
        self._precompute_rotation_fit()

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def update_bind(self, bind_world_transforms: np.ndarray, bind_shape: np.ndarray) -> None:
        """Replace bind data without rebuilding the RBF regressors.

        Use this when only the identity (shape) changes but topology +
        skinning + skeleton are unchanged.
        """
        self.bind_world_transforms = np.asarray(bind_world_transforms, dtype=np.float32)
        self.bind_local_transforms = _joint_world_to_local_np(
            self.bind_world_transforms, np.asarray(self.joint_parent_ids))
        self.bind_shape = np.asarray(bind_shape, dtype=np.float32)
        # The RBF regressors are keyed on ``bind_shape`` (their centers) and
        # queried at ``bind_world_transforms`` joint positions, so both must be
        # rebuilt — this is what upstream ``update_bind`` does. The vectorized
        # rotation fit additionally caches bind-dependent tensors that plain
        # SOMA-X recomputes inside ``fit_joint_rotations``.
        self._precompute_regressors()
        self._precompute_rotation_fit()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    def _precompute_regressors(self) -> None:
        W = self.skinning_weights
        regressor_mask = W > 0.0
        # Initial: vertex influences both the joint AND the joint's parent
        # (bone-tube), so the affine fit has shoulder-of-bone support.
        regressor_mask = regressor_mask & (W[:, self.joint_parent_ids] > 0.0)

        # Joints with no support → broaden to "any vertex on this joint";
        # then propagate up the parent chain until everyone has support.
        zero_weight_ids = np.where(regressor_mask.sum(axis=0) == 0)[0]
        parents = np.asarray(self.joint_parent_ids, dtype=np.int64)
        joint_parent_cur = parents.copy()
        regressor_mask[:, zero_weight_ids] = W[:, zero_weight_ids] > 0.0
        while len(zero_weight_ids) > 1:
            regressor_mask[:, zero_weight_ids] |= (
                W[:, joint_parent_cur][:, zero_weight_ids] > 0.0)
            new_zero = np.where(regressor_mask.sum(axis=0) == 0)[0]
            parent_next = parents[joint_parent_cur]
            if np.array_equal(parent_next, joint_parent_cur):
                break
            joint_parent_cur = parent_next
            zero_weight_ids = new_zero

        # Mirror SOMA-X special-case: if only Root + Hips lack support, fall
        # back to the Hips children's union.
        if np.array_equal(np.where(regressor_mask.sum(axis=0) == 0)[0], np.array([0, 1])):
            children = self.joint_children_ids[1]
            if children:
                regressor_mask[:, 1] = regressor_mask[:, list(children)].any(axis=1)

        if self.vertex_ids_to_exclude is not None:
            regressor_mask[self.vertex_ids_to_exclude] = False
        self.regressor_mask = regressor_mask

        # Build RBF per joint (skip Root — its position is read from bind).
        bind_shape = self.bind_shape
        regressors: list[Optional[RadialBasisFunction]] = [None]
        for j in range(1, self.num_joints):
            ids = np.where(regressor_mask[:, j])[0]
            if len(ids) == 0:
                regressors.append(None)
                continue
            regressors.append(
                RadialBasisFunction(
                    jnp.asarray(bind_shape[ids]),
                    kernel=self.rbf_kernel,
                    include_polynomial=True,
                )
            )
        self.joint_pos_regressors = regressors

        if not self.use_sparse_rbf_matrix:
            self._sparse_rbf_matrix = None
            return

        # Dense (J, V) basis-weight matrix so that
        # ``new_joints = sparse_rbf_matrix @ target_shape``  recovers all
        # joint positions in one matmul. The Root row stays zero; we splice
        # in the bind Root position at call time.
        J = self.num_joints
        V = bind_shape.shape[0]
        mat = np.zeros((J, V), dtype=np.float32)
        for j, rbf in enumerate(regressors):
            if rbf is None:
                continue
            query = self.bind_world_transforms[j, :3, 3].astype(np.float32)
            w = np.asarray(rbf.get_basis_weights(jnp.asarray(query)))
            ids = np.where(regressor_mask[:, j])[0]
            mat[j, ids] = w
        self._sparse_rbf_matrix = jnp.asarray(mat)

    # ------------------------------------------------------------------
    # Position fitting
    # ------------------------------------------------------------------
    def fit_joint_positions(self, target_shapes: jnp.ndarray) -> jnp.ndarray:
        """Regress new joint positions from a deformed mesh.

        Args:
            target_shapes: (V, 3) or (B, V, 3) deformed bind mesh.

        Returns:
            (J, 3) or (B, J, 3) new joint positions.
        """
        target_shapes = jnp.asarray(target_shapes)
        added_batch = target_shapes.ndim == 2
        if added_batch:
            target_shapes = target_shapes[None]
        B, V, D = target_shapes.shape
        J = self.num_joints

        if self._sparse_rbf_matrix is not None:
            # (J, V) @ (V, B*D) → (J, B*D)
            flat = jnp.transpose(target_shapes, (1, 0, 2)).reshape(V, B * D)
            new_joints = self._sparse_rbf_matrix @ flat                    # (J, B*D)
            new_joints = jnp.transpose(new_joints.reshape(J, B, D), (1, 0, 2))
            # Splice the bind Root position back in (its row is zero).
            root_pos = jnp.broadcast_to(
                jnp.asarray(self.bind_world_transforms[0, :3, 3])[None, None, :],
                (B, 1, 3))
            new_joints = new_joints.at[:, 0, :].set(root_pos[:, 0, :])
        else:
            cols = [jnp.broadcast_to(
                jnp.asarray(self.bind_world_transforms[0, :3, 3])[None, None, :], (B, 1, 3))]
            for j in range(1, J):
                rbf = self.joint_pos_regressors[j]
                if rbf is None:
                    cols.append(jnp.broadcast_to(
                        jnp.asarray(self.bind_world_transforms[j, :3, 3])[None, None, :],
                        (B, 1, 3)))
                    continue
                ids = np.where(self.regressor_mask[:, j])[0]
                target_v = target_shapes[:, ids, :]                        # (B, |ids|, 3)
                query = jnp.asarray(self.bind_world_transforms[j:j + 1, :3, 3])
                pred = rbf.interpolate(target_v, query)
                if pred.ndim == 2:
                    pred = pred[:, None, :]
                cols.append(pred)
            new_joints = jnp.concatenate(cols, axis=1)
        return new_joints[0] if added_batch else new_joints

    # ------------------------------------------------------------------
    # Rotation fitting
    # ------------------------------------------------------------------
    def _precompute_rotation_fit(self) -> None:
        """Static (identity-independent) tensors for the vectorized rotation
        fit. Rebuilt whenever bind data changes (``update_bind``).

        Everything here depends only on the canonical bind shape, bind
        skeleton, skinning weights, and topology — never on the target
        identity — so it is computed once in numpy and treated as a constant
        by ``jax.jit``.
        """
        W = self.skinning_weights                       # (V, J)
        J = self.num_joints
        bind_shape = self.bind_shape                    # (V, 3)
        bj = self.bind_world_transforms[:, :3, 3]       # (J, 3)
        parents = np.asarray(self.joint_parent_ids, dtype=np.int64)

        # ---- stage (a): per-joint inverse-LBS support (weights > 0.01) ----
        M = (W > 0.01).astype(np.float32)               # (V, J)
        # Masked-covariance building blocks. We compute the SOURCE-centered
        # form to avoid float32 catastrophic cancellation:
        #   H_i = Σ_v M_vi (t_v − nj_i)(s_v − bj_i)ᵀ
        #       = Σ_v M_vi t_v (s_v − bj_i)ᵀ  −  nj_i (Σ_v M_vi (s_v − bj_i))ᵀ
        #       = einsum(t, cs)_i  −  nj_i ⊗ Sc_i
        # where cs_{v,i} = M_vi (s_v − bj_i) is the pre-centered source offset
        # (identity-independent) and Sc_i = Σ_v cs_{v,i}. Because s_v − bj_i is
        # small (finger-sized) rather than the ~1e2 cm absolute positions, the
        # expand-then-subtract loses far fewer significant digits than the
        # naive four-term expansion — critical for near-rank-2 finger joints
        # whose ambiguous rotation axis is very sensitive to H perturbation.
        cs = M[:, :, None] * (bind_shape[:, None, :] - bj[None, :, :])       # (V, J, 3)
        self._rot_cs = jnp.asarray(cs)
        self._rot_Sc = jnp.asarray(cs.sum(axis=0))                            # (J, 3)
        self._rot_bj = jnp.asarray(bj)
        self._rot_R0 = jnp.asarray(self.bind_world_transforms[:, :3, :3])

        # First two support vertices per joint (ascending order — matches the
        # loop's np.where order) for the virtual-normal correction; zero-filled
        # for joints with <2 support (their H is overridden / degenerate).
        first2 = np.zeros((J, 2), dtype=np.int64)
        sup1_joints, sup1_verts = [], []
        for j in range(J):
            ids = np.where(M[:, j] > 0)[0]
            if len(ids) >= 2:
                first2[j] = ids[:2]
            elif len(ids) == 1:
                # Single-support joints take the loop's N==1 Rodrigues path.
                sup1_joints.append(j)
                sup1_verts.append(ids[0])
        self._rot_first2 = first2
        self._rot_sup1_joints = np.asarray(sup1_joints, dtype=np.int64)
        self._rot_sup1_verts = np.asarray(sup1_verts, dtype=np.int64)

        # ---- stage (b): padded children table -----------------------------
        # joint_children_ids is a {joint_id: [child_ids]} mapping.
        child_lists = [list(self.joint_children_ids[j]) for j in range(J)]
        max_c = max((len(c) for c in child_lists), default=0)
        max_c = max(max_c, 1)
        children_pad = np.zeros((J, max_c), dtype=np.int64)
        child_mask = np.zeros((J, max_c), dtype=np.float32)
        for j, ch in enumerate(child_lists):
            for k, c in enumerate(ch):
                children_pad[j, k] = c
                child_mask[j, k] = 1.0
        self._rot_children_pad = children_pad
        self._rot_child_mask = jnp.asarray(child_mask)
        # Static child offsets in bind space (zero rows where padded).
        pos_orig = (bj[children_pad] - bj[:, None, :]) * child_mask[..., None]
        self._rot_pos_orig = jnp.asarray(pos_orig.astype(np.float32))

        # ---- static joint classification (mirrors the loop's branches) ----
        n_children = np.asarray([len(c) for c in child_lists])
        is_leaf_skip = np.zeros(J, dtype=bool)
        is_frozen = np.zeros(J, dtype=bool)
        for i in range(1, J):
            if n_children[i] == 0 and self.skip_endjoints:
                is_leaf_skip[i] = True
            elif i in self.freeze_rotations:
                is_frozen[i] = True
        is_normal = ~is_leaf_skip & ~is_frozen
        is_normal[0] = False                            # Root keeps bind R.
        self._rot_is_normal = is_normal
        self._rot_leaf_ids = np.where(is_leaf_skip)[0]
        self._rot_leaf_parents = parents[self._rot_leaf_ids]
        self._rot_frozen_ids = np.where(is_frozen)[0]   # ascending (parents first)
        self._rot_parents = parents
        # Normal joints partitioned by child count for stage (b).
        self._rot_child1_ids = np.where(is_normal & (n_children == 1))[0]
        self._rot_childm_ids = np.where(is_normal & (n_children >= 2))[0]

    def fit_joint_rotations(
        self,
        new_joint_positions: jnp.ndarray,
        target_shapes: jnp.ndarray,
        vectorized: bool = True,
    ) -> jnp.ndarray:
        """Recover per-joint world rotations to align the deformed mesh.

        Args:
            new_joint_positions: (J, 3) or (B, J, 3) joints from
                :meth:`fit_joint_positions`.
            target_shapes: (V, 3) or (B, V, 3) deformed mesh.
            vectorized: run the batched implementation (default). It computes
                the same per-joint Kabsch fits as the reference loop but as a
                handful of large masked matmuls + one batched SVD over all
                joints, instead of a 78-iteration Python loop that unrolls
                into many small kernels. Set False for the literal port of
                SOMA-X's per-joint loop (kept for parity testing).

        Returns:
            (J, 4, 4) or (B, J, 4, 4) new bind-world transforms.
        """
        if vectorized:
            return self._fit_joint_rotations_batched(new_joint_positions, target_shapes)
        return self._fit_joint_rotations_loop(new_joint_positions, target_shapes)

    def _fit_joint_rotations_batched(
        self,
        new_joint_positions: jnp.ndarray,
        target_shapes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorized rotation fit — same math as the loop, batched over joints.

        Stage (a) (inverse-LBS Kabsch) builds the per-joint covariance over
        its (ragged) support set as two dense masked matmuls in the
        SOURCE-CENTERED form ``H = einsum(t, cs) − nj ⊗ Sc`` (``cs`` = the
        pre-centered, identity-independent source offsets). The centering is
        numerically critical: the naive four-term expansion
        ``T1 − S_t⊗bj − nj⊗Ss + cnt·nj⊗bj`` differences ~1e6-magnitude terms
        and loses enough float32 precision that near-rank-2 finger joints —
        whose Kabsch axis is ill-conditioned — resolve to a *different* (still
        valid) rotation than the per-joint loop, diverging by ~0.14 and
        shifting posed finger vertices by millimetres. The centered form keeps
        the two paths in lock-step (see
        ``tests/test_skeleton_transfer.py::test_vectorized_matches_loop_on_real_rig``).

        The virtual-normal correction matches
        :func:`~soma_jax.geometry.transforms.compute_covariance` exactly
        (scale each side's triangle normal by its first support point's
        magnitude, drop when collinear). Stage (b) (child-bone alignment) pads
        children to a fixed width — padded rows are zero on the source side so
        they add nothing to the covariance, and the virtual normal uses the
        real first two children. Joints are dispatched by their STATIC class
        (single-support → Rodrigues, multi-child → Kabsch, leaf → inherit,
        frozen → compose with bind local), exactly mirroring the loop.
        """
        nj = jnp.asarray(new_joint_positions)
        t = jnp.asarray(target_shapes)
        added_batch = nj.ndim == 2
        if added_batch:
            nj = nj[None]
        if t.ndim == 2:
            t = t[None]
        B, J, _ = nj.shape
        if J != self.num_joints:
            raise ValueError(
                f"Expected (..., {self.num_joints}, 3); got {new_joint_positions.shape}")
        eps = 1e-8

        # Covariance -> rotation dispatch: pure-JAX (self.rotation_method, i.e.
        # upstream's "auto" by default), or the Warp svd3 kernel (hybrid). The
        # Warp kernel is plain SVD Procrustes — equivalent to "kabsch", and a
        # fast approximation of "auto" that diverges on ill-conditioned
        # covariances. See soma_jax/geometry/warp_kabsch.py.
        def _rot_from_cov(H_):
            if self._warp_kabsch is not None:
                return self._warp_kabsch(H_)
            return rotation_from_covariance(H_, method=self.rotation_method, eps=eps)

        bj = self._rot_bj                                       # (J, 3)
        R0 = self._rot_R0                                       # (J, 3, 3)
        R0_b = jnp.broadcast_to(R0[None], (B, J, 3, 3))

        # ---------------- stage (a): inverse-LBS Kabsch ---------------------
        if self.skip_inverse_lbs:
            R_init = jnp.broadcast_to(jnp.eye(3, dtype=t.dtype), (B, J, 3, 3))
        else:
            # Masked covariance for ALL joints, source-centered (2 terms) to
            # keep float32 conditioning: H = einsum(t, cs) − nj ⊗ Sc.
            H = (jnp.einsum("bva,vic->biac", t, self._rot_cs)
                 - jnp.einsum("bia,ic->biac", nj, self._rot_Sc))

            # Virtual-normal correction — same formula as
            # transforms.compute_covariance (faithful to SOMA-X): scale each
            # side's triangle normal by its first support point's magnitude,
            # and drop the term where a triangle is collinear. f2 = the joint's
            # first two support vertices (ascending, matching the loop's
            # np.where order); a* = target (deformed) offsets, b* = source
            # (bind) offsets.
            f2 = self._rot_first2
            a0 = t[:, f2[:, 0]] - nj
            a1 = t[:, f2[:, 1]] - nj                                     # (B, J, 3)
            b0 = jnp.asarray(self.bind_shape)[f2[:, 0]] - bj
            b1 = jnp.asarray(self.bind_shape)[f2[:, 1]] - bj             # (J, 3)
            n_src = jnp.cross(a0, a1, axis=-1)
            n_dst = jnp.cross(b0, b1, axis=-1)
            len_n_src = jnp.linalg.norm(n_src, axis=-1, keepdims=True)   # (B, J, 1)
            len_n_dst = jnp.linalg.norm(n_dst, axis=-1, keepdims=True)   # (J, 1)
            v_src = n_src * (jnp.linalg.norm(a0, axis=-1, keepdims=True) / (len_n_src + eps))
            v_dst = n_dst * (jnp.linalg.norm(b0, axis=-1, keepdims=True) / (len_n_dst + eps))
            valid = (len_n_src[..., 0] > 1e-9) & (len_n_dst[..., 0] > 1e-9)  # (B, J)
            contrib = jnp.einsum("bia,ic->biac", v_src, v_dst)
            H = H + jnp.where(valid[..., None, None], contrib, 0.0)
            R_init = _rot_from_cov(H)

            # Single-support joints take the loop's N==1 Rodrigues path.
            if len(self._rot_sup1_joints) > 0:
                sj = self._rot_sup1_joints
                sv = self._rot_sup1_verts
                A1 = t[:, sv] - nj[:, sj]                                 # (B, n1, 3)
                B1 = jnp.asarray(self.bind_shape)[sv] - bj[sj]            # (n1, 3)
                B1_b = jnp.broadcast_to(B1[None], A1.shape)
                R_init = R_init.at[:, sj].set(rodrigues_rotation(A1, B1_b))

        # ---------------- stage (b): child-bone alignment -------------------
        pos_orig_rot = jnp.einsum("bjac,jnc->bjna", R_init, self._rot_pos_orig)
        pos_new = (nj[:, self._rot_children_pad] - nj[:, :, None, :]) \
            * self._rot_child_mask[None, :, :, None]                      # (B, J, maxC, 3)

        align = jnp.broadcast_to(jnp.eye(3, dtype=t.dtype), (B, J, 3, 3))
        if len(self._rot_childm_ids) > 0:
            jm = self._rot_childm_ids
            # align_vectors(A, B) for N>=2 == rotation_from_covariance(
            # compute_covariance(A, B, virtual_normal=True)); build the
            # covariance explicitly so the SVD goes through the same
            # (jax|warp) dispatch as stage (a).
            Hb = compute_covariance(pos_new[:, jm], pos_orig_rot[:, jm],
                                    virtual_normal=True, eps=eps)
            align = align.at[:, jm].set(_rot_from_cov(Hb))
        if len(self._rot_child1_ids) > 0:
            j1 = self._rot_child1_ids
            align = align.at[:, j1].set(
                rodrigues_rotation(pos_new[:, j1, 0], pos_orig_rot[:, j1, 0]))

        R_fit = align @ R_init @ R0_b

        # ---------------- composition / overrides ---------------------------
        normal = jnp.asarray(self._rot_is_normal)[None, :, None, None]
        R = jnp.where(normal, R_fit, R0_b)
        # Frozen joints compose the (already final) parent rotation with the
        # bind local — ascending order guarantees parents are resolved first.
        bind_local_R = jnp.asarray(self.bind_local_transforms[:, :3, :3])
        for i in self._rot_frozen_ids:
            p = int(self._rot_parents[i])
            R = R.at[:, i].set(R[:, p] @ bind_local_R[i][None])
        # Leaf joints inherit their parent's fitted rotation (a leaf's parent
        # necessarily has children, so it was fitted above).
        if len(self._rot_leaf_ids) > 0:
            R = R.at[:, self._rot_leaf_ids].set(R[:, self._rot_leaf_parents])

        world_bind_pose = se3_from_rt(R, nj)
        return world_bind_pose[0] if added_batch else world_bind_pose

    def _fit_joint_rotations_loop(
        self,
        new_joint_positions: jnp.ndarray,
        target_shapes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Literal port of SOMA-X's per-joint rotation-fit loop.

        Kept as the reference implementation for parity tests; production
        callers use :meth:`_fit_joint_rotations_batched` via
        :meth:`fit_joint_rotations`.
        """
        new_joint_positions = jnp.asarray(new_joint_positions)
        target_shapes = jnp.asarray(target_shapes)
        added_batch = new_joint_positions.ndim == 2
        if added_batch:
            new_joint_positions = new_joint_positions[None]
        if target_shapes.ndim == 2:
            target_shapes = target_shapes[None]
        B, J, _ = new_joint_positions.shape
        if J != self.num_joints:
            raise ValueError(
                f"Expected (..., {self.num_joints}, 3); got {new_joint_positions.shape}")

        bind_world = jnp.asarray(self.bind_world_transforms)                 # (J, 4, 4)
        bind_local = jnp.asarray(self.bind_local_transforms)                # (J, 4, 4)
        R0 = bind_world[..., :3, :3]                                         # (J, 3, 3)
        R = jnp.broadcast_to(R0[None], (B, J, 3, 3))                         # (B, J, 3, 3)

        bind_shape_j = jnp.asarray(self.bind_shape)

        for i in range(1, J):
            children = self.joint_children_ids[i]
            if not children and self.skip_endjoints:
                p = self.joint_parent_ids[i]
                R = R.at[:, i, :, :].set(R[:, p, :, :])
                continue
            if i in self.freeze_rotations:
                p = self.joint_parent_ids[i]
                R_i = R[:, p, :, :] @ bind_local[i, :3, :3][None]
                R = R.at[:, i, :, :].set(R_i)
                continue

            # ---- (a) Inverse-LBS shoulder rotation from skinned vertices.
            if self.skip_inverse_lbs:
                R_init = jnp.broadcast_to(jnp.eye(3, dtype=R.dtype)[None], (B, 3, 3))
            else:
                # Support mask comes from the STATIC bind skinning weights
                # (numpy attribute) — keeps this method jit-traceable; the
                # mask is identity-independent so it never needs to be traced.
                ids = np.where(self.skinning_weights[:, i] > 0.01)[0]
                if len(ids) == 0:
                    R_init = jnp.broadcast_to(jnp.eye(3, dtype=R.dtype)[None], (B, 3, 3))
                else:
                    skinned_orig = bind_shape_j[ids] - bind_world[i, :3, 3]      # (n, 3)
                    skinned_orig_b = jnp.broadcast_to(skinned_orig[None], (B,) + skinned_orig.shape)
                    skinned_new = target_shapes[:, ids, :] - new_joint_positions[:, i:i + 1, :]
                    R_init = align_vectors(skinned_new, skinned_orig_b, method=self.rotation_method)

            # ---- (b) Child-bone alignment Kabsch.
            children_arr = np.asarray(list(children), dtype=np.int64)
            if len(children_arr) > 0:
                pos_orig = (bind_world[children_arr, :3, 3] - bind_world[i, :3, 3])  # (k, 3)
                pos_orig_b = jnp.broadcast_to(pos_orig[None], (B,) + pos_orig.shape)
                # Apply the inverse-LBS rotation to the canonical child offsets
                # so they're in the "shoulder-aligned" frame before fitting.
                pos_orig_rot = jnp.einsum("bij,bnj->bni", R_init, pos_orig_b)
                pos_new = (new_joint_positions[:, children_arr, :]
                           - new_joint_positions[:, i:i + 1, :])
                align_rot = align_vectors(pos_new, pos_orig_rot, method=self.rotation_method)
                R_i = align_rot @ R_init @ R[:, i, :, :]
            else:
                R_i = R_init @ R[:, i, :, :]
            R = R.at[:, i, :, :].set(R_i)

        world_bind_pose = se3_from_rt(R, new_joint_positions)
        return world_bind_pose[0] if added_batch else world_bind_pose

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    def fit(self, target_shapes: jnp.ndarray) -> jnp.ndarray:
        """Position + rotation fit in one call (matches SOMA-X ``fit``).

        Args:
            target_shapes: (V, 3) or (B, V, 3).

        Returns:
            (J, 4, 4) or (B, J, 4, 4) new bind-world transforms.
        """
        new_joints = self.fit_joint_positions(target_shapes)
        return self.fit_joint_rotations(new_joints, target_shapes)


# ---------------------------------------------------------------------------
# Pose mirror (unchanged from previous version, kept for back-compat callers).
# ---------------------------------------------------------------------------
class PoseMirror:
    """Mirror SOMA poses across the sagittal (left-right symmetry) plane.

    The mirror is defined by a mapping of joint indices (left ↔ right)
    and axis negation signs.

    Args:
        joint_names: list of joint name strings.
        mirror_axis: which world axis is the sagittal normal (default 'x').
    """

    def __init__(self, joint_names: list[str], mirror_axis: str = "x"):
        self.joint_names = joint_names
        self.mirror_axis = mirror_axis
        self._axis_idx = {"x": 0, "y": 1, "z": 2}[mirror_axis.lower()]
        self._build_mirror_map()

    def _build_mirror_map(self) -> None:
        J = len(self.joint_names)
        self.mirror_indices = list(range(J))
        name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        for i, name in enumerate(self.joint_names):
            if "left" in name.lower() or name.lower().endswith("_l"):
                mirror_name = (
                    name.lower()
                    .replace("left", "right")
                    .replace("_l", "_r")
                    .replace("l_", "r_")
                )
                for candidate, idx in name_to_idx.items():
                    if candidate.lower() == mirror_name:
                        self.mirror_indices[i] = idx
                        self.mirror_indices[idx] = i
                        break

    def mirror_rotmats(self, rotmats: np.ndarray) -> np.ndarray:
        was_unbatched = rotmats.ndim == 3
        if was_unbatched:
            rotmats = rotmats[None]
        mirrored = rotmats[:, self.mirror_indices]
        ax = self._axis_idx
        negate_axes = [a for a in (0, 1, 2) if a != ax]
        for a in negate_axes:
            mirrored = mirrored.at[:, :, a, :].multiply(-1)
            mirrored = mirrored.at[:, :, :, a].multiply(-1)
        return mirrored[0] if was_unbatched else mirrored

    def mirror_rotmats_jax(self, rotmats: jnp.ndarray) -> jnp.ndarray:
        was_unbatched = rotmats.ndim == 3
        if was_unbatched:
            rotmats = rotmats[None]
        idx = jnp.array(self.mirror_indices)
        mirrored = rotmats[:, idx]
        ax = self._axis_idx
        negate_axes = [a for a in (0, 1, 2) if a != ax]
        for a in negate_axes:
            mirrored = mirrored.at[:, :, a, :].multiply(-1)
            mirrored = mirrored.at[:, :, :, a].multiply(-1)
        return mirrored[0] if was_unbatched else mirrored
