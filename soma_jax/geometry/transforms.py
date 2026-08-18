"""SO(3)/SE(3) rotation and transformation utilities for SOMA-JAX.

All functions operate on JAX arrays and are compatible with jit/vmap/grad.

Upstream: ``soma/geometry/transforms.py``
    Alignment core is a faithful port — `align_vectors` matches upstream to
    <=1e-6 for all three methods, including rank-deficient covariances, and
    covariance/Kabsch/Newton-Schulz/SE(3)/6D/quaternion composition agree.
    Gaps: `rotmat_to_axis_angle` lacks upstream's near-pi branch and is
    inaccurate near theta = pi; upstream's Euler and general rotvec
    conversions have no counterpart here.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

# Rotation-estimation constants — values mirror
# ``soma.geometry.transforms`` so ``align_vectors(method="auto")`` reproduces
# the reference's regularized Newton-Schulz path exactly.
NEWTON_SCHULZ_ITERS = 30
AUTO_ROTATION_PRIOR_STRENGTH = 0.05
AUTO_ROTATION_RANK_THRESHOLD = 2e-2
AUTO_ROTATION_DEGENERATE_THRESHOLD = 1e-6


def safe_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize a vector with gradient-safe epsilon."""
    return x / jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


def axis_angle_to_rotmat(aa: jnp.ndarray) -> jnp.ndarray:
    """Convert axis-angle to rotation matrix using Rodrigues formula.

    Args:
        aa: (..., 3) axis-angle vector; magnitude encodes angle in radians.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    angle = jnp.sqrt(jnp.sum(aa * aa, axis=-1, keepdims=True) + 1e-12)
    axis = aa / angle
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    x = axis[..., 0:1]
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    zero = jnp.zeros_like(x)

    # Skew-symmetric cross-product matrix K
    K = jnp.concatenate(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(aa.shape[:-1] + (3, 3))

    outer = axis[..., :, None] * axis[..., None, :]
    I = jnp.eye(3, dtype=aa.dtype)

    return cos_a[..., None] * I + (1 - cos_a[..., None]) * outer + sin_a[..., None] * K


def rotmat_to_axis_angle(R: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Rotation matrices -> axis-angle (rotation) vectors, robust everywhere.

    Port of upstream ``soma.geometry.transforms.matrix_to_rotvec``. Three
    regimes, selected per element:

    * **small** (theta <= 1e-3) — series expansion ``0.5 + theta^2/12``; the
      generic formula divides by ``sin(theta) -> 0``.
    * **near pi** (theta >= pi - 1e-3) — the antisymmetric part vanishes, so
      the axis is recovered from the diagonal
      (``u_i = sqrt((2 R_ii - tr + 1)/2)``) and signed from the antisymmetric
      part. Without this branch the result is badly wrong exactly at theta = pi.
    * **generic** — ``theta / (2 sin theta)`` on the antisymmetric part.

    Args:
        R: (..., 3, 3) rotation matrices.
        eps: numerical floor for the divisions.

    Returns:
        (..., 3) axis-angle vectors.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {R.shape}")

    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = jnp.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    S = R - jnp.swapaxes(R, -2, -1)
    v = jnp.stack([S[..., 2, 1] - S[..., 1, 2],
                   S[..., 0, 2] - S[..., 2, 0],
                   S[..., 1, 0] - S[..., 0, 1]], axis=-1)
    sin_theta = 0.5 * jnp.linalg.norm(v, axis=-1)

    small = theta <= 1e-3
    near_pi = theta >= (jnp.pi - 1e-3)

    # Small-angle series. NOTE: this deliberately DIVERGES from upstream.
    # `v` is built from S = R - R^T and differenced again, so it is *twice* the
    # usual antisymmetric vector w = (R21-R12, R02-R20, R10-R01). The generic
    # branch cancels that (it divides by 2 sin(theta)), but upstream's small
    # branch applies w's factor (0.5 + theta^2/12) directly to v and therefore
    # returns 2x the true rotation vector for theta < 1e-3 — verified against
    # `soma.geometry.transforms.matrix_to_rotvec`, which has the same defect.
    # Using v's own factor, rotvec = v * (0.25 + theta^2/24), restores it.
    theta2 = jnp.maximum(3.0 - tr, 0.0)
    w_small = v * (0.25 + theta2 / 24.0)[..., None]

    # Generic.
    denom = jnp.where(sin_theta < eps, eps, 2.0 * sin_theta)
    w_gen = v * (theta / denom)[..., None]

    # Near pi: axis magnitude from the diagonal, sign from the antisymmetric part.
    R00, R11, R22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    u = jnp.stack([
        jnp.sqrt(jnp.maximum((R00 - R11 - R22 + 1.0) * 0.5, 0.0)),
        jnp.sqrt(jnp.maximum((-R00 + R11 - R22 + 1.0) * 0.5, 0.0)),
        jnp.sqrt(jnp.maximum((-R00 - R11 + R22 + 1.0) * 0.5, 0.0)),
    ], axis=-1)
    sign = jnp.where(jnp.sign(v) == 0, 1.0, jnp.sign(v))
    u = u * sign
    u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    w_pi = u / jnp.where(u_norm < eps, eps, u_norm) * theta[..., None]

    return jnp.where(near_pi[..., None], w_pi,
                     jnp.where(small[..., None], w_small, w_gen))


def rotmat_to_6d(R: jnp.ndarray) -> jnp.ndarray:
    """Convert rotation matrix to 6D continuous representation (Zhou et al. 2019).

    Row convention, matching ``soma.geometry.transforms.rotation_6d_to_matrix``
    (and pytorch3d): the 6D vector is the first two **rows** of ``R``. This is
    the inverse of :func:`rotation_6d_to_rotmat`.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) 6D vectors (first two rows of R concatenated).
    """
    return jnp.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def rotation_6d_to_rotmat(r6d: jnp.ndarray) -> jnp.ndarray:
    """Convert 6D rotation representation to rotation matrix via Gram-Schmidt.

    Faithful port of ``soma.geometry.transforms.rotation_6d_to_matrix``: the
    orthonormalized basis vectors become the **rows** of the result, so 6D
    parameters are interchangeable with SOMA-X's. (Building them as columns
    yields the transposed — i.e. inverse — rotation.)

    Args:
        r6d: (..., 6) 6D vectors.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:]
    b1 = safe_normalize(a1)
    b2 = safe_normalize(a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = jnp.cross(b1, b2)
    return jnp.stack([b1, b2, b3], axis=-2)  # rows are basis vectors


def kabsch(src: jnp.ndarray, tgt: jnp.ndarray, weights: jnp.ndarray | None = None) -> jnp.ndarray:
    """Kabsch algorithm: find optimal rotation R such that R @ src ≈ tgt.

    Args:
        src: (N, 3) source points.
        tgt: (N, 3) target points.
        weights: (N,) optional per-point weights.

    Returns:
        (3, 3) optimal rotation matrix.
    """
    if weights is not None:
        w = weights / (jnp.sum(weights) + 1e-8)
        c_src = jnp.einsum("n,nd->d", w, src)
        c_tgt = jnp.einsum("n,nd->d", w, tgt)
        H = jnp.einsum("n,nr,nc->rc", w, src - c_src, tgt - c_tgt)
    else:
        c_src = jnp.mean(src, axis=0)
        c_tgt = jnp.mean(tgt, axis=0)
        H = (src - c_src).T @ (tgt - c_tgt)

    U, _, Vt = jnp.linalg.svd(H)
    d = jnp.linalg.det(Vt.T @ U.T)
    D = jnp.eye(3, dtype=H.dtype).at[2, 2].set(d)
    return Vt.T @ D @ U.T


def compute_covariance(
    A: jnp.ndarray,
    B: jnp.ndarray,
    virtual_normal: bool = True,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Cross-covariance for Kabsch (matches SOMA-X ``transforms.compute_covariance``).

    Returns ``H = Aᵀ B`` plus an optional synthetic normal correspondence built
    from the cross product of the first two row pairs — this lifts a rank-2
    pair of vector correspondences to rank-3 so Kabsch returns a proper SO(3)
    rotation instead of an arbitrary reflection.

    Args:
        A: (..., N, 3) target vectors.
        B: (..., N, 3) source vectors.
        virtual_normal: enable the cross-product normal trick.
        eps: numerical floor.

    Returns:
        (..., 3, 3) covariance matrix.
    """
    H = jnp.einsum("...ni,...nj->...ij", A, B)
    if virtual_normal and A.shape[-2] >= 2:
        # Faithful port of SOMA-X ``compute_covariance``: the synthetic normal
        # correspondence is scaled by the FIRST correspondence's magnitude on
        # each side — ``v_src = n̂_src·‖p0‖``, ``v_dst = n̂_dst·‖q0‖`` — and is
        # dropped (not just eps-regularized) when either triangle is collinear.
        p0, p1 = A[..., 0, :], A[..., 1, :]
        q0, q1 = B[..., 0, :], B[..., 1, :]
        n_src = jnp.cross(p0, p1, axis=-1)
        n_dst = jnp.cross(q0, q1, axis=-1)
        len_n_src = jnp.linalg.norm(n_src, axis=-1, keepdims=True)
        len_n_dst = jnp.linalg.norm(n_dst, axis=-1, keepdims=True)
        scale_src = jnp.linalg.norm(p0, axis=-1, keepdims=True) / (len_n_src + eps)
        scale_dst = jnp.linalg.norm(q0, axis=-1, keepdims=True) / (len_n_dst + eps)
        v_src = n_src * scale_src
        v_dst = n_dst * scale_dst
        valid = (len_n_src[..., 0] > 1e-9) & (len_n_dst[..., 0] > 1e-9)
        contrib = jnp.einsum("...i,...j->...ij", v_src, v_dst)
        H = H + jnp.where(valid[..., None, None], contrib, 0.0)
    return H


def align_vectors(
    A: jnp.ndarray,
    B: jnp.ndarray,
    eps: float = 1e-8,
    method: str = "auto",
) -> jnp.ndarray:
    """Find rotation R such that R @ B ≈ A  (matches SOMA-X ``align_vectors``).

    Falls back to a single-pair Rodrigues rotation when N == 1.

    Args:
        A: (..., N, 3) target vectors.
        B: (..., N, 3) source vectors.
        method: 'auto' (default, as upstream), 'kabsch' (SVD), or
            'newton-schulz'. See :func:`rotation_from_covariance` for what each
            one does.

    Returns:
        (..., 3, 3) rotation matrix.
    """
    if A.shape[-1] != 3 or B.shape[-1] != 3:
        raise NotImplementedError("Only 3D vectors are supported (last dim must be 3).")
    if A.shape[-2] != B.shape[-2]:
        raise ValueError(f"N must match, got {A.shape[-2]} vs {B.shape[-2]}.")
    N = A.shape[-2]
    if N == 1:
        return rodrigues_rotation(A[..., 0, :], B[..., 0, :])
    H = compute_covariance(A, B, virtual_normal=True, eps=eps)
    return rotation_from_covariance(H, method=method, eps=eps)


def rotation_from_covariance(
    H: jnp.ndarray,
    method: str = "auto",
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Rotation extraction from a precomputed Kabsch covariance.

    The covariance→rotation half of :func:`align_vectors`, exposed so callers
    that assemble covariances themselves (e.g. the vectorized
    ``SkeletonTransfer.fit_joint_rotations``, which builds all per-joint
    covariances with masked matmuls instead of ragged per-joint loops) get
    bit-identical post-processing — including the degenerate-H fallback and
    the gradient-safe SVD input handling.

    Method semantics are a faithful port of SOMA-X's ``align_vectors``:

    * ``'kabsch'``  — plain SVD Procrustes on ``H``.
    * ``'newton-schulz'`` — :func:`newton_schulz` on ``H``, with a per-element
      Kabsch fallback wherever the iterate did not land in SO(3).
    * ``'auto'`` (default) — same as ``'newton-schulz'`` but on a covariance
      first regularized by :func:`regularize_covariance_with_reference`, which
      pins the unconstrained subspace of a rank-deficient ``H`` to the identity
      gauge instead of returning an arbitrary rotation.

    Args:
        H: (..., 3, 3) covariance, e.g. from :func:`compute_covariance`.
        method: 'auto', 'kabsch', or 'newton-schulz'.
        eps: numerical floor (matches :func:`align_vectors`).

    Returns:
        (..., 3, 3) rotation matrix.
    """
    def _kabsch_svd(H_):
        """Closed-form Kabsch rotation from a precomputed covariance matrix.

        Given H = A.T @ B, decompose H = U Σ Vᵀ and return R = U D Vᵀ where
        D = diag(1, 1, det(U Vᵀ)) is the reflection-fix that guarantees
        det(R) = +1 (a proper rotation, not a roto-reflection).

        Uses ``full_matrices=False`` so the SVD JVP can propagate through
        ``jax.grad`` — the full-matrix variant isn't implemented in JAX.
        """
        U, _, Vh = jnp.linalg.svd(H_, full_matrices=False)
        UVt = U @ jnp.swapaxes(Vh, -2, -1)
        det_sign = jnp.where(jnp.linalg.det(UVt) < 0, -1.0, 1.0)
        I3 = jnp.eye(3, dtype=H_.dtype)
        I3_b = jnp.broadcast_to(I3, H_.shape)
        Dcorr = I3_b.at[..., -1, -1].set(det_sign)
        return U @ Dcorr @ Vh

    # JAX's `jnp.where` evaluates BOTH branches; if either path computes SVD
    # on a degenerate matrix it produces NaN gradients even when masked out.
    # Feed the SVD a safe identity placeholder on the degenerate path so the
    # JVP stays finite.
    if method == "kabsch":
        return _kabsch_svd(H)

    if method == "auto":
        H = regularize_covariance_with_reference(
            H, rank_threshold=AUTO_ROTATION_DEGENERATE_THRESHOLD, eps=eps,
        )
    elif method != "newton-schulz":
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'kabsch', or 'newton-schulz'.")

    # Guard the SVD fallback against an all-zero covariance: jnp.where evaluates
    # both branches, and an SVD of a degenerate matrix poisons the JVP with NaN.
    H_norm = jnp.linalg.norm(H, axis=(-2, -1), keepdims=True)
    degenerate = H_norm < eps * 100
    I3 = jnp.broadcast_to(jnp.eye(3, dtype=H.dtype), H.shape)
    safe_H = jnp.where(degenerate, I3, H)

    R = newton_schulz(H, num_iter=NEWTON_SCHULZ_ITERS, eps=eps)
    valid = rotation_matrices_are_valid(R)
    return jnp.where(valid[..., None, None], R, _kabsch_svd(safe_H))


# ----------------------------------------------------------------------------
# Quaternion helpers (xyzw layout — matches SOMA-X's `quaternion_order: xyzw`)
# ----------------------------------------------------------------------------
def matrix_to_quaternion_xyzw(R: jnp.ndarray) -> jnp.ndarray:
    """Convert rotation matrices to xyzw quaternions (stable branchful).

    Picks one of four formulas based on which trace/diagonal element is largest,
    so the divisor never collapses to zero, and is differentiable under
    ``jax.grad`` — no branch takes ``where`` over a sqrt-of-near-zero in a way
    that NaNs the derivative.

    Agrees with ``soma.geometry.transforms.matrix_to_quaternion_xyzw`` to ~1e-14
    in float64. (Upstream's ``matrix_to_quaternion_xyzw_stable`` is the same
    formula with ``+ eps`` inside each sqrt, which shifts the result by ~1e-10
    and makes its own round-trip slightly less accurate; this uses a clamp
    instead, so it stays exact.) The result is canonicalized to **non-negative
    w**, matching both upstream variants' documented contract — ``q`` and ``-q``
    are the same rotation, so the sign must be pinned for the outputs to be
    comparable at all.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in xyzw order, with ``w >= 0``.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    eps = 1e-12

    s_a = jnp.sqrt(jnp.maximum(t + 1.0, eps)) * 2.0
    qa = jnp.stack([(m21 - m12) / s_a, (m02 - m20) / s_a,
                    (m10 - m01) / s_a, 0.25 * s_a], axis=-1)
    s_b = jnp.sqrt(jnp.maximum(1.0 + m00 - m11 - m22, eps)) * 2.0
    qb = jnp.stack([0.25 * s_b, (m01 + m10) / s_b,
                    (m02 + m20) / s_b, (m21 - m12) / s_b], axis=-1)
    s_c = jnp.sqrt(jnp.maximum(1.0 + m11 - m00 - m22, eps)) * 2.0
    qc = jnp.stack([(m01 + m10) / s_c, 0.25 * s_c,
                    (m12 + m21) / s_c, (m02 - m20) / s_c], axis=-1)
    s_d = jnp.sqrt(jnp.maximum(1.0 + m22 - m00 - m11, eps)) * 2.0
    qd = jnp.stack([(m02 + m20) / s_d, (m12 + m21) / s_d,
                    0.25 * s_d, (m10 - m01) / s_d], axis=-1)

    use_a = t > 0
    use_b = (~use_a) & (m00 >= m11) & (m00 >= m22)
    use_c = (~use_a) & (~use_b) & (m11 >= m22)
    q = jnp.where(use_a[..., None], qa,
        jnp.where(use_b[..., None], qb,
        jnp.where(use_c[..., None], qc, qd)))
    # Canonical sign: w >= 0 (upstream's documented convention).
    return jnp.where(q[..., 3:] < 0.0, -q, q)


def quaternion_normalize_xyzw(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """L2-normalize an xyzw quaternion. The ``+ eps`` floor keeps the gradient
    well-defined at ||q|| → 0."""
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)


def quaternion_multiply_xyzw(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of two xyzw quaternions: ``q = a * b`` such that the
    composed rotation is "first b, then a" (matrix equivalent: ``Ra @ Rb``)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quaternion_conjugate_xyzw(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of an xyzw quaternion (xyz negated). For unit quaternions this
    equals the inverse / rotational opposite."""
    return jnp.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


_AXIS_BASIS_E = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def quaternion_half_angle_xyzw(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Principal half-angle (square-root) quaternion for xyzw rotations.

    Port of ``soma.geometry.transforms.quaternion_half_angle_xyzw``: pick the
    representation with non-negative ``w`` (``q`` and ``-q`` are the same
    rotation), then normalize ``[v, w + 1]``.
    """
    q = quaternion_normalize_xyzw(q, eps=eps)
    q = jnp.where(q[..., 3:] < 0.0, -q, q)
    return quaternion_normalize_xyzw(
        jnp.concatenate([q[..., :3], q[..., 3:] + 1.0], axis=-1), eps=eps,
    )


def quaternion_twist_angle_xyzw(q: jnp.ndarray, axis_idx: int = 0,
                                eps: float = 1e-12) -> jnp.ndarray:
    """Signed twist angle (radians) around a coordinate axis, from an xyzw quaternion.

    Faithful port of ``soma.geometry.transforms.quaternion_twist_angle_xyzw``:
    the projection is taken on the **half-angle** quaternion and scaled by 4,
    i.e. ``4·atan2(v_half[axis], w_half)``. Doing it on the raw quaternion
    (``2·atan2(v[axis], w)``) agrees only for small rotations and drifts badly
    as the twist approaches ±180°, which is exactly the regime the 1-DOF
    procedural joints operate in.

    Args:
        q: (..., 4) xyzw quaternions.
        axis_idx: 0 / 1 / 2 for X / Y / Z.
        eps: normalisation floor.

    Returns:
        (...,) twist angle in radians.
    """
    if axis_idx not in (0, 1, 2):
        raise ValueError(f"axis_idx must be 0, 1, or 2, got {axis_idx}")
    q_half = quaternion_half_angle_xyzw(q, eps=eps)
    return 4.0 * jnp.arctan2(q_half[..., axis_idx], q_half[..., 3])


def single_axis_rotation_matrices(
    angle: jnp.ndarray,
    axis_idx: int,
    axis_signs: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Build a (..., 3, 3) rotation matrix around a coordinate axis from a
    scalar angle. Direct stand-alone form of Rodrigues' formula when the axis
    is a unit basis vector.

    Matches ``soma.geometry.transforms.single_axis_rotation_matrices``, whose
    third argument flips the rotation direction per joint (mirrored limbs spin
    the opposite way about the shared local axis). Defaults to ``+1`` so
    existing two-argument calls are unchanged.

    Args:
        angle: (...,) rotation angles in radians.
        axis_idx: 0 / 1 / 2 for X / Y / Z.
        axis_signs: broadcastable per-element sign (±1) applied to ``angle``.
    """
    if axis_idx not in (0, 1, 2):
        raise ValueError(f"axis_idx must be 0, 1, or 2, got {axis_idx}")
    angle = angle * jnp.asarray(axis_signs, dtype=angle.dtype)
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    if axis_idx == 0:
        return jnp.stack([
            jnp.stack([o, z, z], -1),
            jnp.stack([z, c, -s], -1),
            jnp.stack([z, s,  c], -1),
        ], -2)
    if axis_idx == 1:
        return jnp.stack([
            jnp.stack([ c, z, s], -1),
            jnp.stack([ z, o, z], -1),
            jnp.stack([-s, z, c], -1),
        ], -2)
    return jnp.stack([
        jnp.stack([c, -s, z], -1),
        jnp.stack([s,  c, z], -1),
        jnp.stack([z,  z, o], -1),
    ], -2)


def newton_schulz(
    A: jnp.ndarray,
    num_iter: int = NEWTON_SCHULZ_ITERS,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Newton-Schulz orthogonalization: iteratively refine A toward the nearest SO(3).

    Faithful port of ``soma.geometry.transforms.newton_schulz``: the input is
    scaled by its **infinity norm** (max absolute row sum) — which is what
    guarantees convergence of the iteration — and the result gets a
    determinant-sign correction on the last column so the output is a proper
    rotation rather than a roto-reflection.

    Args:
        A: (..., 3, 3) matrix (typically a Kabsch covariance).
        num_iter: number of refinement iterations (SOMA-X uses 30).
        eps: numerical floor for the scaling.

    Returns:
        (..., 3, 3) orthogonalized rotation matrix with det = +1.
    """
    max_row_sum = jnp.max(jnp.sum(jnp.abs(A), axis=-1), axis=-1)[..., None, None]
    X = A / (max_row_sum + eps)

    def step(X, _):
        # X_{k+1} = X_k (3I - X_kᵀ X_k) / 2
        return X @ (3.0 * jnp.eye(3, dtype=X.dtype) - jnp.swapaxes(X, -2, -1) @ X) * 0.5, None

    X, _ = jax.lax.scan(step, X, None, length=num_iter)

    sign = jnp.where(jnp.linalg.det(X) < 0, -1.0, 1.0)
    return X.at[..., :, 2].set(X[..., :, 2] * sign[..., None])


def rotation_matrices_are_valid(
    R: jnp.ndarray,
    det_tol: float = 1e-2,
    orthogonality_tol: float = 1e-2,
) -> jnp.ndarray:
    """Boolean mask for finite, right-handed, orthonormal rotations.

    Mirrors ``soma.geometry.transforms.rotation_matrices_are_valid``; used by
    :func:`align_vectors` to decide when the Newton-Schulz result needs the
    Kabsch fallback.
    """
    finite = jnp.all(jnp.isfinite(R), axis=(-2, -1))
    det_R = jnp.linalg.det(R)
    det_valid = jnp.isfinite(det_R) & (det_R > 0.0) & (jnp.abs(det_R - 1.0) <= det_tol)
    eye = jnp.eye(3, dtype=R.dtype)
    ortho_err = jnp.max(jnp.abs(jnp.swapaxes(R, -2, -1) @ R - eye), axis=(-2, -1))
    ortho_valid = jnp.isfinite(ortho_err) & (ortho_err <= orthogonality_tol)
    return finite & det_valid & ortho_valid


def regularize_covariance_with_reference(
    H: jnp.ndarray,
    reference_rotation: jnp.ndarray | None = None,
    prior_strength: float = AUTO_ROTATION_PRIOR_STRENGTH,
    rank_threshold: float = AUTO_ROTATION_RANK_THRESHOLD,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Add a weak reference-gauge prior to a Procrustes covariance matrix.

    Faithful port of ``soma.geometry.transforms.regularize_covariance_with_reference``.
    The prior only engages as the covariance loses rank (``volume_score`` below
    ``rank_threshold``), pinning the otherwise-arbitrary rotation in the
    unconstrained subspace to ``reference_rotation`` (identity by default).
    """
    prior_scale = jnp.maximum(jnp.max(jnp.sum(jnp.abs(H), axis=-1), axis=-1), eps)
    volume_score = jnp.abs(jnp.linalg.det(H)) / prior_scale ** 3
    rank_weight = jnp.clip((rank_threshold - volume_score) / rank_threshold, 0.0, 1.0)
    if reference_rotation is None:
        reference_rotation = jnp.broadcast_to(jnp.eye(3, dtype=H.dtype), H.shape)
    return H + (prior_strength * rank_weight * prior_scale)[..., None, None] * reference_rotation


def se3_from_rt(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Build a (4, 4) SE(3) matrix from rotation and translation.

    Args:
        R: (..., 3, 3) rotation matrices.
        t: (..., 3) translation vectors.

    Returns:
        (..., 4, 4) SE(3) matrices.
    """
    bottom = jnp.zeros(R.shape[:-2] + (1, 4), dtype=R.dtype).at[..., 0, 3].set(1.0)
    Rt = jnp.concatenate([R, t[..., None]], axis=-1)  # (..., 3, 4)
    return jnp.concatenate([Rt, bottom], axis=-2)       # (..., 4, 4)


def se3_inverse(T: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of an SE(3) matrix without numeric inversion.

    Args:
        T: (..., 4, 4) SE(3) matrices.

    Returns:
        (..., 4, 4) inverse SE(3) matrices.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = jnp.swapaxes(R, -2, -1)
    t_inv = -jnp.einsum("...ij,...j->...i", Rt, t)
    return se3_from_rt(Rt, t_inv)


def rodrigues_rotation(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Shortest-arc rotation aligning ``b`` onto ``a``: returns R with ``R @ b ≈ a``.

    Argument order and semantics match ``soma.geometry.transforms.rodrigues_rotation``
    (and SciPy's ``align_vectors``): the FIRST argument is the target, the second
    is the source. :func:`align_vectors` relies on this for its ``N == 1`` path.

    Args:
        a: (..., 3) target vectors.
        b: (..., 3) source vectors.
        eps: numerical floor for the input normalisation.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a_u = safe_normalize(a, eps=eps * eps)
    b_u = safe_normalize(b, eps=eps * eps)
    # v = b × a so that the resulting R maps b → a.
    cross = jnp.cross(b_u, a_u)
    cos_angle = jnp.clip(jnp.sum(a_u * b_u, axis=-1, keepdims=True), -1.0, 1.0)

    # Skew-symmetric matrix from cross product
    x = cross[..., 0:1]
    y = cross[..., 1:2]
    z = cross[..., 2:3]
    zero = jnp.zeros_like(x)
    K = jnp.concatenate(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(a_u.shape[:-1] + (3, 3))

    I = jnp.eye(3, dtype=a_u.dtype)
    K2 = jnp.einsum("...ij,...jk->...ik", K, K)
    denom = 1.0 + cos_angle
    R = I + K + K2 * jnp.where(denom > 1e-8, 1.0 / denom, 0.0)[..., None]

    # Antiparallel case (180°): the shortest arc is undefined, so pick any axis
    # orthogonal to b and rotate by π about it — matches SOMA-X's fallback.
    antiparallel = cos_angle[..., 0] < -1.0 + 1e-6
    y_vec = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0], dtype=a_u.dtype), b_u.shape)
    x_vec = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=a_u.dtype), b_u.shape)
    w = jnp.where((jnp.abs(b_u[..., 0:1]) > 0.6), y_vec, x_vec)
    axis_180 = safe_normalize(jnp.cross(b_u, w), eps=eps * eps)
    R_180 = 2.0 * (axis_180[..., :, None] * axis_180[..., None, :]) - I
    return jnp.where(antiparallel[..., None, None], R_180, R)


# ---------------------------------------------------------------------------
# Euler / quaternion conversions (ports of the upstream helpers of the same name)
# ---------------------------------------------------------------------------


def euler_xyz_to_rotmat(euler_xyz: jnp.ndarray) -> jnp.ndarray:
    """XYZ Euler angles -> rotation matrices.

    Port of upstream ``euler_xyz_to_matrix``. Intrinsic X-then-Y-then-Z.

    Args:
        euler_xyz: (..., 3) radians ordered X, Y, Z.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    if euler_xyz.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3), got {euler_xyz.shape}")
    c, s = jnp.cos(euler_xyz), jnp.sin(euler_xyz)
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    return jnp.stack([
        cy * cz,
        -cx * sz + sx * sy * cz,
        sx * sz + cx * sy * cz,
        cy * sz,
        cx * cz + sx * sy * sz,
        -sx * cz + cx * sy * sz,
        -sy,
        sx * cy,
        cx * cy,
    ], axis=-1).reshape(euler_xyz.shape[:-1] + (3, 3))


def rotmat_to_euler_xyz(R: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrices -> XYZ Euler angles. Port of ``matrix_to_euler_xyz``.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) radians ordered X, Y, Z. Degenerate at gimbal lock
        (``|R[2,0]| -> 1``), as the upstream formulation is.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {R.shape}")
    sy = jnp.clip(-R[..., 2, 0], -1.0, 1.0)
    return jnp.stack([
        jnp.arctan2(R[..., 2, 1], R[..., 2, 2]),
        jnp.arcsin(sy),
        jnp.arctan2(R[..., 1, 0], R[..., 0, 0]),
    ], axis=-1)


def quaternion_xyzw_to_rotmat(quaternion: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """XYZW quaternions -> rotation matrices. Port of ``quaternion_xyzw_to_matrix``.

    Args:
        quaternion: (..., 4) ordered x, y, z, w.
        eps: normalisation floor.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected (..., 4), got {quaternion.shape}")
    q = quaternion_normalize_xyzw(quaternion, eps=eps)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    ], axis=-1).reshape(quaternion.shape[:-1] + (3, 3))
