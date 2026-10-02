"""SO(3)/SE(3) rotation and transformation utilities for SOMA-JAX.

All functions operate on JAX arrays and are compatible with jit/vmap/grad.

Upstream: ``soma/geometry/transforms.py``
    Faithful port of every public function, under upstream's names (listed at
    the end of this module) as well as SOMA-JAX's (``axis_angle_to_rotmat``,
    ``rotmat_to_axis_angle``, ``se3_inverse``, ...). `align_vectors` matches
    upstream to <=1e-6 for all three methods, including rank-deficient
    covariances. Two upstream defects are not reproduced:
    ``matrix_to_rotvec`` doubles the rotation vector below 1e-3 rad, and
    ``rotvec_to_matrix`` (unused upstream) does not return rotations.
    ``kabsch_points`` is a SOMA-JAX extra (upstream's ``kabsch`` takes a
    covariance).
"""
from __future__ import annotations
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

# Rotation-estimation constants — values mirror
# ``soma.geometry.transforms`` so ``align_vectors(method="auto")`` reproduces
# the reference's regularized Newton-Schulz path exactly.
NEWTON_SCHULZ_ITERS = 30
AUTO_ROTATION_PRIOR_STRENGTH = 0.05
AUTO_ROTATION_RANK_THRESHOLD = 2e-2
AUTO_ROTATION_DEGENERATE_THRESHOLD = 1e-6


# Jitted, as `jnp.linalg.norm` is, so eager callers get the same rounding.
@partial(jax.jit, static_argnames=("axis", "keepdims"))
def _vector_norm(x: jnp.ndarray, axis=-1, keepdims: bool = False) -> jnp.ndarray:
    """L2 norm with a zero gradient at the origin, as ``torch.linalg.vector_norm``.

    ``jnp.linalg.norm`` differentiates ``||x||`` at ``x = 0`` as NaN (0/0), and
    the NaN leaks through any ``where`` that masks the value out; torch defines
    that gradient as zero, so upstream's formulas backpropagate finite values
    through degenerate inputs. The value is the same as ``jnp.linalg.norm``.
    """
    sq = jnp.sum(x * x, axis=axis, keepdims=keepdims)
    nonzero = sq > 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, sq, 1.0)), 0.0)


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


def kabsch(H: jnp.ndarray) -> jnp.ndarray:
    """Rotation from a covariance via the Kabsch algorithm (SVD).

    Upstream ``soma.geometry.transforms.kabsch``: ``H = U S Vh`` and
    ``R = U diag(1, 1, ±1) Vh`` with the sign fixing ``det(R) = +1``.

    Args:
        H: (..., 3, 3) covariance, e.g. from :func:`compute_covariance`.

    Returns:
        (..., 3, 3) rotation matrices with ``det(R) = 1``.
    """
    return rotation_from_covariance(H, method="kabsch")


def kabsch_points(src: jnp.ndarray, tgt: jnp.ndarray,
                  weights: jnp.ndarray | None = None) -> jnp.ndarray:
    """Kabsch algorithm on point sets: optimal rotation R such that R @ src ≈ tgt.

    SOMA-JAX extra (centred, optionally weighted); upstream's :func:`kabsch`
    takes a precomputed covariance instead.

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
        # torch-style norms: a collinear triangle (zero normal) or a zero first
        # point backpropagates zeros, not NaN, as upstream's do.
        len_n_src = _vector_norm(n_src, keepdims=True)
        len_n_dst = _vector_norm(n_dst, keepdims=True)
        scale_src = _vector_norm(p0, keepdims=True) / (len_n_src + eps)
        scale_dst = _vector_norm(q0, keepdims=True) / (len_n_dst + eps)
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

    R = newton_schulz(H, num_iters=NEWTON_SCHULZ_ITERS, eps=eps)
    use_kabsch = ~rotation_matrices_are_valid(R)
    if method == "auto":
        # SOMA-X v0.2.2, `align_vectors_warp._create_newton_schulz_auto_kernel`:
        # a reflected covariance needs Kabsch's singular-vector correction.
        # Flipping a fixed Newton-Schulz output column is not the nearest SO(3)
        # projection and can select a substantially different rotation. The sign
        # is read off the REGULARIZED H, exactly where upstream reads it (after
        # the degenerate-rank prior is added to the diagonal, not before).
        use_kabsch = use_kabsch | (jnp.linalg.det(H) < 0)

    # `jnp.where` evaluates the SVD for EVERY covariance, and the SVD's
    # gradient is infinite wherever singular values coincide — so any slot the
    # Kabsch branch does not serve, and any all-zero covariance, gets a
    # placeholder with well-separated singular values. Its Kabsch rotation is
    # the identity, so forward values are unchanged; the placeholder only keeps
    # masked-out gradients finite. (Upstream computes the SVD only where it is
    # used, so torch never sees the placeholder at all.)
    H_norm = jnp.linalg.norm(H, axis=(-2, -1), keepdims=True)
    degenerate = H_norm < eps * 100
    placeholder = jnp.broadcast_to(jnp.diag(jnp.asarray([3.0, 2.0, 1.0], H.dtype)), H.shape)
    svd_input = jnp.where(use_kabsch[..., None, None] & ~degenerate, H, placeholder)
    return jnp.where(use_kabsch[..., None, None], _kabsch_svd(svd_input), R)


# ----------------------------------------------------------------------------
# Quaternion helpers (xyzw layout — matches SOMA-X's `quaternion_order: xyzw`)
# ----------------------------------------------------------------------------
def matrix_to_quaternion_xyzw(R: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Convert rotation matrices to XYZW unit quaternions.

    Port of upstream ``soma.geometry.transforms.matrix_to_quaternion_xyzw``
    (v0.3.3): every row ``4 q_i q`` is formed from the matrix entries, the row
    of the largest squared component is selected — its ``|q_i| >= 1/2``, so it
    normalizes without a small divisor — and the result is standardized to
    unit norm and non-negative ``w``. Recovering all components from one row
    preserves their relative signs near 180 degrees, and the gradient stays
    finite at identity and at half turns.

    Args:
        R: (..., 3, 3) rotation matrices.
        eps: Small constant used when normalizing quaternions.

    Returns:
        (..., 4) quaternions ordered as x, y, z, w with non-negative w.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {R.shape}")
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    squared_components = jnp.stack((
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
        1.0 + m00 + m11 + m22,
    ), axis=-1)
    xx, yy, zz, ww = (squared_components[..., i] for i in range(4))
    xy, xz, yz = m01 + m10, m02 + m20, m12 + m21
    xw, yw, zw = m21 - m12, m02 - m20, m10 - m01
    candidates = jnp.stack((
        jnp.stack((xx, xy, xz, xw), axis=-1),
        jnp.stack((xy, yy, yz, yw), axis=-1),
        jnp.stack((xz, yz, zz, zw), axis=-1),
        jnp.stack((xw, yw, zw, ww), axis=-1),
    ), axis=-2)
    largest = jnp.argmax(squared_components, axis=-1)
    quaternion = jnp.take_along_axis(candidates, largest[..., None, None], axis=-2)[..., 0, :]
    return quaternion_standardize_xyzw(quaternion, eps=eps)


def matrix_to_quaternion_xyzw_stable(R: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Convert rotation matrices to XYZW quaternions with finite branch gradients.

    Upstream keeps this entry point for callers that need gradients through the
    conversion; it is :func:`matrix_to_quaternion_xyzw`, whose largest-component
    branch already avoids zero-valued square roots.
    """
    return matrix_to_quaternion_xyzw(R, eps=eps)


def quaternion_normalize_xyzw(quaternion: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """L2-normalize an xyzw quaternion, flooring the norm at ``eps``.

    Upstream ``soma.geometry.transforms.quaternion_normalize_xyzw``:
    ``quaternion / ||quaternion||.clamp_min(eps)``. (An earlier ``quaternion / (||quaternion|| + eps)`` form shrank
    every unit quaternion by ~1e-12 — negligible once, but the RTS smoother
    normalizes on every frame — and was no better behaved: both forms have a
    finite gradient everywhere except exactly ``quaternion = 0``.)
    """
    return quaternion / jnp.maximum(jnp.linalg.norm(quaternion, axis=-1, keepdims=True), eps)


def quaternion_standardize_xyzw(quaternion: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize an xyzw quaternion and pick the non-negative-``w`` representative.

    Upstream: ``soma.geometry.transforms.quaternion_standardize_xyzw`` (v0.3.1).
    ``quaternion`` and ``-quaternion`` are the same rotation, so a canonical sign is what makes two
    quaternions comparable at all.
    """
    quaternion = quaternion_normalize_xyzw(quaternion, eps=eps)
    return jnp.where(quaternion[..., 3:] < 0.0, -quaternion, quaternion)


def quaternion_log_xyzw(quaternion: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Map xyzw unit quaternions to rotation vectors along the shortest arc.

    Upstream: ``soma.geometry.transforms.quaternion_log_xyzw`` (v0.3.1).
    Standardizing to ``w >= 0`` first is what makes the arc the short one.

    Args:
        quaternion: (..., 4) xyzw quaternions.
        eps: numerical floor for the small-angle branch.

    Returns:
        (..., 3) rotation vectors (axis * angle).
    """
    quaternion = quaternion_standardize_xyzw(quaternion, eps=eps)
    vec = quaternion[..., :3]
    w = jnp.clip(quaternion[..., 3], -1.0, 1.0)
    vec_norm = _vector_norm(vec)   # torch's zero gradient at the identity
    angle = 2.0 * jnp.arctan2(vec_norm, w)
    # jnp.where evaluates both branches, so both divisors are floored to keep
    # the unused branch (and its gradient) finite.
    small = vec_norm < eps
    factor = jnp.where(
        small,
        2.0 / jnp.maximum(w, eps),
        angle / jnp.maximum(vec_norm, eps),
    )
    return vec * factor[..., None]


def quaternion_exp_xyzw(rotvec: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Map rotation vectors to xyzw unit quaternions.

    Upstream: ``soma.geometry.transforms.quaternion_exp_xyzw`` (v0.3.1). The
    small-angle branch uses the same Taylor series as upstream
    (``1/2 - θ²/48 + θ⁴/3840``) so ``θ → 0`` stays finite and accurate.

    Args:
        rotvec: (..., 3) rotation vectors (axis * angle).
        eps: numerical floor for the large-angle divisor.

    Returns:
        (..., 4) xyzw unit quaternions with ``w >= 0``.
    """
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {rotvec.shape}")
    theta = _vector_norm(rotvec)   # torch's zero gradient at the zero vector
    half_theta = 0.5 * theta
    theta2 = theta * theta
    theta4 = theta2 * theta2
    small = theta < 1e-6
    imag_scale = jnp.where(
        small,
        0.5 - theta2 / 48.0 + theta4 / 3840.0,
        jnp.sin(half_theta) / jnp.maximum(theta, eps),
    )
    q = jnp.concatenate(
        [rotvec * imag_scale[..., None], jnp.cos(half_theta)[..., None]], axis=-1
    )
    return quaternion_standardize_xyzw(q, eps=eps)


def project_rotations_to_so3(rotations: jnp.ndarray) -> jnp.ndarray:
    """Project matrices to the nearest proper rotations.

    Upstream: ``soma.geometry.transforms.project_rotations_to_so3`` (v0.3.1).
    Upstream branches on ``torch.any(det < 0)`` as a host-side fast path; that
    would force a device sync under ``jit``, so the sign correction is applied
    unconditionally here — the ``det >= 0`` case multiplies by ``+1`` and is a
    no-op, so the result is identical either way.

    Args:
        rotations: (..., 3, 3) matrices, not necessarily orthonormal.

    Returns:
        (..., 3, 3) nearest matrices in SO(3).
    """
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {rotations.shape}")
    u, _, vh = jnp.linalg.svd(rotations, full_matrices=False)
    det = jnp.linalg.det(u @ vh)
    sign = jnp.where(det < 0, -1.0, 1.0)
    u = u.at[..., :, -1].multiply(sign[..., None])
    return u @ vh


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


def quaternion_conjugate_xyzw(quaternion: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of an xyzw quaternion (xyz negated). For unit quaternions this
    equals the inverse / rotational opposite."""
    return jnp.concatenate([-quaternion[..., :3], quaternion[..., 3:4]], axis=-1)


def quaternion_half_angle_xyzw(quaternion: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Principal half-angle (square-root) quaternion for xyzw rotations.

    Port of ``soma.geometry.transforms.quaternion_half_angle_xyzw``: pick the
    representation with non-negative ``w`` (``quaternion`` and ``-quaternion`` are the same
    rotation), then normalize ``[v, w + 1]``.
    """
    quaternion = quaternion_normalize_xyzw(quaternion, eps=eps)
    quaternion = jnp.where(quaternion[..., 3:] < 0.0, -quaternion, quaternion)
    return quaternion_normalize_xyzw(
        jnp.concatenate([quaternion[..., :3], quaternion[..., 3:] + 1.0], axis=-1), eps=eps,
    )


def quaternion_twist_angle_xyzw(
    quaternion: jnp.ndarray,
    axis_ids=0,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Signed twist angles (radians) around local axes, from xyzw quaternions.

    Faithful port of ``soma.geometry.transforms.quaternion_twist_angle_xyzw``:
    the projection is taken on the **half-angle** quaternion and scaled by 4,
    i.e. ``4·atan2(v_half[axis], w_half)``. Doing it on the raw quaternion
    (``2·atan2(v[axis], w)``) agrees only for small rotations and drifts badly
    as the twist approaches ±180°, which is exactly the regime the 1-DOF
    procedural joints operate in.

    Args:
        quaternion: (..., 4) quaternions ordered as x, y, z, w.
        axis_ids: scalar axis id or array broadcastable to
            ``quaternion.shape[:-1]``; ``0=x``, ``1=y``, ``2=z``.
        eps: normalisation floor.

    Returns:
        ``quaternion.shape[:-1]`` twist angles in radians.
    """
    q_half = quaternion_half_angle_xyzw(quaternion, eps=eps)
    if not isinstance(axis_ids, jax.core.Tracer):
        ids_host = np.asarray(axis_ids)
        if np.any((ids_host < 0) | (ids_host > 2)):
            raise ValueError("axis_ids must contain only 0, 1, or 2")
        if ids_host.ndim == 0:
            return 4.0 * jnp.arctan2(q_half[..., int(ids_host)], q_half[..., 3])
    ids = jnp.asarray(axis_ids, dtype=jnp.int32)
    try:
        ids = jnp.broadcast_to(ids, q_half.shape[:-1])
    except ValueError as e:
        raise ValueError(
            "axis_ids must be broadcastable to quaternion.shape[:-1], "
            f"got {tuple(ids.shape)} for {tuple(q_half.shape[:-1])}") from e
    twist_imag = jnp.take_along_axis(q_half[..., :3], ids[..., None], axis=-1)[..., 0]
    return 4.0 * jnp.arctan2(twist_imag, q_half[..., 3])


def single_axis_rotation_matrices(
    angles: jnp.ndarray,
    axis: int,
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
        angles: (...,) rotation angles in radians.
        axis: 0 / 1 / 2 for X / Y / Z.
        axis_signs: broadcastable per-element sign (±1) applied to ``angles``.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    angle = angles * jnp.asarray(axis_signs, dtype=angles.dtype)
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    if axis == 0:
        return jnp.stack([
            jnp.stack([o, z, z], -1),
            jnp.stack([z, c, -s], -1),
            jnp.stack([z, s,  c], -1),
        ], -2)
    if axis == 1:
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
    H: jnp.ndarray,
    num_iters: int = NEWTON_SCHULZ_ITERS,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Newton-Schulz orthogonalization: iteratively refine A toward the nearest SO(3).

    Faithful port of ``soma.geometry.transforms.newton_schulz``: the input is
    scaled by its **infinity norm** (max absolute row sum) — which is what
    guarantees convergence of the iteration — and the result gets a
    determinant-sign correction on the last column so the output is a proper
    rotation rather than a roto-reflection.

    Args:
        H: (..., 3, 3) matrix (typically a Kabsch covariance).
        num_iters: number of refinement iterations (SOMA-X uses 30).
        eps: numerical floor for the scaling.

    Returns:
        (..., 3, 3) orthogonalized rotation matrix with det = +1.
    """
    return _newton_schulz(H, num_iters, eps)


def _newton_schulz_step(X, _):
    # X_{k+1} = X_k (3I - X_kᵀ X_k) / 2
    return X @ (3.0 * jnp.eye(3, dtype=X.dtype) - jnp.swapaxes(X, -2, -1) @ X) * 0.5, None


# Jitted so eager callers (the pose-inversion refit calls this per joint and
# per iteration) reuse one executable per shape; an eager scan over a fresh
# closure would be retraced and recompiled on every call.
@partial(jax.jit, static_argnums=(1,))
def _newton_schulz(A, num_iter, eps):
    max_row_sum = jnp.max(jnp.sum(jnp.abs(A), axis=-1), axis=-1)[..., None, None]
    X = A / (max_row_sum + eps)
    X, _ = jax.lax.scan(_newton_schulz_step, X, None, length=num_iter)
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


def _cofactor3(M: jnp.ndarray) -> jnp.ndarray:
    """Cofactor matrix of (..., 3, 3) ``M``: ``d det(M) = sum(cofactor(M) * dM)``."""
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    rows = [
        jnp.stack([e * i - f * h, f * g - d * i, d * h - e * g], axis=-1),
        jnp.stack([c * h - b * i, a * i - c * g, b * g - a * h], axis=-1),
        jnp.stack([b * f - c * e, c * d - a * f, a * e - b * d], axis=-1),
    ]
    return jnp.stack(rows, axis=-2)


@jax.custom_jvp
def det3(M: jnp.ndarray) -> jnp.ndarray:
    """``jnp.linalg.det`` for (..., 3, 3), with a gradient that exists everywhere.

    The forward value is ``jnp.linalg.det``'s. Its built-in JVP solves against
    ``M`` and turns NaN on singular matrices — e.g. the all-zero covariance of
    a joint with no support — and the NaN survives the ``jnp.where`` that masks
    that slot out. torch's ``det`` backward stays finite there. The analytic
    derivative of a 3x3 determinant is its cofactor matrix, which is finite for
    every input, so that is the JVP used here.
    """
    return jnp.linalg.det(M)


@det3.defjvp
def _det3_jvp(primals, tangents):
    (M,), (dM,) = primals, tangents
    return jnp.linalg.det(M), jnp.sum(_cofactor3(M) * dM, axis=(-2, -1))


@jax.custom_jvp
def _volume_score(abs_det: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """``abs_det / scale ** 3`` — upstream's expression, with a torch-like JVP.

    ``scale`` bottoms out at ``eps`` (1e-8) for an all-zero covariance. JAX's
    division rule differentiates ``x / y`` as ``x * y**-2`` and ``(1e-24)**-2``
    overflows float32, so the masked-out zero slot turns the whole gradient
    NaN; torch's rule divides twice (``(x / y) / y``) and stays at 0. The JVP
    below keeps that finite structure.
    """
    return abs_det / scale ** 3


@_volume_score.defjvp
def _volume_score_jvp(primals, tangents):
    abs_det, scale = primals
    d_abs_det, d_scale = tangents
    value = abs_det / scale ** 3
    return value, d_abs_det / scale ** 3 - 3.0 * (value / scale) * d_scale


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
    volume_score = _volume_score(jnp.abs(det3(H)), prior_scale)
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


# ---------------------------------------------------------------------------
# Upstream names (``soma.geometry.transforms``)
# ---------------------------------------------------------------------------

#: Rotation-extraction methods accepted by :func:`align_vectors`.
AlignmentMethod = Literal["kabsch", "newton-schulz", "auto"]

SE3_from_Rt = se3_from_rt
SE3_inverse = se3_inverse
euler_xyz_to_matrix = euler_xyz_to_rotmat
matrix_to_euler_xyz = rotmat_to_euler_xyz
quaternion_xyzw_to_matrix = quaternion_xyzw_to_rotmat


def matrix_to_rotvec(R: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 3) rotation vectors (axis * angle).

    Upstream ``matrix_to_rotvec``, robust for small angles and near pi. Its
    small-angle branch (theta <= 1e-3) returns twice the rotation vector;
    this is :func:`rotmat_to_axis_angle`, which returns the true one there
    (see its notes).
    """
    return rotmat_to_axis_angle(R, eps=eps)


def rotvec_to_matrix(rotvec: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """(..., 3) rotation vectors -> (..., 3, 3) rotation matrices, robust near zero.

    Upstream ``rotvec_to_matrix`` (unused upstream) builds ``K`` from the
    *unit* axis but applies the ``sin(theta)/theta`` and
    ``(1 - cos(theta))/theta^2`` coefficients meant for the unnormalized
    vector, so it returns non-rotations for every angle other than about 1
    radian (e.g. det 0.76 at pi/2). This returns the Rodrigues rotation its
    docstring describes, switching to the first-order ``I + [rotvec]_x``
    below ``1e-6`` as upstream switches to ``I + K``.
    """
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {rotvec.shape}")

    def skew(v):
        zero = jnp.zeros_like(v[..., 0])
        return jnp.stack([
            zero, -v[..., 2], v[..., 1],
            v[..., 2], zero, -v[..., 0],
            -v[..., 1], v[..., 0], zero,
        ], axis=-1).reshape(v.shape[:-1] + (3, 3))

    theta = _vector_norm(rotvec)   # torch's zero gradient at the zero vector
    K = skew(rotvec / jnp.where(theta < eps, eps, theta)[..., None])
    eye = jnp.eye(3, dtype=rotvec.dtype)
    R = (eye + jnp.sin(theta)[..., None, None] * K
         + (1.0 - jnp.cos(theta))[..., None, None] * (K @ K))
    small = theta < 1e-6
    return jnp.where(small[..., None, None], eye + skew(rotvec), R)


def _normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """``torch.nn.functional.normalize``: ``x / max(||x||, eps)`` (gradient-safe at 0)."""
    return x / jnp.maximum(_vector_norm(x, keepdims=True), eps)


def rotation_6d_to_matrix(d6: jnp.ndarray) -> jnp.ndarray:
    """Convert the 6D rotation representation of Zhou et al. to rotation matrices.

    Upstream ``rotation_6d_to_matrix``: Gram-Schmidt with
    ``torch.nn.functional.normalize`` (``x / max(||x||, 1e-12)``); the basis
    vectors are the **rows** of the result. :func:`rotation_6d_to_rotmat` is
    the same map with a smooth ``sqrt(||x||^2 + 1e-12)`` normalizer.

    Args:
        d6: (..., 6) 6D rotation representation.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = jnp.cross(b1, b2)
    return jnp.stack((b1, b2, b3), axis=-2)
