"""Optional Warp+JAX hybrid: a 3×3 Kabsch rotation extractor.

The vectorized JAX ``SkeletonTransfer`` (``skeleton_transfer.py``) builds all
per-joint Kabsch covariances as batched masked matmuls — work XLA does well —
then extracts rotations with ``jnp.linalg.svd`` over ``(N, 3, 3)``. That batched
3×3 SVD is XLA's weak spot and dominates at very large batch (see
``benchmarks/README.md``).

This module provides a drop-in replacement for the covariance→rotation step
that runs a hand-written **Warp** ``wp.svd3`` kernel on-device, called from
inside the JAX graph via ``warp.jax_experimental.jax_kernel`` (XLA FFI — no
host round-trip). JAX still builds the covariances and does FK + LBS; only the
SVD crosses into Warp.

What this kernel computes is plain SVD Procrustes, i.e.
``rotation_from_covariance(H, method="kabsch")``:

    R = U @ diag(1, 1, sign(det(U Vᵀ))) @ Vᵀ ,   H = U Σ Vᵀ
    R = I                                        when ‖H‖_F < eps·100  (degenerate)

**It is not ``method="auto"``.** ``auto`` — the ``SkeletonTransfer`` default,
and what upstream SOMA-X runs — is Newton–Schulz on a gauge-regularized
covariance, which on *ill-conditioned* covariances lands on a different (valid,
but not Procrustes-optimal) rotation. On well-conditioned covariances the two
agree; on rank-deficient ones they do not. Upstream ships a dedicated ``auto``
Warp kernel (``soma/geometry/align_vectors_warp.py``,
``_create_newton_schulz_auto_kernel``); this module does not port it, so the
hybrid pipeline is a *fast approximation* of the XLA-SVD one rather than the
same algorithm. Measured end-to-end on the benchmark rig the two posed meshes
agree to 210 µm max / 0.94 µm mean (``benchmarks/verify_fairness.py``).

Warp is an OPTIONAL dependency. Import this module only when the hybrid path is
requested; ``soma_jax`` core never imports it. ``is_available()`` reports
whether ``import warp`` succeeded.

Upstream: ``soma/geometry/align_vectors_warp.py``
    Diverges from that code. Optional Warp svd3 kernel. Implements plain SVD Procrustes (method='kabsch'), NOT upstream's default 'auto'.
"""
from __future__ import annotations
import numpy as np

_WARP_OK = False
_IMPORT_ERROR: Exception | None = None
try:
    import warp as wp
    from warp.jax_experimental import jax_kernel
    _WARP_OK = True
except Exception as e:  # pragma: no cover - depends on optional install
    _IMPORT_ERROR = e


# Degeneracy threshold — must match rotation_from_covariance (eps=1e-8 -> 1e-6).
_DEGEN_THRESH = 1e-6


if _WARP_OK:

    @wp.kernel
    def _kabsch_kernel(
        H: wp.array(dtype=wp.mat33),
        R: wp.array(dtype=wp.mat33),
    ):
        """Per-covariance Kabsch rotation via Warp's built-in 3×3 SVD.

        Mirrors ``soma_jax.geometry.transforms.rotation_from_covariance`` with
        ``method="kabsch"``: proper-rotation reflection fix + identity fallback
        on a near-zero covariance. See the module docstring for how this
        differs from ``method="auto"``.
        """
        tid = wp.tid()
        Hi = H[tid]

        # Frobenius norm of H; degenerate covariance -> identity rotation.
        fro = wp.sqrt(
            Hi[0, 0] * Hi[0, 0] + Hi[0, 1] * Hi[0, 1] + Hi[0, 2] * Hi[0, 2]
            + Hi[1, 0] * Hi[1, 0] + Hi[1, 1] * Hi[1, 1] + Hi[1, 2] * Hi[1, 2]
            + Hi[2, 0] * Hi[2, 0] + Hi[2, 1] * Hi[2, 1] + Hi[2, 2] * Hi[2, 2]
        )
        if fro < float(_DEGEN_THRESH):
            R[tid] = wp.identity(n=3, dtype=wp.float32)
            return

        U = wp.mat33(0.0)
        sigma = wp.vec3(0.0)
        V = wp.mat33(0.0)
        # H = U diag(sigma) Vᵀ  (same convention as numpy U, S, Vh with Vh = Vᵀ).
        wp.svd3(Hi, U, sigma, V)

        Vt = wp.transpose(V)
        # det_sign = sign(det(U Vᵀ)); fold into the last column of the
        # diagonal correction so R is a proper rotation (det +1).
        d = wp.determinant(U @ Vt)
        s = 1.0
        if d < 0.0:
            s = -1.0
        Dcorr = wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, s,
        )
        R[tid] = (U @ Dcorr) @ Vt

    _jax_kabsch = jax_kernel(_kabsch_kernel)


def is_available() -> bool:
    """True when Warp imported and the hybrid kernel is usable."""
    return _WARP_OK


def import_error() -> Exception | None:
    """The exception raised at import time, if Warp is unavailable."""
    return _IMPORT_ERROR


def kabsch_rotation_warp(H):
    """Covariance → rotation via the Warp ``svd3`` kernel (on-device, jit-able).

    Equivalent to ``rotation_from_covariance(H, method="kabsch")``. It stands in
    for ``method="auto"`` in the hybrid pipeline, but is *not* the same
    algorithm there — see the module docstring.

    Args:
        H: (..., 3, 3) JAX array of Kabsch covariances.

    Returns:
        (..., 3, 3) JAX array of proper rotations.
    """
    if not _WARP_OK:
        raise RuntimeError(f"Warp unavailable: {_IMPORT_ERROR!r}")
    import jax.numpy as jnp
    batch_shape = H.shape[:-2]
    H_flat = H.reshape((-1, 3, 3))
    (R_flat,) = _jax_kabsch(H_flat)
    return R_flat.reshape(batch_shape + (3, 3))
