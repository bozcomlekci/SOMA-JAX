"""RBF interpolator for SOMA-JAX — JAX port of SOMA-X's
``soma.geometry.interpolate.RadialBasisFunction``.

Used by ``SkeletonTransfer`` to regress joint positions from arbitrary vertex
deformations. The system matrix is precomputed once (LU-factored) so each
``interpolate(...)`` call is just a back-substitution + one matmul.

Mirrors SOMA-X kernel registry (linear, cubic, quintic, thin-plate-spline,
gaussian, multiquadric, inverse multiquadric, inverse quadratic) and the
optional polynomial augmentation.

Upstream: ``soma/geometry/interpolate.py``
    Faithful port of that code. RadialBasisFunction used by the skeleton transfer's joint regressors.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import numpy as np


def _pairwise_dist(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """(Na, D) vs (Nb, D) → (Na, Nb) Euclidean distances."""
    diff = A[:, None, :] - B[None, :, :]
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1))


def _tps(r, eps=1e-10):
    return (r * r) * jnp.log(r + eps)


def _gaussian(r, eps=0.1):
    return jnp.exp(-((r / eps) ** 2))


def _multiquadric(r, eps=0.1):
    return jnp.sqrt(1.0 + (r / eps) ** 2)


def _inverse_multiquadric(r, eps=0.1):
    return 1.0 / jnp.sqrt(1.0 + (r / eps) ** 2)


def _inverse_quadratic(r, eps=0.1):
    return 1.0 / (1.0 + (r / eps) ** 2)


def _linear(r, eps=1e-10):
    return r


def _cubic(r, eps=1e-10):
    return r ** 3


def _quintic(r, eps=1e-10):
    return r ** 5


_KERNELS = {
    "thin_plate_spline": _tps,
    "gaussian": _gaussian,
    "multiquadric": _multiquadric,
    "inverse_multiquadric": _inverse_multiquadric,
    "inverse_quadratic": _inverse_quadratic,
    "linear": _linear,
    "cubic": _cubic,
    "quintic": _quintic,
}


class RadialBasisFunction:
    """RBF interpolator with precomputed LU factorization.

    Given (N, D) source control points + an optional polynomial term, builds
    the system matrix A and stores its LU factors. Calling ``interpolate(
    target_positions, query_points)`` then solves A·coeffs = b for each
    target deformation and evaluates Φ(query, source)·coeffs.
    """

    KERNELS = _KERNELS

    def __init__(
        self,
        source_control_points: jnp.ndarray,
        kernel: str = "thin_plate_spline",
        kernel_params: Optional[Dict[str, Any]] = None,
        include_polynomial: bool = True,
    ):
        if kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Available: {list(_KERNELS.keys())}")
        scp = jnp.asarray(source_control_points)
        if scp.ndim != 2:
            raise ValueError("source_control_points must be (N, D)")

        self.source_control_points = scp
        self.n_control, self.dim = int(scp.shape[0]), int(scp.shape[1])
        self.dtype = scp.dtype
        self.kernel_name = kernel
        self.kernel_params = kernel_params or {}
        self.include_polynomial = bool(include_polynomial)
        self._rbf_func = _KERNELS[kernel]
        self._precompute_system_matrix()

    def _rbf(self, r):
        return self._rbf_func(r, **self.kernel_params)

    def _precompute_system_matrix(self):
        scp = self.source_control_points
        N, D = self.n_control, self.dim
        K = self._rbf(_pairwise_dist(scp, scp)).astype(self.dtype)
        # Tiny diagonal jitter for conditioning (matches SOMA-X).
        eps = 1e-8 if self.dtype in (jnp.float32, jnp.float64) else 1e-4
        K = K + jnp.eye(N, dtype=self.dtype) * eps

        if self.include_polynomial:
            ones = jnp.ones((N, 1), dtype=self.dtype)
            P = jnp.concatenate([ones, scp], axis=1)          # (N, D+1)
            Z = jnp.zeros((D + 1, D + 1), dtype=self.dtype)
            top = jnp.concatenate([K, P], axis=1)             # (N, N+D+1)
            bot = jnp.concatenate([P.T, Z], axis=1)            # (D+1, N+D+1)
            A = jnp.concatenate([top, bot], axis=0)           # (N+D+1)²
        else:
            A = K
        self.A = A
        # JAX has no in-place LU "factor + solve" split; we keep A and rely on
        # jax.scipy.linalg.lu_factor/lu_solve under jit. For tiny N (face/eye
        # support is small) this is fast enough.
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A)

    def _lu_solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """LU back-solve preserving the legacy (N, BD)-reshape batched path."""
        if b.ndim == 1:
            return jax.scipy.linalg.lu_solve((self._lu, self._piv), b)
        if b.ndim == 2:
            return jax.scipy.linalg.lu_solve((self._lu, self._piv), b)
        # (B, N, D) → solve N×(BD) → reshape back.
        B, N, D = b.shape
        b2 = jnp.transpose(b, (1, 0, 2)).reshape(N, B * D)
        x2 = jax.scipy.linalg.lu_solve((self._lu, self._piv), b2)
        return jnp.transpose(x2.reshape(N, B, D), (1, 0, 2))

    def get_basis_weights(self, query_point: jnp.ndarray) -> jnp.ndarray:
        """Linear weights w such that interpolated_pos = Σ wᵢ · sourceᵢ.

        Args:
            query_point: (D,) point to interpolate to.

        Returns:
            (N,) basis weights for the source control points.
        """
        q = jnp.asarray(query_point)
        if q.ndim == 2:
            q = q.reshape(-1)
        dists = self._rbf_func(
            jnp.linalg.norm(self.source_control_points - q[None, :], axis=1),
            **self.kernel_params,
        ).astype(self.dtype)
        if self.include_polynomial:
            ones = jnp.ones((1,), dtype=self.dtype)
            rhs = jnp.concatenate([dists, ones, q], axis=0)
        else:
            rhs = dists
        w_full = self._lu_solve(rhs)
        return w_full[: self.n_control]

    def interpolate(
        self,
        target_control_positions: jnp.ndarray,
        query_points: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the RBF at ``query_points`` given deformed source positions.

        Args:
            target_control_positions: (N, D) or (B, N, D) new positions for the
                source control points (= deformed mesh vertices at fit-time).
            query_points: (M, D) where to interpolate.

        Returns:
            (M, D) or (B, M, D) interpolated positions.
        """
        N, D = self.n_control, self.dim
        single = False
        if target_control_positions.ndim == 2:
            target_control_positions = target_control_positions[None, ...]
            single = True
        if target_control_positions.shape[1:] != (N, D):
            raise ValueError(
                f"target_control_positions must be (N,D) or (B,N,D), got "
                f"{target_control_positions.shape}")
        if query_points.ndim != 2 or query_points.shape[1] != D:
            raise ValueError(f"query_points must be (M, {D})")
        B = target_control_positions.shape[0]
        if self.include_polynomial:
            zeros_tail = jnp.zeros((B, D + 1, D), dtype=self.dtype)
            b = jnp.concatenate([target_control_positions, zeros_tail], axis=1)
        else:
            b = target_control_positions
        coeffs = self._lu_solve(b)
        Phi = self._rbf(_pairwise_dist(query_points, self.source_control_points)).astype(self.dtype)
        Phi_b = jnp.broadcast_to(Phi[None], (B,) + Phi.shape)
        rbf_contrib = Phi_b @ coeffs[:, :N, :]
        if not self.include_polynomial:
            return rbf_contrib[0] if single else rbf_contrib
        ones = jnp.ones((query_points.shape[0], 1), dtype=self.dtype)
        query_aug = jnp.concatenate([ones, query_points], axis=1)
        QA_b = jnp.broadcast_to(query_aug[None], (B,) + query_aug.shape)
        affine_contrib = QA_b @ coeffs[:, N:, :]
        out = rbf_contrib + affine_contrib
        return out[0] if single else out
