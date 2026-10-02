"""Laplacian mesh editing for SOMA-JAX.

Used to blend inner-face geometry after topology transfer (MHR, SMPL, Garment models).
The Laplacian solve fills in the interior vertices while fixing boundary vertices.

Upstream: ``soma/geometry/laplacian.py``
    Faithful port of the module: ``cotangent_weights``,
    ``build_cotangent_laplacian``, ``build_uniform_laplacian``,
    ``power_laplacian`` (SciPy sparse matrices where upstream returns torch
    sparse tensors) and :class:`LaplacianMesh` with hard or soft constraints —
    what the identity backends call to re-solve the SOMA inner-face vertices
    after topology transfer (order 1, hard constraints).
    :func:`laplacian_solve` is **SOMA-JAX-only** and solves a *different*
    problem (zero Laplacian energy rather than upstream's reference-coordinate
    preservation) — see its docstring.
"""
from __future__ import annotations
from typing import Literal
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp
import jax.scipy.linalg


def _build_cotangent_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> sp.csr_matrix:
    """Build the cotangent-weighted Laplacian matrix.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle face indices.

    Returns:
        (V, V) sparse cotangent Laplacian (positive semi-definite).
    """
    V = vertices.shape[0]
    rows, cols, data = [], [], []

    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3

        vi = vertices[faces[:, i]]
        vj = vertices[faces[:, j]]
        vk = vertices[faces[:, k]]

        # Cotangent at vertex k (opposite to edge i-j)
        u = vi - vk
        v = vj - vk
        cross = np.cross(u, v)
        cot = np.sum(u * v, axis=-1) / (np.linalg.norm(cross, axis=-1) + 1e-12)
        cot = cot * 0.5

        # Off-diagonal entries for edge (i, j)
        fi = faces[:, i]
        fj = faces[:, j]
        rows.extend(fi.tolist())
        cols.extend(fj.tolist())
        data.extend((-cot).tolist())
        rows.extend(fj.tolist())
        cols.extend(fi.tolist())
        data.extend((-cot).tolist())

    L = sp.csr_matrix((data, (rows, cols)), shape=(V, V))
    # Diagonal: sum of negative off-diagonal
    L = L - sp.diags(np.array(L.sum(axis=1)).flatten())
    return L


def _build_uniform_laplacian(
    faces: np.ndarray,
    n_vertices: int,
) -> sp.csr_matrix:
    """Build uniform (combinatorial) Laplacian.

    Args:
        faces: (F, 3) triangle face indices.
        n_vertices: total number of vertices.

    Returns:
        (V, V) sparse uniform Laplacian.
    """
    rows, cols = [], []
    for i in range(3):
        j = (i + 1) % 3
        rows.extend(faces[:, i].tolist())
        cols.extend(faces[:, j].tolist())
        rows.extend(faces[:, j].tolist())
        cols.extend(faces[:, i].tolist())

    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices)
    )
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / (degrees + 1e-8))
    return sp.eye(n_vertices) - D_inv @ adj


def laplacian_solve(
    vertices: np.ndarray,
    faces: np.ndarray,
    constrained_ids: np.ndarray,
    constrained_values: np.ndarray,
    use_cotangent: bool = True,
) -> np.ndarray:
    """Minimum-Laplacian-energy solve — **not** upstream's formulation.

    .. warning::

        This is **not** what SOMA-X does and is not used by the SOMA pipeline.
        It solves ``L_ff x = -L_fc x_c``, i.e. drives the free region's
        Laplacian coordinates to **zero** — a membrane that flattens whatever
        shape was there. Upstream's :class:`LaplacianMesh` instead solves
        ``L_FF x = L_U @ V_ref - L_FG x_G``, preserving the *reference mesh's*
        Laplacian coordinates, so the filled region keeps the template's local
        shape. Use :class:`LaplacianMesh` for anything that must match SOMA-X.

        Kept as a standalone utility for callers that genuinely want the
        membrane solution (it is a different, valid deformation operator).

    Fixes the constrained vertices (boundary) and solves for the free vertices
    to minimize Laplacian energy.

    Args:
        vertices: (V, 3) initial vertex positions (used for cotangent weights).
        faces: (F, 3) triangle face indices.
        constrained_ids: (C,) indices of vertices with fixed positions.
        constrained_values: (C, 3) target positions for constrained vertices.
        use_cotangent: if True, use cotangent weights; else uniform.

    Returns:
        (V, 3) solved vertex positions.
    """
    V = vertices.shape[0]
    all_ids = np.arange(V)
    mask = np.ones(V, dtype=bool)
    mask[constrained_ids] = False
    free_ids = all_ids[mask]

    if len(free_ids) == 0:
        result = vertices.copy()
        result[constrained_ids] = constrained_values
        return result

    if use_cotangent:
        L = _build_cotangent_laplacian(vertices, faces)
    else:
        L = _build_uniform_laplacian(faces, V)

    # Partition: L_ff @ x_f = -L_fc @ x_c
    L_ff = L[free_ids][:, free_ids]
    L_fc = L[free_ids][:, constrained_ids]

    rhs = -L_fc @ constrained_values  # (|free|, 3)

    # Solve with sparse direct solver
    try:
        factor = spla.splu(L_ff.tocsc())
        x_free = factor.solve(rhs)
    except Exception:
        # Fallback: iterative solver
        x_free = np.zeros((len(free_ids), 3))
        for d in range(3):
            x_free[:, d], _ = spla.cg(L_ff, rhs[:, d], x0=vertices[free_ids, d])

    result = vertices.copy()
    result[constrained_ids] = constrained_values
    result[free_ids] = x_free
    return result


# ---------------------------------------------------------------------------
# LaplacianMesh — faithful port of SOMA-X's ``soma.geometry.laplacian.LaplacianMesh``
# ---------------------------------------------------------------------------


def cotangent_weights(
    V: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO ``(rows, cols, values)`` of the cotangent edge weights.

    Port of upstream ``cotangent_weights``: for each triangle, the weight of an
    edge is ``cot`` of the angle opposite it, ``cot(t) = dot(a, b) / |cross(a, b)|``,
    emitted symmetrically for both orientations. Each interior edge therefore
    accumulates the contribution of both incident triangles.
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0        # edge opposite vertex 0 / 1 / 2

    def _cot(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(np.cross(a, b), axis=-1) + 1e-8)

    cot0, cot1, cot2 = _cot(e1, e2), _cot(e2, e0), _cot(e0, e1)
    rows = np.concatenate([F[:, 1], F[:, 2], F[:, 2],
                           F[:, 0], F[:, 0], F[:, 1]])
    cols = np.concatenate([F[:, 2], F[:, 1], F[:, 0],
                           F[:, 2], F[:, 1], F[:, 0]])
    vals = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    return rows, cols, vals


def _weights_to_laplacian(W: sp.spmatrix) -> sp.csr_matrix:
    """Graph Laplacian ``L = D - W`` of a sparse non-negative weight matrix."""
    W = sp.csr_matrix(W)
    return (sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()


def build_cotangent_laplacian(V, F) -> sp.csr_matrix:
    """Cotangent Laplacian ``L = D - W`` (sparse CSR), upstream ``build_cotangent_laplacian``.

    ``W`` holds the symmetrized cotangent edge weights of
    :func:`cotangent_weights`. Computed in the vertices' dtype on the host;
    upstream returns a torch sparse CSR tensor.
    """
    V = np.asarray(V)
    n = V.shape[0]
    rows, cols, vals = cotangent_weights(V, np.asarray(F, np.int64))
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T) / 2                                     # upstream: (W + W.t()) / 2
    return _weights_to_laplacian(W)


def build_cotangent_laplacian_sparse(vertices, faces) -> sp.csr_matrix:
    """:func:`build_cotangent_laplacian` in float64 (SOMA-JAX's earlier name)."""
    return build_cotangent_laplacian(np.asarray(vertices, np.float64), faces)


def build_uniform_laplacian(F, n_verts: int, device=None, dtype=None) -> sp.csr_matrix:
    """Uniform (graph) Laplacian ``L = D - A`` with binary adjacency ``A``.

    Upstream ``build_uniform_laplacian``: geometry-independent, only mesh
    connectivity. ``device`` is accepted for signature compatibility;
    ``dtype`` defaults to float32 as upstream's does.
    """
    F = np.asarray(F, np.int64)
    dtype = np.float32 if dtype is None else dtype
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]],
                            F[:, [1, 0]], F[:, [2, 1]], F[:, [2, 0]]], axis=0)
    A = sp.coo_matrix((np.ones(len(edges), dtype), (edges[:, 0], edges[:, 1])),
                      shape=(n_verts, n_verts)).tocsr()
    A.data = np.minimum(A.data, 1.0).astype(dtype)        # binary adjacency
    return _weights_to_laplacian(A)


def power_laplacian(L: sp.spmatrix, order: int) -> sp.csr_matrix:
    """``L ** order`` for a higher-order Laplacian (upstream ``power_laplacian``)."""
    if order <= 1:
        return L
    result = L
    for _ in range(1, order):
        result = result @ L
    return sp.csr_matrix(result)


#: Upstream's ``LaplacianConstraintMode``.
LaplacianConstraintMode = Literal["hard", "soft"]
#: Upstream's ``LaplacianSolver``: the backend that solves the system.
LaplacianSolver = Literal["cholespy", "pytorch"]


class LaplacianMesh:
    """Laplacian mesh editing with hard or soft constraints — upstream's class.

    Faithful port of SOMA-X's ``LaplacianMesh``. With hard constraints (what
    SOMA uses: ``order=1``) the *anchor* vertices keep the values handed to
    :meth:`solve`, and the remaining (``mask_anchors == False``) vertices are
    re-solved so the mesh keeps the **reference mesh's Laplacian coordinates**::

        L_FF x_U  =  L_U @ V_ref  -  L_FG x_G

    The right-hand side is *not* zero: upstream preserves the reference
    differential coordinates rather than minimising Laplacian energy, so the
    filled region reproduces the template's local shape instead of collapsing
    to a membrane. With soft constraints every vertex is solved from
    ``(L^T L + w C^T C) x = L^T L V_ref + w C^T d``.

    **Where the work happens.** Assembly and factorisation run once, on the
    host, from constant topology. Everything :meth:`solve` does per call is
    **pure JAX**: a gather, a segment-sum (hard), a Cholesky solve and a
    scatter. It is ``jit``-able, ``vmap``-able over the batch axis and
    differentiable w.r.t. the input vertices. The hard system is a dense
    ``(|U|, |U|)`` Cholesky (691 unknowns on the SOMA rig — eye bags + mouth
    bag); the soft system factors the dense ``(V, V)`` matrix, as upstream's
    ``"cholespy"`` path does.

    Args:
        V: (n, 3) reference vertices — the cotangent weights and the target
            Laplacian coordinates.
        F: (F, 3) triangles.
        mask_anchors: (n,) bool — True for anchors, False for free vertices.
        order: Laplacian power (1 or 2).
        constraint_mode: ``"hard"`` or ``"soft"``.
        soft_weight: weight of the soft constraints.
        jitter: diagonal regularisation added to the factored system.
        solver: upstream's backend choice, ``"cholespy"`` or ``"pytorch"``;
            both solve the same system, here with one Cholesky factorisation.
    """

    def __init__(
        self,
        V,
        F,
        mask_anchors,
        order: int = 1,
        constraint_mode: str = "hard",
        soft_weight: float = 1e-5,
        jitter: float = 0.0,
        solver: str = "cholespy",
    ) -> None:
        if solver not in ("cholespy", "pytorch"):
            raise ValueError(f"solver must be 'cholespy' or 'pytorch', got '{solver}'")
        if constraint_mode not in ("hard", "soft"):
            raise ValueError("constraint_mode must be 'hard' or 'soft'")
        self.solver_backend = solver
        self.order = int(order)
        self.constraint_mode = constraint_mode
        self.soft_weight = float(soft_weight)
        self.jitter = float(jitter)
        vertices = np.asarray(V, np.float64)
        faces = np.asarray(F, np.int64)
        mask_anchors = np.asarray(mask_anchors, bool)
        n = vertices.shape[0]
        self.V = vertices
        self.F = faces
        self.num_vertices = n
        self.vid_unknown = np.where(~mask_anchors)[0]
        self.vid_constrained = np.where(mask_anchors)[0]

        L = build_cotangent_laplacian(vertices, faces)
        if self.order > 1:
            L = power_laplacian(L, self.order)
        self.L = L
        if constraint_mode == "hard":
            self._setup_hard_constraints()
        else:
            self._setup_soft_constraints()

    def _setup_hard_constraints(self) -> None:
        if self.vid_unknown.size == 0:
            raise ValueError("mask_anchors leaves no free vertices to solve for.")
        L_U = self.L[self.vid_unknown]                         # (|U|, n)
        btilde = np.asarray(L_U @ self.V)                      # reference coordinates
        L_FF = np.asarray(L_U[:, self.vid_unknown].todense())  # (|U|, |U|) — small
        # The constrained block stays sparse: only a few entries per row are
        # non-zero, so a dense (|U|, |G|) matrix would waste ~46 MB.
        L_FG = L_U[:, self.vid_constrained].tocoo()
        # The cotangent Laplacian is negative semi-definite at odd orders and
        # positive semi-definite at even ones; factor the SPD sign.
        sign = -1.0 if self.order % 2 == 1 else 1.0
        A = sign * L_FF
        if self.jitter > 0:
            A = A + self.jitter * np.eye(A.shape[0])
        self._hard_chol_sign = sign
        self._chol = jnp.asarray(np.linalg.cholesky(A), jnp.float32)
        self._btilde = jnp.asarray(btilde, jnp.float32)
        self._unknown = jnp.asarray(self.vid_unknown, jnp.int32)
        # Global column ids so solve() can gather straight from the input mesh.
        self._fg_rows = jnp.asarray(L_FG.row, jnp.int32)
        self._fg_cols = jnp.asarray(self.vid_constrained[L_FG.col], jnp.int32)
        self._fg_vals = jnp.asarray(L_FG.data, jnp.float32)
        self._n_unknown = int(self.vid_unknown.size)

    def _setup_soft_constraints(self) -> None:
        n = self.num_vertices
        L = sp.csr_matrix(self.L)
        K = (L.T @ L).toarray()
        if self.soft_weight > 0:
            K[self.vid_constrained, self.vid_constrained] += self.soft_weight
        if self.jitter > 0:
            K = K + self.jitter * np.eye(n)
        self._chol = jnp.asarray(np.linalg.cholesky(K), jnp.float32)
        self._core_rhs = jnp.asarray(L.T @ (L @ self.V), jnp.float32)      # (n, 3)
        self._constrained = jnp.asarray(self.vid_constrained, jnp.int32)

    def solve(self, Vdef: jnp.ndarray) -> jnp.ndarray:
        """Solve for the deformed mesh given constraint vertices.

        Args:
            Vdef: (n, 3) or (B, n, 3). Hard mode honours only the anchor
                values; soft mode pulls the solution toward them.

        Returns:
            Same shape: hard mode replaces the free vertices, soft mode returns
            the full solve.
        """
        single = Vdef.ndim == 2
        v = Vdef[None] if single else Vdef
        if v.shape[-2] != self.num_vertices:
            raise ValueError(
                f"Expected {self.num_vertices} vertices, got {v.shape[-2]}.")
        B = v.shape[0]

        if self.constraint_mode == "hard":
            # rhs = btilde - L_FG @ x_G, as a segment-sum over the sparse entries.
            contrib = self._fg_vals[None, :, None] * v[:, self._fg_cols, :]  # (B, nnz, 3)
            fg = jax.ops.segment_sum(
                contrib.transpose(1, 0, 2), self._fg_rows, num_segments=self._n_unknown,
            ).transpose(1, 0, 2)                                             # (B, |U|, 3)
            rhs = self._btilde[None] - fg
            # Fold the batch into the column axis: one triangular solve for all
            # frames at once, as upstream does.
            rhs_2d = rhs.transpose(1, 0, 2).reshape(self._n_unknown, B * 3)
            x_2d = jax.scipy.linalg.cho_solve((self._chol, True),
                                              self._hard_chol_sign * rhs_2d)
            x = x_2d.reshape(self._n_unknown, B, 3).transpose(1, 0, 2)
            out = v.at[:, self._unknown, :].set(x)
        else:
            n = self.num_vertices
            rhs = jnp.broadcast_to(self._core_rhs[None], (B, n, 3))
            if self.soft_weight > 0:
                rhs = rhs.at[:, self._constrained, :].add(
                    self.soft_weight * v[:, self._constrained, :])
            rhs_2d = rhs.transpose(1, 0, 2).reshape(n, B * 3)
            x_2d = jax.scipy.linalg.cho_solve((self._chol, True), rhs_2d)
            out = x_2d.reshape(n, B, 3).transpose(1, 0, 2)
        return out[0] if single else out
