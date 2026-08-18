"""Laplacian mesh editing for SOMA-JAX.

Used to blend inner-face geometry after topology transfer (MHR, SMPL, Garment models).
The Laplacian solve fills in the interior vertices while fixing boundary vertices.

Upstream: ``soma/geometry/laplacian.py``
    :class:`LaplacianMesh` is the faithful port — it re-solves the SOMA
    inner-face vertices after topology transfer (order 1, hard constraints),
    and is what :mod:`soma_jax.identity_model` calls. ``cotangent_weights`` and
    ``build_cotangent_laplacian_sparse`` port upstream's helpers of the same
    name. :func:`laplacian_solve` is **SOMA-JAX-only** and solves a *different*
    problem (zero Laplacian energy rather than upstream's reference-coordinate
    preservation) — see its docstring.
"""
from __future__ import annotations
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
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO ``(rows, cols, values)`` of the cotangent edge weights.

    Port of upstream ``cotangent_weights``: for each triangle, the weight of an
    edge is ``cot`` of the angle opposite it, ``cot(t) = dot(a, b) / |cross(a, b)|``,
    emitted symmetrically for both orientations. Each interior edge therefore
    accumulates the contribution of both incident triangles.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0        # edge opposite vertex 0 / 1 / 2

    def _cot(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(np.cross(a, b), axis=-1) + 1e-8)

    cot0, cot1, cot2 = _cot(e1, e2), _cot(e2, e0), _cot(e0, e1)
    rows = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 2],
                           faces[:, 0], faces[:, 0], faces[:, 1]])
    cols = np.concatenate([faces[:, 2], faces[:, 1], faces[:, 0],
                           faces[:, 2], faces[:, 1], faces[:, 0]])
    vals = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    return rows, cols, vals


def build_cotangent_laplacian_sparse(
    vertices: np.ndarray, faces: np.ndarray
) -> sp.csr_matrix:
    """``L = D - W`` for the symmetrized cotangent weights ``W`` (upstream order)."""
    n = vertices.shape[0]
    rows, cols, vals = cotangent_weights(np.asarray(vertices, np.float64),
                                         np.asarray(faces, np.int64))
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T) * 0.5                                   # upstream: (W + W.t()) / 2
    return sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W


class LaplacianMesh:
    """Laplacian mesh editing with hard constraints — upstream-equivalent.

    Faithful port of SOMA-X's ``LaplacianMesh`` for the configuration SOMA
    actually uses (``order=1``, ``constraint_mode="hard"``): the *anchor*
    vertices keep the values handed to :meth:`solve`, and the remaining
    (``mask_anchors == False``) vertices are re-solved so the mesh keeps the
    **reference mesh's Laplacian coordinates**::

        L_FF x_U  =  L_U @ V_ref  -  L_FG x_G

    Note the right-hand side is *not* zero: upstream preserves the reference
    differential coordinates rather than minimising Laplacian energy, so the
    filled region reproduces the template's local shape instead of collapsing
    to a membrane.

    **Where the work happens.** Assembly and factorisation run once, on the
    host, from constant topology — the cotangent Laplacian is built with a
    Python/SciPy sparse pass because it needs ragged scatter-add over faces.
    Everything :meth:`solve` does per call is **pure JAX**: a gather, a
    segment-sum, a Cholesky solve and a scatter. It is ``jit``-able,
    ``vmap``-able over the batch axis and differentiable w.r.t. ``vertices``.
    ``|U|`` is small (691 vertices on the SOMA rig — eye bags + mouth bag), so
    the factor is a dense ``(|U|, |U|)`` Cholesky rather than a sparse LU.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mask_anchors: np.ndarray,
        order: int = 1,
        jitter: float = 0.0,
    ) -> None:
        """
        Args:
            vertices: (V, 3) reference vertices — sets the cotangent weights and
                the target Laplacian coordinates.
            faces: (F, 3) triangles.
            mask_anchors: (V,) bool — True keeps the input value, False is solved.
            order: Laplacian power. Only 1 is supported (SOMA's setting).
            jitter: optional diagonal regularisation on the SPD system.
        """
        if int(order) != 1:
            raise NotImplementedError(
                f"order={order}: only the order-1 Laplacian SOMA-X uses is ported.")
        vertices = np.asarray(vertices, np.float64)
        faces = np.asarray(faces, np.int64)
        mask_anchors = np.asarray(mask_anchors, bool)
        n = vertices.shape[0]

        self.vid_unknown = np.where(~mask_anchors)[0]
        self.vid_constrained = np.where(mask_anchors)[0]
        if self.vid_unknown.size == 0:
            raise ValueError("mask_anchors leaves no free vertices to solve for.")

        L = build_cotangent_laplacian_sparse(vertices, faces)
        L_U = L[self.vid_unknown]                              # (|U|, n)

        # Target Laplacian coordinates of the reference mesh.
        btilde = np.asarray(L_U @ vertices)                    # (|U|, 3)

        L_FF = np.asarray(L_U[:, self.vid_unknown].todense())  # (|U|, |U|) — small
        # The constrained block stays sparse: only a few entries per row are
        # non-zero, so a (|U|, |G|) dense matrix would waste ~46 MB.
        L_FG = L_U[:, self.vid_constrained].tocoo()

        # Cotangent L is negative semi-definite at odd order; negate for SPD.
        sign = -1.0
        A = sign * L_FF
        if jitter > 0:
            A = A + jitter * np.eye(A.shape[0])

        self._sign = sign
        self._chol = jnp.asarray(np.linalg.cholesky(A), jnp.float32)
        self._btilde = jnp.asarray(btilde, jnp.float32)
        self._unknown = jnp.asarray(self.vid_unknown, jnp.int32)
        # Global column ids so solve() can gather straight from the input mesh.
        self._fg_rows = jnp.asarray(L_FG.row, jnp.int32)
        self._fg_cols = jnp.asarray(self.vid_constrained[L_FG.col], jnp.int32)
        self._fg_vals = jnp.asarray(L_FG.data, jnp.float32)
        self._n_unknown = int(self.vid_unknown.size)
        self.num_vertices = n

    def solve(self, vertices: jnp.ndarray) -> jnp.ndarray:
        """Re-solve the free vertices, keeping the anchors as given.

        Args:
            vertices: (V, 3) or (B, V, 3) mesh whose anchor vertices carry the
                values to honour.

        Returns:
            Same shape, with the free vertices replaced by the solve.
        """
        single = vertices.ndim == 2
        v = vertices[None] if single else vertices
        if v.shape[-2] != self.num_vertices:
            raise ValueError(
                f"Expected {self.num_vertices} vertices, got {v.shape[-2]}.")

        # rhs = btilde - L_FG @ x_G, as a segment-sum over the sparse entries.
        contrib = self._fg_vals[None, :, None] * v[:, self._fg_cols, :]   # (B, nnz, 3)
        fg = jax.ops.segment_sum(
            contrib.transpose(1, 0, 2), self._fg_rows,
            num_segments=self._n_unknown,
        ).transpose(1, 0, 2)                                              # (B, |U|, 3)
        rhs = self._btilde[None] - fg                                     # (B, |U|, 3)

        # cho_solve wants (m, k); fold the batch into the column axis and do a
        # single triangular solve for all frames at once (as upstream does).
        B = rhs.shape[0]
        rhs_2d = rhs.transpose(1, 0, 2).reshape(self._n_unknown, B * 3)
        x_2d = jax.scipy.linalg.cho_solve((self._chol, True), self._sign * rhs_2d)
        x = x_2d.reshape(self._n_unknown, B, 3).transpose(1, 0, 2)        # (B, |U|, 3)

        out = v.at[:, self._unknown, :].set(x)
        return out[0] if single else out
