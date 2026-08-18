"""Chamfer distance between two point sets, in pure JAX.

Used as a loss for pose inversion when point correspondences are unknown.

Upstream: none — SOMA-JAX-only.
    **Not a port.** Upstream's `ChamferLoss` is a one-way source-point to
    closest-point-on-*triangle* query. These functions are a JAX-only
    point-cloud loss: bidirectional mean-squared distance between vertex
    sets. Different direction, different target geometry, different value.
"""
from __future__ import annotations
import jax
from typing import Optional

import jax.numpy as jnp


def chamfer_distance(
    x: jnp.ndarray,
    y: jnp.ndarray,
    bidirectional: bool = True,
    weights_x: jnp.ndarray | None = None,
    weights_y: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Chamfer distance between two point sets.

    Args:
        x: (N, 3) first point set.
        y: (M, 3) second point set.
        bidirectional: if True, sum distances in both directions.
        weights_x: optional (N,) per-point weights for x.
        weights_y: optional (M,) per-point weights for y.

    Returns:
        Scalar Chamfer distance.
    """
    # Pairwise squared distances: (N, M)
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = jnp.sum(diff * diff, axis=-1)

    # x -> nearest y
    nn_x_to_y = jnp.min(sq_dist, axis=1)
    if weights_x is not None:
        loss_x = jnp.sum(weights_x * nn_x_to_y) / (jnp.sum(weights_x) + 1e-8)
    else:
        loss_x = jnp.mean(nn_x_to_y)

    if not bidirectional:
        return loss_x

    # y -> nearest x
    nn_y_to_x = jnp.min(sq_dist, axis=0)
    if weights_y is not None:
        loss_y = jnp.sum(weights_y * nn_y_to_x) / (jnp.sum(weights_y) + 1e-8)
    else:
        loss_y = jnp.mean(nn_y_to_x)

    return loss_x + loss_y


def chamfer_distance_batched(
    x: jnp.ndarray,
    y: jnp.ndarray,
    bidirectional: bool = True,
) -> jnp.ndarray:
    """Batched Chamfer distance.

    Args:
        x: (B, N, 3) batched point sets.
        y: (B, M, 3) batched point sets.
        bidirectional: if True, sum both directions.

    Returns:
        (B,) per-batch Chamfer distances.
    """
    return jax.vmap(lambda a, b: chamfer_distance(a, b, bidirectional))(x, y)


def nearest_neighbor_indices(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """For each point in x, find the index of its nearest neighbor in y.

    Args:
        x: (N, 3) query points.
        y: (M, 3) reference points.

    Returns:
        (N,) indices into y.
    """
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = jnp.sum(diff * diff, axis=-1)
    return jnp.argmin(sq_dist, axis=1)


# ---------------------------------------------------------------------------
# Upstream-equivalent one-way point-to-mesh Chamfer (soma.geometry.chamfer_warp)
# ---------------------------------------------------------------------------


def _closest_point_on_triangle(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
                               c: jnp.ndarray) -> jnp.ndarray:
    """Closest point to ``p`` on triangle ``(a, b, c)`` — Ericson's region test.

    Branch-free so it stays ``vmap``/``jit`` friendly.

    Args:
        p: (..., 3) query points.
        a, b, c: (..., 3) triangle corners.

    Returns:
        (..., 3) closest points on the triangle.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = jnp.sum(ab * ap, -1)
    d2 = jnp.sum(ac * ap, -1)
    bp = p - b
    d3 = jnp.sum(ab * bp, -1)
    d4 = jnp.sum(ac * bp, -1)
    cp = p - c
    d5 = jnp.sum(ab * cp, -1)
    d6 = jnp.sum(ac * cp, -1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = 1.0 / jnp.maximum(va + vb + vc, 1e-20)

    # Interior (barycentric) solution, then override per Voronoi region.
    v_i, w_i = vb * denom, vc * denom
    out = a + ab * v_i[..., None] + ac * w_i[..., None]

    e = 1e-20
    out = jnp.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
                    a + ab * (d1 / jnp.maximum(d1 - d3, e))[..., None], out)   # edge AB
    out = jnp.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
                    a + ac * (d2 / jnp.maximum(d2 - d6, e))[..., None], out)   # edge AC
    out = jnp.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None],
                    b + (c - b) * ((d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), e))[..., None],
                    out)                                                       # edge BC
    out = jnp.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)                # vertex A
    out = jnp.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)               # vertex B
    out = jnp.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)               # vertex C
    return out


def chamfer_distance_to_mesh(
    src_points: jnp.ndarray,
    target_verts: jnp.ndarray,
    target_faces: Optional[jnp.ndarray] = None,
    face_chunk: int = 4096,
    max_distance: float = 1e6,
) -> jnp.ndarray:
    """One-way Chamfer from points to a target **triangle mesh**.

    Port of upstream ``soma.geometry.chamfer_warp.ChamferLoss``: for every
    source point, find the closest point on the target *surface* (not the
    closest target vertex) and average the squared distances. Upstream uses a
    Warp BVH query; this evaluates exact point-to-triangle distances, chunked
    over faces to bound memory.

    Passing ``target_faces=None`` reproduces upstream's degenerate-triangle
    fallback, i.e. a point-to-point query against the target vertices.

    Args:
        src_points: (B, N, 3) or (N, 3) query points.
        target_verts: (B, M, 3) or (M, 3) target vertices.
        target_faces: (F, 3) target topology, or None for a point cloud.
        face_chunk: faces evaluated per step; trades memory for steps.
        max_distance: queries farther than this contribute 0, matching
            upstream's Warp ``mesh_query_point`` search radius (its kernel
            simply records nothing when no hit is found within it).

    Returns:
        Scalar for unbatched input; ``(B,)`` otherwise. Note upstream returns a
        scalar for a batched ``B=1`` input as well; this keeps the batch axis so
        the shape is a function of the input rank alone.
    """
    src_b = src_points.ndim == 3
    tgt_b = target_verts.ndim == 3
    S = src_points if src_b else src_points[None]
    T = target_verts if tgt_b else target_verts[None]
    if S.shape[0] != T.shape[0]:
        if T.shape[0] == 1:
            T = jnp.broadcast_to(T, (S.shape[0],) + T.shape[1:])
        elif S.shape[0] == 1:
            S = jnp.broadcast_to(S, (T.shape[0],) + S.shape[1:])
        else:
            raise ValueError(f"Batch mismatch: src {S.shape[0]} vs target {T.shape[0]}")

    if target_faces is None:
        d2 = jnp.sum((S[:, :, None, :] - T[:, None, :, :]) ** 2, -1)
        best = jnp.min(d2, axis=-1)
    else:
        F = jnp.asarray(target_faces)
        best = jnp.full(S.shape[:2], jnp.inf)
        for start in range(0, F.shape[0], face_chunk):
            f = F[start:start + face_chunk]
            a, b, c = T[:, f[:, 0]], T[:, f[:, 1]], T[:, f[:, 2]]   # (B, Fc, 3)
            q = _closest_point_on_triangle(
                S[:, :, None, :], a[:, None], b[:, None], c[:, None])
            best = jnp.minimum(best, jnp.min(jnp.sum((S[:, :, None, :] - q) ** 2, -1), -1))

    # Upstream's BVH query ignores anything beyond its search radius.
    best = jnp.where(best > max_distance ** 2, 0.0, best)

    out = jnp.mean(best, axis=-1)
    return out[0] if not (src_b or tgt_b) else out
