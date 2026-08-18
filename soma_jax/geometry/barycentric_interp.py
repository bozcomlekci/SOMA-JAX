"""Barycentric interpolation for topology transfer in SOMA-JAX.

Used to transfer vertex positions from one mesh topology to another
by computing barycentric coordinates within tetrahedra formed from
surface triangles (plus a normal-offset 4th vertex).

Upstream: ``soma/geometry/barycentric_interp.py``
    Port of ``fabricate_tet``, ``barycentric_interpolation`` and the deformed-
    mesh path of ``BarycentricInterpolator.forward``. Target vertices are
    embedded in tetrahedra (source triangle + normal-offset 4th point), and all
    four coordinates are interpolated so the out-of-plane offset survives the
    transfer. ``compute_barycentric_coords`` is the SOMA-JAX equivalent of
    upstream's ``compute_correspondence`` (nearest-face search + tet embedding);
    it uses trimesh rather than upstream's search and is not bit-comparable.
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp



def fabricate_tet(p0, p1, p2, normal_scale: str = "area"):
    """Fourth tetrahedron point for a triangle — port of upstream ``fabricate_tet``.

    ``"area"`` (upstream's default) offsets by the **raw** cross product, so the
    height scales with triangle area; ``"edge"`` offsets by a unit normal scaled
    by the mean edge length. Works on NumPy or JAX arrays.

    Args:
        p0, p1, p2: (..., 3) triangle corners.
        normal_scale: ``"area"`` or ``"edge"``.

    Returns:
        (..., 3) the fabricated point ``p3``.
    """
    xp = jnp if isinstance(p0, jnp.ndarray) else np
    n = xp.cross(p1 - p0, p2 - p0)
    if normal_scale == "edge":
        edge = (
            xp.linalg.norm(p1 - p0, axis=-1, keepdims=True)
            + xp.linalg.norm(p2 - p1, axis=-1, keepdims=True)
            + xp.linalg.norm(p0 - p2, axis=-1, keepdims=True)
        ) / 3.0
        n_norm = xp.linalg.norm(n, axis=-1, keepdims=True)
        n = xp.where(n_norm > 1e-12, n / xp.maximum(n_norm, 1e-12) * edge, n)
    elif normal_scale != "area":
        raise ValueError(f"Unsupported normal_scale: {normal_scale}")
    return p0 + n


def _build_tetrahedra(
    vertices: np.ndarray, faces: np.ndarray, normal_scale: str = "area"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pseudo-tetrahedra from surface triangles.

    Each triangle (v0, v1, v2) becomes a tetrahedron by adding a 4th vertex
    offset along the face normal.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle face indices.

    Returns:
        Tuple of:
        - tet_verts: (F, 4, 3) tetrahedra vertices
        - normals: (F, 3) face normals
        - scale: (F,) average edge length per face (used as offset magnitude)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    scale = (
        np.linalg.norm(e1, axis=-1)
        + np.linalg.norm(e2, axis=-1)
        + np.linalg.norm(v2 - v1, axis=-1)
    ) / 3.0

    # Upstream anchors the fabricated point at p0 with the RAW cross product
    # ("area" scale). Anchoring at the centroid, or normalising the normal,
    # yields a different tetrahedron and therefore different barycentric
    # coordinates — which would silently disagree with any asset whose
    # coordinates were produced by SOMA-X.
    v3 = fabricate_tet(v0, v1, v2, normal_scale)
    normals = normals / (norms + 1e-12)

    tet_verts = np.stack([v0, v1, v2, v3], axis=1)  # (F, 4, 3)
    return tet_verts, normals, scale


def _point_in_tet_bary(point: np.ndarray, tet: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates of a point within a tetrahedron.

    Args:
        point: (3,) query point.
        tet: (4, 3) tetrahedron vertices.

    Returns:
        (4,) barycentric coordinates (may be outside [0,1] for exterior points).
    """
    T = tet[1:] - tet[0]            # (3, 3)
    b = point - tet[0]              # (3,)
    try:
        coords = np.linalg.solve(T.T, b)
    except np.linalg.LinAlgError:
        coords = np.zeros(3)
    bary = np.concatenate([[1.0 - coords.sum()], coords])
    return bary


def compute_barycentric_coords(
    query_points: np.ndarray,
    src_vertices: np.ndarray,
    src_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute barycentric face assignments for query points.

    For each query point, finds the nearest face on the source mesh and
    computes its barycentric coordinates in the associated tetrahedron.

    Args:
        query_points: (N, 3) target mesh vertices to embed in source topology.
        src_vertices: (V_src, 3) source mesh vertices.
        src_faces: (F_src, 3) source mesh face indices.

    Returns:
        Tuple of:
        - face_ids: (N,) face index for each query point.
        - bary_coords: (N, 4) barycentric coordinates in the tetrahedron.
    """
    try:
        import trimesh
        src_mesh = trimesh.Trimesh(vertices=src_vertices, faces=src_faces, process=False)
        _, _, face_ids = trimesh.proximity.closest_point(src_mesh, query_points)
    except ImportError:
        # Fallback: brute-force nearest triangle centroid
        centroids = src_vertices[src_faces].mean(axis=1)  # (F, 3)
        dists = np.sum((query_points[:, None] - centroids[None]) ** 2, axis=-1)
        face_ids = np.argmin(dists, axis=-1)

    tet_verts, _, _ = _build_tetrahedra(src_vertices, src_faces)

    N = query_points.shape[0]
    bary_coords = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        bary_coords[i] = _point_in_tet_bary(query_points[i], tet_verts[face_ids[i]])

    return face_ids.astype(np.int32), bary_coords.astype(np.float32)


def barycentric_interpolate(
    src_verts: jnp.ndarray,
    src_faces: jnp.ndarray,
    face_ids: jnp.ndarray,
    bary_coords: jnp.ndarray,
    normal_scale: str = "area",
) -> jnp.ndarray:
    """Transfer vertex positions from source to target topology.

    Port of upstream ``BarycentricInterpolator.forward`` +
    ``barycentric_interpolation``. Each target vertex was embedded, offline, in
    a **tetrahedron** fabricated from a source triangle plus a fourth point
    offset along the face normal. Interpolating all four coordinates is what
    carries the target vertex's **out-of-plane offset** through the
    deformation; using only the triangle's three coordinates would project
    every transferred vertex onto the source surface.

    The fourth point is rebuilt from the *deformed* source vertices on every
    call, exactly as upstream does, so the offset follows the surface.

    Args:
        src_verts: (..., V_src, 3) source vertex positions (batched OK).
        src_faces: (F_src, 3) source face indices.
        face_ids: (N,) source face index for each target vertex.
        bary_coords: (N, 4) tetrahedral coordinates, or (N, 3) for a plain
            surface-triangle embedding.
        normal_scale: ``"area"`` (upstream default) or ``"edge"`` — must match
            whatever produced ``bary_coords``.

    Returns:
        (..., N, 3) interpolated positions on the target topology.
    """
    tri = src_faces[face_ids]                        # (N, 3)
    p0 = src_verts[..., tri[:, 0], :]
    p1 = src_verts[..., tri[:, 1], :]
    p2 = src_verts[..., tri[:, 2], :]

    if bary_coords.shape[-1] == 3:
        b = bary_coords / (bary_coords.sum(axis=-1, keepdims=True) + 1e-8)
        return (b[..., 0:1] * p0 + b[..., 1:2] * p1 + b[..., 2:3] * p2)

    if bary_coords.shape[-1] != 4:
        raise ValueError(
            f"bary_coords must have 3 or 4 columns, got {bary_coords.shape[-1]}.")

    p3 = fabricate_tet(p0, p1, p2, normal_scale)
    b = bary_coords
    return (b[..., 0:1] * p0 + b[..., 1:2] * p1
            + b[..., 2:3] * p2 + b[..., 3:4] * p3)
