"""Barycentric interpolation for topology transfer in SOMA-JAX.

Used to transfer vertex positions from one mesh topology to another
by computing barycentric coordinates within tetrahedra formed from
surface triangles (plus a normal-offset 4th vertex).

Upstream: ``soma/geometry/barycentric_interp.py``
    Port of ``fabricate_tet``, ``barycentric_interpolation`` and the deformed-
    mesh path of ``BarycentricInterpolator.forward``. Target vertices are
    embedded in tetrahedra (source triangle + normal-offset 4th point), and all
    four coordinates are interpolated so the out-of-plane offset survives the
    transfer. ``compute_barycentric_coords`` performs upstream's
    ``compute_correspondence`` (trimesh nearest face + tet embedding) and
    returns the ``(face_ids, bary)`` pair SOMA-JAX passes around;
    :class:`BarycentricInterpolator`, ``barycentric_interpolation`` and
    ``compute_barycentric_coords_3d`` keep upstream's object API on top.
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
    normal_scale: str = "area",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed query points in tetrahedra fabricated from their nearest source faces.

    Port of upstream ``BarycentricInterpolator.compute_correspondence``: the
    nearest face from ``trimesh.Trimesh(vertices, faces).nearest.on_surface``
    (upstream's call, default processing included), a fourth point fabricated
    off every face (:func:`fabricate_tet`), and a batched solve of each point's
    four tetrahedral coordinates.

    The arithmetic runs in the **inputs' dtype**, as upstream's NumPy does.
    Upstream always builds its interpolators from float32 tensors, so pass
    float32 arrays to reproduce its coordinates: on near-degenerate source
    triangles the coordinates are ill-conditioned, and a float64 solve lands
    measurably elsewhere (up to tenths of a millimetre after transfer).

    Args:
        query_points: (N, 3) target mesh vertices to embed in source topology.
        src_vertices: (V_src, 3) source mesh vertices.
        src_faces: (F_src, 3) source mesh face indices.
        normal_scale: ``"area"`` (upstream default) or ``"edge"``.

    Returns:
        Tuple of:
        - face_ids: (N,) face index for each query point.
        - bary_coords: (N, 4) tetrahedral coordinates.
    """
    V_src = np.asarray(src_vertices)
    F_src = np.asarray(src_faces).astype(np.int64)
    V_dst = np.asarray(query_points)
    try:
        import trimesh
        mesh_src = trimesh.Trimesh(vertices=V_src, faces=F_src)
        _, _, face_ids = mesh_src.nearest.on_surface(V_dst)
    except ImportError:
        # SOMA-JAX extra: brute-force nearest triangle centroid without trimesh.
        centroids = V_src[F_src].mean(axis=1)
        dists = np.sum((V_dst[:, None] - centroids[None]) ** 2, axis=-1)
        face_ids = np.argmin(dists, axis=-1)
    face_ids = np.asarray(face_ids).astype(np.int64)

    V_src_P3 = fabricate_tet(V_src[F_src[:, 0]], V_src[F_src[:, 1]], V_src[F_src[:, 2]],
                             normal_scale)
    V_src_tet = np.concatenate([V_src, V_src_P3], axis=0)
    F_src_tet = np.concatenate(
        [F_src, np.arange(F_src.shape[0])[:, None] + V_src.shape[0]], axis=1)
    tet = F_src_tet[face_ids]
    v0, v1, v2, v3 = (V_src_tet[tet[:, k]] for k in range(4))
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)
    rhs = V_dst - v0
    try:
        b123 = np.linalg.solve(T, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # SOMA-JAX extra: upstream's batched solve raises on a singular
        # tetrahedron; solve per point and give degenerate ones zeros instead.
        b123 = np.zeros_like(rhs)
        for i in range(rhs.shape[0]):
            try:
                b123[i] = np.linalg.solve(T[i], rhs[i])
            except np.linalg.LinAlgError:
                pass
    bary = np.concatenate([1.0 - b123.sum(axis=-1, keepdims=True), b123], axis=-1)
    return face_ids.astype(np.int32), bary


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


def compute_barycentric_coords_3d(p, v0, v1, v2, v3) -> np.ndarray:
    """3D barycentric coordinates of points ``p`` in tetrahedra ``(v0, v1, v2, v3)``.

    Upstream ``compute_barycentric_coords_3d`` (NumPy): solves
    ``[v1 - v0, v2 - v0, v3 - v0] b = p - v0`` and prepends ``1 - sum(b)``.

    Args:
        p, v0, v1, v2, v3: (..., 3) query points and tetrahedron corners.

    Returns:
        (..., 4) barycentric coordinates.
    """
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)
    b123 = np.linalg.solve(T, (p - v0)[..., None])[..., 0]
    return np.concatenate([1.0 - b123.sum(axis=-1, keepdims=True), b123], axis=-1)


def barycentric_interpolation(V_tet, F_tet, face_ids, bary_coords) -> jnp.ndarray:
    """Interpolate vertices with precomputed tetrahedral coordinates.

    Upstream ``barycentric_interpolation``.

    Args:
        V_tet: (B, V + F, 3) or (V + F, 3) tetrahedralized vertices — the mesh
            vertices followed by one fabricated point per face.
        F_tet: (F, 4) tetrahedron indices.
        face_ids: (N,) face of each target point.
        bary_coords: (N, 4) tetrahedral coordinates.

    Returns:
        (B, N, 3) or (N, 3) interpolated points.
    """
    V_tet = jnp.asarray(V_tet)
    has_batch = V_tet.ndim == 3
    if not has_batch:
        V_tet = V_tet[None]
    tet = jnp.asarray(F_tet)[jnp.asarray(face_ids)]
    bc = jnp.asarray(bary_coords)[None]
    result = (V_tet[:, tet[:, 0]] * bc[..., 0:1] + V_tet[:, tet[:, 1]] * bc[..., 1:2]
              + V_tet[:, tet[:, 2]] * bc[..., 2:3] + V_tet[:, tet[:, 3]] * bc[..., 3:4])
    return result if has_batch else result[0]


class BarycentricInterpolator:
    """Transfer deformations from a source mesh to target points.

    Upstream ``BarycentricInterpolator`` (a ``torch.nn.Module``) as a plain
    callable: each target vertex is embedded once in a tetrahedron built from
    its nearest source triangle plus a fabricated fourth point
    (:func:`fabricate_tet`), and calling the interpolator rebuilds those
    tetrahedra on deformed source vertices and re-applies the coordinates.

    Args:
        V_src: (V_src, 3) source vertices.
        F_src: (F_src, 3) source triangles.
        V_dst: (N, 3) target vertices.
        correspondence_path: optional ``.npz`` written by
            :meth:`save_correspondence` to load instead of computing.
        tet_normal_scale: ``"area"`` (upstream default) or ``"edge"`` height
            for the fabricated tetrahedra.

    Upstream checks for torch tensors; any array-like is accepted here, and the
    correspondence is computed in the inputs' dtype, as upstream's NumPy does.
    """

    def __init__(self, V_src, F_src, V_dst, correspondence_path=None,
                 tet_normal_scale: str = "area") -> None:
        if tet_normal_scale not in {"area", "edge"}:
            raise ValueError(f"Unsupported tet_normal_scale: {tet_normal_scale}")
        self.V_src = np.asarray(V_src)
        self.F_src = np.asarray(F_src)
        self.V_dst = np.asarray(V_dst)
        self.tet_normal_scale = tet_normal_scale
        self.face_ids = None
        self.bary_coords = None
        self.F_src_tet = None
        if correspondence_path is not None:
            self.load_correspondence(correspondence_path)
        else:
            self.compute_correspondence()

    @property
    def dtype(self):
        """The coordinates' dtype; upstream keeps its buffers in float32."""
        return jnp.float32

    def compute_correspondence(self) -> None:
        """Embed every target vertex in its nearest source triangle's tetrahedron."""
        face_ids, bary = compute_barycentric_coords(
            self.V_dst, self.V_src, self.F_src, normal_scale=self.tet_normal_scale)
        F_src = self.F_src.astype(np.int64)
        new_vert_indices = np.arange(F_src.shape[0])[:, None] + self.V_src.shape[0]
        self.face_ids = jnp.asarray(face_ids)
        self.bary_coords = jnp.asarray(bary)
        self.F_src_tet = jnp.asarray(np.concatenate([F_src, new_vert_indices], axis=1))

    def load_correspondence(self, path) -> None:
        """Load a correspondence saved by :meth:`save_correspondence`."""
        with np.load(path, allow_pickle=False) as correspondence:
            self.face_ids = jnp.asarray(correspondence["face_ids"])
            self.bary_coords = jnp.asarray(correspondence["bary_coords"], dtype=self.dtype)
            self.F_src_tet = jnp.asarray(correspondence["F_src_tet"])

    def save_correspondence(self, path) -> None:
        """Save the correspondence (``face_ids``, ``bary_coords``, ``F_src_tet``)."""
        np.savez(path, face_ids=np.asarray(self.face_ids),
                 bary_coords=np.asarray(self.bary_coords),
                 F_src_tet=np.asarray(self.F_src_tet))

    def __call__(self, V_src_deformed) -> jnp.ndarray:
        """Deformed target vertices, (B, N, 3) or (N, 3) like the input."""
        V = jnp.asarray(V_src_deformed)
        has_batch = V.ndim == 3
        if not has_batch:
            V = V[None]
        faces = jnp.asarray(self.F_src)
        P3 = fabricate_tet(V[:, faces[:, 0]], V[:, faces[:, 1]], V[:, faces[:, 2]],
                           self.tet_normal_scale)
        V_tet = jnp.concatenate([V, P3], axis=1)
        result = barycentric_interpolation(V_tet, self.F_src_tet, self.face_ids,
                                           self.bary_coords.astype(V.dtype))
        return result if has_batch else result[0]

    forward = __call__
