"""USD (UsdSkel) I/O for SOMA-JAX — faithful port of SOMA-X's ``soma.io`` USD half.

Reads and writes SOMA rigs, skinned meshes and skeletal animation as USD, so
SOMA-JAX output drops straight into DCC tools (Maya, Houdini, Omniverse) and
USD template rigs can be read back in.

Everything here needs the optional ``usd-core`` package::

    pip install usd-core

The NPZ half of upstream's ``io.py`` lives in :mod:`soma_jax.io`.

Typical export::

    from soma_jax.usd_io import export_soma_usd

    layer.prepare_identity(coeffs)                    # or use the returned binds
    export_soma_usd("anim.usda", layer, rotations, root_translation,
                    bind_transforms_world=bind_transforms, rest_shape=rest_verts)

Upstream: ``soma/io.py (USD half)``
    Faithful port of that code. UsdSkel rig/animation read+write. The LOD-discovery chain used to BUILD assets is not ported.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .units import Unit

logger = logging.getLogger(__name__)

DEFAULT_SKIN_MESH_NAME = "c_skin_mid"
_HIPS_IDX = 1

_USD_IMPORT_ERROR = (
    "USD I/O requires the 'usd-core' package. Install it with: pip install usd-core"
)


def _pxr():
    """Import pxr lazily so the rest of SOMA-JAX works without usd-core."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt
    except ImportError:
        raise ImportError(_USD_IMPORT_ERROR) from None
    return Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt


def _to_np(x) -> np.ndarray:
    """Accept jax arrays, torch tensors or numpy without importing either."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _to_f32(x) -> np.ndarray:
    return _to_np(x).astype(np.float32)


def _open_stage(usd_file_path):
    _, _, Usd, _, _, _ = _pxr()
    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")
    return stage


class UVPrimvarEntry(dict):
    """A UV set from :func:`load_usd_mesh`.

    Dict-like (``entry["coordinates"]``) with attribute access
    (``entry.coordinates``), matching upstream.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------


def list_usd_meshes(usd_file_path) -> list[str]:
    """Return the prim paths of every Mesh prim in a USD file."""
    _, _, _, UsdGeom, _, _ = _pxr()
    stage = _open_stage(usd_file_path)
    return [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]


def _is_uv_primvar(pv) -> bool:
    """Whether a primvar is a UV set — port of upstream ``_is_uv_primvar``.

    Both the value type *and* the interpolation must match: upstream accepts
    the scalar forms as well as the arrays, and only for vertex / varying /
    faceVarying interpolation. Accepting any array-typed float2 would pick up
    constant or uniform primvars that are not UV sets.
    """
    _, Sdf, _, _, _, _ = _pxr()
    if pv.GetAttr().GetTypeName() not in (
        Sdf.ValueTypeNames.TexCoord2fArray,
        Sdf.ValueTypeNames.TexCoord2f,
        Sdf.ValueTypeNames.Float2Array,
        Sdf.ValueTypeNames.Float2,
    ):
        return False
    return pv.GetInterpolation() in ("vertex", "varying", "faceVarying")


def load_usd_mesh(usd_file_path, mesh_name: str):
    """Load a mesh from a USD file.

    Args:
        usd_file_path: path to the USD file.
        mesh_name: prim path of the mesh (a leading ``/`` is added if missing).

    Returns:
        ``(vertices, face_vert_indices, face_vert_counts, uv_data)`` — vertices
        ``(V, 3)`` float32, flattened face-vertex indices, per-face vertex
        counts, and a ``{name: UVPrimvarEntry}`` mapping.
    """
    _, _, _, UsdGeom, _, _ = _pxr()
    stage = _open_stage(usd_file_path)

    if not mesh_name.startswith("/"):
        mesh_name = "/" + mesh_name
    prim = stage.GetPrimAtPath(mesh_name)
    if not prim:
        raise RuntimeError(
            f"Mesh '{mesh_name}' not found in '{usd_file_path}'. "
            f"Available meshes: {list_usd_meshes(usd_file_path)}"
        )

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        raise ValueError(f"Mesh '{mesh_name}' has no points")
    vertices = np.array(pts, dtype=np.float32)

    fvi_raw = mesh.GetFaceVertexIndicesAttr().Get()
    fvc_raw = mesh.GetFaceVertexCountsAttr().Get()
    if fvi_raw is None or fvc_raw is None:
        raise ValueError(f"Mesh '{mesh_name}' has no face topology")

    uv_data: dict[str, UVPrimvarEntry] = {}
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if not _is_uv_primvar(pv):
            continue
        coords = pv.GetAttr().Get()
        if not coords or len(coords) == 0:
            continue
        uvs = np.array(coords, dtype=np.float32)
        if uvs.ndim == 1:
            if uvs.size % 2 != 0:
                continue
            uvs = uvs.reshape(-1, 2)
        uv_indices = None
        if pv.IsIndexed():
            idx = pv.GetIndicesAttr().Get()
            if idx:
                uv_indices = np.array(idx, dtype=np.int32)
        uv_data[pv.GetPrimvarName()] = UVPrimvarEntry(
            coordinates=uvs, indices=uv_indices, interpolation=pv.GetInterpolation()
        )

    return (
        vertices,
        np.array(fvi_raw, dtype=np.int32),
        np.array(fvc_raw, dtype=np.int32),
        uv_data,
    )


def _write_uv_primvars(primvars, uv_data: Mapping[str, Any]) -> None:
    Gf, Sdf, _, UsdGeom, _, _ = _pxr()
    interp_tokens = {
        "vertex": UsdGeom.Tokens.vertex,
        "faceVarying": UsdGeom.Tokens.faceVarying,
        "uniform": UsdGeom.Tokens.uniform,
        "constant": UsdGeom.Tokens.constant,
    }
    for name, info in uv_data.items():
        pv = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.TexCoord2fArray)
        pv.Set([Gf.Vec2f(float(u[0]), float(u[1])) for u in info["coordinates"]])
        pv.SetInterpolation(
            interp_tokens.get(info.get("interpolation", "faceVarying"),
                              UsdGeom.Tokens.faceVarying)
        )
        if info.get("indices") is not None:
            pv.SetIndices(np.asarray(info["indices"], dtype=np.int32).tolist())


def write_usd_mesh(
    usd_file_path,
    mesh_name: str,
    vertices,
    face_vert_indices,
    face_vert_counts,
    uv_data: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a bare mesh (no skeleton or skinning) to a new USD file."""
    Gf, _, Usd, UsdGeom, _, _ = _pxr()
    stage = Usd.Stage.CreateNew(str(usd_file_path))
    if not mesh_name.startswith("/"):
        mesh_name = "/" + mesh_name

    mesh = UsdGeom.Mesh(stage.DefinePrim(mesh_name, "Mesh"))
    verts = _to_f32(vertices)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*map(float, v)) for v in verts])
    mesh.CreateFaceVertexIndicesAttr().Set(_to_np(face_vert_indices).astype(np.int32).tolist())
    mesh.CreateFaceVertexCountsAttr().Set(_to_np(face_vert_counts).astype(np.int32).tolist())

    if uv_data:
        _write_uv_primvars(UsdGeom.PrimvarsAPI(mesh), uv_data)

    stage.GetRootLayer().Save()


def fan_triangulate(face_vert_indices, face_vert_counts) -> np.ndarray:
    """Convert a polygon soup to triangles by fan triangulation.

    Returns:
        ``(F_tri, 3)`` int32 triangles.
    """
    fvi = _to_np(face_vert_indices)
    triangles = []
    offset = 0
    for count in _to_np(face_vert_counts):
        count = int(count)
        v0 = fvi[offset]
        for j in range(1, count - 1):
            triangles.append([v0, fvi[offset + j], fvi[offset + j + 1]])
        offset += count
    return np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), np.int32)


# ---------------------------------------------------------------------------
# Skeleton / animation / skinning readers
# ---------------------------------------------------------------------------


def load_usd_skeleton(usd_file_path):
    """Extract the skeleton from a USD file.

    Returns:
        ``(joint_paths, bind_transforms, parent_ids)`` — J path strings,
        a ``(J, 4, 4)`` float32 array of world bind transforms in USD
        row-major convention (``point * M``), and J parent indices with
        ``-1`` for the root.
    """
    _, _, _, _, UsdSkel, _ = _pxr()
    stage = _open_stage(usd_file_path)

    skel_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No Skeleton prim found in '{usd_file_path}'")

    skel = UsdSkel.Skeleton(skel_prim)
    joint_paths = list(skel.GetJointsAttr().Get())
    bind_xforms = skel.GetBindTransformsAttr().Get()
    if bind_xforms is None:
        raise RuntimeError(f"No bindTransforms in '{usd_file_path}'")
    bind_transforms = np.array(bind_xforms, dtype=np.float32).reshape(len(joint_paths), 4, 4)

    path_to_idx = {j: i for i, j in enumerate(joint_paths)}
    parent_ids = [
        path_to_idx.get(j.rsplit("/", 1)[0] if "/" in j else "", -1) for j in joint_paths
    ]
    return joint_paths, bind_transforms, parent_ids


def load_usd_animation(usd_file_path):
    """Extract SkelAnimation local rotations and translations.

    Returns:
        ``(rot_mats, translations)`` — ``(J, 3, 3)`` float32 rotation matrices
        and ``(J, 3)`` float32 translations, or ``None`` when the file has no
        SkelAnimation prim.
    """
    _, _, _, _, UsdSkel, _ = _pxr()
    from scipy.spatial.transform import Rotation

    stage = _open_stage(usd_file_path)
    anim_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Animation)), None)
    if anim_prim is None:
        return None

    anim = UsdSkel.Animation(anim_prim)
    quats = anim.GetRotationsAttr().Get()
    trans = anim.GetTranslationsAttr().Get()
    if quats is None:
        return None

    # USD Quatf stores real=w with imaginary=(x, y, z); scipy wants (x, y, z, w).
    quats_xyzw = np.array(
        [[q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()]
         for q in quats],
        dtype=np.float64,
    )
    rot_mats = Rotation.from_quat(quats_xyzw).as_matrix().astype(np.float32)

    translations = np.zeros((len(quats), 3), dtype=np.float32)
    if trans is not None:
        translations = np.array([[t[0], t[1], t[2]] for t in trans], dtype=np.float32)
    return rot_mats, translations


def load_usd_skinning(usd_file_path, mesh_prim_path: Optional[str] = None):
    """Extract dense skinning weights from a skinned mesh.

    The mesh binding may declare its own joint subset; weights are remapped to
    the **skeleton** joint order so they line up with :func:`load_usd_skeleton`.

    Returns:
        ``(skinning_weights, num_joints)`` — ``(V, J)`` float32 and J.
    """
    _, _, _, UsdGeom, UsdSkel, _ = _pxr()
    stage = _open_stage(usd_file_path)

    if mesh_prim_path is not None:
        if not mesh_prim_path.startswith("/"):
            mesh_prim_path = "/" + mesh_prim_path
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if not mesh_prim:
            raise RuntimeError(f"Mesh '{mesh_prim_path}' not found in '{usd_file_path}'")
    else:
        mesh_prim = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)), None)
        if mesh_prim is None:
            raise RuntimeError(f"No Mesh prim found in '{usd_file_path}'")

    skel_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No Skeleton prim found in '{usd_file_path}'")

    binding = UsdSkel.BindingAPI(mesh_prim)
    ji_pv = binding.GetJointIndicesPrimvar()
    jw_pv = binding.GetJointWeightsPrimvar()
    if not ji_pv or not jw_pv:
        raise RuntimeError(
            "No skinning primvars (skel:jointIndices / skel:jointWeights) found on "
            f"'{mesh_prim.GetPath()}'"
        )

    pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    if not pts:
        raise ValueError(f"Mesh '{mesh_prim.GetPath()}' has no points")
    V = len(pts)
    K = ji_pv.GetElementSize()
    ji = np.array(ji_pv.Get(), dtype=np.int32).reshape(V, K)
    jw = np.array(jw_pv.Get(), dtype=np.float32).reshape(V, K)

    skel_joints = list(UsdSkel.Skeleton(skel_prim).GetJointsAttr().Get())
    J = len(skel_joints)
    skel_joint_to_idx = {name: i for i, name in enumerate(skel_joints)}

    binding_joints = binding.GetJointsAttr().Get()
    if binding_joints and len(binding_joints) > 0:
        binding_to_skel = np.array(
            [skel_joint_to_idx.get(str(j), -1) for j in binding_joints], dtype=np.int32
        )
    else:
        binding_to_skel = np.arange(J, dtype=np.int32)

    v_idx = np.repeat(np.arange(V, dtype=np.int32), K)
    j_idx = binding_to_skel[ji.ravel()]
    w_vals = jw.ravel()
    valid = (w_vals > 0) & (j_idx >= 0)

    W = np.zeros((V, J), dtype=np.float32)
    np.add.at(W, (v_idx[valid], j_idx[valid]), w_vals[valid])
    return W, J


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _build_joint_paths(joint_names, joint_parent_ids) -> list[str]:
    """Build UsdSkel path tokens from flat joint names and parent IDs."""
    names = [str(n) for n in joint_names]
    paths = [""] * len(joint_parent_ids)
    paths[0] = names[0]
    for j in range(1, len(joint_parent_ids)):
        paths[j] = paths[int(joint_parent_ids[j])] + "/" + names[j]
    return paths


def _rotmats_to_quats_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert (N, J, 3, 3) rotations to (N, J, 4) wxyz quaternions.

    Near-degenerate matrices are projected to the nearest proper rotation via
    SVD, and quaternion signs are made continuous between frames so the USD
    interpolation does not take the long way round.
    """
    from scipy.spatial.transform import Rotation as ScipyR

    N, J = R.shape[:2]
    flat = R.reshape(-1, 3, 3).copy()
    U, _S, Vt = np.linalg.svd(flat)
    R_clean = U @ Vt
    neg = np.linalg.det(R_clean) < 0
    if neg.any():
        U[neg, :, -1] *= -1
        R_clean[neg] = U[neg] @ Vt[neg]

    q_xyzw = ScipyR.from_matrix(R_clean).as_quat()
    q_wxyz = np.empty_like(q_xyzw)
    q_wxyz[:, 0] = q_xyzw[:, 3]
    q_wxyz[:, 1:] = q_xyzw[:, :3]
    q_wxyz = q_wxyz.reshape(N, J, 4)
    for n in range(1, N):
        flip = (q_wxyz[n] * q_wxyz[n - 1]).sum(axis=-1) < 0
        q_wxyz[n, flip] *= -1
    return q_wxyz.astype(np.float32)


def save_soma_usd(
    out_path,
    rotations=None,
    root_translation=None,
    *,
    joint_names: Sequence[str],
    joint_parent_ids,
    bind_transforms_world,
    bind_transforms_local,
    rest_shape,
    faces=None,
    face_vert_indices=None,
    face_vert_counts=None,
    uv_data: Optional[Mapping[str, Any]] = None,
    skinning_weights,
    unit: str = "meters",
    fps: float = 30.0,
    topk: int = 8,
    root_joint_idx: Optional[int] = None,
    skin_mesh_name: str = DEFAULT_SKIN_MESH_NAME,
) -> None:
    """Save a SOMA rig (and optionally an animation) as a UsdSkel file.

    With ``rotations`` and ``root_translation`` omitted, only the static rig is
    written — skeleton bind pose plus skinned mesh, no SkelAnimation prim.
    Rotations are expected in the absolute local-space convention that
    :meth:`~soma_jax.SOMAPoseInversion.fit` returns.

    Args:
        out_path: destination ``.usd`` / ``.usda`` / ``.usdc`` path.
        rotations: (N, J, 3, 3) local rotation matrices, or None for a static rig.
        root_translation: (N, 3) root joint local translation, or None.
        joint_names: J joint names, including the Root.
        joint_parent_ids: (J,) parent indices.
        bind_transforms_world: (J, 4, 4) or (1, J, 4, 4) world bind transforms.
        bind_transforms_local: (J, 4, 4) or (1, J, 4, 4) local bind transforms.
        rest_shape: (V, 3) bind-pose vertices.
        faces: (F, 3) triangles; used when polygon topology is not supplied.
        face_vert_indices: flattened polygon face-vertex indices (takes
            priority over ``faces``, e.g. to keep the template's quads).
        face_vert_counts: per-face vertex counts matching ``face_vert_indices``.
        uv_data: optional UV sets in :func:`load_usd_mesh` format.
        skinning_weights: (V, J) dense skinning weights.
        unit: ``"meters"``, ``"centimeters"`` or ``"millimeters"``.
        fps: frames per second (timeCodesPerSecond).
        topk: max joint influences per vertex written to the mesh.
        root_joint_idx: joint receiving ``root_translation`` (default 1, the
            SOMA Hips; pass 0 for hand-only rigs).
        skin_mesh_name: leaf name for the skinned mesh prim.
    """
    from .geometry.batched_skinning import topk_skinning

    Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt = _pxr()

    if root_joint_idx is None:
        root_joint_idx = _HIPS_IDX
    if not isinstance(skin_mesh_name, str) or not skin_mesh_name or "/" in skin_mesh_name:
        raise ValueError(
            f"skin_mesh_name must be a non-empty leaf name (no '/'), got {skin_mesh_name!r}"
        )

    has_animation = rotations is not None
    bind_world = _to_np(bind_transforms_world).astype(np.float64)
    bind_local = _to_np(bind_transforms_local).astype(np.float64)
    rest = _to_f32(rest_shape)
    weights = _to_f32(skinning_weights)
    joint_parent_ids = _to_np(joint_parent_ids)

    if bind_world.ndim == 4:
        bind_world = bind_world[0]
    if bind_local.ndim == 4:
        bind_local = bind_local[0]
    if rest.ndim == 3:
        rest = rest[0]

    J = len(joint_parent_ids)
    V = rest.shape[0]
    joint_paths = _build_joint_paths(joint_names, joint_parent_ids)
    bind_local_t = bind_local[:, :3, 3].astype(np.float32)

    idx, wts = topk_skinning(weights, topk)
    idx_np = np.asarray(idx, dtype=np.int32).reshape(-1)
    wts_np = np.asarray(wts, dtype=np.float32).reshape(-1)
    K = np.asarray(idx).shape[1]

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(out_path)

    meters_per_unit = Unit.from_name(unit).meters_per_unit
    stage.SetMetadata("metersPerUnit", meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    skel_root = UsdSkel.Root.Define(stage, "/Root")
    skel = UsdSkel.Skeleton.Define(stage, skel_root.GetPath().AppendChild("Skeleton"))
    skel.CreateJointsAttr(Vt.TokenArray(joint_paths))
    # USD matrices are row-major (point * M), so transpose from our column-major.
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray.FromNumpy(bind_world.swapaxes(-2, -1)))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray.FromNumpy(bind_local.swapaxes(-2, -1)))

    if has_animation:
        rots = _to_f32(rotations)
        root_t = _to_f32(root_translation)
        N = rots.shape[0]
        quats = _rotmats_to_quats_wxyz(rots)

        stage.SetTimeCodesPerSecond(fps)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(N - 1)

        skel_anim = UsdSkel.Animation.Define(stage, skel.GetPath().AppendChild("Anim"))
        skel_anim.CreateJointsAttr(Vt.TokenArray(joint_paths))
        rot_attr = skel_anim.CreateRotationsAttr()
        transl_attr = skel_anim.CreateTranslationsAttr()
        scales_attr = skel_anim.CreateScalesAttr()

        rot_attr.Set(Vt.QuatfArray([Gf.Quatf(1, 0, 0, 0)] * J))
        transl_attr.Set(Vt.Vec3fArray.FromNumpy(bind_local_t))
        scales_attr.Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * J))

        skel_bind = UsdSkel.BindingAPI.Apply(skel.GetPrim())
        skel_bind.CreateAnimationSourceRel().SetTargets([skel_anim.GetPath()])

        for frame_idx in range(N):
            tc = Usd.TimeCode(float(frame_idx))
            rot_attr.Set(
                Vt.QuatfArray([
                    Gf.Quatf(*(float(v) for v in quats[frame_idx, j])) for j in range(J)
                ]),
                tc,
            )
            frame_t = bind_local_t.copy()
            frame_t[root_joint_idx] = root_t[frame_idx]
            transl_attr.Set(Vt.Vec3fArray.FromNumpy(frame_t), tc)
            scales_attr.Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * J), tc)

    mesh = UsdGeom.Mesh.Define(stage, skel_root.GetPath().AppendChild(skin_mesh_name))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(rest))
    if face_vert_indices is not None and face_vert_counts is not None:
        counts = _to_np(face_vert_counts).astype(np.int32)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts.tolist()))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray(_to_np(face_vert_indices).astype(np.int32).tolist())
        )
        F_count = len(counts)
    elif faces is not None:
        faces = _to_np(faces).astype(np.int32)
        F_count = faces.shape[0]
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * F_count))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
    else:
        raise ValueError("Either faces or (face_vert_indices, face_vert_counts) must be provided.")

    mesh_bind = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    mesh_bind.CreateSkeletonRel().SetTargets([skel.GetPath()])
    mesh_bind.CreateSkinningMethodAttr().Set(UsdSkel.Tokens.classicLinear)
    mesh_bind.CreateGeomBindTransformAttr().Set(Gf.Matrix4d(1.0))

    primvars = UsdGeom.PrimvarsAPI(mesh)
    ji_pv = primvars.CreatePrimvar("skel:jointIndices", Sdf.ValueTypeNames.IntArray, "vertex", K)
    ji_pv.Set(Vt.IntArray.FromNumpy(idx_np))
    jw_pv = primvars.CreatePrimvar("skel:jointWeights", Sdf.ValueTypeNames.FloatArray, "vertex", K)
    jw_pv.Set(Vt.FloatArray.FromNumpy(wts_np))

    if uv_data:
        _write_uv_primvars(primvars, uv_data)

    stage.SetDefaultPrim(skel_root.GetPrim())
    stage.GetRootLayer().Save()

    logger.info(
        "Saved USD: %s\n  joints: %s%s\n  vertices: %s, faces: %s\n"
        "  skinning: top-%s influences/vertex\n  unit: %s (metersPerUnit=%s)",
        out_path, J,
        f", frames: {rots.shape[0]} @ {fps} fps" if has_animation else " (static rig)",
        V, F_count, K, unit, meters_per_unit,
    )


def save_vertex_animation_usd(
    out_path, vertices, faces, *, unit: str = "meters", fps: float = 30.0,
    prim_path: str = "/Mesh",
) -> None:
    """Save per-frame vertex positions as an animated Mesh — no skeleton.

    Useful for exporting target meshes, blendshape sequences, or any
    topology-constant vertex animation.

    Args:
        out_path: destination USD path.
        vertices: (N, V, 3) per-frame vertex positions.
        faces: (F, 3) triangles, constant across frames.
        unit: ``"meters"``, ``"centimeters"`` or ``"millimeters"``.
        fps: frames per second.
        prim_path: USD prim path for the mesh.
    """
    _, _, Usd, UsdGeom, _, Vt = _pxr()

    verts = _to_f32(vertices)
    faces = _to_np(faces).astype(np.int32)
    N, V = verts.shape[:2]
    F_count = faces.shape[0]

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(out_path)

    meters_per_unit = Unit.from_name(unit).meters_per_unit
    stage.SetMetadata("metersPerUnit", meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(N - 1)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * F_count))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))

    pts_attr = mesh.CreatePointsAttr()
    for frame_idx in range(N):
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(verts[frame_idx]), Usd.TimeCode(float(frame_idx)))

    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()
    logger.info(
        "Saved vertex animation USD: %s\n  vertices: %s, faces: %s, frames: %s @ %s fps\n"
        "  unit: %s (metersPerUnit=%s)",
        out_path, V, F_count, N, fps, unit, meters_per_unit,
    )


def export_soma_usd(
    output_path,
    soma_layer,
    rotations,
    root_translation,
    *,
    bind_transforms_world,
    rest_shape,
    fps: float = 30.0,
    unit: str = "meters",
    root_joint_idx: Optional[int] = None,
    skin_mesh_name: str = DEFAULT_SKIN_MESH_NAME,
) -> None:
    """Export a skeletal animation to USD from a SOMA-JAX layer.

    Convenience wrapper around :func:`save_soma_usd` that pulls joint names,
    parents, skinning weights and faces off the layer's public rig view.

    Because SOMA-JAX layers are immutable and do not cache identity state, the
    fitted rig is passed explicitly — take both from
    ``layer.prepare_identity(..., return_bind_transforms=True)``. (Upstream
    reads them from the layer's ``_cached_*`` attributes instead.)

    Args:
        output_path: destination USD path.
        soma_layer: a :class:`~soma_jax.SOMALayer`.
        rotations: (N, J, 3, 3) absolute local rotation matrices, e.g. from
            :meth:`~soma_jax.SOMAPoseInversion.fit`.
        root_translation: (N, 3) root translation.
        bind_transforms_world: (J, 4, 4) or (1, J, 4, 4) fitted bind transforms.
        rest_shape: (V, 3) or (1, V, 3) fitted rest mesh.
        fps: animation frame rate.
        unit: unit string written to the USD metadata.
        root_joint_idx: joint receiving ``root_translation`` (default 1).
        skin_mesh_name: leaf name for the skinned mesh prim.
    """
    from .geometry.rig_utils import joint_world_to_local

    bw = _to_np(bind_transforms_world)
    if bw.ndim == 4 and bw.shape[0] == 1:
        bw = bw[0]
    rs = _to_np(rest_shape)
    if rs.ndim == 3 and rs.shape[0] == 1:
        rs = rs[0]

    parents = np.asarray(soma_layer._parents_np)
    bl = _to_np(joint_world_to_local(bw, parents))

    rots = _to_np(rotations)
    expected = len(soma_layer.joint_names)
    if rots.shape[-3] != expected:
        raise ValueError(
            f"Expected rotations for {expected} joints, got {rots.shape[-3]}."
        )

    save_soma_usd(
        output_path,
        rots,
        _to_np(root_translation),
        joint_names=[str(n) for n in soma_layer.joint_names],
        joint_parent_ids=parents,
        bind_transforms_world=bw,
        bind_transforms_local=bl,
        rest_shape=rs,
        faces=_to_np(soma_layer.faces),
        skinning_weights=_to_np(soma_layer.public_skinning_weights()),
        unit=unit,
        fps=fps,
        root_joint_idx=(
            root_joint_idx if root_joint_idx is not None else _HIPS_IDX
        ),
        skin_mesh_name=skin_mesh_name,
    )


def load_template_rig(usd_path=None, mesh_name: str = "c_skin_mid") -> dict:
    """Load the **expanded** template rig (skeleton + skinning) from the USD.

    The runtime archive this repo builds carries the 78-joint *public* rig. The
    template USD additionally authors the full skeleton — 122 joints with bind
    transforms — and binds the skin mesh to a subset of them. Upstream skins
    with that expanded rig by default (its procedural transforms expand the
    public pose to fill it); reading it is the prerequisite for doing the same
    here.

    Part of upstream's LOD/rig-discovery chain (``load_rig_from_usd``), which
    ``docs/INSTALL.md`` §4.2 otherwise bypasses by building through the
    upstream layer.

    Args:
        usd_path: template rig; defaults to the resolved asset.
        mesh_name: skin mesh to read the binding from. ``"c_skin_mid"`` is the
            mid-LOD body used by the SOMA runtime archive.

    Returns:
        dict with ``joint_names`` (J,), ``parents`` (J,), ``bind_transforms``
        (J, 4, 4), ``bound_joint_names`` (K,) — the subset the mesh binds to,
        in binding order — ``joint_indices`` / ``joint_weights`` (V, n_inf)
        indexing into ``bound_joint_names``, and ``bound_to_skeleton`` (K,)
        mapping those onto the skeleton.

    Raises:
        ValueError: when the skeleton or the named mesh is absent.
    """
    _, _, Usd, UsdGeom, UsdSkel, _ = _pxr()
    if usd_path is None:
        from .assets import resolve
        usd_path = resolve("SOMA_template_rig.usda")

    stage = Usd.Stage.Open(str(usd_path))
    skels = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    if not skels:
        raise ValueError(f"No UsdSkel.Skeleton in {usd_path}")
    skel = UsdSkel.Skeleton(skels[0])

    paths = [str(p) for p in skel.GetJointsAttr().Get()]
    names = [p.split("/")[-1] for p in paths]
    index = {p: i for i, p in enumerate(paths)}
    parents = np.full(len(paths), -1, dtype=np.int32)
    for i, p in enumerate(paths):
        # A top-level joint has no "/" — rsplit would return the joint itself
        # and make the root its own parent, so guard on the separator.
        if "/" not in p:
            continue
        parent = p.rsplit("/", 1)[0]
        if parent in index:
            parents[i] = index[parent]

    def _xforms(attr):
        # USD stores row-vector matrices; transpose into the column-vector
        # convention the rest of the port and upstream's rig arrays use.
        v = attr.Get()
        return (np.asarray([np.asarray(m, dtype=np.float64).T for m in v], np.float32)
                if v is not None else None)

    bind_transforms = _xforms(skel.GetBindTransformsAttr())
    # `restTransforms` are the T-pose *local* transforms, i.e. upstream's
    # ``t_pose_local``. Both attributes are authored on the shipped template.
    rest_transforms = _xforms(skel.GetRestTransformsAttr())

    mesh = next((p for p in stage.Traverse()
                 if p.IsA(UsdGeom.Mesh) and p.GetName() == mesh_name), None)
    if mesh is None:
        raise ValueError(f"Mesh {mesh_name!r} not found in {usd_path}")
    binding = UsdSkel.BindingAPI(mesh)
    ji = binding.GetJointIndicesAttr().Get()
    jw = binding.GetJointWeightsAttr().Get()
    if ji is None or jw is None:
        raise ValueError(f"Mesh {mesh_name!r} carries no skinning binding")
    n_inf = binding.GetJointIndicesPrimvar().GetElementSize() or 1
    joint_indices = np.asarray(ji, dtype=np.int32).reshape(-1, n_inf)
    joint_weights = np.asarray(jw, dtype=np.float32).reshape(-1, n_inf)

    bound = binding.GetJointsAttr().Get()
    bound_names = [str(b).split("/")[-1] for b in bound] if bound else names
    name_to_skel = {n: i for i, n in enumerate(names)}
    bound_to_skeleton = np.asarray([name_to_skel.get(n, -1) for n in bound_names], np.int32)

    return {
        "joint_names": names,
        "parents": parents,
        "bind_transforms": bind_transforms,
        "rest_transforms": rest_transforms,
        "bound_joint_names": bound_names,
        "joint_indices": joint_indices,
        "joint_weights": joint_weights,
        "bound_to_skeleton": bound_to_skeleton,
    }


# ---------------------------------------------------------------------------
# LOD skin-mesh discovery (upstream `soma/io.py`)
# ---------------------------------------------------------------------------
#: Preferred skin-mesh prim names per LOD — upstream's
#: ``_LOD_SKIN_MESH_CANDIDATES``. The shipped ``SOMA_template_rig.usda`` uses
#: ``c_skin_mid`` / ``c_skin_lo`` / ``c_skin_xlo``.
LOD_SKIN_MESH_CANDIDATES = {
    "mid": ("c_skin_mid", "c_bodyRig_mid", "c_skin"),
    "low": ("c_skin_lo", "c_bodyRig_lo", "c_skin_low", "c_bodyRig_low"),
    "xlo": ("c_skin_xlo", "c_bodyRig_xlo", "c_skin_extra_low",
            "c_bodyRig_extra_low", "c_body_xlo", "skin_xlo"),
}

#: Fallback substrings when no candidate name is present — upstream's
#: ``_LOD_NAME_TOKENS``. Order matters: ``xlo`` is checked before ``low`` so an
#: ``xlo`` mesh is not claimed by the ``low`` token ``"lo"``.
LOD_NAME_TOKENS = {
    "mid": ("mid",),
    "low": ("_lo", "_low", "low"),
    "xlo": ("xlo", "extra_low", "extralow", "extra-low"),
}

#: Vertex counts upstream documents for the shipped rig, for validation.
LOD_VERTEX_COUNTS = {"mid": 18056, "low": 4505, "xlo": 612}


def find_lod_skin_mesh_name(usd_path, lod: str) -> str:
    """Name of the skinned body mesh for ``lod`` in a template rig USD.

    Port of upstream ``soma.io.find_lod_skin_mesh_name``. Tries the per-LOD
    candidate names first, then falls back to substring matching over skinned
    meshes. This is the discovery step ``SOMALayer(lod="xlo")`` needs: the
    runtime npz carries mid and low data only (``triangles_low``,
    ``lod_mid_to_low``) and has **no xlo arrays at all**, so the xlo topology can
    only come from the template USD.

    Args:
        usd_path: the template rig ``.usda``.
        lod: ``"mid"``, ``"low"`` or ``"xlo"``.

    Returns:
        The mesh prim name.

    Raises:
        ValueError: for an unknown ``lod``, or when no skinned mesh matches.
    """
    if lod not in LOD_SKIN_MESH_CANDIDATES:
        raise ValueError(
            f"Unknown lod {lod!r}; expected one of {sorted(LOD_SKIN_MESH_CANDIDATES)}.")
    _, _, Usd, UsdGeom, UsdSkel, _ = _pxr()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Could not open USD stage: {usd_path}")

    skinned = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        binding = UsdSkel.BindingAPI(prim)
        if binding.GetJointIndicesPrimvar() and binding.GetJointWeightsPrimvar():
            skinned[prim.GetName()] = prim

    for name in LOD_SKIN_MESH_CANDIDATES[lod]:
        if name in skinned:
            return name

    # Substring fallback. Check the more specific LODs first so "lo" inside
    # "xlo" does not misclassify.
    for name in skinned:
        lowered = name.lower()
        claimed = next((l for l in ("xlo", "mid", "low")
                        if any(t in lowered for t in LOD_NAME_TOKENS[l])), None)
        if claimed == lod:
            return name

    raise ValueError(
        f"No skinned mesh for lod {lod!r} in {usd_path}. "
        f"Skinned meshes present: {sorted(skinned)}."
    )


def load_lod_rig(usd_path=None, lod: str = "mid") -> dict:
    """Template rig plus the skin binding for one LOD.

    Convenience wrapper: resolves the LOD's mesh name with
    :func:`find_lod_skin_mesh_name`, then reads it with
    :func:`load_template_rig`. This is what makes the **xlo** topology (612
    vertices) reachable at all — see :data:`LOD_VERTEX_COUNTS`.
    """
    if usd_path is None:
        from .assets import resolve
        usd_path = resolve("SOMA_template_rig.usda")
    name = find_lod_skin_mesh_name(usd_path, lod)
    rig = load_template_rig(usd_path, mesh_name=name)
    rig["lod"] = lod
    rig["mesh_name"] = name
    return rig
