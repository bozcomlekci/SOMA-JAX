"""Hand-only SOMA layer (JAX port).

Upstream: ``soma/hand/soma.py`` (SOMA-X v0.3.0, reference poses v0.3.1).

:class:`SOMAHandLayer` is a hand-only parametric model operating in wrist-local
coordinate space. The hand mesh and its 25 joints are a **strict subset of the
full-body SOMA topology and skeleton** (the 78-joint ``SOMALayer`` skeleton).
``SOMAHand.npz`` stores the hand mapping (vertex IDs, remapped faces, hand-joint
indices), the SOMA hand identity PCA and a bind-relative articulation-pose PCA;
rig vertices and skinning weights are sliced from ``SOMA_template_rig.usda`` at
construction time.

LOD vertex counts: ``"mid"`` 2,859, ``"low"`` 718, ``"xlo"`` 134 per hand
(``low_lod=True`` is the legacy alias for ``lod="low"``).

Skeleton (25 joints, wrist = root at the output origin)::

    0      Wrist
    1-4    Thumb1, Thumb2, Thumb3, ThumbEnd
    5-9    Index1..Index4, IndexEnd
    10-14  Middle1..Middle4, MiddleEnd
    15-19  Ring1..Ring4, RingEnd
    20-24  Pinky1..Pinky4, PinkyEnd

Poses are ``(B, 25, 3)`` axis-angle or ``(B, 25, 3, 3)`` rotation matrices
(``pose2rot=False``). Joint 0 is the global wrist rotation; joints 1-24 are
finger articulation, relative to the T-pose joint orient unless
``absolute_pose=True``. Identity coefficients: ``soma`` 20 (PCA),
``mano`` 10 (betas), ``mhr`` 5 (MHR dims 40-44). ``scale_params``: ``soma``
``(B, 24)`` bone-length scales applied at pose time; ``mhr`` ``(B, 26)`` baked
into the rest shape; ``mano`` unused. Native unit centimetres; outputs in
``output_unit`` (default metres).

API difference (the same one :class:`~soma_jax.SOMALayer` makes): upstream's
``prepare_identity`` caches the identity on the module and ``pose`` reads that
cache. JAX layers are immutable, so here :meth:`SOMAHandLayer.prepare_identity`
**returns** a :class:`SOMAHandIdentity` and :meth:`SOMAHandLayer.pose` takes it
explicitly; :meth:`SOMAHandLayer.forward` (also ``__call__``) does both, exactly
like upstream's ``forward``.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Optional

import logging

import jax
import jax.numpy as jnp
import numpy as np

from ..correctives_model import (
    _DEFAULT_CORRECTIVES_MODEL_PATH,
    CorrectivesMLP,
    resolve_correctives_model_path,
)
from ..geometry.barycentric_interp import barycentric_interpolate, compute_barycentric_coords
from ..geometry.batched_skinning import pose_from_bind, topk_skinning
from ..geometry.lbs import batch_rodrigues, compute_skeleton_levels
from ..geometry.skeleton_transfer import SkeletonTransfer
from ..reference_poses import (
    ReferencePoseHistory,
    _apply_orient,
    _orient_pair,
    convert_reference_rotations,
    validate_reference_pose,
)
from ..io import SOMA_TEMPLATE_RIG_FILENAME  # noqa: F401  (upstream re-exports it here)
from ..procedural_transforms import SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME  # noqa: F401
from ..units import Unit
from .identity_model import SOMAHandIdentityModel

logger = logging.getLogger(__name__)

__all__ = ["SOMAHandLayer", "SOMAHandPoseOutput", "SOMAHandIdentity"]

_VALID_HAND_LODS = ("mid", "low", "xlo")


class SOMAHandPoseOutput(dict):
    """Output of :meth:`SOMAHandLayer.pose` / :meth:`SOMAHandLayer.forward`.

    Behaves like a ``dict`` (``out["vertices"]``) and supports attribute access
    (``out.vertices``). ``vertices`` is absent when ``fk_only=True``.
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class SOMAHandIdentity(NamedTuple):
    """A prepared hand identity — what upstream caches in ``prepare_identity``.

    Attributes:
        rest_shape: (B, Vh, 3) wrist-local rest shape, ``output_unit``.
        bind_transforms_world: (B, 25, 4, 4) fitted (and possibly reposed) binds.
        scale_params: SOMA-backend (B, 24) bone scales for :meth:`pose`, else None.
        global_scale: the uniform scale used, for corrective scaling.
        correctives_rest_shape: rest shape correctives are evaluated on (the low
            LOD for an xlo layer), else ``rest_shape``.
    """

    rest_shape: jnp.ndarray
    bind_transforms_world: jnp.ndarray
    scale_params: Optional[jnp.ndarray]
    global_scale: Any
    correctives_rest_shape: jnp.ndarray


def _resolve_hand_lod(low_lod: bool, lod: Optional[str]) -> str:
    if lod is None:
        return "low" if low_lod else "mid"
    lod = lod.lower()
    if lod not in _VALID_HAND_LODS:
        raise ValueError(f"Unsupported hand LOD {lod!r}; expected one of {_VALID_HAND_LODS}")
    if low_lod and lod != "low":
        raise ValueError("low_lod=True is only compatible with lod='low'")
    return lod


def _localize_points(points: np.ndarray, wrist_inv: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return (wrist_inv @ np.hstack([points, ones]).T).T[:, :3]


def _remap_faces(faces: np.ndarray, selected_vert_ids: np.ndarray, num_verts: int) -> np.ndarray:
    inverse = np.full((num_verts,), -1, dtype=np.int64)
    inverse[selected_vert_ids] = np.arange(selected_vert_ids.shape[0], dtype=np.int64)
    remapped = inverse[faces]
    keep = (remapped >= 0).all(axis=1)
    return remapped[keep].astype(np.int32)


def _boundary_vertices_from_faces(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.int32)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(boundary_edges).astype(np.int32)


def _hand_weights(weights: np.ndarray, body_vert_ids, hand_joint_ids, boundary_loop) -> np.ndarray:
    w = np.asarray(weights, np.float32)[body_vert_ids][:, hand_joint_ids].copy()
    if boundary_loop.size:
        w[boundary_loop] = 0.0
        w[boundary_loop, 0] = 1.0
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-6] = 1.0
    return w / row_sums


def _fan_triangulate(face_vert_indices, face_vert_counts) -> np.ndarray:
    from ..usd_io import fan_triangulate
    return np.asarray(fan_triangulate(np.asarray(face_vert_indices),
                                      np.asarray(face_vert_counts)), np.int64)


def _derive_low_hand_lod(rig, low_rig, mid_vert_ids):
    mid_to_hand_local = np.full((rig["bind_shape"].shape[0],), -1, dtype=np.int64)
    mid_to_hand_local[mid_vert_ids] = np.arange(mid_vert_ids.shape[0], dtype=np.int64)
    mid_to_low = np.asarray(rig["lod_mid_to_low"], np.int64)
    keep = mid_to_hand_local[mid_to_low] >= 0
    body_vert_ids = np.flatnonzero(keep).astype(np.int32)
    source_mid_local = mid_to_hand_local[mid_to_low[body_vert_ids]].astype(np.int64)
    body_faces = _fan_triangulate(low_rig["face_vert_indices"], low_rig["face_vert_counts"])
    faces = _remap_faces(body_faces, body_vert_ids, low_rig["bind_shape"].shape[0])
    return body_vert_ids, source_mid_local, faces, _boundary_vertices_from_faces(faces)


def _derive_xlo_hand_lod(rig, xlo_rig, mid_vert_ids):
    # Upstream builds a BarycentricInterpolator(mid, mid_faces, xlo) and keeps
    # the xlo vertices whose embedding face touches the hand.
    face_ids, _ = compute_barycentric_coords(
        np.asarray(xlo_rig["bind_shape"], np.float32),
        np.asarray(rig["bind_shape"], np.float32),
        np.asarray(rig["triangles"], np.int64))
    hand_mid = np.zeros(rig["bind_shape"].shape[0], dtype=bool)
    hand_mid[np.asarray(mid_vert_ids, np.int64)] = True
    source_faces = np.asarray(rig["triangles"], np.int64)[np.asarray(face_ids, np.int64)]
    keep = hand_mid[source_faces].any(axis=1)
    body_vert_ids = np.flatnonzero(keep).astype(np.int32)
    body_faces = _fan_triangulate(xlo_rig["face_vert_indices"], xlo_rig["face_vert_counts"])
    faces = _remap_faces(body_faces, body_vert_ids, xlo_rig["bind_shape"].shape[0])
    return body_vert_ids, faces, _boundary_vertices_from_faces(faces)


def _barycentric_setup(src_verts, src_faces, dst_verts):
    """Embed ``dst_verts`` in the tet-extended ``src`` mesh (upstream ``BarycentricInterpolator``).

    Pass coordinates in the units upstream builds the interpolator in (native
    centimetres here). The embedding is **not** scale-invariant: building the
    same xlo transfer from metre coordinates reassigns 5 of 134 vertices to other
    faces and moves the mesh by ~0.6 mm. It is computed in float32, the
    precision of upstream's tensors. The resulting unitless coordinates are then
    applied to output-unit shapes, as upstream does.
    """
    face_ids, bary = compute_barycentric_coords(
        np.asarray(dst_verts, np.float32), np.asarray(src_verts, np.float32),
        np.asarray(src_faces, np.int64))
    return (jnp.asarray(np.asarray(src_faces, np.int32)), jnp.asarray(face_ids),
            jnp.asarray(bary, jnp.float32))


def _barycentric_apply(transfer, src_verts):
    faces, face_ids, bary = transfer
    return barycentric_interpolate(src_verts, faces, face_ids, bary)


def _world_to_local(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    out = np.zeros_like(world)
    for j in range(world.shape[0]):
        p = int(parents[j])
        out[j] = world[j] if p < 0 or p == j else np.linalg.inv(world[p]) @ world[j]
    return out


def _public_rig(usd_path, lod: str, public_joint_names) -> dict:
    """Upstream ``derive_soma_rig_without_procedural_joints(template_lod_rig, public)``."""
    from ..rig_build import merge_template_rig, prune_procedural_joints
    return prune_procedural_joints(merge_template_rig(usd_path, lod), public_joint_names)


class SOMAHandLayer:
    """Hand-only parametric model in wrist-local coordinates. See the module docstring.

    Two-phase API matching :class:`~soma_jax.SOMALayer`:

    1. ``identity = layer.prepare_identity(identity_coeffs, scale_params=None)``
    2. ``out = layer.pose(poses, identity)``

    :meth:`forward` / ``layer(poses, identity_coeffs)`` does both.
    """

    NATIVE_UNIT = Unit.CENTIMETERS
    NUM_BONE_SCALE_PARAMS = 24   # joints 1-24 (SOMA backend scale_params dim)

    def __init__(
        self,
        data_root: str | Path | None = None,
        hand_type: str = "left",
        device=None,
        identity_model_type: str = "soma",
        mode: str = "warp",
        output_unit: Unit = Unit.METERS,
        identity_model_kwargs: Mapping[str, Any] | None = None,
        lod: str | None = None,
        low_lod: bool = False,
        load_correctives_model: bool | None = None,
        correctives_model_path=_DEFAULT_CORRECTIVES_MODEL_PATH,
        *,
        reference_pose=None,
    ) -> None:
        """Build a SOMAHandLayer with the selected identity backend.

        Args:
            data_root: directory with ``SOMAHand.npz``, ``SOMA_neutral.npz``,
                the template rig, the procedural JSON and the per-backend
                folders. Defaults to :func:`soma_jax.assets.data_root`.
            hand_type: ``"left"`` or ``"right"``.
            device: accepted in upstream's (third) position for call
                compatibility and ignored: arrays live on JAX's default device.
            identity_model_type: ``"soma"`` (default), ``"mano"`` or ``"mhr"``.
            mode: ``"warp"`` (default) skins with upstream's top-8 sparse
                weights (what its Warp kernel uses); ``"dense"`` uses all 25.
            output_unit: unit of every translational output. Default metres.
            identity_model_kwargs: forwarded to the identity backend, e.g.
                MANO's ``model_path``.
            lod: ``"mid"`` (2,859 verts), ``"low"`` (718) or ``"xlo"`` (134).
            low_lod: legacy alias for ``lod="low"``.
            load_correctives_model: deprecated alias, as upstream — use
                ``correctives_model_path=None`` to disable loading.
            correctives_model_path: pose-corrective checkpoint (``.pt`` or a
                converted ``.npz``). Defaults to ``data_root/correctives_model.pt``
                (a missing default file loads nothing); ``None`` skips loading;
                an explicit path must exist. Reading a ``.pt`` needs ``torch``;
                without it the default is skipped (correctives are off by default
                anyway) and an explicit path raises.
            reference_pose: default reference for :meth:`pose` / :meth:`forward`
                — a ``(25, 3, 3)`` / ``(25, 4, 4)`` array in this hand's joint
                order and wrist bind frame, or a :meth:`get_reference_pose`
                argument dict, resolved once. ``None`` uses the current T-pose.
        """
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got '{hand_type}'")
        if mode not in ("warp", "dense"):
            raise ValueError(f"mode must be 'warp' or 'dense', got {mode!r}")
        self.lod = _resolve_hand_lod(low_lod, lod)
        if data_root is None or not Path(data_root).exists():
            from ..assets import data_root as _data_root
            if data_root is not None:
                logger.info("data_root '%s' not found, using the default SOMA-X assets", data_root)
            data_root = _data_root()
        data_root = Path(data_root)
        # Resolved before any asset is read, as upstream.
        self.correctives_model_path = resolve_correctives_model_path(
            data_root=data_root, correctives_model_path=correctives_model_path,
            load_correctives_model=load_correctives_model)
        self._unit_conversion = self.NATIVE_UNIT.meters_per_unit / output_unit.meters_per_unit

        # -- SOMAHand.npz mapping --------------------------------------------
        hand_asset = data_root / "SOMAHand.npz"
        if not hand_asset.exists():
            raise FileNotFoundError(f"Hand asset not found: {hand_asset}")
        _map = np.load(hand_asset, allow_pickle=False)
        p = f"{hand_type}_"
        mid_vert_ids = np.asarray(_map[f"{p}vert_ids"], np.int64)
        hand_joint_ids = np.asarray(_map[f"{p}hand_joint_ids_global"], np.int64)
        hand_parent_ids = [int(x) for x in _map[f"{p}joint_parent_ids"].tolist()]
        mid_boundary_loop = np.asarray(_map[f"{p}boundary_loop"], np.int64)

        self.hand_type = hand_type
        self.identity_model_type = identity_model_type
        self.identity_model_kwargs = dict(identity_model_kwargs or {})
        self.mode = mode
        self.output_unit = output_unit
        self.data_root = data_root
        self.low_lod = self.lod == "low"
        self.nv_lod_mid_to_low = None
        self.excluded_vert_ids = None
        self.root_joint_idx = 0                  # wrist is the root (no virtual root)
        self.hand_joint_ids_global = hand_joint_ids.tolist()
        self.wrist_global_id = int(_map[f"{p}wrist_global_id"])

        # -- core npz + template rig (SOMA-X v0.3 asset contract) -------------
        core_asset = data_root / "SOMA_neutral.npz"
        if not core_asset.exists():
            raise FileNotFoundError(f"Core asset not found: {core_asset}")
        core = dict(np.load(core_asset, allow_pickle=False))
        self._reference_pose_history = ReferencePoseHistory(core)
        if "joint_names" in core:
            public_joint_names = [str(n) for n in core["joint_names"]]
        else:
            from ..procedural_transforms import load_definition
            public_joint_names = list(load_definition(
                data_root / "SOMA_procedural_transforms.json").main_joint_names)
        usd_rig = data_root / "SOMA_template_rig.usda"
        if not usd_rig.exists():
            raise FileNotFoundError(
                f"Template rig asset not found: {usd_rig}. Since SOMA-X v0.3 the rig, "
                "bind pose, bind shape and skinning come only from SOMA_template_rig.usda.")
        rig = _public_rig(usd_rig, "mid", public_joint_names)
        rig["triangles"] = np.asarray(core["triangles"], np.int64)
        rig["lod_mid_to_low"] = np.asarray(core["lod_mid_to_low"], np.int64)
        body_parents = np.asarray(rig["parents"], np.int64).copy()
        body_parents[body_parents < 0] = np.arange(len(body_parents))[body_parents < 0]
        self._reference_body_joint_names = tuple(str(n) for n in rig["joint_names"])
        self._reference_body_parent_ids = body_parents   # upstream convention: root self

        hj = hand_joint_ids
        bind_pose = np.asarray(rig["bind_pose_world"], np.float64)       # (78,4,4) cm
        t_pose = np.asarray(rig["t_pose_world"], np.float64)
        # Body-world -> hand-world (wrist at origin). Wrist-local IS this
        # model's world frame, hence the `_world` names below.
        wrist_inv = np.linalg.inv(bind_pose[self.wrist_global_id])
        hand_bind_pose_world = wrist_inv[None] @ bind_pose[hj]           # (25,4,4) cm
        hand_t_pose_world = wrist_inv[None] @ t_pose[hj]

        mid_triangles = np.asarray(_map[f"{p}triangles"], np.int32)
        body_lod_vert_ids = mid_vert_ids.astype(np.int32)
        identity_lod_mid_ids = None
        identity_lod_transfer = None
        correctives_vertex_index_map = mid_vert_ids
        correctives_lod_transfer = None
        skeleton_lod_mid_ids = None
        skeleton_bind_shape_world = None
        skeleton_W = None

        if self.lod == "mid":
            hand_faces = mid_triangles
            boundary_loop = mid_boundary_loop.astype(np.int32)
            hand_bind_shape_world = _localize_points(
                np.asarray(rig["bind_shape"], np.float64)[body_lod_vert_ids], wrist_inv)
            hand_W = _hand_weights(rig["weights"], body_lod_vert_ids, hj, boundary_loop)
        else:
            low_rig = _public_rig(usd_rig, "low", public_joint_names)
            low_body_vert_ids, low_source_mid_local, low_faces, low_boundary_loop = (
                _derive_low_hand_lod(rig, low_rig, mid_vert_ids))
            if self.lod == "low":
                body_lod_vert_ids = low_body_vert_ids
                identity_lod_mid_ids = low_source_mid_local
                correctives_vertex_index_map = mid_vert_ids[low_source_mid_local]
                hand_faces = low_faces
                boundary_loop = low_boundary_loop
                hand_bind_shape_world = _localize_points(
                    np.asarray(low_rig["bind_shape"], np.float64)[body_lod_vert_ids], wrist_inv)
                hand_W = _hand_weights(low_rig["weights"], body_lod_vert_ids, hj, boundary_loop)
            else:
                xlo_rig = _public_rig(usd_rig, "xlo", public_joint_names)
                body_lod_vert_ids, hand_faces, boundary_loop = _derive_xlo_hand_lod(
                    rig, xlo_rig, mid_vert_ids)
                hand_bind_shape_world = _localize_points(
                    np.asarray(xlo_rig["bind_shape"], np.float64)[body_lod_vert_ids], wrist_inv)
                hand_W = _hand_weights(xlo_rig["weights"], body_lod_vert_ids, hj, boundary_loop)
                skeleton_lod_mid_ids = low_source_mid_local
                skeleton_bind_shape_world = _localize_points(
                    np.asarray(low_rig["bind_shape"], np.float64)[low_body_vert_ids], wrist_inv)
                skeleton_W = _hand_weights(low_rig["weights"], low_body_vert_ids, hj,
                                           low_boundary_loop)
                correctives_vertex_index_map = mid_vert_ids[low_source_mid_local]
                # Built in native centimetres, exactly as upstream: the tet
                # embedding is not scale-invariant (see _barycentric_setup).
                correctives_lod_transfer = _barycentric_setup(
                    skeleton_bind_shape_world, low_faces, hand_bind_shape_world)

        if self.lod == "xlo":
            identity_lod_transfer = _barycentric_setup(
                _localize_points(np.asarray(rig["bind_shape"], np.float64)[mid_vert_ids],
                                 wrist_inv),
                mid_triangles, hand_bind_shape_world)

        # -- to output units ---------------------------------------------------
        uc = self._unit_conversion
        bind_w = hand_bind_pose_world.copy(); bind_w[..., :3, 3] *= uc
        t_w = hand_t_pose_world.copy(); t_w[..., :3, 3] *= uc
        bind_shape = hand_bind_shape_world * uc
        skel_bind_shape = skeleton_bind_shape_world * uc if skeleton_bind_shape_world is not None \
            else bind_shape
        skel_W = skeleton_W if skeleton_W is not None else hand_W

        parents_self = np.asarray(hand_parent_ids, np.int64)
        parents_int = parents_self.copy()
        parents_int[parents_int == np.arange(len(parents_int))] = -1   # SOMA-JAX root = -1
        self._parents_int = parents_int
        self._levels = compute_skeleton_levels(parents_int)

        # SkeletonTransfer in hand-world output units, wrist a real root.
        self.skeleton_transfer = SkeletonTransfer(
            parents_int, bind_w.astype(np.float32), skel_bind_shape.astype(np.float32),
            skel_W.astype(np.float32), rotation_method="auto",
            use_sparse_rbf_matrix=False, root_joint_idx=0)
        self.identity_lod_transfer = identity_lod_transfer
        self.correctives_lod_transfer = correctives_lod_transfer

        # -- identity model ----------------------------------------------------
        if identity_model_type == "soma":
            self.identity_model = SOMAHandIdentityModel(
                data_root, False, hand_map=_map, hand_type=hand_type, output_unit=output_unit)
        elif identity_model_type == "mano":
            from .identity_model import MANOHandIdentityModel
            self.identity_model = MANOHandIdentityModel(
                data_root, False, hand_type=hand_type, output_unit=output_unit,
                **self.identity_model_kwargs)
            # Exclude the Laplacian-blended wrist verts from pose inversion.
            self.excluded_vert_ids = self.identity_model.no_correspondence_ids
        elif identity_model_type == "mhr":
            from .identity_model import MHRHandIdentityModel
            self.identity_model = MHRHandIdentityModel(
                data_root, False, hand_type=hand_type, output_unit=output_unit,
                **self.identity_model_kwargs)
        else:
            raise ValueError(
                f"Unknown identity_model_type '{identity_model_type}'. "
                "Supported: 'soma', 'mano', 'mhr'")

        # -- correctives (shared body checkpoint on the 25 hand joints) --------
        correctives_model_path = self.correctives_model_path
        self.correctives_model = None
        if correctives_model_path is not None:
            self.correctives_model = CorrectivesMLP.load_checkpoint(
                str(correctives_model_path), v_index_map=correctives_vertex_index_map,
                joint_indices=hand_joint_ids)
            if (self.correctives_model is not None
                    and output_unit is not Unit.METERS):   # checkpoint arrays are in metres
                s = Unit.METERS.meters_per_unit / output_unit.meters_per_unit
                self.correctives_model = _scale_correctives(self.correctives_model, s)

        # -- topology / provenance / rig ----------------------------------------
        self.hand_vert_ids = jnp.asarray(body_lod_vert_ids.astype(np.int64))
        self.hand_mid_vert_ids = jnp.asarray(mid_vert_ids)
        self.identity_lod_mid_ids = (None if identity_lod_mid_ids is None
                                     else jnp.asarray(identity_lod_mid_ids))
        self.xlo_skeleton_mid_to_low = (None if skeleton_lod_mid_ids is None
                                        else jnp.asarray(skeleton_lod_mid_ids))
        self.faces = jnp.asarray(np.asarray(hand_faces, np.int32))
        self.joint_parent_ids = parents_self             # upstream convention (root = 0)
        self.bind_pose_world = jnp.asarray(bind_w, jnp.float32)
        self.bind_pose_local = jnp.asarray(_world_to_local(bind_w, parents_int), jnp.float32)
        self.t_pose_world = jnp.asarray(t_w, jnp.float32)
        self.t_pose_local = jnp.asarray(_world_to_local(t_w, parents_int), jnp.float32)
        self.bind_shape = jnp.asarray(bind_shape, jnp.float32)
        self.skinning_weights = jnp.asarray(hand_W, jnp.float32)
        if mode == "warp":
            idx, val = topk_skinning(np.asarray(hand_W, np.float32), 8)
            self._weight_indices, self._weight_values = jnp.asarray(idx), jnp.asarray(val)
        else:
            self._weight_indices = self._weight_values = None
        self._correctives_to_hand_frame = jnp.asarray(wrist_inv[:3, :3], jnp.float32)
        self._t_pose_orient, self._t_pose_orient_parent_T = _orient_pair(
            self.t_pose_world, parents_self)

        hand_joint_names = [str(rig["joint_names"][gid]) for gid in hand_joint_ids]
        self.rig_data = {"joint_names": hand_joint_names}

        self._default_reference_pose = None
        if reference_pose is not None:
            if isinstance(reference_pose, dict):
                reference_pose = self.get_reference_pose(**reference_pose)
            self._default_reference_pose = jnp.asarray(validate_reference_pose(
                reference_pose, 25, require_identity_root=False), jnp.float32)

    # ------------------------------------------------------------------
    @property
    def default_skin_mesh_name(self) -> str:
        """Default USD skin-mesh prim name for this hand's topology."""
        side = "l" if self.hand_type == "left" else "r"
        suffix = {"mid": "mid", "low": "lo", "xlo": "xlo"}[self.lod]
        return f"{side}_hand_{suffix}"

    @property
    def num_shape_components(self) -> int:
        """Number of identity coefficients."""
        return self.identity_model.num_identity_coeffs

    # ---- reference poses (v0.3.1) -------------------------------------
    def list_reference_poses(self) -> list[dict]:
        """List reference keys and metadata in the current core npz."""
        return self._reference_pose_history.list_reference_poses()

    def get_reference_pose(self, reference_id: str | None = None, *, version: str | None = None,
                           data_key: str = "t_pose_world", asset_revision: str | None = None,
                           alias: str | None = None) -> jnp.ndarray:
        """Saved orientations for this hand, including the wrist.

        Select one of ``version``, ``asset_revision``, ``alias`` or
        ``reference_id``. Returns a fresh ``(25, 3, 3)`` array in this layer's
        joint order and current wrist bind frame; translations are not included.
        """
        reference_id = self._reference_pose_history.resolve_reference_id(
            reference_id, soma_version=version, data_key=data_key,
            asset_revision=asset_revision, alias=alias)
        body = self._reference_pose_history.get_reference_pose(
            reference_id, self._reference_body_joint_names, self._reference_body_parent_ids,
            dtype=jnp.float32)
        hand = body[jnp.asarray(self.hand_joint_ids_global)]
        # Elementwise products (not a matmul) as upstream does, to keep SO(3)
        # precision under reduced-precision matmul policies.
        return (self._correctives_to_hand_frame[None, :, :, None] * hand[:, None, :, :]).sum(axis=2)

    def convert_reference(self, rotations, from_ref, to_ref) -> jnp.ndarray:
        """Re-express ``(B, 25, 3, 3)`` rotations in another reference.

        References are explicit arrays or :meth:`get_reference_pose` dicts in this
        layer's wrist bind frame, wrist included; the constructor default is not
        used. Pass the result to ``pose(..., pose2rot=False, reference_pose=to_ref)``.
        """
        if isinstance(from_ref, dict):
            from_ref = self.get_reference_pose(**from_ref)
        if isinstance(to_ref, dict):
            to_ref = self.get_reference_pose(**to_ref)
        return convert_reference_rotations(rotations, from_ref, to_ref, self.joint_parent_ids,
                                           virtual_root=False)

    # ---- identity ---------------------------------------------------------
    def _apply_lod_transfer(self, mid_rest_shape: jnp.ndarray) -> jnp.ndarray:
        if self.identity_lod_mid_ids is not None:
            return mid_rest_shape[:, self.identity_lod_mid_ids, :]
        if self.identity_lod_transfer is not None:
            return _barycentric_apply(self.identity_lod_transfer, mid_rest_shape)
        return mid_rest_shape

    def get_rest_shape(self, identity_coeffs, scale_params=None, global_scale=1.0,
                       kwargs=None) -> jnp.ndarray:
        """(B, Vh, 3) wrist-local rest shape in ``output_unit``."""
        mid = self.identity_model(jnp.asarray(identity_coeffs), scale_params=scale_params,
                                  kwargs=kwargs, global_scale=global_scale)
        return self._apply_lod_transfer(mid)

    def _skin(self, bind, rest, rotations_abs, translation, local_scales=None, fk_only=False):
        return pose_from_bind(
            bind, rest, self.skinning_weights, self._levels, self._parents_int,
            rotations_abs, translation, hips_idx=0,
            weight_values=self._weight_values, weight_indices=self._weight_indices,
            local_translation_scales=local_scales, skip_lbs=fk_only)

    def prepare_identity(self, identity_coeffs, scale_params=None, repose_to_bind_pose: bool = True,
                         global_scale=1.0, kwargs=None) -> SOMAHandIdentity:
        """Rest shape and fitted skeleton for an identity (upstream ``prepare_identity``).

        Args:
            identity_coeffs: (B, K) identity coefficients.
            scale_params: SOMA (B, 24) — kept for :meth:`pose`; MHR (B, 26) —
                consumed here; MANO unused.
            repose_to_bind_pose: rebind to the template bind pose after fitting.
            global_scale: uniform scale scalar or (B,) array.
            kwargs: forwarded to the identity model.

        Returns:
            The prepared :class:`SOMAHandIdentity`.
        """
        identity_rest = self.identity_model(jnp.asarray(identity_coeffs), scale_params=scale_params,
                                            kwargs=kwargs, global_scale=global_scale)
        hand_rest = self._apply_lod_transfer(identity_rest)
        skeleton_rest = correctives_rest = hand_rest
        if self.xlo_skeleton_mid_to_low is not None:
            skeleton_rest = correctives_rest = identity_rest[:, self.xlo_skeleton_mid_to_low, :]
        bind = self.skeleton_transfer.fit(skeleton_rest)                   # (B, 25, 4, 4)
        rest = hand_rest
        if repose_to_bind_pose:
            B = bind.shape[0]
            rot = jnp.broadcast_to(self.bind_pose_local[None, :, :3, :3], (B, 25, 3, 3))
            trans = jnp.broadcast_to(self.bind_pose_local[None, self.root_joint_idx, :3, 3], (B, 3))
            rest, bind = self._skin(bind, rest, rot, trans)
        return SOMAHandIdentity(
            rest_shape=rest, bind_transforms_world=bind,
            scale_params=(None if scale_params is None or self.identity_model_type != "soma"
                          else jnp.atleast_2d(jnp.asarray(scale_params))),
            global_scale=global_scale, correctives_rest_shape=correctives_rest)

    def _apply_joint_orient(self, poses_rot_relative: jnp.ndarray) -> jnp.ndarray:
        """T-pose-relative -> absolute local skinning rotations (pair form)."""
        return _apply_orient(poses_rot_relative, self._t_pose_orient, self._t_pose_orient_parent_T)

    def _full_bone_scales(self, bone_scales: jnp.ndarray) -> jnp.ndarray:
        """(B, 24) scales for joints 1-24 -> (B, 25) with the wrist at 1."""
        B = bone_scales.shape[0]
        return jnp.concatenate([jnp.ones((B, 1), bone_scales.dtype), bone_scales], axis=1)

    # ---- posing -----------------------------------------------------------
    def pose(self, poses, identity: SOMAHandIdentity | None = None, pose2rot: bool = True,
             apply_correctives: bool = False, absolute_pose: bool = False,
             global_translation=None, fk_only: bool = False, *,
             reference_pose=None) -> SOMAHandPoseOutput:
        """Pose a prepared identity (upstream ``pose``).

        Args:
            poses: (B, 25, 3) axis-angle, or (B, 25, 3, 3) with ``pose2rot=False``.
            identity: from :meth:`prepare_identity`.
            apply_correctives: add the shared body correctives (hand joints).
            absolute_pose: rotations are absolute (skip the T-pose joint orient).
            global_translation: (B, 3) or (3,) wrist translation; default origin.
            fk_only: forward kinematics only.
            reference_pose: ``None`` uses the constructor default, else the
                current T-pose; otherwise a :meth:`get_reference_pose` dict or a
                ``(25, 3, 3)`` / ``(25, 4, 4)`` array in this hand's joint order
                and wrist bind frame. Rejected with ``absolute_pose=True``.

        Returns:
            ``vertices`` (B, Vh, 3; omitted when ``fk_only``), ``joints``
            (B, 25, 3), ``transforms`` (B, 25, 4, 4), all in ``output_unit``.
        """
        if identity is None:
            raise RuntimeError(
                "No identity: pass the result of prepare_identity() (JAX layers keep no cache).")
        if reference_pose is not None and absolute_pose:
            raise ValueError("reference_pose cannot be combined with absolute_pose=True.")
        if reference_pose is None and not absolute_pose:
            reference_pose = self._default_reference_pose
        if isinstance(reference_pose, dict):
            reference_pose = self.get_reference_pose(**reference_pose)
        if reference_pose is not None:
            reference_pose = validate_reference_pose(reference_pose, 25, require_identity_root=False)

        poses = jnp.asarray(poses)
        B = poses.shape[0]
        if pose2rot:
            poses_rot = batch_rodrigues(poses.reshape(-1, 3)).reshape(B, 25, 3, 3)
        else:
            poses_rot = poses.reshape(B, 25, 3, 3)
        if reference_pose is not None:
            orient, parent_t = _orient_pair(jnp.asarray(reference_pose, poses_rot.dtype),
                                            self.joint_parent_ids)
            poses_rot = _apply_orient(poses_rot, orient, parent_t)
            absolute_pose = True
        rotations_abs = poses_rot if absolute_pose else self._apply_joint_orient(poses_rot)

        if global_translation is None:
            global_translation = jnp.zeros((B, 3), poses_rot.dtype)
        global_translation = jnp.broadcast_to(jnp.asarray(global_translation, poses_rot.dtype),
                                              (B, 3))

        # SOMA backend: bone-length scales stretch the local translations.
        local_scales = None
        if identity.scale_params is not None and isinstance(self.identity_model,
                                                            SOMAHandIdentityModel):
            local_scales = self._full_bone_scales(identity.scale_params)

        rest = identity.rest_shape
        if apply_correctives and not fk_only:
            if self.correctives_model is None:
                raise RuntimeError(
                    "apply_correctives=True but no corrective model is loaded. Construct with "
                    "a valid correctives_model_path or pass apply_correctives=False.")
            out = self.correctives_model.offsets(rotations_abs)          # body frame
            out = jnp.einsum("ij,bvj->bvi", self._correctives_to_hand_frame, out)
            gs = identity.global_scale
            if isinstance(gs, (jnp.ndarray, np.ndarray)):
                out = out * jnp.asarray(gs).reshape(-1, 1, 1)
            elif gs != 1.0:
                out = out * gs
            if self.correctives_lod_transfer is not None:
                out = _barycentric_apply(self.correctives_lod_transfer,
                                         identity.correctives_rest_shape + out) - rest
            rest = rest + out

        vertices, T_world = self._skin(identity.bind_transforms_world, rest, rotations_abs,
                                       global_translation, local_scales, fk_only=fk_only)
        if fk_only:
            return SOMAHandPoseOutput(joints=T_world[..., :3, 3], transforms=T_world)
        return SOMAHandPoseOutput(vertices=vertices, joints=T_world[..., :3, 3],
                                  transforms=T_world)

    def forward(self, poses, identity_coeffs, pose2rot: bool = True,
                apply_correctives: bool = False, absolute_pose: bool = False,
                global_translation=None, global_scale=1.0, scale_params=None,
                kwargs=None, *, reference_pose=None) -> SOMAHandPoseOutput:
        """Combined :meth:`prepare_identity` + :meth:`pose` (upstream ``forward``).

        Like upstream, the identity is reposed to the bind pose only when
        correctives are applied.
        """
        if reference_pose is not None and absolute_pose:
            raise ValueError("reference_pose cannot be combined with absolute_pose=True.")
        if isinstance(reference_pose, dict):
            reference_pose = self.get_reference_pose(**reference_pose)
        identity = self.prepare_identity(identity_coeffs, scale_params=scale_params,
                                         repose_to_bind_pose=apply_correctives,
                                         global_scale=global_scale, kwargs=kwargs)
        return self.pose(poses, identity, pose2rot=pose2rot, apply_correctives=apply_correctives,
                         absolute_pose=absolute_pose, global_translation=global_translation,
                         reference_pose=reference_pose)

    __call__ = forward


def _scale_correctives(model: CorrectivesMLP, scale: float) -> CorrectivesMLP:
    """Rescale a metre-unit corrective model's output layer to another unit."""
    import equinox as eqx
    return eqx.tree_at(lambda m: m.W2, model, model.W2 * scale)
