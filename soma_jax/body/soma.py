"""SOMALayer: main entry point for SOMA-JAX body model.

SOMALayer wraps the full pipeline:
  1. Identity model → rest vertices + skeleton
  2. FK (forward kinematics) on the skeleton
  3. Corrective MLP → pose-dependent displacements
  4. LBS → posed vertices

Usage::

    import numpy as np
    import jax.numpy as jnp
    from soma_jax import SOMALayer, SOMAParams

    layer = SOMALayer.load("path/to/SOMA_neutral.npz")
    rest_verts, rest_joints = layer.prepare_identity(identity_coeffs)
    params = SOMAParams(
        poses=jnp.eye(3)[None, None].repeat(B * J, axis=0).reshape(B, J, 3, 3),
        transl=jnp.zeros((B, 3)),
        identity_coeffs=identity_coeffs,
    )
    output = layer(params)

Upstream: ``soma/body/soma.py :: SOMALayer``
    Forward pass is a faithful port: identity blend -> skeleton fit -> repose ->
    joint-orient remap -> FK + LBS, pinned by tests/test_layer_parity.py to
    3.2e-6 m (mid LOD) and 2.4e-6 m (low LOD) — both LBS-only with
    identity_model_type="soma". **Constructor defaults differ from upstream**
    (upstream defaults to MHR + Warp + procedural transforms + a real
    corrective checkpoint), and `rebind()` updates only `v_template`, leaving
    the identity model and skeleton-transfer caches stale.
"""
from __future__ import annotations
import copy
from typing import Optional
from dataclasses import dataclass
import logging
import warnings
from typing import TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from ..types import SOMAParams, SOMAOutput
from ..units import Unit
from ..geometry.transforms import rotation_6d_to_rotmat
from ..geometry.lbs import (
    batch_rodrigues,
    forward_kinematics,
    lbs_transforms,
    lbs_blend,
    lbs_sparse,
    compute_skeleton_levels,
)
from ..geometry.rig_utils import (
    apply_joint_orient_local,
    joint_world_to_local as _joint_world_to_local,
    remove_joint_orient_local as _remove_joint_orient_local,
)
from ..correctives_model import _DEFAULT_CORRECTIVES_MODEL_PATH, CorrectivesMLP
from ..identity_packs import BaseIdentityModel, create_identity_model
from ..io import SOMA_TEMPLATE_RIG_FILENAME, SOMA_XLO_TEMPLATE_RIG_FILENAME  # noqa: F401
from ..procedural_transforms import SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME  # noqa: F401
from ..reference_poses import (
    ReferencePoseHistory,
    convert_reference_rotations,
    validate_reference_pose,
)


#: Body levels of detail, as upstream ``BODY_LODS``: 18,056 / 4,505 / 612 vertices.
BODY_LODS = ("mid", "low", "xlo")

#: Sentinel for keyword aliases that were not passed.
_UNSET = object()

#: ``sparse_k`` that keeps every skinning influence (upstream's dense mode).
_ALL_INFLUENCES = 1 << 30


def _resolve_body_lod(low_lod: bool, lod: Optional[str]) -> str:
    """Resolve legacy ``low_lod`` and the explicit body LOD (upstream ``_resolve_body_lod``)."""
    if lod is None:
        return "low" if low_lod else "mid"
    lod = lod.lower()
    if lod not in BODY_LODS:
        raise ValueError(f"lod must be one of {BODY_LODS}, got {lod!r}")
    if low_lod and lod != "low":
        raise ValueError("low_lod=True is only compatible with lod='low'")
    return lod


def _nearest_lod_vertex_ids(source_vertices, target_vertices, source_vertex_ids) -> np.ndarray:
    """Map source vertex IDs to nearest target vertex IDs and drop duplicates.

    Port of upstream ``_nearest_lod_vertex_ids`` (``soma/body/soma.py``): how an
    xlo layer carries the mid-LOD facial inner-geometry exclusion over to its
    own mesh, which shares no vertex indexing with the mid mesh.
    """
    source_vertex_ids = np.asarray(source_vertex_ids, np.int64)
    if source_vertex_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(target_vertices))
    _, nearest = tree.query(np.asarray(source_vertices)[source_vertex_ids])
    return np.unique(nearest).astype(np.int64)


def _mid_ids_to_lod(mid_ids, mid_to_low, n_mid: int) -> np.ndarray:
    """Remap mid-LOD vertex ids into a ``mid_to_low`` subset, dropping absentees.

    Upstream's ``inverse_lod_map[ids]`` followed by ``[>= 0]``.
    """
    inv = np.full((int(n_mid),), -1, dtype=np.int64)
    mid_to_low = np.asarray(mid_to_low, np.int64)
    inv[mid_to_low] = np.arange(mid_to_low.shape[0], dtype=np.int64)
    mapped = inv[np.asarray(mid_ids).astype(np.int64).ravel()]
    return mapped[mapped >= 0]


def _slice_rig_to_low_lod(soma_data: dict) -> dict:
    """Return a copy of ``soma_data`` restricted to the low-LOD vertex subset.

    Mirrors upstream ``SOMALayer(lod="low")``
    (``third_party/SOMA-X/soma/soma.py``): every per-vertex array is indexed by
    ``lod_mid_to_low``, faces are replaced by ``triangles_low``, and the facial
    inner-geometry exclusion lists are remapped from mid- into low-LOD indices
    with entries outside the subset dropped. Because the SkeletonTransfer and
    the identity model are both built from this dict, they are rebuilt *on* the
    low-LOD mesh — which is what upstream does, and what makes the result a
    consistent layer rather than a mesh subset bolted onto a full-res rig.
    """
    if "lod_mid_to_low" not in soma_data or "triangles_low" not in soma_data:
        raise RuntimeError(
            "lod='low' requires 'lod_mid_to_low' and 'triangles_low' in the "
            "asset; rebuild it from the upstream rig (docs/INSTALL.md §4.2)."
        )
    idx = np.asarray(soma_data["lod_mid_to_low"], dtype=np.int64)
    n_mid = int(np.asarray(soma_data["v_template"]).shape[0])

    # mid -> low inverse map; -1 where a mid vertex has no low counterpart.
    inv = np.full((n_mid,), -1, dtype=np.int64)
    inv[idx] = np.arange(idx.shape[0], dtype=np.int64)

    out = dict(soma_data)
    for key in ("v_template", "weights", "shapedirs", "bind_shape"):
        if key in out:
            out[key] = np.asarray(out[key])[idx]
    if out.get("J_regressor") is not None:
        out["J_regressor"] = np.asarray(out["J_regressor"])[:, idx]
    out["faces"] = np.asarray(soma_data["triangles_low"])

    for seg in ("segment_eye_bags", "segment_mouth_bag"):
        if seg in out:
            out[seg] = _mid_ids_to_lod(out[seg], idx, n_mid)

    if "mirror_vert_indices" in out:
        # Remap through the subset; vertices whose mirror is absent map to
        # themselves, which is what a symmetric subset degrades to.
        mirror = np.asarray(out["mirror_vert_indices"]).astype(np.int64)[idx]
        mapped = inv[np.clip(mirror, 0, n_mid - 1)]
        out["mirror_vert_indices"] = np.where(
            mapped >= 0, mapped, np.arange(idx.shape[0], dtype=np.int64))

    # Vertex count of the mesh this subset came from. `lod_mid_to_low` cannot
    # be trusted to reveal it: in the shipped SOMA rig it is exactly
    # `arange(4505)` (the mid mesh is ordered with the low-LOD subset leading),
    # so `max() + 1` gives the LOW count, not the mid one.
    out["lod_mid_num_verts"] = np.asarray(n_mid, dtype=np.int64)

    # `lod_mid_to_low` is deliberately KEPT: a low-LOD layer uses it to accept
    # full-resolution SOMA vertices and subsample them (upstream's
    # `_soma_full_num_verts` path, used by SOMAPoseInversion). `triangles_low`
    # has already been applied as `faces`, so it would be redundant.
    out.pop("triangles_low", None)
    return out


def _scale_correctives(offsets: jnp.ndarray, global_scale) -> jnp.ndarray:
    """Scale corrective offsets by the identity's global scale.

    Mirrors upstream ``SOMALayer.pose``: correctives are learned in unscaled
    units, so a globally-scaled identity needs its offsets scaled to match.
    Accepts a scalar or a per-batch array.
    """
    gs = jnp.asarray(global_scale, dtype=offsets.dtype)
    if gs.ndim == 0:
        return offsets * gs
    return offsets * gs.reshape(-1, 1, 1)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..procedural_transforms import ProceduralTransforms


class _HostArray:
    """Holds a numpy array that is deliberately *static* on an eqx.Module.

    These arrays are host-side rig structure (joint parents, bind-pose locals,
    LOD index maps) consumed by NumPy control flow at trace time, not traced
    data — so they must be static. Storing them bare would make equinox warn
    ("A JAX array is being set as static"), because ``equinox.is_array`` counts
    NumPy arrays too and a bare array is its own pytree leaf. Wrapping puts a
    non-array object at the leaf, which is the accurate description: this is
    structure, not a tensor.

    Hashed and compared by identity, which is what a static field needs (NumPy's
    elementwise ``__eq__`` would otherwise return an array).
    """

    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return f"_HostArray(shape={getattr(self.a, 'shape', None)})"


def _host(a):
    """Wrap a numpy array for a static field, passing None straight through."""
    return None if a is None else _HostArray(a)


class _BuildRecipe:
    """How a layer was constructed, so it can be rebuilt at another LOD.

    Upstream's ``PoseInversion(soma, low_lod=True)`` constructs a second,
    low-LOD ``SOMALayer`` from the original's ``data_root``, identity backend,
    ``output_unit``, ``identity_model_kwargs``, template rig and procedural
    flag. SOMA-JAX layers do not keep those constructor arguments, so the
    factories record them here. Static and hashed by identity, like
    :class:`_HostArray`.
    """

    __slots__ = ("factory", "kwargs")

    def __init__(self, factory: str, kwargs: dict):
        self.factory = factory
        self.kwargs = kwargs

    def __repr__(self):
        return f"_BuildRecipe({self.factory})"


class _LazyCache:
    """Memo for derived host-side views of one layer (static, hashed by identity)."""

    __slots__ = ("values",)

    def __init__(self):
        self.values = {}


class SOMAPoseOutput(dict):
    """Upstream ``SOMAPoseOutput``: :meth:`SOMALayer.forward`'s result.

    A ``dict`` (``out["vertices"]``) with attribute access (``out.vertices``).
    ``vertices`` is absent with ``fk_only``; ``joints`` and ``transforms`` are
    always present.
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


@dataclass(frozen=True)
class SOMAPublicRigView:
    """Public-joint view of a SOMA target skinning rig (upstream's dataclass).

    Procedural layers keep an expanded target skeleton for LBS, but their
    public API is the 78-joint SOMA body skeleton; this packages the public
    hierarchy, fitted bind transforms and target-weight folding. Items can also
    be read dict-style (``view["joint_names"]``), as SOMA-JAX's earlier dict
    return allowed.
    """

    joint_names: tuple
    joint_parent_ids: np.ndarray
    target_joint_indices: np.ndarray
    target_to_public_joint_indices: np.ndarray
    bind_transforms_world: jnp.ndarray
    bind_transforms_local: jnp.ndarray
    t_pose_world: jnp.ndarray
    skinning_weights: jnp.ndarray

    def __getitem__(self, key: str):
        return getattr(self, key)


def _target_to_public_map(target_names, parents, public_names) -> np.ndarray:
    """For each expanded-rig joint, the public joint whose bone scale it follows.

    Port of upstream's ``target_to_public_joint_indices``. Public joints map to
    themselves; procedural and helper joints inherit from the nearest public
    ancestor so a stretched bone carries its whole subtree.
    """
    public_at = {n: i for i, n in enumerate(public_names)}
    out = np.zeros(len(target_names), np.int32)
    for j in range(len(target_names)):
        a = j
        while target_names[a] not in public_at and 0 <= int(parents[a]) != a:
            a = int(parents[a])
        out[j] = public_at.get(target_names[a], 0)
    return out


def _rig_transforms_to_metres(transforms) -> np.ndarray:
    """Scale a (J, 4, 4) rig transform's translation column into metres.

    ``rig_build`` keeps the USD/npz transforms in their native centimetres (only
    ``v_template``/``shapedirs`` are converted), while the per-identity
    ``bind_transforms`` the layer composes them with are metres. Only the
    translation column is a length; the rotation block is unitless.
    """
    out = np.asarray(transforms, np.float32).copy()
    out[..., :3, 3] *= Unit.CENTIMETERS.meters_per_unit
    return out


class SOMALayer(eqx.Module):
    """SOMA body model layer — universal human body pivot in JAX.

    Supports five identity model types (SMPL/SMPL-X, MHR, Anny, SOMA, GarmentMeasurement)
    and produces fully posed meshes via linear blend skinning with pose correctives.

    Attributes:
        v_template: (V, 3) SOMA neutral rest template.
        faces: (F, 3) triangle face indices.
        J_regressor: (J, V) joint position regressor.
        weights: (V, J) LBS skinning weights.
        joint_names: list of J joint name strings.
        correctives: CorrectivesMLP for pose-dependent deformations, or
            ``None`` when no checkpoint was loaded.
        identity_model: active BaseIdentityModel.
    """

    v_template: jnp.ndarray          # (V, 3)
    faces: jnp.ndarray                # (F, 3)
    J_regressor: Optional[jnp.ndarray]  # (J, V); None when not fitted (skeleton_fit="linear" off)
    weights: jnp.ndarray              # (V, J)
    weight_indices: Optional[jnp.ndarray]  # (V, K) sparse top-K indices
    weight_values: Optional[jnp.ndarray]   # (V, K) sparse top-K weights
    joint_names: list = eqx.field(static=True)
    skeleton_levels: list = eqx.field(static=True)
    correctives: Optional[CorrectivesMLP]  # None without a checkpoint, as upstream
    identity_model: BaseIdentityModel = eqx.field(static=True)
    _parents_host: _HostArray = eqx.field(static=True)
    # The PUBLIC rig's T-pose / bind-pose world transforms (native units), or
    # None. Upstream's `t_pose_world` / `bind_pose_world` are the skinning
    # (target) rig's — the properties of those names below.
    _public_t_pose_world: Optional[jnp.ndarray]    # (J, 4, 4)
    _public_bind_pose_world: Optional[jnp.ndarray]  # (J, 4, 4)
    _bind_pose_local_host: Optional[_HostArray] = eqx.field(static=True)  # numpy mirror used by _repose_to_bind_pose
    _lod_mid_to_low_host: Optional[_HostArray] = eqx.field(static=True)   # (V_low,) vertex subset for low-LOD
    _lod_mid_num_verts: Optional[int] = eqx.field(static=True)            # source mesh size, low-LOD layers only
    _has_trained_correctives: bool = eqx.field(static=True)               # False when no checkpoint was loaded
    _triangles_low_host: Optional[_HostArray] = eqx.field(static=True)    # (F_low, 3) face indices into the subset
    #: Body LOD this layer skins: ``"mid"``, ``"low"`` or ``"xlo"`` (upstream ``lod``).
    _lod: str = eqx.field(static=True)
    # Upstream-named views (see the properties of the same names).
    _num_shape_components: Optional[int] = eqx.field(static=True)
    _correctives_model_path: Optional[str] = eqx.field(static=True)
    _xlo_skeleton_mid_to_low_host: Optional[_HostArray] = eqx.field(static=True)
    #: Constructor arguments recorded by :meth:`load` / :meth:`from_upstream_assets`
    #: (see :class:`_BuildRecipe`); ``None`` for a layer built any other way.
    _build_recipe: Optional[_BuildRecipe] = eqx.field(static=True)
    _lazy: _LazyCache = eqx.field(static=True)
    #: xlo only — upstream's ``identity_lod_transfer``: the barycentric embedding
    #: of the xlo mesh in the low-LOD mesh, as ``(low_faces, face_ids, bary)``.
    #: The identity model and skeleton fit run on the low LOD and this carries
    #: their rest shape (and the low-LOD correctives) onto the xlo vertices.
    _lod_transfer_host: Optional[_HostArray] = eqx.field(static=True)
    #: Upstream ``excluded_vert_ids``: the facial inner geometry (eye bags +
    #: mouth bag) in *this layer's* mesh indexing. Pose inversion excludes it.
    _excluded_vert_ids_host: Optional[_HostArray] = eqx.field(static=True)
    # SOMA-X's per-identity skeleton fit (RBF joint regression + two-stage
    # Kabsch). Built when the asset carries ``bind_shape`` + ``bind_pose_world``;
    # ``prepare_identity(skeleton_fit="full")`` then matches upstream
    # ``SOMALayer.prepare_identity`` exactly. None on legacy assets (the linear
    # J_regressor alternative is used instead).
    skeleton_transfer: Optional[object] = eqx.field(static=True)
    # Public-joint bone-scale controls (SOMA-X's `scale_params` for the SOMA
    # identity backend): the ordered active child joints and their (parent,
    # child) local-translation edges.
    _bone_scale_joint_indices_host: Optional[_HostArray] = eqx.field(static=True)
    #: The SOMA bone-length controls (upstream ``soma_bone_scale_param_names`` /
    #: ``_segments``): 60 public joints on the stock rig.
    bone_scale_param_names: tuple = eqx.field(static=True)
    bone_scale_param_segments: tuple = eqx.field(static=True)
    #: Upstream ``scale_param_names`` / ``scale_param_segments``: the bone
    #: controls on the SOMA backend, the identity model's own controls (and no
    #: segments) on every other backend.
    scale_param_names: tuple = eqx.field(static=True)
    scale_param_segments: tuple = eqx.field(static=True)
    #: Upstream ``identity_model_type``: which backend feeds ``identity_coeffs``.
    identity_model_type: str = eqx.field(static=True)
    #: Upstream ``output_unit``: the unit of every translational input and
    #: output (vertices, joints, transform translations, ``transl``). The layer
    #: computes in metres and converts at the boundary.
    output_unit: Unit = eqx.field(static=True)

    # ---- procedural (expanded twist) rig ---------------------------------
    # Upstream's default rig is the expanded template skeleton — 110 joints on
    # the v0027 template (78 public + 32 twist; v0026 added 12 USD-only helpers
    # for 122) — but its *public* pose contract stays at 78 joints: the 32 twist
    # rotations are derived from the public ones through the procedural
    # parameter matrix. On the two-rig layer `from_upstream_assets` builds,
    # `joint_names` stays the 78-joint public rig (identity, skeleton fit, bind)
    # and the expanded FK/LBS skeleton lives in `_skin_rig`.
    _procedural: Optional[object] = eqx.field(static=True)
    _public_idx_host: Optional[_HostArray] = eqx.field(static=True)
    _public_names: tuple = eqx.field(static=True)
    #: The expanded skinning rig used for FK/LBS only, as a plain dict of host
    #: arrays: ``joint_names``, ``parents``, ``levels``, ``weights``,
    #: ``weight_values``, ``weight_indices``, ``t_pose_world``, ``public_idx``,
    #: ``twist_idx``. ``None`` on a single-rig layer.
    _skin_rig: Optional[dict] = eqx.field(static=True)

    # ---- reference poses (SOMA-X v0.3.1) --------------------------------
    #: Historical reference T-poses read from the core npz (empty on assets
    #: that predate v0.3). Host-side and never traced, so static.
    _reference_pose_history: object = eqx.field(static=True)
    #: Constructor-default reference: (J, 3, 3) public world rotations
    #: including the identity virtual Root, or None for the current T-pose.
    _default_reference_pose: Optional[jnp.ndarray]

    # Public joints whose local translation `scale_params` may stretch.
    # SOMA-X v0.2.2 appended four native foot controls to the original 56;
    # legacy (B, 56) tensors are still accepted and get unit foot scales.
    LEGACY_NUM_BONE_SCALE_PARAMS = 56
    NUM_BONE_SCALE_PARAMS = 60
    BODY_BONE_SCALE_JOINT_NAMES = (
        "LeftArm", "LeftForeArm", "LeftHand",
        "RightArm", "RightForeArm", "RightHand",
        "LeftShin", "RightShin",
    )
    FINGER_BONE_SCALE_JOINT_PREFIXES = (
        "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle",
        "LeftHandRing", "LeftHandPinky",
        "RightHandThumb", "RightHandIndex", "RightHandMiddle",
        "RightHandRing", "RightHandPinky",
    )
    FOOT_BONE_SCALE_JOINT_NAMES = (
        "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
    )
    #: Upstream ``root_joint_idx``: Hips, the child of the virtual Root.
    root_joint_idx = 1

    # ---- host-side rig structure (see _HostArray) ---------------------------
    @property
    def _parents_np(self):
        return self._parents_host.a

    @property
    def _bind_pose_local_np(self):
        h = self._bind_pose_local_host
        return None if h is None else h.a

    @property
    def _lod_mid_to_low_np(self):
        h = self._lod_mid_to_low_host
        return None if h is None else h.a

    @property
    def _triangles_low_np(self):
        h = self._triangles_low_host
        return None if h is None else h.a

    @property
    def _unit_scale(self) -> float:
        """Metres -> ``output_unit``."""
        return Unit.METERS.meters_per_unit / self.output_unit.meters_per_unit

    @property
    def lod(self) -> str:
        """Body level of detail: ``"mid"`` (18,056 verts), ``"low"`` (4,505) or ``"xlo"`` (612)."""
        return self._lod

    @property
    def low_lod(self) -> bool:
        """Upstream's legacy flag: ``lod == "low"``."""
        return self._lod == "low"

    @property
    def procedural_transforms_enabled(self) -> bool:
        """Upstream's flag: FK/LBS run on the expanded procedural twist rig."""
        return self._procedural is not None

    @property
    def excluded_vert_ids(self) -> Optional[np.ndarray]:
        """Facial inner-geometry vertex ids in this layer's mesh (upstream ``excluded_vert_ids``).

        Eye bags + mouth bag: mid-LOD ids on a mid layer, remapped into the
        subset on a low layer, and the nearest xlo vertices on an xlo layer.
        """
        h = self._excluded_vert_ids_host
        return None if h is None else h.a

    @property
    def default_skin_mesh_name(self) -> str:
        """USD skin-mesh prim name for this topology (upstream ``default_skin_mesh_name``)."""
        return {"mid": "c_skin_mid", "low": "c_skin_lo", "xlo": "c_skin_xlo"}[self._lod]

    # -- upstream attribute names --------------------------------------------
    @property
    def num_shape_components(self) -> int:
        """Upstream ``num_shape_components``: the SOMA shape-PCA size (128),
        whatever the identity backend."""
        if self._num_shape_components is not None:
            return self._num_shape_components
        return int(self.identity_model.num_identity_coeffs)

    @property
    def correctives_model(self):
        """Upstream ``correctives_model``: the loaded checkpoint, or ``None``."""
        return self.correctives if self._has_trained_correctives else None

    @property
    def correctives_model_path(self):
        """Upstream ``correctives_model_path``: the checkpoint loaded, or ``None``."""
        from pathlib import Path as _Path
        return None if self._correctives_model_path is None else _Path(self._correctives_model_path)

    @property
    def nv_lod_mid_to_low(self) -> Optional[np.ndarray]:
        """Upstream ``nv_lod_mid_to_low``: the mid-LOD ids of a low layer's vertices."""
        return self._lod_mid_to_low_np if self._lod == "low" else None

    @property
    def xlo_skeleton_mid_to_low(self) -> Optional[np.ndarray]:
        """Upstream ``xlo_skeleton_mid_to_low``: an xlo layer's low-LOD skeleton-fit
        mesh, as mid-LOD ids."""
        h = self._xlo_skeleton_mid_to_low_host
        return None if h is None else h.a

    @property
    def identity_lod_transfer(self):
        """Upstream ``identity_lod_transfer``: an xlo layer's low -> xlo transfer
        (``(faces, face_ids, bary)`` here), else ``None``."""
        h = self._lod_transfer_host
        return None if (self._lod != "xlo" or h is None) else h.a

    @property
    def xlo_skeleton_transfer(self):
        """Upstream ``xlo_skeleton_transfer``: an xlo layer's (low-LOD) skeleton fit."""
        return self.skeleton_transfer if self._lod == "xlo" else None

    @property
    def soma_bone_scale_param_names(self) -> tuple:
        """Upstream ``soma_bone_scale_param_names``: the SOMA bone-scale layout."""
        return self.bone_scale_param_names

    @property
    def soma_bone_scale_param_segments(self) -> tuple:
        """Upstream ``soma_bone_scale_param_segments``."""
        return self.bone_scale_param_segments

    @property
    def _bone_scale_joint_indices(self):
        h = self._bone_scale_joint_indices_host
        return None if h is None else h.a

    # -- upstream's structural attributes (read-only views) --------------------
    # Upstream registers these as buffers of its mutable layer. Here they are
    # derived from the immutable rig on access. On a procedural layer the
    # "target" rig is the expanded twist skeleton FK/LBS run on; SOMA-JAX's own
    # `joint_names`, `weights`, `t_pose_world` and `bind_pose_world` describe the
    # public 78-joint rig, where upstream's `t_pose_world` / `bind_pose_world`
    # buffers describe the target rig (see `public_rig_view`).
    @property
    def target_joint_names(self) -> tuple[str, ...]:
        """Upstream ``target_joint_names``: the joints FK/LBS run on."""
        rig = self._skin_rig
        if rig is not None:
            return tuple(str(n) for n in rig["joint_names"])
        return tuple(str(n) for n in self.joint_names)

    @property
    def joint_parent_ids(self) -> np.ndarray:
        """Upstream ``joint_parent_ids``: parents of :attr:`target_joint_names`,
        the root self-parented as upstream's rig loader writes it."""
        rig = self._skin_rig
        p = np.asarray(rig["parents"].a if rig is not None else self._parents_np, np.int64)
        return np.where(p < 0, np.arange(len(p)), p)

    @property
    def parents(self) -> list[int]:
        """Upstream ``parents``: :attr:`joint_parent_ids` without the virtual
        Root, every index shifted down by one."""
        return [int(i) - 1 for i in self.joint_parent_ids][1:]

    @property
    def skinning_weights(self) -> jnp.ndarray:
        """Upstream ``skinning_weights``: dense (V, J_target) weights."""
        rig = self._skin_rig
        return jnp.asarray(rig["weights"]) if rig is not None else self.weights

    @property
    def public_transform_joint_indices(self) -> np.ndarray:
        """Upstream ``public_transform_joint_indices``: each public joint's
        index in :attr:`target_joint_names`."""
        rig = self._skin_rig
        if rig is not None:
            return np.asarray(rig["public_idx"].a, np.int64)
        if self._public_idx is not None:
            return np.asarray(self._public_idx, np.int64)
        return np.arange(len(self.public_joint_names), dtype=np.int64)

    @property
    def public_joint_indices(self) -> np.ndarray:
        """Upstream ``public_joint_indices``: target indices of the 77 posable
        public joints (the virtual Root excluded)."""
        return self.public_transform_joint_indices[1:]

    @property
    def public_joint_parent_ids(self) -> np.ndarray:
        """Upstream ``public_joint_parent_ids``: parents of :attr:`public_joint_names`."""
        return np.asarray(self.output_joint_parent_ids, np.int64)

    @property
    def target_to_public_joint_indices(self) -> np.ndarray:
        """Upstream ``target_to_public_joint_indices``: the public joint each
        target joint folds onto (itself, else its nearest public ancestor)."""
        rig = self._skin_rig
        if rig is not None:
            return np.asarray(rig["target_to_public"].a, np.int64)
        return np.arange(len(self.public_joint_names), dtype=np.int64)

    @property
    def bone_scale_public_joint_indices(self) -> np.ndarray:
        """Upstream ``bone_scale_public_joint_indices``: the public joints the
        SOMA bone-scale parameters stretch, in parameter order."""
        ids = self._bone_scale_joint_indices
        return np.zeros(0, np.int64) if ids is None else np.asarray(ids, np.int64)

    @property
    def facial_inner_geometry(self) -> Optional[np.ndarray]:
        """Upstream's backward-compatible alias of :attr:`excluded_vert_ids`."""
        return self.excluded_vert_ids

    @property
    def procedural_transform_definition(self):
        """Upstream ``procedural_transform_definition``: the parsed JSON, or ``None``.

        Upstream parses the packaged definition whenever it exists — on a
        legacy (``procedural=False``) layer too, where it still names the
        public joints.
        """
        if self._procedural is not None:
            return self._procedural.definition
        path = self._asset_path("SOMA_procedural_transforms.json")
        if path is None or not path.exists():
            return None
        cache = self._lazy.values
        if "procedural_transform_definition" not in cache:
            from ..procedural_transforms import load_soma_procedural_transform_definition
            cache["procedural_transform_definition"] = (
                load_soma_procedural_transform_definition(path))
        return cache["procedural_transform_definition"]

    @property
    def procedural_transforms(self):
        """Upstream ``procedural_transforms``: the compiled
        :class:`~soma_jax.procedural_transforms.SOMAProceduralParameterTransform`
        for this layer's target rig, or ``None`` on the legacy rig.

        Built on first access from the expanded rig (translations in metres);
        the layer itself evaluates the same math through its internal
        :class:`~soma_jax.procedural_transforms.ProceduralTransforms`.
        """
        rig = self._skin_rig
        if self._procedural is None or rig is None:
            return None
        cache = self._lazy.values
        if "procedural_transforms" not in cache:
            from ..procedural_transforms import SOMAProceduralParameterTransform
            definition = self._procedural.definition
            cache["procedural_transforms"] = SOMAProceduralParameterTransform(
                self.public_joint_names, self.target_joint_names,
                rotation_extraction_modes=definition.rotation_extraction_modes,
                segments=definition.segments,
                rotation_entries=definition.rotation_entries,
                translation_entries=definition.translation_entries,
                target_t_pose_world=_rig_transforms_to_metres(np.asarray(rig["t_pose_world"])),
                target_joint_parent_ids=self.joint_parent_ids,
                target_bind_pose_world=np.asarray(rig["bind_pose_world_m"]),
            )
        return cache["procedural_transforms"]

    def _core_asset_array(self, key: str) -> jnp.ndarray:
        """An array of the upstream core asset ``SOMA_neutral.npz``, memoized."""
        cache = self._lazy.values
        if key not in cache:
            recipe = self._build_recipe
            npz = None if recipe is None else recipe.kwargs.get("npz_path")
            if npz is None:
                npz = self.data_root / "SOMA_neutral.npz"
            with np.load(npz, allow_pickle=False) as core:
                cache[key] = jnp.asarray(core[key])
        return cache[key]

    @property
    def shape_pca(self) -> jnp.ndarray:
        """Upstream ``shape_pca``: the core asset's SOMA shape PCA (``shapedirs``,
        (128, 3V) in its native centimetres), whatever the identity backend."""
        return self._core_asset_array("shapedirs")

    @property
    def shape_mean(self) -> jnp.ndarray:
        """Upstream ``shape_mean``: the core asset's mean shape (``mean``)."""
        return self._core_asset_array("mean")

    @property
    def shape_eigenvalues(self) -> jnp.ndarray:
        """Upstream ``shape_eigenvalues``: the core asset's PCA eigenvalues."""
        return self._core_asset_array("eigenvalues")

    @property
    def data_root(self):
        """Upstream ``data_root``: the asset directory the layer was built from."""
        from pathlib import Path as _Path
        recipe = self._build_recipe
        root = None if recipe is None else recipe.kwargs.get("data_root")
        if root is None:
            from ..assets import data_root as _assets_root
            root = _assets_root()
        return _Path(root)

    @property
    def identity_model_kwargs(self) -> dict:
        """Upstream ``identity_model_kwargs``: the backend options the layer was built with."""
        recipe = self._build_recipe
        kwargs = None if recipe is None else recipe.kwargs.get("identity_model_kwargs")
        return dict(kwargs or {})

    @property
    def procedural_template_rig_path(self):
        """Upstream ``procedural_template_rig_path``: the template USD the rig was
        merged from (upstream records it in both modes), else ``None``."""
        path = self._asset_path("SOMA_template_rig.usda", "usd_path")
        if path is not None and path.exists():
            return path
        if self._procedural is None:
            return None
        return self.data_root / "SOMA_template_rig.usda"

    def _asset_path(self, name: str, recipe_key: Optional[str] = None):
        """Where a layer built by :meth:`from_upstream_assets` read ``name``, else ``None``."""
        from pathlib import Path as _Path
        recipe = self._build_recipe
        if recipe is None or recipe.factory != "from_upstream_assets":
            return None
        explicit = recipe.kwargs.get(recipe_key) if recipe_key else None
        if explicit is not None:
            return _Path(explicit)
        data_root = recipe.kwargs.get("data_root")
        if data_root is not None:
            return _Path(data_root) / name
        from ..assets import resolve
        return resolve(name, required=False)

    # ------------------------------------------------------------------
    # Upstream's rig buffers. Upstream registers these from `rig_data`: the
    # skinning (target) rig — the expanded twist rig on a procedural layer —
    # in the asset's native centimetres. The public rig's own transforms are
    # in `public_rig_view()`.
    # ------------------------------------------------------------------
    @property
    def rig_data(self) -> dict:
        """Upstream ``rig_data``: ``SOMA_neutral.npz`` merged with the template rig.

        Assembled as upstream's constructor assembles it, from the same asset
        files, on first access: the core npz's arrays, updated with the public
        rig derived from the mid template (procedural joints pruned), then on a
        procedural layer with the mid template rig itself, then on an xlo layer
        with the xlo rig. Only layers built by :meth:`from_upstream_assets`
        carry one.
        """
        data = self._rig_data_or_none()
        if data is None:
            raise AttributeError(
                "rig_data is assembled from SOMA_neutral.npz and the template rig; only "
                "layers built by SOMALayer.from_upstream_assets() carry it.")
        return data

    def _rig_data_or_none(self) -> Optional[dict]:
        npz = self._asset_path("SOMA_neutral.npz", "npz_path")
        usd = self._asset_path("SOMA_template_rig.usda", "usd_path")
        if npz is None or usd is None:
            return None
        cache = self._lazy.values
        if "rig_data" not in cache:
            from ..procedural_transforms import derive_soma_rig_without_procedural_joints
            from ..usd_io import load_lod_rigs_from_usd
            with np.load(npz, allow_pickle=False) as core:
                rig_data = {key: core[key] for key in core.files}
            definition = self.procedural_transform_definition
            segments = None if definition is None else definition.segments
            public_names = list(self.public_joint_names)
            rigs = load_lod_rigs_from_usd(usd, ("mid", "low", "xlo") if self.lod == "xlo"
                                          else ("mid",))
            rig_data.update(derive_soma_rig_without_procedural_joints(
                rigs["mid"], public_names, segments=segments))
            if self._procedural is not None:
                rig_data.update(rigs["mid"])
            if self.lod == "xlo":
                xlo = rigs["xlo"]
                if self._procedural is None:
                    xlo = derive_soma_rig_without_procedural_joints(
                        xlo, public_names, segments=segments)
                rig_data.update(xlo)
            cache["rig_data"] = rig_data
        return cache["rig_data"]

    def _rig_buffer(self, key: str) -> Optional[jnp.ndarray]:
        """``rig_data[key]`` as upstream's float32 buffer, or ``None`` without rig data."""
        data = self._rig_data_or_none()
        if data is None or key not in data:
            return None
        cache = self._lazy.values
        if ("buffer", key) not in cache:
            value = np.asarray(data[key], np.float32)
            if key == "bind_shape" and self.lod == "low":
                value = value[np.asarray(self._lod_mid_to_low_np)]
            cache[("buffer", key)] = jnp.asarray(value)
        return cache[("buffer", key)]

    @property
    def bind_pose_world(self) -> Optional[jnp.ndarray]:
        """Upstream ``bind_pose_world``: the skinning rig's (J_target, 4, 4) bind-pose
        world transforms, native units. Layers without :attr:`rig_data` report
        their public rig, which then is their skinning rig too."""
        value = self._rig_buffer("bind_pose_world")
        return self._public_bind_pose_world if value is None else value

    @property
    def t_pose_world(self) -> Optional[jnp.ndarray]:
        """Upstream ``t_pose_world``: the skinning rig's (J_target, 4, 4) T-pose
        world transforms, native units (see :attr:`bind_pose_world`)."""
        value = self._rig_buffer("t_pose_world")
        return self._public_t_pose_world if value is None else value

    @property
    def bind_pose_local(self) -> Optional[jnp.ndarray]:
        """Upstream ``bind_pose_local``: the skinning rig's parent-relative bind pose."""
        value = self._rig_buffer("bind_pose_local")
        if value is None and self._bind_pose_local_np is not None:
            value = jnp.asarray(np.asarray(self._bind_pose_local_np, np.float32))
        return value

    @property
    def t_pose_local(self) -> Optional[jnp.ndarray]:
        """Upstream ``t_pose_local``: the skinning rig's parent-relative T-pose."""
        value = self._rig_buffer("t_pose_local")
        if value is None and self._public_t_pose_world is not None:
            value = _joint_world_to_local(self._public_t_pose_world, self._parents_np)
        return value

    @property
    def bind_shape(self) -> Optional[jnp.ndarray]:
        """Upstream ``bind_shape``: the template rig's skin-mesh points at this LOD
        (native centimetres; the low LOD takes the mid points' ``lod_mid_to_low``
        subset, as upstream does), or ``None`` without :attr:`rig_data`."""
        return self._rig_buffer("bind_shape")

    @property
    def mode(self) -> str:
        """Upstream ``mode``: ``"warp"`` (sparse top-K LBS) or ``"dense"``."""
        return "warp" if self.weight_values is not None else "dense"

    @classmethod
    def _is_body_bone_scale_joint(cls, name: str) -> bool:
        return name in cls.BODY_BONE_SCALE_JOINT_NAMES or any(
            name.startswith(prefix) for prefix in cls.FINGER_BONE_SCALE_JOINT_PREFIXES
        )

    def __init__(
        self,
        soma_data: dict,
        identity_model: BaseIdentityModel,
        correctives: Optional[CorrectivesMLP] = None,
        sparse_k: int = 8,   # top-K sparse LBS; 8 matches SOMA-X's Warp path (topk_skinning K=8)
        *,
        reference_pose=None,
        lod: str = "mid",
        skeleton_rig: Optional[dict] = None,
        lod_transfer: Optional[tuple] = None,
        excluded_vert_ids=None,
        identity_model_type: str = "soma",
        output_unit: Unit = Unit.METERS,
    ):
        """
        Args:
            soma_data: dict with keys: v_template, faces, parents, weights,
                       joint_names, optionally J_regressor (and correctives
                       data). Reference-history arrays from the core npz, when
                       present, back :meth:`get_reference_pose`.
            identity_model: pre-constructed BaseIdentityModel.
            correctives: optional pre-constructed CorrectivesMLP.
            sparse_k: number of top-K joints to keep in sparse LBS.
            reference_pose: default reference for :meth:`__call__` — a
                ``(J, 3, 3)`` / ``(J, 4, 4)`` array of public world orientations
                (identity virtual Root included), or a dict of
                :meth:`get_reference_pose` arguments, resolved once. ``None``
                uses the current T-pose. Upstream: ``SOMALayer(reference_pose=)``
                (v0.3.1).
            lod: the body LOD ``soma_data`` describes (``"mid"``/``"low"``/``"xlo"``).
            skeleton_rig: the rig the per-identity skeleton fit runs on, when it
                is not ``soma_data`` itself — upstream's xlo layer fits on the
                low LOD (``xlo_skeleton_transfer``) while skinning the xlo mesh.
                Needs ``parents``, ``bind_pose_world``, ``bind_shape``,
                ``weights`` and the facial segments in its own indexing.
            lod_transfer: xlo only — ``(low_faces, face_ids, bary)`` embedding
                of this mesh in the skeleton rig's mesh (upstream
                ``identity_lod_transfer``).
            excluded_vert_ids: the facial inner geometry in this mesh's
                indexing (upstream ``excluded_vert_ids``); defaults to the
                eye-bag + mouth-bag segments of ``soma_data``.
            identity_model_type: the backend name behind ``identity_model``.
                Decides what ``scale_params`` mean, as upstream: bone-length
                controls on ``"soma"``, the identity model's own scales else.
            output_unit: unit of translational inputs and outputs (upstream
                ``output_unit``); metres by default.
        """
        if lod not in BODY_LODS:
            raise ValueError(f"lod must be one of {BODY_LODS}, got {lod!r}")
        self._lod = lod
        self._num_shape_components = (int(np.shape(soma_data["shapedirs"])[-1])
                                      if "shapedirs" in soma_data else None)
        self._correctives_model_path = None
        self._xlo_skeleton_mid_to_low_host = None
        self._build_recipe = None
        self._lazy = _LazyCache()
        self.output_unit = output_unit if isinstance(output_unit, Unit) else Unit.from_name(output_unit)
        self._lod_transfer_host = _host(lod_transfer)
        self.v_template = jnp.array(soma_data["v_template"], dtype=jnp.float32)
        self.faces = jnp.array(soma_data["faces"], dtype=jnp.int32)
        # The linear regressor is a SOMA-JAX extra (`skeleton_fit="linear"`);
        # the faithful skeleton fit never reads it.
        self.J_regressor = (None if soma_data.get("J_regressor") is None
                            else jnp.array(soma_data["J_regressor"], dtype=jnp.float32))
        parents_np = np.array(soma_data["parents"], dtype=np.int32)
        self._parents_host = _HostArray(parents_np)
        # Plain Python str, not the numpy str_ scalars np.load hands back —
        # those are pytree leaves that equinox flags inside a static field,
        # and they leak into every public joint-name API.
        self.joint_names = [str(n) for n in soma_data.get("joint_names", [])]

        weights_np = np.array(soma_data["weights"], dtype=np.float32)
        self.weights = jnp.array(weights_np)

        # Precompute sparse top-K skinning weights
        if sparse_k < weights_np.shape[1]:
            top_k_idx = np.argsort(weights_np, axis=1)[:, -sparse_k:][:, ::-1]
            top_k_val = np.take_along_axis(weights_np, top_k_idx, axis=1)
            top_k_val = top_k_val / (top_k_val.sum(axis=1, keepdims=True) + 1e-8)
            self.weight_indices = jnp.array(top_k_idx, dtype=jnp.int32)
            self.weight_values = jnp.array(top_k_val, dtype=jnp.float32)
        else:
            self.weight_indices = None
            self.weight_values = None

        self.skeleton_levels = compute_skeleton_levels(parents_np)
        self.identity_model = identity_model

        # Upstream rig orientation arrays (when present in the asset).
        # Used as default joint_orient for pose(), so the correctives input is
        # in the SAME absolute-skinning frame the trained checkpoint expects.
        if "t_pose_world" in soma_data:
            self._public_t_pose_world = jnp.array(soma_data["t_pose_world"], dtype=jnp.float32)
        else:
            self._public_t_pose_world = None
        if "bind_pose_world" in soma_data:
            self._public_bind_pose_world = jnp.array(soma_data["bind_pose_world"], dtype=jnp.float32)
        else:
            self._public_bind_pose_world = None
        # Bind-pose LOCAL transforms (parent-relative) — needed for the
        # `repose_to_bind_pose` step in prepare_identity. Stored as numpy so
        # it stays static under eqx.
        if "bind_pose_local" in soma_data:
            self._bind_pose_local_host = _HostArray(np.asarray(
                soma_data["bind_pose_local"], dtype=np.float32,
            ))
        else:
            self._bind_pose_local_host = None
        # Low-LOD vertex subset + face indices, applied by load(lod='low')
        self._lod_mid_to_low_host = _host(
            np.asarray(soma_data["lod_mid_to_low"], dtype=np.int32)
            if "lod_mid_to_low" in soma_data else None
        )
        # Vertex count of the mesh a low-LOD subset was taken from (None on a
        # mid-LOD layer). A plain int, so it stays hashable as a static field.
        self._lod_mid_num_verts = (
            int(np.asarray(soma_data["lod_mid_num_verts"]))
            if "lod_mid_num_verts" in soma_data else None
        )
        self._triangles_low_host = _host(
            np.asarray(soma_data["triangles_low"], dtype=np.int32)
            if "triangles_low" in soma_data else None
        )

        # No checkpoint -> no corrective network, as upstream
        # (`self.correctives_model = None`); pose() refuses when correctives
        # are explicitly requested.
        self._has_trained_correctives = correctives is not None
        self.correctives = correctives

        def _facial_ids(data) -> list[int]:
            excl: list[int] = []
            for seg in ("segment_eye_bags", "segment_mouth_bag"):
                if seg in data:
                    excl.extend(np.asarray(data[seg]).astype(int).ravel().tolist())
            return excl

        # Full SOMA-X skeleton fit (RBF + two-stage Kabsch), constructed with
        # the exact upstream arguments (third_party/SOMA-X/soma/body/soma.py):
        #   SkeletonTransfer(parents, bind_pose_world, bind_shape, weights,
        #                    rotation_method="auto",
        #                    vertex_ids_to_exclude=eye_bags + mouth_bag)
        # on this layer's own rig, or on `skeleton_rig` when the fit runs on
        # another LOD (xlo fits on the low mesh). The upstream rig stores bind
        # data in centimeters while this layer works in meters — normalise on
        # load (same heuristic as ``_repose_to_bind_pose``).
        fit_rig = soma_data if skeleton_rig is None else skeleton_rig
        if "bind_shape" in fit_rig and "bind_pose_world" in fit_rig:
            from ..geometry.skeleton_transfer import SkeletonTransfer
            bind_shape = np.asarray(fit_rig["bind_shape"], dtype=np.float32)
            bind_world = np.asarray(fit_rig["bind_pose_world"], dtype=np.float32).copy()
            if float(np.abs(bind_world[..., :3, 3]).max()) > 10.0:   # cm-scale rig
                bind_world[..., :3, 3] *= 0.01
                bind_shape = bind_shape * 0.01
            excl = _facial_ids(fit_rig)
            self.skeleton_transfer = SkeletonTransfer(
                np.asarray(fit_rig["parents"], dtype=np.int32),
                bind_world,
                bind_shape,
                np.asarray(fit_rig["weights"], dtype=np.float32),
                rotation_method="auto",
                vertex_ids_to_exclude=excl or None,
            )
        else:
            self.skeleton_transfer = None

        if excluded_vert_ids is None:
            excluded_vert_ids = _facial_ids(soma_data) or None
        self._excluded_vert_ids_host = _host(
            None if excluded_vert_ids is None
            else np.asarray(excluded_vert_ids, dtype=np.int64).ravel())

        # Bone-scale control layout (SOMA-X SOMALayer.soma_bone_scale_param_*):
        # the legacy controls in public-joint order, then the four foot
        # controls in FOOT_BONE_SCALE_JOINT_NAMES order (v0.2.2).
        names = [str(n) for n in self.joint_names]
        scale_ids = [i for i, n in enumerate(names) if self._is_body_bone_scale_joint(n)]
        if scale_ids:
            # The SOMA public rig — hold it to upstream's layout checks. Rigs
            # with no bone-scale joints at all (synthetic test rigs) simply get
            # no controls, as before.
            if len(scale_ids) != self.LEGACY_NUM_BONE_SCALE_PARAMS:
                raise RuntimeError(
                    "Unexpected legacy SOMA bone-scale parameter layout: "
                    f"expected {self.LEGACY_NUM_BONE_SCALE_PARAMS}, got {len(scale_ids)}")
            name_to_idx = {n: i for i, n in enumerate(names)}
            missing_feet = [n for n in self.FOOT_BONE_SCALE_JOINT_NAMES if n not in name_to_idx]
            if missing_feet:
                raise RuntimeError(
                    f"SOMA public rig is missing foot-scaling joints: {missing_feet}")
            scale_ids = scale_ids + [name_to_idx[n] for n in self.FOOT_BONE_SCALE_JOINT_NAMES]
        self._bone_scale_joint_indices_host = _host(
            np.asarray(scale_ids, dtype=np.int32) if scale_ids else None
        )
        self.bone_scale_param_names = tuple(names[i] for i in scale_ids)
        safe_parents = np.where(parents_np < 0, np.arange(len(parents_np)), parents_np)
        self.bone_scale_param_segments = tuple(
            (names[int(safe_parents[i])], names[i]) for i in scale_ids
        )
        self.identity_model_type = str(identity_model_type).lower()
        if self.identity_model_type == "soma":
            self.scale_param_names = self.bone_scale_param_names
            self.scale_param_segments = self.bone_scale_param_segments
        else:
            self.scale_param_names = tuple(getattr(identity_model, "scale_param_names", ()) or ())
            self.scale_param_segments = ()

        # No procedural rig unless `attach_procedural_rig` is called; the layer
        # then poses against its full joint list, as it always has.
        self._procedural = None
        self._public_idx_host = _host(None)
        self._public_names = tuple(names)
        self._skin_rig = None

        # Reference poses (SOMA-X v0.3.1). `attach_procedural_rig` copies the
        # module, so both carry over to the procedural layer unchanged.
        self._reference_pose_history = ReferencePoseHistory(soma_data)
        self._default_reference_pose = None
        if reference_pose is not None:
            self._default_reference_pose = self._resolve_reference_pose(reference_pose)

    def attach_procedural_rig(self, procedural, public_joint_names,
                              skin_rig: Optional[dict] = None) -> "SOMALayer":
        """Drive this rig from the 78-joint public pose contract.

        Upstream's default (``enable_procedural_transforms=True``) skins with the
        expanded template skeleton while keeping a public contract of 78 joints /
        77 posable ones: ``SOMAProceduralParameterTransform`` derives the 32
        twist joints' local rotations from the public rotations before FK
        (any USD-only helper bones — 12 on the v0026 template, none on v0027 —
        take identity) (``soma/body/soma.py``: "The expanded twist skeleton is used internally for
        FK/LBS ... but is not returned from pose()/forward()").

        Args:
            procedural: a :class:`~soma_jax.procedural_transforms.ProceduralTransforms`.
            public_joint_names: the 78 public joint names, in the order callers
                pose in. Must all appear in this layer's ``joint_names``.

        Returns:
            A new layer (this is an ``eqx.Module``; nothing is mutated).
        """
        names = [str(n) for n in self.joint_names]
        public = [str(n) for n in public_joint_names]
        missing = [n for n in public if n not in names]
        if missing:
            raise ValueError(
                f"public joints absent from this rig: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}")
        idx = np.asarray([names.index(n) for n in public], dtype=np.int32)

        out = eqx.tree_at(lambda m: m.weights, self, self.weights)
        object.__setattr__(out, "_lazy", _LazyCache())
        object.__setattr__(out, "_procedural", procedural)
        object.__setattr__(out, "_public_idx_host", _host(idx))
        object.__setattr__(out, "_public_names", tuple(public))
        object.__setattr__(out, "_skin_rig", skin_rig)
        return out

    @staticmethod
    def build_skinning_rig(skin_data: dict, public_joint_names, procedural,
                           sparse_k: int = 8) -> dict:
        """Package an expanded rig for FK/LBS from ``rig_build`` output.

        Upstream drives FK/LBS with the 110-joint template skeleton while the
        identity model, skeleton fit and bind data stay on the 78-joint public
        rig (``skeleton_transfer.skinning_weights`` is ``(18056, 78)`` in *both*
        upstream modes). This holds the FK/LBS half.

        Args:
            skin_data: output of :func:`soma_jax.rig_build.build_soma_asset` —
                needs ``joint_names``, ``parents``, ``weights``, ``t_pose_world``.
            public_joint_names: the public contract, to locate those joints
                inside the expanded rig.
            procedural: the :class:`ProceduralTransforms` whose twist-joint order
                fixes where derived binds land.
            sparse_k: top-K sparse LBS width, as for the public rig.
        """
        from ..geometry.batched_skinning import topk_skinning

        names = [str(n) for n in skin_data["joint_names"]]
        parents = np.asarray(skin_data["parents"], np.int32)
        weights = jnp.asarray(skin_data["weights"], jnp.float32)
        wi, wv = topk_skinning(weights, K=min(int(sparse_k), weights.shape[1]))
        twist = [str(n) for n in procedural.definition.twist_joint_names]
        return {
            "joint_names": tuple(names),
            "parents": _host(parents),
            "levels": compute_skeleton_levels(parents),
            "weights": weights,
            "weight_indices": wi,
            "weight_values": wv,
            "t_pose_world": jnp.asarray(skin_data["t_pose_world"], jnp.float32),
            # Upstream starts its bind expansion from `self._public_bind_pose_world`
            # (`_expand_public_bind_transforms`), in the layer's own units.
            "bind_pose_world_m": jnp.asarray(
                _rig_transforms_to_metres(skin_data["bind_pose_world"]), jnp.float32),
            "public_idx": _host(np.asarray(
                [names.index(str(n)) for n in public_joint_names], np.int32)),
            "twist_idx": _host(np.asarray([names.index(n) for n in twist], np.int32)),
            # Upstream's target_t_pose_local_rotations / local translations: the
            # expanded rig's own bind-local step, which each twist joint composes
            # its twist rotation onto.
            "base_rotations": jnp.asarray(
                np.asarray(skin_data["t_pose_local"])[..., :3, :3], jnp.float32),
            # UNITS: rig_build keeps the USD/npz transforms in their native
            # centimetres (only `v_template`/`shapedirs` are converted), while the
            # per-identity `bind_transforms` the expander composes these onto are
            # in metres. Mixing them put the twist binds ~17 m out.
            "local_translations": jnp.asarray(
                np.asarray(skin_data["t_pose_local"])[..., :3, 3]
                * Unit.CENTIMETERS.meters_per_unit, jnp.float32),
            "twist_parent_idx": _host(np.asarray(
                [int(parents[names.index(n)]) for n in twist], np.int32)),
            # Upstream's `target_to_public_joint_indices`: which public joint's
            # bone scale each expanded joint follows. Public joints follow
            # themselves; anything else follows its nearest public ancestor.
            "target_to_public": _host(_target_to_public_map(
                names, parents, [str(n) for n in public_joint_names])),
            # `_apply_target_bone_scales` then overrides the twist joints to
            # follow their segment's **end** joint rather than their parent.
            "twist_end_public": _host(np.asarray(
                [[str(n) for n in public_joint_names].index(seg.end_joint)
                 for seg in procedural.definition.segments
                 for _ in seg.twist_joints], np.int32)),
        }

    def _pose_procedural(self, bind_transforms, rest_verts, rotmats, transl,
                         *, use_sparse: bool = True, fk_only: bool = False,
                         bone_scales=None):
        """Pose through the expanded rig — upstream's procedural path.

        Order matters and is upstream's (``soma/soma.py:1569``):

        1. FK on the **public** joints only.
        2. Expand those world transforms onto the 110-joint rig, each twist joint
           a single local step off its public parent
           (``expand_world_transforms_from_source_fk``).
        3. LBS with the 110-joint weights against the correspondingly expanded
           bind.

        The bind is expanded through the *same* function with identity twist
        rotations, so at rest ``T_world @ inv(bind)`` is exactly identity and the
        twist joints contribute nothing — a consistency the previous
        translation-matrix bind did not have.

        ``joints`` / ``transforms`` come from the public FK result, matching
        upstream's ``output_transforms = public_world_transforms``.
        """
        from ..geometry.batched_skinning import pose_from_bind
        from ..geometry.lbs import lbs_sparse
        from ..geometry.transforms import se3_inverse

        rig = self._skin_rig

        # 1. public FK
        _, T_world_public = pose_from_bind(
            bind_transforms, rest_verts, self.weights, self.skeleton_levels,
            self._parents_np, rotmats, transl, hips_idx=1, skip_lbs=True,
            local_translation_scales=bone_scales,
        )

        # 2a. This identity's expanded bind — upstream's
        # `_expand_public_bind_transforms`. The barrier guards the parent
        # gathers below against jaxlib <= 0.6.2's constant-folding bug (see
        # `pose_from_bind`) when the bind is a closed-over constant.
        bind_full = jax.lax.optimization_barrier(
            self._expanded_bind_transforms(bind_transforms))

        # 2b. The local step each joint composes comes from **that** bind, not
        # from the static template: upstream passes
        # `BatchedSkinning.local_rotations / local_translations`, which
        # `rebind()` recomputes as `joint_world_to_local(bind_world)` on every
        # identity (`batched_skinning.py:312`).
        parents = rig["parents"].a
        safe = np.maximum(parents, 0)
        R_all, t_all = bind_full[..., :3, :3], bind_full[..., :3, 3]
        R_par = R_all[:, safe]
        is_root = jnp.asarray(parents < 0)[None, :, None]
        local_t = jnp.einsum("bjnm,bjn->bjm", R_par, t_all - t_all[:, safe])
        local_t = jnp.where(is_root, t_all, local_t)
        base_rot = jnp.einsum("bjnm,bjnp->bjmp", R_par, R_all)
        base_rot = jnp.where(is_root[..., None], R_all, base_rot)

        # Bone-length controls: upstream's `_apply_target_bone_scales` maps each
        # public scale onto the expanded rig through `target_to_public`, then
        # overrides every twist joint to follow its segment's **end** joint —
        # a stretched forearm must carry its twist helpers with it — and scales
        # the local translations before FK.
        if bone_scales is not None:
            target_scales = bone_scales[:, jnp.asarray(rig["target_to_public"].a)]
            target_scales = target_scales.at[:, jnp.asarray(rig["twist_idx"].a)].set(
                bone_scales[:, jnp.asarray(rig["twist_end_public"].a)])
            local_t = local_t * target_scales[..., None]

        def _expand(source_rot, source_world):
            return self._procedural.expand_world_transforms_from_source_fk(
                source_rot, source_world, base_rot, local_t,
                rig["public_idx"].a, rig["twist_idx"].a,
                rig["twist_parent_idx"].a, parents,
            )

        # SOMA-X v0.3 fix: public FK broadcasts a singleton pose over the
        # identity batch, but the expansion needs rotations of that same batch.
        if rotmats.shape[0] == 1 and T_world_public.shape[0] > 1:
            rotmats = jnp.broadcast_to(rotmats, (T_world_public.shape[0],) + rotmats.shape[1:])
        T_world = _expand(rotmats, T_world_public)
        if fk_only:
            return SOMAOutput(vertices=None, joints=T_world_public[..., :3, 3],
                              transforms=T_world_public)

        # 3. LBS on the expanded rig
        bone_Rt = jnp.einsum("bjmn,bjnp->bjmp", T_world, se3_inverse(bind_full))[..., :3, :]
        zeros = jnp.zeros_like(rest_verts)
        if use_sparse:
            posed = lbs_sparse(rest_verts, zeros, bone_Rt,
                               rig["weight_values"], rig["weight_indices"])
        else:
            posed = lbs_blend(rest_verts, zeros, bone_Rt, rig["weights"])
        return SOMAOutput(vertices=posed, joints=T_world_public[..., :3, 3],
                          transforms=T_world_public)

    def _expanded_bind_transforms(self, bind_transforms: jnp.ndarray) -> jnp.ndarray:
        """Scatter the fitted public binds into the expanded rig.

        The public joints keep their per-identity fitted binds; the 32 twist
        joints take the translation-matrix combination of public positions with
        identity rotation (upstream's ``full_rig_bind_world``); any USD-only
        helpers keep the template bind. The v0026 template had 12, all with
        **zero** skinning weight, so their binds could not move a vertex; the
        v0027 template (SOMA-X v0.2.2+) has none.
        """
        rig = self._skin_rig
        pub_idx = jnp.asarray(rig["public_idx"].a)
        twist_idx = jnp.asarray(rig["twist_idx"].a)
        B, J = bind_transforms.shape[0], len(rig["joint_names"])

        # Template bind for every joint, then override with the fitted values.
        full = jnp.broadcast_to(rig["bind_pose_world_m"][None], (B, J, 4, 4))
        full = full.at[:, pub_idx].set(bind_transforms)

        # Upstream's `_apply_translation_parameters` rewrites only the
        # **translation column** (``out[..., :3, 3] = matrix @ positions``) and
        # leaves every rotation block as it found it. Zeroing the twist bind
        # rotation to identity instead makes the bind inconsistent with the posed
        # transform that is derived from the same local step.
        pub_pos = bind_transforms[..., :3, 3]                       # (B, 78, 3)
        twist_pos = self._procedural.emit_twist_world_positions(pub_pos)
        # Gather whole 4x4 blocks: `full.at[:, twist_idx, :3, 3]` would mix an
        # advanced index with slices and reorder the gathered axis to the front,
        # the same trap `pose_from_bind` documents for `[:, parents, :3, 3]`.
        blocks = full[:, twist_idx].at[..., :3, 3].set(twist_pos)
        return full.at[:, twist_idx].set(blocks)

    @property
    def _public_idx(self) -> Optional[np.ndarray]:
        h = self._public_idx_host
        return None if h is None else h.a

    @property
    def num_bone_scale_params(self) -> int:
        """Number of active bone-scale controls (60 on the stock SOMA rig)."""
        return len(self.bone_scale_param_names)

    @property
    def num_scale_params(self) -> Optional[int]:
        """Upstream ``num_scale_params``: what ``scale_params`` must hold.

        The bone-length controls on the SOMA backend; the identity model's own
        count (e.g. 68 for MHR, ``None`` when unused) on every other backend.
        """
        if self.identity_model_type == "soma":
            return self.num_bone_scale_params
        return getattr(self.identity_model, "num_scale_params", None)

    def normalize_bone_scales(self, bone_scales: jnp.ndarray) -> jnp.ndarray:
        """Accept current or legacy SOMA bone-scale tensors, return the current width.

        Port of upstream ``SOMALayer._normalize_soma_bone_scales`` (v0.2.2):
        a legacy ``(B, 56)`` tensor gets unit scales appended for the four foot
        controls; ``(B, 60)`` passes through. Any other width is an error.
        """
        bone_scales = jnp.asarray(bone_scales)
        if bone_scales.ndim == 1:
            bone_scales = bone_scales[None]
        expected, legacy = self.num_bone_scale_params, self.LEGACY_NUM_BONE_SCALE_PARAMS
        if bone_scales.ndim != 2 or bone_scales.shape[1] not in (expected, legacy):
            raise ValueError(
                "SOMA scale_params must have shape "
                f"(B, {expected}) or legacy shape (B, {legacy}); "
                f"got {tuple(bone_scales.shape)}. "
                "Use layer.scale_param_names for the active control order."
            )
        if bone_scales.shape[1] == legacy and expected != legacy:
            pad = jnp.ones((bone_scales.shape[0], expected - legacy), bone_scales.dtype)
            bone_scales = jnp.concatenate([bone_scales, pad], axis=1)
        return bone_scales

    def full_bone_scales(self, bone_scales: jnp.ndarray) -> jnp.ndarray:
        """Scatter (B, S) active bone scales into a full (B, J) multiplier array.

        Mirrors SOMA-X's ``_full_public_bone_scales``: joints without a control
        keep a scale of 1.0. ``scale_param_names`` gives the expected order and
        ``scale_param_segments`` the (parent, child) edge each value stretches.
        """
        bone_scales = self.normalize_bone_scales(bone_scales)
        J = self.weights.shape[1]
        full = jnp.ones((bone_scales.shape[0], J), dtype=bone_scales.dtype)
        if self._bone_scale_joint_indices is None:
            return full
        return full.at[:, jnp.asarray(self._bone_scale_joint_indices)].set(bone_scales)

    # ------------------------------------------------------------------
    # Public rig view
    # ------------------------------------------------------------------
    @property
    def public_joint_names(self) -> tuple:
        """Names of the public SOMA joints.

        The 78-joint contract callers pose against. On the layers
        :meth:`from_upstream_assets` builds this is all of ``joint_names`` —
        the two-rig procedural layer keeps its expanded FK/LBS skeleton in
        ``_skin_rig`` — and it is the public subset when
        :meth:`attach_procedural_rig` is applied to a larger rig.
        """
        return tuple(self._public_names)

    def public_skinning_weights(self) -> jnp.ndarray:
        """Skinning weights folded onto the public SOMA hierarchy.

        SOMA-X folds an expanded twist-joint rig down to the 78 public joints
        (``derive_soma_rig_without_procedural_joints`` aggregates each dropped
        joint's weights onto its nearest kept parent). On a non-procedural layer
        the weights already *are* the public rig and pass through.
        """
        pub = self._public_idx
        if pub is None or self.weights.shape[1] == len(pub):
            return self.weights
        # Aggregate every non-public column onto its nearest kept ancestor.
        parents = self._parents_np
        keep = {int(i): k for k, i in enumerate(pub)}
        folded = np.zeros((self.weights.shape[0], len(pub)), np.float64)
        W = np.asarray(self.weights, np.float64)
        for j in range(W.shape[1]):
            a = j
            while a not in keep and 0 <= int(parents[a]) != a:
                a = int(parents[a])
            if a in keep:
                folded[:, keep[a]] += W[:, j]
        return jnp.asarray(folded, self.weights.dtype)

    def to_public_rotations(self, rotations: jnp.ndarray) -> jnp.ndarray:
        """Reduce target-joint rotations to public SOMA joint order.

        An identity mapping on the non-procedural rig; validates the joint
        count so a mismatched rotation tensor fails loudly rather than
        silently skinning the wrong joints.
        """
        public_count = len(self.public_joint_names)
        target_count = len(self.target_joint_names)
        count = rotations.shape[-3]
        if count == public_count:
            return rotations
        if count == target_count:
            return rotations[..., self.public_transform_joint_indices, :, :]
        raise ValueError(
            f"Expected rotations for {public_count} public joints or {target_count} "
            f"target joints, got {count}."
        )

    def public_bind_transforms_world(self, bind_transforms_world=None) -> jnp.ndarray:
        """Select public-joint world bind transforms (upstream method).

        Target-rig (expanded) binds are reduced to the public joints; public
        binds — what :meth:`prepare_identity` returns here — pass through.
        Upstream reads its cached identity when called without an argument;
        this immutable layer holds none and returns the public rig's bind pose.
        """
        if bind_transforms_world is None:
            return self._public_bind_pose_world
        count = bind_transforms_world.shape[-3]
        if count == len(self.public_joint_names):
            return bind_transforms_world
        return bind_transforms_world[..., self.public_transform_joint_indices, :, :]

    def public_rig_view(self, bind_transforms_world: Optional[jnp.ndarray] = None
                        ) -> SOMAPublicRigView:
        """Public-joint view of the current rig — upstream's ``SOMAPublicRigView``.

        Args:
            bind_transforms_world: optional (B, J, 4, 4) fitted binds from
                ``prepare_identity(return_bind_transforms=True)`` (public, or the
                expanded target rig's). Upstream defaults to its cached
                identity; this immutable layer defaults to the public rig's
                bind pose.

        Returns:
            :class:`SOMAPublicRigView` (also readable dict-style).
        """
        bind_world = self.public_bind_transforms_world(bind_transforms_world)
        if bind_world is None:
            raise RuntimeError(
                "No bind transforms available: pass bind_transforms_world, or load "
                "an asset carrying bind_pose_world (see docs/INSTALL.md §4.2)."
            )
        return SOMAPublicRigView(
            joint_names=self.public_joint_names,
            joint_parent_ids=self._parents_np,
            target_joint_indices=self.public_transform_joint_indices,
            target_to_public_joint_indices=self.target_to_public_joint_indices,
            bind_transforms_world=bind_world,
            bind_transforms_local=_joint_world_to_local(bind_world, self._parents_np),
            t_pose_world=self._public_t_pose_world,
            skinning_weights=self.public_skinning_weights(),
        )

    # ------------------------------------------------------------------
    # Reference poses (SOMA-X v0.3.1)
    # ------------------------------------------------------------------
    @property
    def output_joint_parent_ids(self) -> np.ndarray:
        """Parent ids of :attr:`public_joint_names`, root self-parented.

        Upstream's ``SOMALayer.output_joint_parent_ids``: the hierarchy of the
        joints :meth:`__call__` returns, with the virtual Root as its own parent
        (the convention reference-pose hierarchies are checked against).
        Parents are walked up through any joint outside the public set.
        """
        names = [str(n) for n in self.joint_names]
        parents = self._parents_np
        pub = [str(n) for n in self.public_joint_names]
        at = {n: i for i, n in enumerate(names)}
        pub_at = {n: k for k, n in enumerate(pub)}
        out = np.arange(len(pub), dtype=np.int64)
        for k, n in enumerate(pub):
            j = at[n]
            p = int(parents[j])
            while 0 <= p != j and names[p] not in pub_at:
                j, p = p, int(parents[p])
            if 0 <= p and names[p] in pub_at and names[p] != n:
                out[k] = pub_at[names[p]]
        return out

    def list_reference_poses(self) -> list[dict]:
        """List reference IDs and descriptive metadata shipped in the core npz.

        Upstream ``SOMALayer.list_reference_poses``. Each record includes its npz
        key, data key and reference revision or asset-revision selector; the
        dictionaries are independent copies. Assets without a reference history
        return an empty list; nothing is downloaded.
        """
        return self._reference_pose_history.list_reference_poses()

    def get_reference_pose(
        self,
        reference_id: Optional[str] = None,
        *,
        version: Optional[str] = None,
        data_key: str = "t_pose_world",
        asset_revision: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> jnp.ndarray:
        """Return a saved reference's world rotations in public joint order.

        Upstream ``SOMALayer.get_reference_pose`` (v0.3.1). Select exactly one of
        ``version`` (e.g. ``"v0.3.0"``: the newest stored revision of
        ``data_key`` at or before that semantic version), an ``asset_revision``,
        an ``alias`` from the catalogue, or a direct ``reference_id``. Full npz
        keys and asset revisions match exactly. No downloads.

        Returns:
            A fresh ``(J, 3, 3)`` float32 array including the virtual Root. No
            prepared identity is needed.

        Raises:
            KeyError: unknown lookup.
            ValueError: conflicting selectors, or a reference whose joint names
                or hierarchy do not match this layer's public rig.
        """
        reference_id = self._reference_pose_history.resolve_reference_id(
            reference_id, soma_version=version, data_key=data_key,
            asset_revision=asset_revision, alias=alias)
        return self._reference_pose_history.get_reference_pose(
            reference_id, self.public_joint_names, self.output_joint_parent_ids,
            dtype=jnp.float32)

    def convert_reference(self, rotations, from_ref, to_ref) -> jnp.ndarray:
        """Re-express ``(B, 77, 3, 3)`` posable-joint rotations in another reference.

        Upstream ``SOMALayer.convert_reference`` (v0.3.1). Both references accept
        an array or a :meth:`get_reference_pose` argument dict: public world
        orientations including the identity virtual Root. Absolute local
        rotations are preserved, so the result posed against ``to_ref`` matches
        ``rotations`` posed against ``from_ref``. Differentiable, ``jit``-safe,
        and independent of any constructor default — references are explicit.
        Pass the result on as ``poses`` (rotation matrices) with
        ``reference_pose=to_ref``.
        """
        if isinstance(from_ref, dict):
            from_ref = self.get_reference_pose(**from_ref)
        if isinstance(to_ref, dict):
            to_ref = self.get_reference_pose(**to_ref)
        return convert_reference_rotations(
            rotations, from_ref, to_ref, self.output_joint_parent_ids, virtual_root=True)

    def _resolve_reference_pose(self, reference_pose) -> jnp.ndarray:
        """Dict -> lookup; validate; return the (J, 3, 3) rotation block."""
        if isinstance(reference_pose, dict):
            reference_pose = self.get_reference_pose(**reference_pose)
        rot = validate_reference_pose(reference_pose, len(self.public_joint_names))
        return jnp.asarray(rot, dtype=jnp.float32)

    @classmethod
    def load(
        cls,
        path: str,
        identity_model_type: str = "soma",
        identity_model_path: Optional[str] = None,
        correctives_path: Optional[str] = None,
        sparse_k: int = 8,   # top-K sparse LBS; 8 matches SOMA-X's Warp path (topk_skinning K=8)
        lod: str = "mid",
        *,
        reference_pose=None,
        output_unit: Unit = Unit.METERS,
    ) -> "SOMALayer":
        """Load a SOMALayer from a SOMA_neutral.npz asset file.

        Args:
            path: path to SOMA_neutral.npz (contains v_template, weights, etc.)
            identity_model_type: which identity model to instantiate.
            identity_model_path: path to identity model parameters (optional).
            correctives_path: path to correctives checkpoint (optional).
            sparse_k: top-K joints for sparse LBS.
            lod: body mesh level of detail — ``"mid"`` (18,056 vertices, the
                default) or ``"low"`` (4,505). ``"low"`` mirrors upstream
                ``SOMALayer(lod="low")``: the whole rig, identity model and
                skeleton fit are built on the low-LOD subset, not subsampled
                afterwards. Requires ``lod_mid_to_low`` + ``triangles_low`` in
                the asset. ``"xlo"`` needs the template USD's low and xlo skin
                meshes, which the archive does not carry — use
                :meth:`from_upstream_assets` for it.
            reference_pose: constructor-default reference, as for
                :class:`SOMALayer` (SOMA-X v0.3.1). Dict lookups need the
                archive to carry the reference history.
            output_unit: unit of translational inputs and outputs, as
                upstream's ``output_unit`` (metres by default).

        Returns:
            Instantiated SOMALayer.
        """
        if lod not in BODY_LODS:
            raise ValueError(f"lod must be one of {BODY_LODS}, got {lod!r}")
        if lod == "xlo":
            raise ValueError(
                "lod='xlo' reads the low and xlo skin meshes of SOMA_template_rig.usda, "
                "which a runtime archive does not carry; use "
                "SOMALayer.from_upstream_assets(lod='xlo').")
        soma_data = dict(np.load(path, allow_pickle=True))

        # Flatten any object arrays
        for k in list(soma_data.keys()):
            v = soma_data[k]
            if isinstance(v, np.ndarray) and v.dtype == object:
                soma_data[k] = v.item()

        layer = cls._from_soma_data(
            soma_data, identity_model_type=identity_model_type,
            identity_model_path=identity_model_path,
            correctives_path=correctives_path, sparse_k=sparse_k, lod=lod,
            reference_pose=reference_pose, output_unit=output_unit)
        object.__setattr__(layer, "_build_recipe", _BuildRecipe("load", dict(
            path=path, identity_model_type=identity_model_type,
            identity_model_path=identity_model_path, sparse_k=sparse_k,
            output_unit=output_unit)))
        return layer

    @classmethod
    def from_upstream_assets(
        cls,
        data_root=None,
        low_lod: bool = False,
        device=None,
        identity_model_type: str = "soma",
        mode: str = "warp",
        output_unit: Unit = Unit.METERS,
        identity_model_kwargs: Optional[dict] = None,
        lod: Optional[str] = None,
        template_rig_path=None,
        enable_procedural_transforms: bool = True,
        load_correctives_model: Optional[bool] = None,
        correctives_model_path=_DEFAULT_CORRECTIVES_MODEL_PATH,
        *,
        reference_pose=None,
        npz_path: Optional[str] = None,
        identity_model_path: Optional[str] = None,
        sparse_k: Optional[int] = None,
        fit_joint_regressor: bool = True,
        procedural: Optional[bool] = None,
        correctives_path=_UNSET,
        usd_path: Optional[str] = None,
    ) -> "SOMALayer":
        """Upstream's ``SOMALayer(...)`` constructor, as a classmethod.

        Takes upstream's parameters in upstream's order and builds the layer
        upstream's constructor builds. Upstream (SOMA-X v0.3) reads the rig,
        bind pose, bind shape and skinning from ``SOMA_template_rig.usda`` alone
        and only shape/topology data from ``SOMA_neutral.npz``, refusing to run
        without the USD; this performs that merge with numpy/scipy/pxr — no
        ``torch`` and no ``SOMA_neutral_fixed.npz`` — reproducing upstream's
        ``rig_data`` exactly (see :mod:`soma_jax.rig_build`).

        Use :meth:`load` instead when you have the cached archive and would
        rather not depend on ``usd-core`` at runtime.

        Args:
            data_root: an upstream-layout asset directory (upstream's
                ``data_root``) to read ``SOMA_neutral.npz``, the template, the
                procedural JSON and the identity packs from. ``None`` resolves
                them through :mod:`soma_jax.assets`; a directory that does not
                exist falls back to the default assets, as upstream's does.
            low_lod: legacy alias for ``lod="low"``, resolved as upstream's
                ``_resolve_body_lod``.
            device: upstream's torch device: accepted for call compatibility and
                ignored (arrays live on JAX's default device).
            identity_model_type: identity backend. The default is ``"soma"``
                where upstream's is ``"mhr"``: the MHR backend reads its
                TorchScript archive and so needs ``torch``, which a default
                layer should not.
            mode: upstream's skinning backend: ``"warp"`` skins with the top-8
                sparse weights its Warp kernel uses; any other value uses every
                influence, as upstream's dense fallback does.
            output_unit: unit of translational inputs and outputs, as
                upstream's ``output_unit`` (metres by default).
            identity_model_kwargs: options for the identity backend, as
                upstream's ``identity_model_kwargs`` (e.g. ``{"model_path": ...}``
                for the licensed SMPL-family files). Every backend is built from
                the asset directory exactly as upstream builds it
                (:mod:`soma_jax.body.identity_model`) unless
                ``identity_model_path`` names a SOMA-JAX identity pack.
            lod: ``"mid"``, ``"low"`` or ``"xlo"``, as upstream's ``lod``
                (``None``: ``"low"`` if ``low_lod`` else ``"mid"``).
                Every LOD starts from the **mid** template, as upstream's does:
                ``"low"`` slices it by the npz's ``lod_mid_to_low`` (faces
                ``triangles_low``); ``"xlo"`` skins the template's 612-vertex
                xlo mesh while the identity model and skeleton fit run on the
                low LOD, whose rest shape — and, when applied, the low-LOD
                correctives — reach the xlo vertices through a barycentric
                embedding (upstream ``identity_lod_transfer``).
            template_rig_path: ``SOMA_template_rig.usda``; resolved when
                omitted, as upstream's ``template_rig_path``.
            enable_procedural_transforms: keep the expanded 110-joint twist
                skeleton (upstream's default); ``False`` requests the pruned
                78-joint legacy rig.
            load_correctives_model: deprecated alias, as upstream — use
                ``correctives_model_path=None`` to disable loading.
            correctives_model_path: pose-corrective checkpoint: by default
                ``<data_root>/correctives_model.pt`` on a procedural layer (a
                missing default file loads nothing) and none on the legacy rig;
                ``None`` skips loading; an explicit path must exist and needs the
                procedural rig.
            reference_pose: constructor-default reference, as upstream's
                (SOMA-X v0.3.1).

        SOMA-JAX keyword-only extras:
            npz_path: ``SOMA_neutral.npz`` to read instead of
                ``<data_root>/SOMA_neutral.npz``.
            identity_model_path: a SOMA-JAX identity pack
                (:mod:`soma_jax.identity_packs`) to use instead of the
                data-root backend.
            sparse_k: top-K sparse LBS width, overriding what ``mode`` selects.
            fit_joint_regressor: fit the ``skeleton_fit="linear"`` regressor.
                The faithful path uses ``SkeletonTransfer`` and does not need it.
            procedural, correctives_path, usd_path: SOMA-JAX's earlier names
                for ``enable_procedural_transforms``, ``correctives_model_path``
                and ``template_rig_path``; when given they take precedence.

        Returns:
            An instantiated :class:`SOMALayer`.
        """
        # Upstream's parameter names, normalized onto the builder below.
        if procedural is None:
            procedural = bool(enable_procedural_transforms)
        if correctives_path is _UNSET:
            correctives_path = correctives_model_path
        if usd_path is None and template_rig_path is not None:
            usd_path = template_rig_path
        lod = _resolve_body_lod(low_lod, lod)
        if sparse_k is None:
            sparse_k = 8 if mode == "warp" else _ALL_INFLUENCES
        del device  # torch placement; JAX arrays live on the default device
        from pathlib import Path as _Path

        from ..assets import resolve as _resolve
        from ..procedural_transforms import ProceduralTransforms, load_definition
        from ..rig_build import _fit_joint_regressor, build_soma_asset, prune_procedural_joints

        from ..correctives_model import resolve_correctives_model_path
        if data_root is not None:
            data_root = _Path(data_root)
            if not data_root.exists():
                # Upstream falls back to its default assets (`get_assets_dir()`)
                # when the directory does not exist.
                from ..assets import data_root as _assets_data_root
                logger.info("data_root '%s' not found, using the default SOMA-X assets",
                            data_root)
                data_root = _assets_data_root()
        # Upstream resolves the checkpoint right after the asset root, before
        # validating any asset.
        if data_root is not None:
            correctives_root = data_root
        else:
            from ..assets import data_root as _assets_root
            correctives_root = _assets_root()
        correctives_path = resolve_correctives_model_path(
            data_root=correctives_root, correctives_model_path=correctives_path,
            load_correctives_model=load_correctives_model, default_enabled=procedural)
        if correctives_path is not None:
            correctives_path = str(correctives_path)
        if data_root is not None:
            npz_path = npz_path if npz_path is not None else data_root / "SOMA_neutral.npz"
            usd_path = usd_path if usd_path is not None else data_root / "SOMA_template_rig.usda"
            definition_path = data_root / "SOMA_procedural_transforms.json"
            # Upstream validates the core asset first, naming it and the root,
            # then reads the procedural definition, then the template rig.
            if not _Path(npz_path).exists():
                raise FileNotFoundError(
                    f"Core asset 'SOMA_neutral.npz' not found in '{data_root}'.\n"
                    "Please ensure the assets are correctly downloaded and extracted.")
            definition = (load_definition(definition_path)
                          if _Path(definition_path).exists() else None)
            if definition is None and procedural:
                raise FileNotFoundError(
                    "Procedural transforms require the SOMA procedural transform definition "
                    f"at '{definition_path}'.")
            if not _Path(usd_path).exists():
                if procedural:
                    raise FileNotFoundError(
                        "Procedural transforms require the universal SOMA template rig with "
                        f"twist joints at '{usd_path}'.")
                from ..io import SOMA_TEMPLATE_RIG_FILENAME, missing_soma_neutral_rig_keys
                with np.load(npz_path, allow_pickle=False) as _core:
                    missing = missing_soma_neutral_rig_keys(_core)
                if missing:
                    raise FileNotFoundError(
                        f"Template rig asset not found: {usd_path}. "
                        f"Core asset '{npz_path}' is a slim SOMA_neutral.npz and no longer "
                        f"contains rig fields: {', '.join(missing)}. Install "
                        f"'{SOMA_TEMPLATE_RIG_FILENAME}' next to the core asset.")
        else:
            npz_path = _resolve("SOMA_neutral.npz") if npz_path is None else npz_path
            definition_path = _resolve("SOMA_procedural_transforms.json")
            definition = load_definition(definition_path)

        # Two-rig construction, mirroring upstream `body/soma.py`: the public
        # 78-joint rig carries the identity model, the skeleton fit and the bind
        # data, while the expanded 110-joint template skeleton is used for FK/LBS
        # only (procedural layers) or pruned away (`procedural=False`). Both
        # halves come from the SAME template read: upstream derives the public
        # rig with `derive_soma_rig_without_procedural_joints` (ported as
        # `prune_procedural_joints`, which aggregates each dropped joint's
        # weights onto its nearest kept parent). Earlier versions of this port
        # read the public half from a cached `SOMA_neutral_fixed.npz`, which
        # silently pinned it to whatever template the cache was built from.
        with np.load(npz_path, allow_pickle=False) as _core:
            npz_joint_names = (tuple(map(str, _core["joint_names"]))
                               if "joint_names" in _core.files else None)
        if definition is not None:
            if (npz_joint_names is not None
                    and npz_joint_names != tuple(definition.public_joint_names)):
                raise ValueError(
                    "SOMA procedural transform definition public joints do not match "
                    "SOMA_neutral.npz joint_names.")
            public_names = definition.main_joint_names
        elif npz_joint_names is not None:
            public_names = npz_joint_names
        else:
            raise FileNotFoundError(
                f"Core asset '{npz_path}' does not contain joint_names. Install "
                f"'{_Path(definition_path).name}' next to it so the public SOMA joint contract "
                "can be derived from the procedural definition.")
        expanded = build_soma_asset(npz_path, usd_path, "mid", fit_joint_regressor=False)
        if procedural:
            from ..procedural_transforms import has_soma_twist_joints
            if not has_soma_twist_joints(expanded["joint_names"], segments=definition.segments):
                raise ValueError(
                    "Procedural transforms require a SOMA template rig with twist joints.")
        public = prune_procedural_joints(expanded, public_names)
        if fit_joint_regressor:
            public["J_regressor"] = _fit_joint_regressor(
                public["bind_shape"], public["bind_pose_world"],
                public["weights"], public["parents"])

        common = dict(identity_model_type=identity_model_type,
                      identity_model_path=identity_model_path,
                      correctives_path=correctives_path, sparse_k=sparse_k,
                      reference_pose=reference_pose, output_unit=output_unit)
        # Every backend, SOMA's own PCA included, is built from ``data_root`` as
        # upstream's constructor builds it (``create_identity_model``), so
        # ``layer.identity_model`` is upstream's class with upstream's
        # ``forward``. Only a SOMA-JAX identity pack (``identity_model_path``)
        # takes the pack route.
        use_data_root_backend = identity_model_path is None
        if lod == "xlo":
            layer, skin_source = cls._build_xlo_layer(
                public, usd_path, public_names,
                fit_joint_regressor=fit_joint_regressor,
                identity_model_kwargs=(identity_model_kwargs or {}) if use_data_root_backend
                else None, data_root=data_root, **common)
        else:
            identity = None
            if use_data_root_backend:
                identity = cls._upstream_identity_model(
                    identity_model_type, public, lod, identity_model_kwargs or {},
                    data_root=data_root)
            layer = cls._from_soma_data(public, lod=lod, identity_model=identity, **common)
            skin_source = expanded
            if lod == "low":
                # Upstream slices the mid template's weights by `lod_mid_to_low`
                # (`skinning_weights[nv_lod_mid_to_low]`); the skeleton and every
                # other skinning-rig array are vertex-independent.
                mid_to_low = np.asarray(expanded["lod_mid_to_low"], np.int64)
                skin_source = dict(expanded,
                                   weights=np.asarray(expanded["weights"])[mid_to_low])
        # Upstream-named views: the checkpoint loaded, the SOMA PCA size, and an
        # xlo layer's low-LOD skeleton-fit ids (static host data; the procedural
        # copy below carries them).
        object.__setattr__(layer, "_correctives_model_path",
                           correctives_path if layer._has_trained_correctives else None)
        object.__setattr__(layer, "_num_shape_components",
                           int(np.shape(public["shapedirs"])[-1]))
        if lod == "xlo":
            object.__setattr__(layer, "_xlo_skeleton_mid_to_low_host",
                               _host(np.asarray(public["lod_mid_to_low"], np.int64)))
        # What upstream's `PoseInversion` carries over to its internal low-LOD
        # layer; the corrective checkpoint and reference pose are left out so
        # they take the constructor defaults there, as upstream's do.
        object.__setattr__(layer, "_build_recipe", _BuildRecipe("from_upstream_assets", dict(
            npz_path=npz_path, usd_path=usd_path, identity_model_type=identity_model_type,
            identity_model_path=identity_model_path, sparse_k=sparse_k, procedural=procedural,
            fit_joint_regressor=fit_joint_regressor, identity_model_kwargs=identity_model_kwargs,
            data_root=data_root, output_unit=output_unit)))
        if not procedural:
            return layer

        procedural_transforms = ProceduralTransforms(definition)
        # `aligned_x_swing_twist` (what the shipped JSON asks for) measures
        # twist in the bind-aligned frame: `q_current * conj(q_bind) * q_align`.
        # Without bind data the extractor silently falls back to a
        # start-joint scalar that is not upstream-equivalent.
        #
        # SOMA-X v0.2.2 ("Fixes procedural twist evaluation to use the USD
        # skin bind pose") takes BOTH `bind_quaternions` and the segment
        # alignment quaternions from `target_bind_pose_world`; before it
        # they came from `target_t_pose_world`. The two differ on the
        # template, so this choice is visible even at rest: with the
        # T-pose, `q_current * conj(q_bind)` is identity at rest and the
        # twist vanishes; with the bind pose it does not, and upstream's
        # rest mesh moves by ~2 cm on the limb twist segments.
        # The joint-orient step is unaffected — it stays on the T-pose.
        if layer._public_bind_pose_world is not None:
            procedural_transforms.set_bind_data(layer._public_bind_pose_world)
        skin_rig = cls.build_skinning_rig(
            skin_source, layer.joint_names, procedural_transforms, sparse_k=sparse_k)
        return layer.attach_procedural_rig(
            procedural_transforms, layer.joint_names, skin_rig=skin_rig)

    @classmethod
    def _build_xlo_layer(cls, public_mid: dict, usd_path, public_names, *,
                         fit_joint_regressor: bool, identity_model_kwargs=None,
                         data_root=None, **common):
        """The public half of an xlo layer, and the expanded xlo rig for skinning.

        Port of upstream's ``lod == "xlo"`` construction (``soma/body/soma.py``):

        * the layer skins the template's **xlo** mesh — its points are the bind
          shape, its polygons (fan-triangulated) the faces, its binding the
          weights (``rig_data.update(xlo_rig_data)``);
        * the identity model runs on the **low** LOD
          (``identity_uses_low_lod``, ``nv_lod_mid_to_low``) with the low mesh's
          own fan-triangulated faces (``soma_low_lod_faces``);
        * the skeleton fit runs on the public-pruned **low** mesh rig
          (``xlo_skeleton_transfer``), excluding the facial inner geometry
          remapped through ``lod_mid_to_low``;
        * ``identity_lod_transfer`` embeds the xlo points in the low mesh
          (``BarycentricInterpolator(low_points, low_faces, xlo_points)``),
          built in the template's native centimetres — the tetrahedral
          embedding is not scale-invariant;
        * correctives are sliced to the low LOD (``v_index_map=mid_to_low``);
        * ``excluded_vert_ids`` are the nearest xlo vertices to the mid-LOD
          facial geometry (``_nearest_lod_vertex_ids``).

        Returns:
            ``(layer, expanded_xlo_rig)``.
        """
        from ..geometry.barycentric_interp import compute_barycentric_coords
        from ..rig_build import _fit_joint_regressor, merge_template_rig, prune_procedural_joints
        from ..usd_io import fan_triangulate

        mid_to_low = np.asarray(public_mid["lod_mid_to_low"], np.int64)
        n_mid = int(np.asarray(public_mid["bind_shape"]).shape[0])
        facial_mid = np.concatenate([
            np.asarray(public_mid["segment_eye_bags"]).astype(np.int64).ravel(),
            np.asarray(public_mid["segment_mouth_bag"]).astype(np.int64).ravel(),
        ])

        low_rig = merge_template_rig(usd_path, "low")
        xlo_expanded = merge_template_rig(usd_path, "xlo")
        for name, rig in (("low", low_rig), ("xlo", xlo_expanded)):
            if rig.get("face_vert_indices") is None or rig.get("face_vert_counts") is None:
                raise RuntimeError(
                    f"The {name} LOD of SOMA_template_rig.usda carries no face topology, "
                    "which lod='xlo' needs.")
        low_faces = np.asarray(fan_triangulate(np.asarray(low_rig["face_vert_indices"]),
                                               np.asarray(low_rig["face_vert_counts"])), np.int64)
        xlo_faces = np.asarray(fan_triangulate(np.asarray(xlo_expanded["face_vert_indices"]),
                                               np.asarray(xlo_expanded["face_vert_counts"])), np.int64)

        # Identity model input: the mid shape data sliced to the low LOD, with
        # the low mesh's own triangulation (upstream `soma_low_lod_faces`).
        identity_data = _slice_rig_to_low_lod(dict(public_mid))
        identity_data["faces"] = low_faces.astype(np.int32)

        # Skeleton-fit rig: the public-pruned low mesh rig.
        low_public = prune_procedural_joints(low_rig, public_names)
        skeleton_rig = {
            "parents": low_public["parents"],
            "bind_pose_world": low_public["bind_pose_world"],
            "bind_shape": low_public["bind_shape"],
            "weights": low_public["weights"],
            "segment_eye_bags": _mid_ids_to_lod(public_mid["segment_eye_bags"], mid_to_low, n_mid),
            "segment_mouth_bag": _mid_ids_to_lod(public_mid["segment_mouth_bag"], mid_to_low, n_mid),
        }

        # The layer's own (public) rig on the xlo mesh.
        xlo_public = prune_procedural_joints(xlo_expanded, public_names)
        xlo_points_cm = np.asarray(xlo_public["bind_shape"], np.float64)
        layer_data = {k: v for k, v in public_mid.items()
                      if k not in ("lod_mid_to_low", "triangles_low", "mirror_vert_indices",
                                   "segment_eye_bags", "segment_mouth_bag", "shapedirs",
                                   "J_regressor", "face_vert_indices", "face_vert_counts")}
        layer_data.update(
            joint_names=xlo_public["joint_names"],
            parents=xlo_public["parents"],
            weights=xlo_public["weights"],
            bind_pose_world=xlo_public["bind_pose_world"],
            bind_pose_local=xlo_public["bind_pose_local"],
            t_pose_world=xlo_public["t_pose_world"],
            t_pose_local=xlo_public["t_pose_local"],
            bind_shape=xlo_public["bind_shape"],
            v_template=(xlo_points_cm * Unit.CENTIMETERS.meters_per_unit).astype(np.float32),
            faces=xlo_faces.astype(np.int32),
        )
        if fit_joint_regressor:
            layer_data["J_regressor"] = _fit_joint_regressor(
                xlo_public["bind_shape"], xlo_public["bind_pose_world"],
                xlo_public["weights"], xlo_public["parents"])

        # float32, as upstream's `BarycentricInterpolator(....float(), ...)`.
        face_ids, bary = compute_barycentric_coords(
            xlo_points_cm.astype(np.float32),
            np.asarray(low_rig["bind_shape"], np.float32), low_faces)
        lod_transfer = (low_faces.astype(np.int32), np.asarray(face_ids, np.int32),
                        np.asarray(bary, np.float32))
        excluded = _nearest_lod_vertex_ids(
            np.asarray(public_mid["bind_shape"], np.float64), xlo_points_cm, facial_mid)

        identity = None
        if identity_model_kwargs is not None:
            identity = cls._upstream_identity_model(
                common["identity_model_type"], public_mid, "xlo", identity_model_kwargs,
                low_faces=low_faces, data_root=data_root)
        layer = cls._from_soma_data(
            layer_data, lod="xlo", identity_data=identity_data,
            skeleton_rig=skeleton_rig, lod_transfer=lod_transfer,
            excluded_vert_ids=excluded, correctives_v_index_map=mid_to_low,
            identity_model=identity, **common)
        return layer, xlo_expanded

    @staticmethod
    def _upstream_identity_model(identity_model_type: str, public_mid: dict, lod: str,
                                 identity_model_kwargs: dict, low_faces=None, data_root=None):
        """A non-SOMA identity backend built from ``data_root`` as upstream does.

        Upstream ``SOMALayer.__init__``: ``create_identity_model(type, data_root,
        identity_uses_low_lod, ..., nv_lod_mid_to_low=..., soma_low_lod_faces=...,
        vertex_ids_to_exclude=...)``. The low LOD uses the npz's
        ``triangles_low``; xlo evaluates identity on the low LOD too, with the
        template's own low-mesh triangulation; the facial exclusion is remapped
        into the identity LOD's indexing.
        """
        from ..assets import data_root as data_root_default
        from ..body.identity_model import create_identity_model as _create
        facial = np.concatenate([
            np.asarray(public_mid["segment_eye_bags"]).astype(np.int64).ravel(),
            np.asarray(public_mid["segment_mouth_bag"]).astype(np.int64).ravel(),
        ])
        kwargs = dict(identity_model_kwargs)
        if lod == "mid":
            low_lod, excluded = False, facial
        else:
            mid_to_low = np.asarray(public_mid["lod_mid_to_low"], np.int64)
            n_mid = int(np.asarray(public_mid["bind_shape"]).shape[0])
            low_lod, excluded = True, _mid_ids_to_lod(facial, mid_to_low, n_mid)
            kwargs.update(nv_lod_mid_to_low=mid_to_low,
                          soma_low_lod_faces=(public_mid["triangles_low"] if lod == "low"
                                              else low_faces))
        root = data_root_default() if data_root is None else data_root
        return _create(identity_model_type, root, low_lod,
                       vertex_ids_to_exclude=excluded, **kwargs)

    @classmethod
    def _from_soma_data(
        cls,
        soma_data: dict,
        *,
        identity_model_type: str = "soma",
        identity_model_path: Optional[str] = None,
        correctives_path: Optional[str] = None,
        sparse_k: int = 8,
        lod: str = "mid",
        reference_pose=None,
        identity_data: Optional[dict] = None,
        skeleton_rig: Optional[dict] = None,
        lod_transfer: Optional[tuple] = None,
        excluded_vert_ids=None,
        correctives_v_index_map=None,
        identity_model: Optional[BaseIdentityModel] = None,
        output_unit: Unit = Unit.METERS,
    ) -> "SOMALayer":
        """Shared construction from an assembled ``soma_data`` dict.

        Args:
            soma_data: the mid-LOD rig for ``lod`` ``"mid"``/``"low"`` (a low
                layer slices it by ``lod_mid_to_low``, as upstream does), or the
                finished xlo rig for ``"xlo"``.
            identity_data: the data the identity model is built on, when it is
                not ``soma_data`` (xlo: the low LOD).
            skeleton_rig, lod_transfer, excluded_vert_ids: forwarded to
                :class:`SOMALayer` (xlo).
            correctives_v_index_map: the vertex map the corrective checkpoint's
                output layer is sliced by; a low layer uses ``lod_mid_to_low``.
            identity_model: a ready identity backend (the data-root backends of
                :mod:`soma_jax.body.identity_model`); built from ``soma_data`` /
                ``identity_model_path`` when omitted.
        """
        if lod == "low":
            correctives_v_index_map = np.asarray(soma_data["lod_mid_to_low"], dtype=np.int64)
            soma_data = _slice_rig_to_low_lod(soma_data)

        if identity_model is None:
            # SOMA-JAX's own backends: the SOMA PCA from `soma_data`, or a
            # pre-built identity pack (`identity_model_path`).
            model_data = None
            if identity_model_path is not None:
                model_data = dict(np.load(identity_model_path, allow_pickle=True))
            identity_model = create_identity_model(
                identity_model_type, soma_data if identity_data is None else identity_data,
                model_data)

        # Load correctives. A low-LOD layer needs the checkpoint's output layer
        # sliced onto the same vertex subset (upstream's
        # `correctives_vertex_index_map`), or it emits full-resolution offsets
        # against a 4,505-vertex rest shape. An xlo layer evaluates them on the
        # low LOD too and transfers the result.
        correctives = None
        if correctives_path is not None:
            correctives = CorrectivesMLP.load_checkpoint(
                correctives_path, v_index_map=correctives_v_index_map)

        return cls(soma_data, identity_model, correctives, sparse_k=sparse_k,
                   reference_pose=reference_pose, lod=lod, skeleton_rig=skeleton_rig,
                   lod_transfer=lod_transfer, excluded_vert_ids=excluded_vert_ids,
                   identity_model_type=identity_model_type, output_unit=output_unit)

    def prepare_identity(
        self,
        identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
        repose_to_bind_pose: bool = True,
        global_scale: float | jnp.ndarray = 1.0,
        kwargs: Optional[dict] = None,
        *,
        skeleton_fit: str = "auto",
        return_bind_transforms: bool = False,
        return_identity_rest_shape: bool = False,
    ):
        """Compute rest-pose vertices and joint positions for given identity.

        Called once per identity (not per pose); cache the result across calls
        to pose().

        Args:
            identity_coeffs: (B, C) or (C,) identity shape coefficients.
            scale_params: optional (B, S) or (S,) body-part scale parameters.
            repose_to_bind_pose: if True (default, matching SOMA-X), run the
                identity-fit skeleton through the bind-pose local rotations
                stored in ``bind_pose_local``, so the returned rest mesh +
                joints are *posed to the bind pose* rather than the T-pose —
                required for the trained correctives to operate in their
                training frame. No-op when ``bind_pose_local`` is unavailable.
            global_scale: uniform scale scalar applied to rest verts and
                joints. Matches SOMA-X's ``global_scale`` arg.
            skeleton_fit: ``"full"`` — SOMA-X's exact per-identity skeleton
                fit (``SkeletonTransfer.fit``: RBF joint regression +
                two-stage Kabsch; requires an asset with ``bind_shape``).
                ``"linear"`` — the fast linear ``J_regressor`` approximation
                (a SOMA-JAX alternative, NOT what upstream does).
                ``"auto"`` (default) — full when available, else linear.
            return_bind_transforms: if True, also return the fitted bind
                world transforms (B, J, 4, 4) for the faithful
                ``pose(bind_transforms=...)`` path (None on the linear path).
            kwargs: optional dict forwarded to the identity model's
                ``get_rest_shape`` (upstream's ``kwargs``; e.g. MHR's
                ``bone_length_flexibles``). Data-root backends only.
            return_identity_rest_shape: if True, also return the identity
                model's own rest shape before any LOD transfer or repose —
                upstream's ``_cached_identity_rest_shape``. An xlo layer needs
                it to apply correctives (see :meth:`pose`); on other LODs it is
                the un-reposed ``rest_verts``.

        Returns:
            (rest_verts, rest_joints) — plus ``bind_transforms`` when
            ``return_bind_transforms=True``, then the identity rest shape when
            ``return_identity_rest_shape=True``. Batched (B, ...) or unbatched
            matching the input.
        """
        unbatched = not isinstance(identity_coeffs, dict) and identity_coeffs.ndim == 1
        if unbatched:
            identity_coeffs = identity_coeffs[None]
            if scale_params is not None:
                scale_params = scale_params[None]

        rest_verts, rest_joints = self._identity_forward(identity_coeffs, scale_params, kwargs)

        # Uniform scale applied to BOTH verts and joints so the skeleton stays
        # rigidly attached. SOMA-X applies global_scale inside the identity
        # forward; we apply it here once for any backend (smpl/mhr/anny/...).
        # A per-batch (B,) scale must broadcast over the BATCH axis. Multiplying
        # (B, V, 3) by (B,) broadcasts against xyz instead: it raises for most B
        # and, when B happens to equal 3, silently scales x/y/z differently.
        # Upstream reshapes for exactly this reason.
        gs = jnp.asarray(global_scale, dtype=rest_verts.dtype)
        if gs.ndim == 1:
            gs = gs.reshape(-1, 1, 1)
        elif gs.ndim > 1:
            raise ValueError(f"global_scale must be scalar or (B,), got {gs.shape}")
        rest_verts = rest_verts * gs
        if rest_joints is not None:
            rest_joints = rest_joints * gs

        # Upstream: `_cached_identity_rest_shape` is the identity model's output
        # (low LOD on an xlo layer); `identity_lod_transfer` carries it onto the
        # skinned mesh. The skeleton is fitted on the former.
        identity_rest = rest_verts
        if self._lod_transfer_host is not None:
            rest_verts = self._apply_lod_transfer(identity_rest)

        if skeleton_fit not in ("auto", "full", "linear"):
            raise ValueError(f"skeleton_fit must be auto|full|linear, got {skeleton_fit!r}")
        if skeleton_fit == "full" and self.skeleton_transfer is None:
            raise ValueError(
                "skeleton_fit='full' needs an asset with bind_shape + "
                "bind_pose_world (see docs/INSTALL.md); this asset has neither."
            )
        use_full = skeleton_fit == "full" or (
            skeleton_fit == "auto" and self.skeleton_transfer is not None
        )
        if not use_full and rest_joints is None:
            raise ValueError(
                "skeleton_fit='linear' needs the linear J_regressor, which this layer "
                "was built without (fit_joint_regressor=False).")

        bind_transforms = None
        if use_full:
            # SOMA-X: skeleton_transfer.fit(rest_shape) -> per-identity bind
            # world transforms; joints are their translation components.
            bind_transforms = self.skeleton_transfer.fit(identity_rest)  # (B, J, 4, 4)
            rest_joints = bind_transforms[..., :3, 3]

        if repose_to_bind_pose and self._has_bind_pose_local():
            if bind_transforms is not None:
                rest_verts, bind_transforms = self._repose_full(rest_verts, bind_transforms)
                rest_joints = bind_transforms[..., :3, 3]
            else:
                rest_verts, rest_joints = self._repose_to_bind_pose(
                    rest_verts, rest_joints,
                )

        if unbatched:
            rest_verts = rest_verts[0]
            rest_joints = rest_joints[0]
            identity_rest = identity_rest[0]
            if bind_transforms is not None:
                bind_transforms = bind_transforms[0]

        s = self._unit_scale
        if s != 1.0:
            rest_verts, rest_joints, identity_rest = rest_verts * s, rest_joints * s, identity_rest * s
            if bind_transforms is not None:
                bind_transforms = bind_transforms.at[..., :3, 3].multiply(s)
        out = (rest_verts, rest_joints)
        if return_bind_transforms:
            out = out + (bind_transforms,)
        if return_identity_rest_shape:
            out = out + (identity_rest,)
        return out

    def _identity_forward(self, identity_coeffs, scale_params, kwargs=None):
        """Rest vertices (metres) and, when a regressor exists, rest joints.

        Upstream passes ``scale_params`` to the identity model on every backend
        but SOMA, whose scales are bone-length controls applied at pose time.
        """
        from ..body.identity_model import BaseIdentityModel as _DataRootModel
        identity_scales = None if self.identity_model_type == "soma" else scale_params
        if isinstance(self.identity_model, _DataRootModel):
            verts = self.identity_model.forward(identity_coeffs, identity_scales, kwargs)
            joints = None
            if self.J_regressor is not None and self.J_regressor.shape[1] == verts.shape[1]:
                joints = jnp.einsum("jv,bvd->bjd", self.J_regressor, verts)
            return verts, joints
        if kwargs:
            raise ValueError("kwargs are forwarded to data-root identity backends only.")
        return self.identity_model.forward(identity_coeffs, identity_scales)

    def _apply_lod_transfer(self, verts: jnp.ndarray) -> jnp.ndarray:
        """Carry low-LOD vertices onto an xlo layer's mesh (upstream ``identity_lod_transfer``)."""
        from ..geometry.barycentric_interp import barycentric_interpolate
        faces, face_ids, bary = self._lod_transfer_host.a
        return barycentric_interpolate(verts, jnp.asarray(faces), jnp.asarray(face_ids),
                                       jnp.asarray(bary))

    def _corrective_offsets(self, rotmats, rest_verts, global_scale, identity_rest_shape):
        """Per-vertex corrective offsets on this layer's mesh.

        Upstream ``SOMALayer.pose``: the network output is scaled by the cached
        global scale; on an xlo layer it lives on the low LOD and becomes
        ``identity_lod_transfer(identity_rest + correctives) - rest_shape``.
        That replaces the (possibly reposed) xlo rest shape with the transfer
        of the *un-reposed* low-LOD identity plus correctives — upstream's
        formula, reproduced as-is.
        """
        offsets = _scale_correctives(self.correctives.offsets(rotmats), global_scale)
        if self._lod_transfer_host is None:
            return offsets
        if identity_rest_shape is None:
            raise ValueError(
                "Correctives on an xlo layer need identity_rest_shape: pass the value "
                "prepare_identity(..., return_identity_rest_shape=True) returns.")
        return self._apply_lod_transfer(identity_rest_shape + offsets) - rest_verts

    def _repose_full(
        self,
        rest_verts: jnp.ndarray,
        bind_transforms: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Re-pose the identity into the bind pose against the FITTED skeleton.

        The faithful mirror of both upstream ``prepare_identity(
        repose_to_bind_pose=True)`` branches (``soma/body/soma.py``; the
        procedural one, ``_repose_public_bind_pose``, runs the same pose on the
        public rig): rebind to the fitted transforms, pose with the
        ``bind_pose_local`` rotations in absolute mode with
        ``align_translation=[0, 0, 0]`` — Hips X/Z anchored at the origin, the
        lowest joint floored at Y=0 — then ``_pin_virtual_root_to_origin``.
        Returns the reposed rest mesh and the NEW bind world transforms.

        The frame matters beyond LBS: posing is invariant to a rigid offset of
        the (rest, bind) pair, but an xlo layer's correctives replace the rest
        shape with a transfer of the *un-reposed* identity, so the binds must
        sit where upstream's do.
        """
        from ..geometry.batched_skinning import pose_from_bind
        B, J = bind_transforms.shape[:2]
        bind_local = np.array(self._bind_pose_local_np, dtype=np.float32)
        if float(np.abs(bind_local[..., :3, 3]).max()) > 10.0:   # cm-scale rig
            bind_local[..., :3, 3] *= 0.01
        R = jnp.broadcast_to(jnp.asarray(bind_local[None, :, :3, :3]), (B, J, 3, 3))
        # Ignored under `align_translation`, as upstream ignores its
        # `global_translation=bind_pose_local[..., 1, :3, 3]` there.
        hips_t = jnp.broadcast_to(jnp.asarray(bind_local[None, 1, :3, 3]), (B, 3))
        posed_verts, T_world = pose_from_bind(
            bind_transforms, rest_verts, self.weights, self.skeleton_levels,
            self._parents_np, R, hips_t, hips_idx=1,
            align_translation=jnp.zeros((3,), R.dtype),
        )
        # `_pin_virtual_root_to_origin`: the dummy Root must not carry the
        # identity fit's offsets (or the floor shift).
        T_world = T_world.at[:, 0].set(jnp.eye(4, dtype=T_world.dtype))
        return posed_verts, T_world

    def _has_bind_pose_local(self) -> bool:
        """True when the asset shipped both ``bind_pose_world`` (as a JAX
        attribute) and ``bind_pose_local`` (as the numpy mirror used by
        :py:meth:`_repose_to_bind_pose`). Older soma_jax assets that predate
        the augmentation step don't carry either field, in which case
        ``prepare_identity(repose_to_bind_pose=True)`` silently no-ops."""
        return self._public_bind_pose_world is not None and getattr(
            self, "_bind_pose_local_np", None,
        ) is not None

    def _repose_to_bind_pose(
        self,
        rest_verts: jnp.ndarray,
        rest_joints: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Re-pose the T-pose identity into the bind pose via RestJointSkinning.

        Mirrors SOMA-X's ``prepare_identity(repose_to_bind_pose=True)``
        non-procedural branch (third_party/SOMA-X/soma/soma.py:1408): drive
        BatchedSkinning with ``bind_pose_local[..., :3, :3]`` rotations and
        the hips translation, in absolute-pose mode. The returned vertices
        live in the bind-pose frame, which is the frame the trained
        correctives expect as input.
        """
        from ..geometry.batched_skinning import RestJointSkinning
        B = rest_verts.shape[0]
        J = rest_joints.shape[1]
        # Upstream bind transforms ship in centimeters; identity_model output
        # is in meters. Apply the same cm->m scale SOMA-X does inside
        # `_convert_units` so the reposed mesh stays in meters.
        bind_local = np.array(self._bind_pose_local_np, dtype=np.float32)
        if float(np.abs(bind_local[..., :3, 3]).max()) > 10.0:  # clearly cm-scale
            bind_local[..., :3, 3] = bind_local[..., :3, 3] * 0.01
        bind_local = jnp.asarray(bind_local)
        joint_orient = self._public_t_pose_world[..., :3, :3] if self._public_t_pose_world is not None else None

        bs = RestJointSkinning(
            rest_verts=np.asarray(rest_verts[0]),
            rest_joints=np.asarray(rest_joints[0]),
            weights=np.asarray(self.weights),
            parents=self._parents_np,
            joint_orient=None if joint_orient is None else np.asarray(joint_orient),
            sparse_k=5,
        )
        bind_R = jnp.broadcast_to(bind_local[None, :, :3, :3], (B, J, 3, 3))
        # SOMA-X uses bind_pose_local[1, :3, 3] (Hips' local translation) as
        # the global_translation argument and floor-locks via
        # align_translation=[0,0,0].
        hips_t = jnp.broadcast_to(bind_local[None, 1, :3, 3], (B, 3))
        align_t = jnp.zeros((B, 3), dtype=jnp.float32)
        posed_verts, posed_joints = bs.pose(
            bind_R, hips_t, absolute_pose=True, align_translation=align_t,
        )
        return posed_verts, posed_joints

    # TODO: give pose() upstream's signature and semantics (poses, transl=None,
    # pose2rot=True, ..., identity=...) and move this low-level form to its own
    # name; see TODO.md.
    def pose(
        self,
        rotmats: jnp.ndarray,
        transl: jnp.ndarray,
        rest_verts: jnp.ndarray,
        rest_joints: jnp.ndarray,
        joint_orient: Optional[jnp.ndarray] = None,
        use_sparse: bool = True,
        absolute_pose: bool = False,
        apply_correctives: Optional[bool] = None,
        bind_transforms: Optional[jnp.ndarray] = None,
        bone_scales: Optional[jnp.ndarray] = None,
        fk_only: bool = False,
        global_scale: float | jnp.ndarray = 1.0,
        *,
        reference_pose=None,
        identity_rest_shape: Optional[jnp.ndarray] = None,
    ) -> SOMAOutput:
        """Apply pose (rotations + translation) to rest vertices.

        Args:
            identity_rest_shape: xlo layers only — the low-LOD identity rest
                shape from ``prepare_identity(..., return_identity_rest_shape=
                True)``. Correctives are evaluated on the low LOD there, and
                upstream applies them as ``rest = transfer(identity_rest +
                correctives)``, so they cannot be added to the xlo
                ``rest_verts`` directly. Required with ``apply_correctives``
                on an xlo layer, ignored elsewhere.
            global_scale: the same uniform scale passed to
                :meth:`prepare_identity`. Corrective offsets are trained in
                unscaled units, so upstream multiplies them by the cached
                global scale before adding them to the rest shape
                (``soma/soma.py``); pass it here or scaled identities get
                unscaled correctives.
            rotmats: (B, J, 3, 3) local rotation matrices.
            transl: (B, 3) root translation. On the faithful
                ``bind_transforms`` path this drives the hips slot inside FK
                (SOMA-X's "hips world position" semantic). On the legacy path
                (``bind_transforms=None``) it is an additive post-LBS shift
                (SMPL semantic) — a SOMA-JAX alternative.
            rest_verts: (B, V, 3) rest-pose vertices from prepare_identity().
            rest_joints: (B, J, 3) rest joint positions from prepare_identity().
            joint_orient: optional (J, 3, 3) T-pose joint orientation correction.
            use_sparse: if True, use sparse top-K LBS (faster when K<<J).
            absolute_pose: if True, treat ``rotmats`` as absolute skinning-frame
                rotations and SKIP the joint-orient remap (mirrors SOMA-X's
                ``BatchedSkinning.pose(absolute_pose=True)`` path — used for
                BVH input and PoseInversion output).
            apply_correctives: if True (default), run the pose-corrective MLP and
                add its per-vertex displacement before LBS. Mirrors SOMA-X's
                ``SOMALayer.pose(apply_correctives=...)``. Set False to skip the
                MLP entirely (LBS-only forward — e.g. for runtime benchmarking
                or when no trained checkpoint is loaded).
            bind_transforms: optional (B, J, 4, 4) per-identity bind world
                transforms from ``prepare_identity(return_bind_transforms=
                True)``. When given, skinning runs against these binds via
                ``pose_from_bind`` — the faithful mirror of SOMA-X's
                ``BatchedSkinning.rebind + pose``. When None, the simplified
                rest-joint LBS path runs instead (SOMA-JAX alternative).
            bone_scales: optional (B, S) bone-length multipliers for the active
                controls listed by ``scale_param_names`` — SOMA-X's SOMA-backend
                ``scale_params``. Upstream caches these in ``prepare_identity``;
                this layer is immutable, so they are passed per pose call.
                Requires the faithful ``bind_transforms`` path.
            fk_only: run forward kinematics only and skip skinning, as in
                SOMA-X's ``pose(fk_only=True)``. ``vertices`` is then None.
            reference_pose: SOMA-X v0.3.1 reference (array or
                :meth:`get_reference_pose` dict), used as ``joint_orient``.
                Rejected with ``absolute_pose`` or an explicit ``joint_orient``.
                Unlike upstream's ``pose()`` — which takes a prepared identity
                and always orients — this low-level entry point orients only
                when asked, so the constructor default applies in
                :meth:`__call__`, not here.

        Lengths — ``transl``, ``rest_verts``, ``rest_joints``, the
        translations of ``bind_transforms`` and ``identity_rest_shape`` — are in
        the layer's ``output_unit``, as :meth:`prepare_identity` returns them,
        and so are the outputs.

        Returns:
            SOMAOutput with posed ``vertices`` (B, V, 3; None when
            ``fk_only``), ``joints`` (B, J, 3) and ``transforms``
            (B, J, 4, 4) world joint transforms.
        """
        s = self._unit_scale
        if s != 1.0:
            m = 1.0 / s
            transl = jnp.asarray(transl) * m
            rest_verts = jnp.asarray(rest_verts) * m
            rest_joints = None if rest_joints is None else jnp.asarray(rest_joints) * m
            if bind_transforms is not None:
                bind_transforms = jnp.asarray(bind_transforms).at[..., :3, 3].multiply(m)
            if identity_rest_shape is not None:
                identity_rest_shape = jnp.asarray(identity_rest_shape) * m
        out = self._pose_m(
            rotmats, transl, rest_verts, rest_joints, joint_orient, use_sparse, absolute_pose,
            apply_correctives, bind_transforms, bone_scales, fk_only, global_scale,
            reference_pose=reference_pose, identity_rest_shape=identity_rest_shape)
        if s == 1.0:
            return out
        return SOMAOutput(
            vertices=None if out.vertices is None else out.vertices * s,
            joints=None if out.joints is None else out.joints * s,
            transforms=None if out.transforms is None else out.transforms.at[..., :3, 3].multiply(s),
        )

    def _pose_m(
        self,
        rotmats: jnp.ndarray,
        transl: jnp.ndarray,
        rest_verts: jnp.ndarray,
        rest_joints: jnp.ndarray,
        joint_orient: Optional[jnp.ndarray] = None,
        use_sparse: bool = True,
        absolute_pose: bool = False,
        apply_correctives: Optional[bool] = None,
        bind_transforms: Optional[jnp.ndarray] = None,
        bone_scales: Optional[jnp.ndarray] = None,
        fk_only: bool = False,
        global_scale: float | jnp.ndarray = 1.0,
        *,
        reference_pose=None,
        identity_rest_shape: Optional[jnp.ndarray] = None,
    ) -> SOMAOutput:
        """:meth:`pose` in metres (every length already converted)."""
        if reference_pose is not None:
            if absolute_pose:
                raise ValueError("reference_pose cannot be combined with absolute_pose=True.")
            if joint_orient is not None:
                raise ValueError("Pass either reference_pose or joint_orient, not both.")
            joint_orient = self._resolve_reference_pose(reference_pose)
        # None -> apply correctives when a trained checkpoint is loaded.
        # Upstream can default this to True because its default constructor
        # loads a real checkpoint; this layer defaults to none, so an
        # unconditional True would silently add zeros. Asking for them
        # explicitly without a checkpoint is still an error.
        if apply_correctives is None:
            apply_correctives = self._has_trained_correctives

        # Apply joint orient correction (T-pose alignment).
        # Aligns local rotations to bone-aligned frames defined by joint_orient.
        # SOMA-X formula: R_out[j] = orient[parent[j]].T @ R_in[j] @ orient[j].
        # Callers that drive trained correctives should pass
        # joint_orient=layer.public_rig_view().t_pose_world[..., :3, :3] explicitly so the input
        # frame matches the checkpoint's training frame.
        # On the two-rig procedural path the rotations are already expanded, so the
        # orient must come from the expanded rig too — the public `t_pose_world`
        # has the wrong joint count and the wrong parent chain.
        _rig = self._skin_rig
        _expanded = _rig is not None and rotmats.shape[-3] == len(_rig["joint_names"])
        if joint_orient is not None and not absolute_pose:
            orient_parents = _rig["parents"].a if _expanded else self._parents_np
            if _expanded and joint_orient.shape[0] != rotmats.shape[-3]:
                joint_orient = _rig["t_pose_world"][..., :3, :3]
            rotmats = apply_joint_orient_local(rotmats, joint_orient, orient_parents)

        if bind_transforms is not None:
            # ---- faithful SOMA-X path: rebind + pose against the fitted bind.
            from ..geometry.batched_skinning import pose_from_bind
            if apply_correctives and not fk_only:
                # Upstream (v0.3) `SOMALayer.pose`: correctives are trained on the
                # twist rig and refused on the legacy public-rig layer.
                if self._skin_rig is None and self._procedural is None:
                    raise RuntimeError(
                        "SOMALayer correctives require procedural transforms; construct with "
                        "procedural=True (upstream's enable_procedural_transforms=True) or "
                        "pass apply_correctives=False.")
                if not self._has_trained_correctives:
                    raise RuntimeError(
                        "apply_correctives=True but no corrective model is loaded. Construct with "
                        "a valid correctives_path (upstream correctives_model_path) or pass "
                        "apply_correctives=False.")
                rest_verts = rest_verts + self._corrective_offsets(
                    rotmats, rest_verts, global_scale, identity_rest_shape)
            wv = self.weight_values if use_sparse else None
            wi = self.weight_indices if use_sparse else None
            weights, levels, parents = self.weights, self.skeleton_levels, self._parents_np
            scales = None if bone_scales is None else self.full_bone_scales(bone_scales)

            # Two-rig procedural path: FK and LBS run on the expanded skeleton,
            # everything upstream of here (identity, skeleton fit, bind) stayed on
            # the public rig — which is what upstream does
            # (`skeleton_transfer.skinning_weights` is (V, 78) in both its modes).
            rig = self._skin_rig
            if rig is not None and rotmats.shape[-3] == len(self.joint_names):
                return self._pose_procedural(
                    bind_transforms, rest_verts, rotmats, transl,
                    use_sparse=use_sparse, fk_only=fk_only, bone_scales=scales)

            posed_verts, T_world = pose_from_bind(
                bind_transforms, rest_verts, weights, levels,
                parents, rotmats, transl, hips_idx=1,
                weight_values=wv, weight_indices=wi,
                local_translation_scales=scales, skip_lbs=fk_only,
            )
            return SOMAOutput(
                vertices=posed_verts,
                joints=T_world[..., :3, 3],
                transforms=T_world,
            )

        if bone_scales is not None:
            raise ValueError(
                "bone_scales require the faithful bind path; pass bind_transforms "
                "from prepare_identity(return_bind_transforms=True)."
            )

        # FK: compute global transforms per sample.
        # Use numpy parents in the closure — numpy arrays are treated as static
        # constants by JAX's tracer, avoiding vmap shape confusion.
        parents_np = self._parents_np
        G = jax.vmap(
            lambda R, j: forward_kinematics(R, j, parents_np)
        )(rotmats, rest_joints)  # (B, J, 4, 4)

        # Bone transforms: (B, J, 3, 4)
        bone_T = lbs_transforms(G, rest_joints)

        if fk_only:
            return SOMAOutput(
                vertices=None,
                joints=G[:, :, :3, 3] + transl[:, None, :],
                transforms=G,
            )

        # Pose correctives: (B, V, 3). Skipped entirely when disabled — the
        # zero array keeps the LBS call's signature identical without paying
        # for the (B, K) @ (K, 3V) corrective matmul.
        if apply_correctives:
            if not self._has_trained_correctives:
                raise RuntimeError(
                    "apply_correctives=True but no corrective model is loaded. Construct with "
                    "a valid correctives_path (upstream correctives_model_path) or pass "
                    "apply_correctives=False.")
            correctives = self._corrective_offsets(
                rotmats, rest_verts, global_scale, identity_rest_shape)
        else:
            correctives = jnp.zeros_like(rest_verts)

        # LBS
        if use_sparse and self.weight_indices is not None:
            posed_verts = lbs_sparse(
                rest_verts, correctives, bone_T,
                self.weight_values, self.weight_indices
            )
        else:
            posed_verts = lbs_blend(rest_verts, correctives, bone_T, self.weights)

        # Apply root translation
        posed_verts = posed_verts + transl[:, None, :]

        # Global joint positions
        posed_joints = G[:, :, :3, 3] + transl[:, None, :]

        return SOMAOutput(vertices=posed_verts, joints=posed_joints, transforms=G)

    def __call__(
        self,
        params: SOMAParams,
        apply_correctives: Optional[bool] = None,
        absolute_pose: bool = False,
        fk_only: bool = False,
        *,
        global_scale: float | jnp.ndarray = 1.0,
        reference_pose=None,
        kwargs: Optional[dict] = None,
        repose_to_bind_pose: Optional[bool] = None,
    ) -> SOMAOutput:
        """Full forward pass: identity + pose (mirrors SOMA-X ``forward``).

        ``global_scale`` is applied once to the identity (rest verts + joints)
        and forwarded to :meth:`pose` so corrective offsets, which are trained
        in unscaled units, are scaled to match — upstream does both from its
        cached scale, so passing it here keeps the two in step.

        Faithful to upstream ``SOMALayer.forward``: the identity is prepared
        with ``repose_to_bind_pose=apply_correctives or procedural`` (SOMA-X
        v0.2.2 — procedural layers always repose, which is what makes SMPL /
        SMPL-X identities pose correctly there) and the full skeleton fit when
        the asset supports it, the reference joint orient is applied (unless
        ``absolute_pose``), and skinning runs against the fitted bind
        transforms. On legacy assets without bind data this degrades to the
        simplified linear path.

        Args:
            params: SOMAParams with poses, transl, identity_coeffs, etc.
            apply_correctives: run the pose-corrective MLP (upstream default).
                An untrained model contributes exactly zero.
            absolute_pose: treat rotations as absolute skinning-frame (skip
                the T-pose joint-orient remap), as in upstream.
            fk_only: skip skinning and return joints/transforms only.
            reference_pose: SOMA-X v0.3.1. The reference the pose rotations are
                relative to: ``(J, 3, 3)`` world orientations or ``(J, 4, 4)``
                transforms in ``public_joint_names`` order, identity virtual
                Root included, shared across the batch — or a dict of
                :meth:`get_reference_pose` arguments. Rotation blocks must be
                finite SO(3) (absolute tolerance 1e-4); translations are
                ignored. Overrides the T-pose convention without changing bind
                geometry or bone lengths. ``None`` uses the constructor default,
                else the current T-pose. ``absolute_pose=True`` bypasses the
                default but rejects an explicit reference. Mutually exclusive
                with ``params.joint_orient`` (a SOMA-JAX-only override).
            kwargs: optional dict forwarded to the identity model's
                ``get_rest_shape``, as upstream's ``forward(kwargs=...)``.
            repose_to_bind_pose: override the identity repose. ``None`` keeps
                upstream ``forward``'s ``apply_correctives or procedural``;
                ``True`` is upstream's ``prepare_identity()`` default, which
                callers use for its ``prepare_identity()`` + ``pose()`` sequence
                (this layer does not cache an identity between the two).

        Returns:
            SOMAOutput with posed vertices, joints and world transforms.
        """
        # Reference pose (SOMA-X v0.3.1): explicit + absolute is an error; the
        # constructor default is bypassed by absolute_pose (and yields to an
        # explicit params.joint_orient); failing both, the current T-pose.
        if reference_pose is not None:
            if absolute_pose:
                raise ValueError("reference_pose cannot be combined with absolute_pose=True.")
            if params.joint_orient is not None:
                raise ValueError("Pass either reference_pose or params.joint_orient, not both.")
            reference_pose = self._resolve_reference_pose(reference_pose)
        elif not absolute_pose and params.joint_orient is None:
            reference_pose = self._default_reference_pose

        # None -> apply correctives when a trained checkpoint is loaded.
        # Upstream can default this to True because its default constructor
        # loads a real checkpoint; this layer defaults to none, so an
        # unconditional True would silently add zeros. Asking for them
        # explicitly without a checkpoint is still an error.
        if apply_correctives is None:
            apply_correctives = self._has_trained_correctives

        # Handle rotation representation.
        # Check rotation-matrix shape first since (J,3,3) also has ndim==3,
        # which would otherwise be misidentified as axis-angle (B,J,3).
        if params.poses.shape[-2:] == (3, 3):
            # Rotation matrices: (B, J, 3, 3) batched or (J, 3, 3) unbatched
            rotmats = params.poses
        elif params.poses.shape[-1] == 6:
            # 6D continuous representation: (B, J, 6) or (J, 6)
            # batch_shape = all dims except the last (6D feature dim)
            batch_shape = params.poses.shape[:-1]
            flat = params.poses.reshape(-1, 6)
            rotmats_flat = jax.vmap(rotation_6d_to_rotmat)(flat)
            rotmats = rotmats_flat.reshape(batch_shape + (3, 3))
        elif params.poses.ndim >= 2 and params.poses.shape[-1] == 3:
            # Axis-angle: (B, J, 3) or (J, 3)
            batch_shape = params.poses.shape[:-1]
            flat = params.poses.reshape(-1, 3)
            rotmats_flat = batch_rodrigues(flat)
            rotmats = rotmats_flat.reshape(batch_shape + (3, 3))
        else:
            raise ValueError(
                f"Unsupported pose shape: {params.poses.shape}. "
                "Expected (B,J,3) axis-angle, (B,J,3,3) rotmat, or (B,J,6) 6D."
            )

        unbatched_poses = rotmats.ndim == 3
        if unbatched_poses:
            rotmats = rotmats[None]
            transl = params.transl[None] if params.transl.ndim == 1 else params.transl
            identity_coeffs = (params.identity_coeffs[None]
                               if not isinstance(params.identity_coeffs, dict)
                               and params.identity_coeffs.ndim == 1
                               else params.identity_coeffs)
            scale_params = (
                params.scale_params[None]
                if params.scale_params is not None and params.scale_params.ndim == 1
                else params.scale_params
            )
        else:
            transl = params.transl
            identity_coeffs = params.identity_coeffs
            scale_params = params.scale_params

        # Expand the public pose onto the procedural rig before FK/LBS. Upstream
        # does this inside `pose()`; here it sits in front of the shared machinery
        # so `pose()` stays a plain "rotations for every rig joint" entry point.
        n_public = len(self.public_joint_names)
        if rotmats.shape[-3] == n_public - 1:
            # Upstream's public contract is the 77 *posable* joints — Root is in
            # the rig but never posed, so `_pad_poses` prepends identity for it,
            # on every layer. (All 78, Root included, are accepted too.)
            eye = jnp.broadcast_to(
                jnp.eye(3, dtype=rotmats.dtype), rotmats.shape[:-3] + (1, 3, 3))
            rotmats = jnp.concatenate([eye, rotmats], axis=-3)
        if self._procedural is not None and rotmats.shape[-3] != len(self.joint_names):
            if rotmats.shape[-3] != n_public:
                raise ValueError(
                    f"Expected {n_public} public joints (or {n_public - 1} posable, "
                    f"excluding Root) for the procedural rig, got {rotmats.shape[-3]}."
                )
            rotmats, _ = self._procedural.extend_to_template_rig(
                rotmats, list(self.joint_names))

        rest_verts, rest_joints, bind_transforms, identity_rest = self.prepare_identity(
            identity_coeffs, scale_params,
            # upstream forward() since v0.2.2: procedural layers always repose
            repose_to_bind_pose=(apply_correctives or self._procedural is not None
                                 if repose_to_bind_pose is None else repose_to_bind_pose),
            return_bind_transforms=True,
            global_scale=global_scale,
            return_identity_rest_shape=True,
            kwargs=kwargs,
        )

        # Upstream applies the reference joint orient unless absolute_pose: an
        # explicit/default reference if any, else the asset's T-pose.
        joint_orient = params.joint_orient
        if reference_pose is not None:
            joint_orient = reference_pose
        elif joint_orient is None and not absolute_pose and self._public_t_pose_world is not None:
            joint_orient = self._public_t_pose_world[..., :3, :3]

        # SOMA-backend `scale_params` are bone-length controls consumed at pose
        # time (upstream caches them in prepare_identity); other identity
        # backends consume them inside the identity model instead.
        bone_scales = None
        if (
            self.identity_model_type == "soma"
            and scale_params is not None
            and bind_transforms is not None
            and self.num_bone_scale_params
        ):
            # Legacy (B, 56) tensors get unit foot scales, as upstream; any
            # other width is rejected, as upstream's `_normalize_soma_bone_scales`.
            bone_scales = self.normalize_bone_scales(scale_params)

        out = self.pose(
            rotmats, transl, rest_verts, rest_joints, joint_orient,
            absolute_pose=absolute_pose,
            apply_correctives=apply_correctives,
            bind_transforms=bind_transforms,
            bone_scales=bone_scales,
            fk_only=fk_only,
            global_scale=global_scale,      # correctives are trained unscaled
            identity_rest_shape=identity_rest,
        )

        # Upstream returns only the public joints: "The expanded twist skeleton is
        # used internally for FK/LBS and cached bind data, but is not returned
        # from pose() / forward()." Vertices are unaffected — they are skinned
        # with the full rig.
        pub = self._public_idx
        if pub is not None and out.joints is not None:
            sel = jnp.asarray(pub)
            out = SOMAOutput(
                vertices=out.vertices,
                joints=out.joints[..., sel, :],
                transforms=None if out.transforms is None else out.transforms[..., sel, :, :],
            )

        if unbatched_poses:
            return SOMAOutput(
                vertices=None if out.vertices is None else out.vertices[0],
                joints=out.joints[0],
                transforms=None if out.transforms is None else out.transforms[0],
            )
        return out

    def forward(
        self,
        poses: jnp.ndarray,
        identity_coeffs,
        scale_params: Optional[jnp.ndarray] = None,
        transl: Optional[jnp.ndarray] = None,
        pose2rot: bool = True,
        apply_correctives: bool = True,
        absolute_pose: bool = False,
        global_scale: float | jnp.ndarray = 1.0,
        kwargs: Optional[dict] = None,
        return_transforms: Optional[bool] = None,
        *,
        reference_pose=None,
    ) -> SOMAPoseOutput:
        """Combined ``prepare_identity`` + ``pose`` with upstream's signature.

        Upstream ``SOMALayer.forward``. Equivalent to :meth:`__call__` with
        upstream's input and output conventions: 77 posable joints in, and a
        :class:`SOMAPoseOutput` whose ``joints`` exclude the virtual Root.

        Args:
            poses: (B, 77, 3) axis-angle, or (B, 77, 3, 3) rotation matrices
                with ``pose2rot=False``; ``poses[:, 0]`` is the Hips rotation.
            identity_coeffs: (B, K) identity coefficients (or a backend's dict).
            scale_params: backend-dependent scales, as for :meth:`__call__`.
            transl: (B, 3) Hips translation in ``output_unit``; ``None`` keeps
                the Hips at the origin.
            pose2rot: convert axis-angle input to rotation matrices.
            apply_correctives: apply the pose-corrective offsets (upstream's
                default ``True`` needs a loaded checkpoint).
            absolute_pose: rotations are absolute, not relative to the T-pose.
            global_scale: uniform scale scalar or (B,) array.
            kwargs: forwarded to the identity model's ``get_rest_shape``.
            return_transforms: deprecated; ``transforms`` is always returned.
            reference_pose: a :meth:`get_reference_pose` selector or world
                orientations, as for :meth:`__call__`.

        Returns:
            :class:`SOMAPoseOutput` with ``vertices`` (B, V, 3), ``joints``
            (B, 77, 3) and ``transforms`` (B, 78, 4, 4) — the public skeleton,
            Root included.
        """
        if reference_pose is not None and absolute_pose:
            raise ValueError("reference_pose cannot be combined with absolute_pose=True.")
        if return_transforms is not None:
            warnings.warn(
                "return_transforms is deprecated; 'transforms' is always included in the result.",
                DeprecationWarning, stacklevel=2)
        poses = jnp.asarray(poses)
        if pose2rot:
            rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(poses.shape[:-1] + (3, 3))
        else:
            rotations = poses
        if transl is None:
            transl = jnp.zeros((rotations.shape[0], 3), rotations.dtype)
        out = self(
            SOMAParams(poses=rotations, transl=jnp.asarray(transl),
                       identity_coeffs=identity_coeffs, scale_params=scale_params),
            apply_correctives=apply_correctives, absolute_pose=absolute_pose,
            global_scale=global_scale, reference_pose=reference_pose, kwargs=kwargs)
        return SOMAPoseOutput(vertices=out.vertices, joints=out.joints[..., 1:, :],
                              transforms=out.transforms)

    def extend_rig_with_procedural_transforms(
        self,
        procedural_def_path: str,
        mode: str = "aligned_x_swing_twist",
    ) -> tuple[
        "ProceduralTransforms",
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
    ]:
        """Load the procedural twist-joint definition and build the full-rig
        bind world transforms + parents + joint names.

        Mirrors SOMA-X's ``enable_procedural_transforms=True`` SOMALayer
        construction (third_party/SOMA-X/soma/soma.py around L600) but as a
        post-init helper since soma_jax's SOMALayer is eqx-immutable.

        Args:
            procedural_def_path: path to ``SOMA_procedural_transforms.json``.
            mode: one of ``"aligned_x_swing_twist"``, ``"local_x_swing_twist"``,
                ``"local_x_euler"``.

        Returns:
            ``(transforms, full_bind_world, full_joint_names, full_parents)``:

            * ``transforms`` — :class:`ProceduralTransforms` instance used
              at evaluation time (call ``extend_public_rotations`` on it).
            * ``full_bind_world`` — (n_public + n_twist, 4, 4) bind world
              transforms; twist joints sit at the translation-matrix
              positions, with identity rotation at rest.
            * ``full_joint_names`` — tuple of 78 + n_twist joint names.
            * ``full_parents`` — (n_public + n_twist,) parent indices; twist
              joints attach to their segment's start joint.
        """
        from ..procedural_transforms import ProceduralTransforms, load_definition
        defn = load_definition(procedural_def_path)
        pt = ProceduralTransforms(defn, mode=mode)
        if self._public_bind_pose_world is None:
            raise RuntimeError(
                "extend_rig_with_procedural_transforms() needs bind_pose_world "
                "in the SOMA asset (augment SOMA_neutral_fixed.npz from the v0.1 HF dump)."
            )
        pub_bind = np.asarray(self._public_bind_pose_world)
        full_bind = pt.full_rig_bind_world(pub_bind)
        full_parents = pt.full_rig_parents(self._parents_np)
        full_names = pt.full_rig_joint_names()
        return pt, full_bind, full_names, full_parents

    def downsample_to_low_lod(self) -> "SOMALayer":
        """Deprecated — use ``SOMALayer.load(..., lod="low")`` instead.

        A low-LOD layer cannot be produced by subsetting an already-built
        mid-LOD layer: the identity model and the skeleton transfer (RBF
        regressors, sparse RBF matrix, rotation-fit precompute) are built from
        the full-resolution rig and would stay at 18,056 vertices while the
        mesh arrays dropped to 4,505 — `prepare_identity()` would then hand
        back full-resolution rest vertices for a low-LOD skinning matrix.
        Upstream builds the whole rig at the chosen LOD instead, which is what
        ``load(lod="low")`` now does.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            "downsample_to_low_lod() cannot build a consistent layer; the "
            "identity model and skeleton transfer would stay at full "
            "resolution. Use SOMALayer.load(path, lod='low') instead, which "
            "builds the entire rig on the low-LOD subset as upstream does."
        )

    def rebind(
        self,
        new_v_template: jnp.ndarray,
        new_bind_world: Optional[jnp.ndarray] = None,
    ) -> "SOMALayer":
        """Rebind to a new rest template, updating the state that depends on it.

        Upstream's rebind refreshes the skinning object's cached bind
        transforms and rest shape (``soma/soma.py`` ->
        ``batched_skinning.rebind``). Swapping only ``v_template`` here would
        leave the skeleton transfer's RBF regressors keyed on the *previous*
        bind shape — they are centred on it and queried at its joint positions —
        so every later fit would be silently biased toward the old identity.

        The skeleton transfer is **copied** before updating, so the layer this
        is called on is left untouched (equinox modules are immutable, but the
        transfer is a plain object shared by reference).

        Args:
            new_v_template: (V, 3) new rest-pose template vertices.
            new_bind_world: optional (J, 4, 4) bind transforms matching the new
                template. Defaults to the existing ones, which is right when
                only the shape changed.

        Returns:
            New SOMALayer with the template and dependent state updated.
        """
        layer = eqx.tree_at(lambda m: m.v_template, self, new_v_template)
        # Rebuilding from the recorded constructor arguments would drop the new
        # template, so a rebound layer cannot seed another LOD.
        object.__setattr__(layer, "_build_recipe", None)
        object.__setattr__(layer, "_lazy", _LazyCache())

        if self.skeleton_transfer is not None:
            transfer = copy.copy(self.skeleton_transfer)
            bind_world = (self.skeleton_transfer.bind_world_transforms
                          if new_bind_world is None else np.asarray(new_bind_world))
            transfer.update_bind(bind_world, np.asarray(new_v_template))
            # `skeleton_transfer` is a *static* field, so it is not a pytree
            # leaf and `tree_at` cannot target it. `layer` is already a fresh
            # object from the tree_at above, so setting it here leaves `self`
            # untouched.
            object.__setattr__(layer, "skeleton_transfer", transfer)
        return layer


# Alias for compatibility
SomaLayer = SOMALayer


def get_assets_dir() -> str:
    """Return the default assets directory path."""
    import os
    return os.path.join(os.path.dirname(__file__), "..", "assets")


# Re-export for backwards compatibility / public API
remove_joint_orient_local = _remove_joint_orient_local
