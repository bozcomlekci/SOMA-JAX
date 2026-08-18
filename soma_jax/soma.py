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

Upstream: ``soma/soma.py :: SOMALayer``
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
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .types import SOMAParams, SOMAOutput
from .units import Unit
from .geometry.transforms import axis_angle_to_rotmat, rotation_6d_to_rotmat
from .geometry.lbs import (
    forward_kinematics,
    lbs_transforms,
    lbs,
    lbs_sparse,
    compute_skeleton_levels,
    fk_levelorder,
)
from .geometry.rig_utils import (
    apply_joint_orient_local,
    joint_world_to_local as _joint_world_to_local,
    remove_joint_orient_local as _remove_joint_orient_local,
)
from .correctives_model import CorrectivesMLP
from .identity_model import BaseIdentityModel, create_identity_model


BODY_LODS = ("mid", "low")


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
    if "J_regressor" in out:
        out["J_regressor"] = np.asarray(out["J_regressor"])[:, idx]
    out["faces"] = np.asarray(soma_data["triangles_low"])

    for seg in ("segment_eye_bags", "segment_mouth_bag"):
        if seg in out:
            mapped = inv[np.asarray(out[seg]).astype(np.int64).ravel()]
            out[seg] = mapped[mapped >= 0]

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


def _target_to_public_map(target_names, parents, public_names) -> np.ndarray:
    """For each expanded-rig joint, the public joint whose bone scale it follows.

    Port of upstream's ``target_to_public_joint_indices``. Public joints map to
    themselves; procedural and helper joints inherit from the nearest public
    ancestor so a stretched bone carries its whole subtree.
    """
    public_at = {n: i for i, n in enumerate(public_names)}
    out = np.zeros(len(target_names), np.int32)
    for j, name in enumerate(target_names):
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
        correctives: CorrectivesMLP for pose-dependent deformations.
        identity_model: active BaseIdentityModel.
    """

    v_template: jnp.ndarray          # (V, 3)
    faces: jnp.ndarray                # (F, 3)
    J_regressor: jnp.ndarray          # (J, V)
    weights: jnp.ndarray              # (V, J)
    weight_indices: Optional[jnp.ndarray]  # (V, K) sparse top-K indices
    weight_values: Optional[jnp.ndarray]   # (V, K) sparse top-K weights
    joint_names: list = eqx.field(static=True)
    skeleton_levels: list = eqx.field(static=True)
    correctives: CorrectivesMLP
    identity_model: BaseIdentityModel = eqx.field(static=True)
    _parents_host: _HostArray = eqx.field(static=True)
    t_pose_world: Optional[jnp.ndarray]  # (J, 4, 4) — upstream T-pose world transform; None if unavailable
    bind_pose_world: Optional[jnp.ndarray]  # (J, 4, 4) — upstream bind-pose world transform; None if unavailable
    _bind_pose_local_host: Optional[_HostArray] = eqx.field(static=True)  # numpy mirror used by _repose_to_bind_pose
    _lod_mid_to_low_host: Optional[_HostArray] = eqx.field(static=True)   # (V_low,) vertex subset for low-LOD
    _lod_mid_num_verts: Optional[int] = eqx.field(static=True)            # source mesh size, low-LOD layers only
    _has_trained_correctives: bool = eqx.field(static=True)               # False when no checkpoint was loaded
    _triangles_low_host: Optional[_HostArray] = eqx.field(static=True)    # (F_low, 3) face indices into the subset
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
    scale_param_names: tuple = eqx.field(static=True)
    scale_param_segments: tuple = eqx.field(static=True)

    # ---- procedural (expanded twist) rig ---------------------------------
    # Upstream's default rig is the 122-joint template skeleton, but its *public*
    # pose contract stays at 78 joints: the 32 twist rotations are derived from
    # the public ones through the procedural parameter matrix, and the 12
    # USD-only helpers take identity. When these are set, `joint_names` is the
    # 122-joint rig and `public_joint_names` is the 78-joint contract callers
    # pose against.
    _procedural: Optional[object] = eqx.field(static=True)
    _public_idx_host: Optional[_HostArray] = eqx.field(static=True)
    _public_names: tuple = eqx.field(static=True)
    #: The expanded skinning rig used for FK/LBS only, as a plain dict of host
    #: arrays: ``joint_names``, ``parents``, ``levels``, ``weights``,
    #: ``weight_values``, ``weight_indices``, ``t_pose_world``, ``public_idx``,
    #: ``twist_idx``. ``None`` on a single-rig layer.
    _skin_rig: Optional[dict] = eqx.field(static=True)

    # Public joints whose local translation `scale_params` may stretch.
    NUM_BONE_SCALE_PARAMS = 56
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
    def _bone_scale_joint_indices(self):
        h = self._bone_scale_joint_indices_host
        return None if h is None else h.a

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
    ):
        """
        Args:
            soma_data: dict with keys: v_template, faces, J_regressor, parents,
                       weights, joint_names (and optionally correctives data).
            identity_model: pre-constructed BaseIdentityModel.
            correctives: optional pre-constructed CorrectivesMLP.
            sparse_k: number of top-K joints to keep in sparse LBS.
        """
        self.v_template = jnp.array(soma_data["v_template"], dtype=jnp.float32)
        self.faces = jnp.array(soma_data["faces"], dtype=jnp.int32)
        self.J_regressor = jnp.array(soma_data["J_regressor"], dtype=jnp.float32)
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
            self.t_pose_world = jnp.array(soma_data["t_pose_world"], dtype=jnp.float32)
        else:
            self.t_pose_world = None
        if "bind_pose_world" in soma_data:
            self.bind_pose_world = jnp.array(soma_data["bind_pose_world"], dtype=jnp.float32)
        else:
            self.bind_pose_world = None
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

        # No checkpoint -> an untrained (effectively zero) network. Upstream
        # raises instead of silently applying nothing, so record the fact and
        # let pose() refuse when correctives are explicitly requested.
        self._has_trained_correctives = correctives is not None
        if correctives is not None:
            self.correctives = correctives
        else:
            V, J = weights_np.shape
            self.correctives = CorrectivesMLP(
                n_joints=J,
                n_vertices=V,
                cors_per_joint=24,
            )

        # Full SOMA-X skeleton fit (RBF + two-stage Kabsch), constructed with
        # the exact upstream arguments (third_party/SOMA-X/soma/soma.py):
        #   SkeletonTransfer(parents, bind_pose_world, bind_shape, weights,
        #                    rotation_method="auto",
        #                    vertex_ids_to_exclude=eye_bags + mouth_bag)
        # The upstream rig stores bind data in centimeters while this layer
        # works in meters — normalise on load (same heuristic as
        # ``_repose_to_bind_pose``).
        if "bind_shape" in soma_data and "bind_pose_world" in soma_data:
            from .geometry.skeleton_transfer import SkeletonTransfer
            bind_shape = np.asarray(soma_data["bind_shape"], dtype=np.float32)
            bind_world = np.asarray(soma_data["bind_pose_world"], dtype=np.float32).copy()
            if float(np.abs(bind_world[..., :3, 3]).max()) > 10.0:   # cm-scale rig
                bind_world[..., :3, 3] *= 0.01
                bind_shape = bind_shape * 0.01
            excl: list[int] = []
            for seg in ("segment_eye_bags", "segment_mouth_bag"):
                if seg in soma_data:
                    excl.extend(np.asarray(soma_data[seg]).astype(int).ravel().tolist())
            self.skeleton_transfer = SkeletonTransfer(
                parents_np,
                bind_world,
                bind_shape,
                weights_np,
                rotation_method="auto",
                vertex_ids_to_exclude=excl or None,
            )
        else:
            self.skeleton_transfer = None

        # Bone-scale control layout (SOMA-X SOMALayer.soma_bone_scale_param_*).
        names = [str(n) for n in self.joint_names]
        scale_ids = [i for i, n in enumerate(names) if self._is_body_bone_scale_joint(n)]
        self._bone_scale_joint_indices_host = _host(
            np.asarray(scale_ids, dtype=np.int32) if scale_ids else None
        )
        self.scale_param_names = tuple(names[i] for i in scale_ids)
        safe_parents = np.where(parents_np < 0, np.arange(len(parents_np)), parents_np)
        self.scale_param_segments = tuple(
            (names[int(safe_parents[i])], names[i]) for i in scale_ids
        )

        # No procedural rig unless `attach_procedural_rig` is called; the layer
        # then poses against its full joint list, as it always has.
        self._procedural = None
        self._public_idx_host = _host(None)
        self._public_names = tuple(names)
        self._skin_rig = None

    def attach_procedural_rig(self, procedural, public_joint_names,
                              skin_rig: Optional[dict] = None) -> "SOMALayer":
        """Drive this (122-joint) rig from the 78-joint public pose contract.

        Upstream's default (``enable_procedural_transforms=True``) skins with the
        expanded template skeleton while keeping a public contract of 78 joints /
        77 posable ones: ``SOMAProceduralParameterTransform`` derives the 32
        twist joints' local rotations from the public rotations before FK, and
        the 12 USD-only helper bones take identity
        (``soma/soma.py``: "The expanded twist skeleton is used internally for
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
        object.__setattr__(out, "_procedural", procedural)
        object.__setattr__(out, "_public_idx_host", _host(idx))
        object.__setattr__(out, "_public_names", tuple(public))
        object.__setattr__(out, "_skin_rig", skin_rig)
        return out

    @staticmethod
    def build_skinning_rig(skin_data: dict, public_joint_names, procedural,
                           sparse_k: int = 8) -> dict:
        """Package an expanded rig for FK/LBS from ``rig_build`` output.

        Upstream drives FK/LBS with the 122-joint template skeleton while the
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
        from .geometry.batched_skinning import topk_skinning

        names = [str(n) for n in skin_data["joint_names"]]
        parents = np.asarray(skin_data["parents"], np.int32)
        weights = jnp.asarray(skin_data["weights"], jnp.float32)
        wi, wv = topk_skinning(weights, k=sparse_k)
        twist = [str(n) for n in procedural.definition.twist_joint_names]
        return {
            "joint_names": tuple(names),
            "parents": _host(parents),
            "levels": compute_skeleton_levels(parents),
            "weights": weights,
            "weight_indices": wi,
            "weight_values": wv,
            "t_pose_world": jnp.asarray(skin_data["t_pose_world"], jnp.float32),
            # Upstream starts its bind expansion from `self.bind_pose_world`
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
        2. Expand those world transforms onto the 122-joint rig, each twist joint
           a single local step off its public parent
           (``expand_world_transforms_from_source_fk``).
        3. LBS with the 122-joint weights against the correspondingly expanded
           bind.

        The bind is expanded through the *same* function with identity twist
        rotations, so at rest ``T_world @ inv(bind)`` is exactly identity and the
        twist joints contribute nothing — a consistency the previous
        translation-matrix bind did not have.

        ``joints`` / ``transforms`` come from the public FK result, matching
        upstream's ``output_transforms = public_world_transforms``.
        """
        from .geometry.batched_skinning import pose_from_bind
        from .geometry.lbs import lbs, lbs_sparse
        from .geometry.transforms import se3_inverse

        rig = self._skin_rig
        eye3 = jnp.broadcast_to(jnp.eye(3, dtype=rotmats.dtype), rotmats.shape)

        # 1. public FK
        _, T_world_public = pose_from_bind(
            bind_transforms, rest_verts, self.weights, self.skeleton_levels,
            self._parents_np, rotmats, transl, hips_idx=1, skip_lbs=True,
            local_translation_scales=bone_scales,
        )

        # 2a. This identity's expanded bind — upstream's
        # `_expand_public_bind_transforms`.
        bind_full = self._expanded_bind_transforms(bind_transforms)

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
            posed = lbs(rest_verts, zeros, bone_Rt, rig["weights"])
        return SOMAOutput(vertices=posed, joints=T_world_public[..., :3, 3],
                          transforms=T_world_public)

    def _expanded_bind_transforms(self, bind_transforms: jnp.ndarray) -> jnp.ndarray:
        """Scatter the fitted public binds into the expanded rig.

        The public joints keep their per-identity fitted binds; the 32 twist
        joints take the translation-matrix combination of public positions with
        identity rotation (upstream's ``full_rig_bind_world``); the 12 USD-only
        helpers keep the template bind. Those helpers carry **zero** skinning
        weight on the shipped rig, so their binds cannot move a vertex — they
        only need to exist so FK indices line up.
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
        """Number of active bone-scale controls (56 on the stock SOMA rig)."""
        return len(self.scale_param_names)

    def full_bone_scales(self, bone_scales: jnp.ndarray) -> jnp.ndarray:
        """Scatter (B, S) active bone scales into a full (B, J) multiplier array.

        Mirrors SOMA-X's ``_full_public_bone_scales``: joints without a control
        keep a scale of 1.0. ``scale_param_names`` gives the expected order and
        ``scale_param_segments`` the (parent, child) edge each value stretches.
        """
        bone_scales = jnp.asarray(bone_scales)
        if bone_scales.ndim == 1:
            bone_scales = bone_scales[None]
        expected = self.num_bone_scale_params
        if bone_scales.shape[-1] != expected:
            raise ValueError(
                f"bone_scales must have shape (B, {expected}); got "
                f"{tuple(bone_scales.shape)}. Use layer.scale_param_names for the order."
            )
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

        All of ``joint_names`` on the non-procedural rig; the 78-joint contract
        when a procedural rig is attached, where ``joint_names`` is the 122-joint
        skinning skeleton.
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
        count = rotations.shape[-3]
        if count != public_count:
            raise ValueError(
                f"Expected rotations for {public_count} public joints, got {count}."
            )
        return rotations

    def public_rig_view(self, bind_transforms_world: Optional[jnp.ndarray] = None) -> dict:
        """Public-joint view of the current rig — SOMA-X's ``SOMAPublicRigView``.

        Args:
            bind_transforms_world: optional (B, J, 4, 4) fitted binds from
                ``prepare_identity(return_bind_transforms=True)``. Defaults to
                the canonical ``bind_pose_world``.

        Returns:
            Dict with ``joint_names``, ``joint_parent_ids``,
            ``bind_transforms_world``, ``bind_transforms_local``,
            ``t_pose_world`` and ``skinning_weights``.
        """
        bind_world = (
            self.bind_pose_world if bind_transforms_world is None else bind_transforms_world
        )
        if bind_world is None:
            raise RuntimeError(
                "No bind transforms available: pass bind_transforms_world, or load "
                "an asset carrying bind_pose_world (see docs/INSTALL.md §4.2)."
            )
        return {
            "joint_names": self.public_joint_names,
            "joint_parent_ids": self._parents_np,
            "bind_transforms_world": bind_world,
            "bind_transforms_local": _joint_world_to_local(bind_world, self._parents_np),
            "t_pose_world": self.t_pose_world,
            "skinning_weights": self.public_skinning_weights(),
        }

    @classmethod
    def load(
        cls,
        path: str,
        identity_model_type: str = "soma",
        identity_model_path: Optional[str] = None,
        correctives_path: Optional[str] = None,
        sparse_k: int = 8,   # top-K sparse LBS; 8 matches SOMA-X's Warp path (topk_skinning K=8)
        lod: str = "mid",
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
                the asset.

        Returns:
            Instantiated SOMALayer.
        """
        if lod not in BODY_LODS:
            raise ValueError(f"lod must be one of {BODY_LODS}, got {lod!r}")
        soma_data = dict(np.load(path, allow_pickle=True))

        # Flatten any object arrays
        for k in list(soma_data.keys()):
            v = soma_data[k]
            if isinstance(v, np.ndarray) and v.dtype == object:
                soma_data[k] = v.item()

        return cls._from_soma_data(
            soma_data, identity_model_type=identity_model_type,
            identity_model_path=identity_model_path,
            correctives_path=correctives_path, sparse_k=sparse_k, lod=lod)

    @classmethod
    def from_upstream_assets(
        cls,
        npz_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        identity_model_type: str = "soma",
        identity_model_path: Optional[str] = None,
        correctives_path: Optional[str] = None,
        sparse_k: int = 8,
        lod: str = "mid",
        *,
        procedural: bool = True,
        fit_joint_regressor: bool = True,
    ) -> "SOMALayer":
        """Build directly from upstream's own two assets — no derived archive.

        Upstream merges ``SOMA_template_rig.usda`` over ``SOMA_neutral.npz`` at
        load time and refuses to run without the USD; the npz's own rig arrays
        differ from the merged result across ~10k of 18,056 vertices. This
        constructor performs that merge with numpy/scipy/pxr — no ``torch`` and
        no ``SOMA_neutral_fixed.npz`` — reproducing upstream's ``rig_data``
        exactly (see :mod:`soma_jax.rig_build`).

        Use :meth:`load` instead when you have the cached archive and would
        rather not depend on ``usd-core`` at runtime.

        Args:
            npz_path: full-schema ``SOMA_neutral.npz``; resolved when omitted.
            usd_path: ``SOMA_template_rig.usda``; resolved when omitted.
            identity_model_type, identity_model_path, correctives_path, sparse_k,
                lod: as for :meth:`load`.
            procedural: keep the expanded 122-joint twist skeleton, which is
                upstream's default (``enable_procedural_transforms=True``).
                ``False`` requests the pruned 78-joint legacy rig.
            fit_joint_regressor: fit the ``skeleton_fit="linear"`` regressor.
                The faithful path uses ``SkeletonTransfer`` and does not need it.

        Returns:
            An instantiated :class:`SOMALayer`.
        """
        from .rig_build import build_soma_asset

        if lod not in BODY_LODS:
            raise ValueError(f"lod must be one of {BODY_LODS}, got {lod!r}")
        if not procedural:
            # Upstream's legacy rig is *derived* from the same template, not read
            # from the npz: `derive_soma_rig_without_procedural_joints` prunes the
            # procedural and auxiliary joints and aggregates each dropped joint's
            # skin weights onto its nearest kept parent.
            from .assets import resolve as _resolve
            from .procedural_transforms import load_definition
            from .rig_build import build_soma_asset as _build, prune_procedural_joints

            definition = load_definition(_resolve("SOMA_procedural_transforms.json"))
            pruned = prune_procedural_joints(
                _build(npz_path, usd_path, lod, fit_joint_regressor=False),
                definition.main_joint_names)
            if fit_joint_regressor:
                from .rig_build import _fit_joint_regressor
                pruned["J_regressor"] = _fit_joint_regressor(
                    pruned["bind_shape"], pruned["bind_pose_world"],
                    pruned["weights"], pruned["parents"])
            return cls._from_soma_data(
                pruned, identity_model_type=identity_model_type,
                identity_model_path=identity_model_path,
                correctives_path=correctives_path, sparse_k=sparse_k,
                lod="mid" if lod == "mid" else lod, already_lod_sliced=lod != "mid")
        # The rig is read at the requested LOD directly rather than sliced after
        # the fact, matching upstream's per-LOD template read.
        if procedural:
            # Two-rig construction, mirroring upstream: the public 78-joint rig
            # carries the identity model, the skeleton fit and the bind data,
            # while the expanded 122-joint template skeleton is used for FK/LBS
            # only. The public half comes from the cached archive because the
            # 78-joint rig is upstream's *pruned* template merge and that prune
            # (`derive_soma_rig_without_procedural_joints`, which aggregates each
            # dropped joint's weights onto its nearest kept parent) is not ported.
            from .assets import resolve as _resolve
            from .procedural_transforms import ProceduralTransforms, load_definition

            public_asset = npz_path if npz_path is not None and str(npz_path).endswith(
                "SOMA_neutral_fixed.npz") else _resolve("SOMA_neutral_fixed.npz")
            layer = cls.load(
                str(public_asset), identity_model_type=identity_model_type,
                identity_model_path=identity_model_path,
                correctives_path=correctives_path, sparse_k=sparse_k, lod=lod)

            definition = load_definition(_resolve("SOMA_procedural_transforms.json"))
            procedural_transforms = ProceduralTransforms(definition)
            # `aligned_x_swing_twist` (what the shipped JSON asks for) measures
            # twist in the bind-aligned frame, so it needs the public T-pose
            # transforms. Without them the extractor silently falls back to a
            # start-joint scalar that is not upstream-equivalent, and the twist
            # contribution collapses — the posed result then reproduces the
            # non-procedural rig to ~1 mm instead of upstream's procedural one.
            if layer.t_pose_world is not None:
                procedural_transforms.set_bind_data(layer.t_pose_world)
            skin_data = build_soma_asset(
                None, usd_path, lod, fit_joint_regressor=False)
            skin_rig = cls.build_skinning_rig(
                skin_data, layer.joint_names, procedural_transforms, sparse_k=sparse_k)
            return layer.attach_procedural_rig(
                procedural_transforms, layer.joint_names, skin_rig=skin_rig)

        soma_data = build_soma_asset(
            npz_path, usd_path, lod if lod != "mid" else "mid",
            fit_joint_regressor=fit_joint_regressor)
        layer = cls._from_soma_data(
            soma_data, identity_model_type=identity_model_type,
            identity_model_path=identity_model_path,
            correctives_path=correctives_path, sparse_k=sparse_k,
            lod="mid" if lod == "mid" else lod, already_lod_sliced=lod != "mid")

        # Drive the expanded rig from the public contract, as upstream does.
        from .assets import resolve as _resolve
        from .procedural_transforms import ProceduralTransforms, load_definition
        definition = load_definition(_resolve("SOMA_procedural_transforms.json"))
        return layer.attach_procedural_rig(
            ProceduralTransforms(definition), definition.main_joint_names)

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
        already_lod_sliced: bool = False,
    ) -> "SOMALayer":
        """Shared construction from an assembled ``soma_data`` dict.

        Args:
            already_lod_sliced: the rig was already read at the target LOD (the
                :meth:`from_upstream_assets` path reads the LOD's own skin mesh),
                so skip the mid->low slice but still pass the vertex map to the
                corrective checkpoint loader.
        """
        lod_v_index_map = None
        if lod == "low":
            lod_v_index_map = np.asarray(soma_data["lod_mid_to_low"], dtype=np.int64)
            if not already_lod_sliced:
                soma_data = _slice_rig_to_low_lod(soma_data)

        # Load identity model parameters
        model_data = None
        if identity_model_path is not None:
            model_data = dict(np.load(identity_model_path, allow_pickle=True))

        identity_model = create_identity_model(identity_model_type, soma_data, model_data)

        # Load correctives. A low-LOD layer needs the checkpoint's output layer
        # sliced onto the same vertex subset (upstream's
        # `correctives_vertex_index_map`), or it emits full-resolution offsets
        # against a 4,505-vertex rest shape.
        correctives = None
        if correctives_path is not None:
            correctives = CorrectivesMLP.load_checkpoint(
                correctives_path, v_index_map=lod_v_index_map)

        return cls(soma_data, identity_model, correctives, sparse_k=sparse_k)

    def prepare_identity(
        self,
        identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
        repose_to_bind_pose: bool = True,
        global_scale: float | jnp.ndarray = 1.0,
        skeleton_fit: str = "auto",
        return_bind_transforms: bool = False,
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

        Returns:
            (rest_verts, rest_joints) — plus ``bind_transforms`` when
            ``return_bind_transforms=True``. Batched (B, ...) or unbatched
            matching the input.
        """
        unbatched = identity_coeffs.ndim == 1
        if unbatched:
            identity_coeffs = identity_coeffs[None]
            if scale_params is not None:
                scale_params = scale_params[None]

        rest_verts, rest_joints = self.identity_model.forward(identity_coeffs, scale_params)

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
        rest_joints = rest_joints * gs

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

        bind_transforms = None
        if use_full:
            # SOMA-X: skeleton_transfer.fit(rest_shape) -> per-identity bind
            # world transforms; joints are their translation components.
            bind_transforms = self.skeleton_transfer.fit(rest_verts)     # (B, J, 4, 4)
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
            if bind_transforms is not None:
                bind_transforms = bind_transforms[0]

        if return_bind_transforms:
            return rest_verts, rest_joints, bind_transforms
        return rest_verts, rest_joints

    def _repose_full(
        self,
        rest_verts: jnp.ndarray,
        bind_transforms: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Re-pose the T-pose identity into the bind pose against the FITTED
        skeleton — the faithful mirror of SOMA-X's non-procedural
        ``prepare_identity(repose_to_bind_pose=True)`` branch
        (third_party/SOMA-X/soma/soma.py:1424): rebind to the fit transforms,
        then pose with ``bind_pose_local`` rotations in absolute mode. Returns
        the reposed rest mesh and the NEW bind world transforms."""
        from .geometry.batched_skinning import pose_from_bind
        B, J = bind_transforms.shape[:2]
        bind_local = np.array(self._bind_pose_local_np, dtype=np.float32)
        if float(np.abs(bind_local[..., :3, 3]).max()) > 10.0:   # cm-scale rig
            bind_local[..., :3, 3] *= 0.01
        R = jnp.broadcast_to(jnp.asarray(bind_local[None, :, :3, :3]), (B, J, 3, 3))
        hips_t = jnp.broadcast_to(jnp.asarray(bind_local[None, 1, :3, 3]), (B, 3))
        posed_verts, T_world = pose_from_bind(
            bind_transforms, rest_verts, self.weights, self.skeleton_levels,
            self._parents_np, R, hips_t, hips_idx=1,
        )
        return posed_verts, T_world

    def _has_bind_pose_local(self) -> bool:
        """True when the asset shipped both ``bind_pose_world`` (as a JAX
        attribute) and ``bind_pose_local`` (as the numpy mirror used by
        :py:meth:`_repose_to_bind_pose`). Older soma_jax assets that predate
        the augmentation step don't carry either field, in which case
        ``prepare_identity(repose_to_bind_pose=True)`` silently no-ops."""
        return self.bind_pose_world is not None and getattr(
            self, "_bind_pose_local_np", None,
        ) is not None

    def _repose_to_bind_pose(
        self,
        rest_verts: jnp.ndarray,
        rest_joints: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Re-pose the T-pose identity into the bind pose via BatchedSkinning.

        Mirrors SOMA-X's ``prepare_identity(repose_to_bind_pose=True)``
        non-procedural branch (third_party/SOMA-X/soma/soma.py:1408): drive
        BatchedSkinning with ``bind_pose_local[..., :3, :3]`` rotations and
        the hips translation, in absolute-pose mode. The returned vertices
        live in the bind-pose frame, which is the frame the trained
        correctives expect as input.
        """
        from .geometry.batched_skinning import BatchedSkinning
        B = rest_verts.shape[0]
        J = rest_joints.shape[1]
        # Upstream bind transforms ship in centimeters; identity_model output
        # is in meters. Apply the same cm->m scale SOMA-X does inside
        # `_convert_units` so the reposed mesh stays in meters.
        bind_local = np.array(self._bind_pose_local_np, dtype=np.float32)
        if float(np.abs(bind_local[..., :3, 3]).max()) > 10.0:  # clearly cm-scale
            bind_local[..., :3, 3] = bind_local[..., :3, 3] * 0.01
        bind_local = jnp.asarray(bind_local)
        joint_orient = self.t_pose_world[..., :3, :3] if self.t_pose_world is not None else None

        bs = BatchedSkinning(
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
    ) -> SOMAOutput:
        """Apply pose (rotations + translation) to rest vertices.

        Args:
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

        Returns:
            SOMAOutput with posed ``vertices`` (B, V, 3; None when
            ``fk_only``), ``joints`` (B, J, 3) and ``transforms``
            (B, J, 4, 4) world joint transforms.
        """
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
        # joint_orient=layer.t_pose_world[..., :3, :3] explicitly so the input
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
            from .geometry.batched_skinning import pose_from_bind
            if apply_correctives and not fk_only:
                if not self._has_trained_correctives:
                    raise RuntimeError(
                        "apply_correctives=True but no corrective checkpoint is "
                        "loaded; the network is untrained and would contribute "
                        "nothing. Pass correctives_path= to SOMALayer.load(), or "
                        "apply_correctives=False."
                    )
                rest_verts = rest_verts + _scale_correctives(
                    self.correctives(rotmats), global_scale)
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
                    "apply_correctives=True but no corrective checkpoint is loaded; "
                    "the network is untrained and would contribute nothing. Pass "
                    "correctives_path= to SOMALayer.load(), or apply_correctives=False."
                )
            correctives = _scale_correctives(self.correctives(rotmats), global_scale)
        else:
            correctives = jnp.zeros_like(rest_verts)

        # LBS
        if use_sparse and self.weight_indices is not None:
            posed_verts = lbs_sparse(
                rest_verts, correctives, bone_T,
                self.weight_values, self.weight_indices
            )
        else:
            posed_verts = lbs(rest_verts, correctives, bone_T, self.weights)

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
    ) -> SOMAOutput:
        """Full forward pass: identity + pose (mirrors SOMA-X ``forward``).

        ``global_scale`` is applied once to the identity (rest verts + joints)
        and forwarded to :meth:`pose` so corrective offsets, which are trained
        in unscaled units, are scaled to match — upstream does both from its
        cached scale, so passing it here keeps the two in step.

        Faithful to upstream ``SOMALayer.forward``: the identity is prepared
        with ``repose_to_bind_pose=apply_correctives`` and the full skeleton
        fit when the asset supports it, the T-pose joint orient is applied
        (unless ``absolute_pose``), and skinning runs against the fitted bind
        transforms. On legacy assets without bind data this degrades to the
        simplified linear path.

        Args:
            params: SOMAParams with poses, transl, identity_coeffs, etc.
            apply_correctives: run the pose-corrective MLP (upstream default).
                An untrained model contributes exactly zero.
            absolute_pose: treat rotations as absolute skinning-frame (skip
                the T-pose joint-orient remap), as in upstream.
            fk_only: skip skinning and return joints/transforms only.

        Returns:
            SOMAOutput with posed vertices, joints and world transforms.
        """
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
            rotmats_flat = jax.vmap(axis_angle_to_rotmat)(flat)
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
            identity_coeffs = params.identity_coeffs[None] if params.identity_coeffs.ndim == 1 else params.identity_coeffs
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
        if self._procedural is not None and rotmats.shape[-3] != len(self.joint_names):
            if rotmats.shape[-3] == n_public - 1:
                # Upstream's public contract is the 77 *posable* joints — Root is
                # in the rig but never posed, so prepend identity for it.
                eye = jnp.broadcast_to(
                    jnp.eye(3, dtype=rotmats.dtype), rotmats.shape[:-3] + (1, 3, 3))
                rotmats = jnp.concatenate([eye, rotmats], axis=-3)
            if rotmats.shape[-3] != n_public:
                raise ValueError(
                    f"Expected {n_public} public joints (or {n_public - 1} posable, "
                    f"excluding Root) for the procedural rig, got {rotmats.shape[-3]}."
                )
            rotmats, _ = self._procedural.extend_to_template_rig(
                rotmats, list(self.joint_names))

        rest_verts, rest_joints, bind_transforms = self.prepare_identity(
            identity_coeffs, scale_params,
            repose_to_bind_pose=apply_correctives,       # upstream forward()
            return_bind_transforms=True,
            global_scale=global_scale,
        )

        # Upstream applies the T-pose joint orient unless absolute_pose; use
        # the asset's t_pose_world when the caller didn't override it.
        joint_orient = params.joint_orient
        if joint_orient is None and not absolute_pose and self.t_pose_world is not None:
            joint_orient = self.t_pose_world[..., :3, :3]

        # SOMA-backend `scale_params` are bone-length controls consumed at pose
        # time (upstream caches them in prepare_identity); other identity
        # backends consume them inside the identity model instead.
        bone_scales = None
        if (
            scale_params is not None
            and bind_transforms is not None
            and scale_params.shape[-1] == self.num_bone_scale_params
        ):
            bone_scales = scale_params

        out = self.pose(
            rotmats, transl, rest_verts, rest_joints, joint_orient,
            absolute_pose=absolute_pose,
            apply_correctives=apply_correctives,
            bind_transforms=bind_transforms,
            bone_scales=bone_scales,
            fk_only=fk_only,
            global_scale=global_scale,      # correctives are trained unscaled
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
        from .procedural_transforms import ProceduralTransforms, load_definition
        defn = load_definition(procedural_def_path)
        pt = ProceduralTransforms(defn, mode=mode)
        if self.bind_pose_world is None:
            raise RuntimeError(
                "extend_rig_with_procedural_transforms() needs bind_pose_world "
                "in the SOMA asset (augment SOMA_neutral_fixed.npz from the v0.1 HF dump)."
            )
        pub_bind = np.asarray(self.bind_pose_world)
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
