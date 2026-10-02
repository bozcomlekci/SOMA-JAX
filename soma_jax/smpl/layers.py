"""SMPL-family rig layers backed by SMPL, SMPL-X and MANO assets (JAX port).

Upstream: ``soma/smpl/__init__.py`` (``_SMPLFamilyLBSLayer``,
``_SMPLFamilyRigLayer``, ``SMPLLayer``, ``SMPLXLayer``,
``create_smpl_family_layer``) and ``soma/_smpl_family_loader.py``
(``load_smpl_family_model``).

These are native SMPL-family LBS rigs implementing upstream's PoseInversion
*layer contract* — the interface its ``smpl2soma`` / ``mano2soma`` /
``soma2mano`` tools drive. The bind pose is the shaped model's regressed
joints with identity rotations, so the T-pose orient is the identity and the
root (index 0) receives ``global_translation``.

API difference (the one every SOMA-JAX layer makes): upstream caches the
prepared identity on the module; here :meth:`prepare_identity` **returns** a
:class:`SMPLFamilyIdentity` and :meth:`pose` takes it. The neutral identity is
prepared at construction, as upstream does, and used when none is passed, and
``bind_shape`` / ``bind_pose_world`` / ``t_pose_world`` describe it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from ..body_models.model_io import _load_npz, _load_pickle, parent_ids_from_kintree
from ..geometry.batched_skinning import pose_from_bind, topk_skinning
from ..geometry.lbs import batch_rodrigues, compute_skeleton_levels
from ..units import Unit

__all__ = [
    "SMPL_JOINT_NAMES",
    "SMPLX_JOINT_NAMES",
    "SMPLFamilyIdentity",
    "SMPLLayer",
    "SMPLXLayer",
    "create_smpl_family_layer",
    "load_smpl_family_model",
]

SMPL_JOINT_NAMES = [
    "Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee", "Spine2",
    "LeftAnkle", "RightAnkle", "Spine3", "LeftFoot", "RightFoot", "Neck",
    "LeftCollar", "RightCollar", "Head", "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow", "LeftWrist", "RightWrist", "LeftHand", "RightHand",
]

SMPLX_JOINT_NAMES = [
    "Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee", "Spine2",
    "LeftAnkle", "RightAnkle", "Spine3", "LeftFoot", "RightFoot", "Neck",
    "LeftCollar", "RightCollar", "Head", "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow", "LeftHand", "RightHand", "Jaw", "LeftEye", "RightEye",
    "LeftIndex1", "LeftIndex2", "LeftIndex3", "LeftMiddle1", "LeftMiddle2", "LeftMiddle3",
    "LeftPinky1", "LeftPinky2", "LeftPinky3", "LeftRing1", "LeftRing2", "LeftRing3",
    "LeftThumb1", "LeftThumb2", "LeftThumb3",
    "RightIndex1", "RightIndex2", "RightIndex3", "RightMiddle1", "RightMiddle2",
    "RightMiddle3", "RightPinky1", "RightPinky2", "RightPinky3", "RightRing1",
    "RightRing2", "RightRing3", "RightThumb1", "RightThumb2", "RightThumb3",
]


# ---------------------------------------------------------------------------
# loading (upstream soma/_smpl_family_loader.py)
# ---------------------------------------------------------------------------
def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=dtype)


def _read_model_file(model_path: Path) -> dict[str, Any]:
    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(str(model_path))
    if suffix == ".pkl":
        return _load_pickle(str(model_path))
    raise ValueError(f"Unsupported SMPL-family model file extension: {model_path.suffix!r}.")


def _get_required(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"SMPL-family model is missing required key; tried {keys}.")


def _normalize_shapedirs(shapedirs, *, vertex_count: int, num_betas: Optional[int]) -> np.ndarray:
    shapedirs = _to_numpy(shapedirs, np.float32)
    if shapedirs.ndim != 3:
        raise ValueError(f"Expected shapedirs rank 3, got shape {shapedirs.shape}.")
    if shapedirs.shape[0] != vertex_count and shapedirs.shape[1:] == (vertex_count, 3):
        shapedirs = np.transpose(shapedirs, (1, 2, 0))
    if shapedirs.shape[0] != vertex_count or shapedirs.shape[1] != 3:
        raise ValueError(
            f"Expected shapedirs shape (V, 3, B), got {shapedirs.shape} for V={vertex_count}.")
    if num_betas is not None:
        shapedirs = shapedirs[:, :, :num_betas]
    return shapedirs


def _normalize_posedirs(posedirs, *, vertex_count: int) -> np.ndarray:
    posedirs = _to_numpy(posedirs, np.float32)
    if posedirs.ndim == 3:
        if posedirs.shape[0] == vertex_count and posedirs.shape[1] == 3:
            return posedirs.reshape(vertex_count * 3, posedirs.shape[2]).T
        if posedirs.shape[1] == vertex_count and posedirs.shape[2] == 3:
            return posedirs.reshape(posedirs.shape[0], vertex_count * 3)
    if posedirs.ndim == 2:
        if posedirs.shape[1] == vertex_count * 3:
            return posedirs
        if posedirs.shape[0] == vertex_count * 3:
            return posedirs.T
    raise ValueError(f"Could not normalize posedirs with shape {posedirs.shape}.")


def load_smpl_family_model(model_path, *, model_type: str,
                           num_betas: Optional[int] = 10) -> dict[str, np.ndarray]:
    """Load an SMPL, SMPL-H or SMPL-X model into plain NumPy arrays.

    Upstream ``load_smpl_family_model``: template vertices, shape directions
    ``(V, 3, B)``, joint regressor, skinning weights, parents (root = 0), faces
    and pose directions ``(P, 3V)``. Pickles load without ``chumpy``.
    """
    model_type = model_type.lower()
    if model_type not in {"smpl", "smplh", "smplx"}:
        raise ValueError(f"Unsupported SMPL-family model type: {model_type!r}.")
    data = _read_model_file(Path(model_path))
    v_template = _to_numpy(_get_required(data, "v_template"), np.float32)
    vertex_count = v_template.shape[0]
    parents = parent_ids_from_kintree(_get_required(data, "kintree_table")).astype(np.int64)
    parents[0] = 0
    return {
        "v_template": v_template,
        "shapedirs": _normalize_shapedirs(_get_required(data, "shapedirs"),
                                          vertex_count=vertex_count, num_betas=num_betas),
        "J_regressor": _to_numpy(_get_required(data, "J_regressor"), np.float32),
        "lbs_weights": _to_numpy(_get_required(data, "weights", "lbs_weights"), np.float32),
        "parents": parents,
        "faces": _to_numpy(_get_required(data, "f", "faces"), np.int64),
        "posedirs": _normalize_posedirs(_get_required(data, "posedirs"), vertex_count=vertex_count),
    }


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------
class SMPLFamilyIdentity(NamedTuple):
    """A prepared SMPL-family identity: rest shape + bind world transforms."""

    rest_shape: jnp.ndarray              # (B, V, 3) output_unit
    bind_transforms_world: jnp.ndarray   # (B, J, 4, 4)


def _coerce_unit(unit) -> Unit:
    return unit if isinstance(unit, Unit) else Unit.from_name(unit)


def _identity_coeffs(values, *, num_coeffs: int, dtype) -> jnp.ndarray:
    """Upstream ``_identity_coeffs``: None -> zeros; pad or truncate to width."""
    if values is None:
        return jnp.zeros((1, num_coeffs), dtype)
    coeffs = jnp.asarray(values, dtype)
    if coeffs.ndim == 1:
        coeffs = coeffs[None]
    if coeffs.ndim != 2:
        raise ValueError(f"Expected identity coefficients with shape (B, C), got {coeffs.shape}.")
    if coeffs.shape[1] >= num_coeffs:
        return coeffs[:, :num_coeffs]
    return jnp.concatenate(
        [coeffs, jnp.zeros((coeffs.shape[0], num_coeffs - coeffs.shape[1]), coeffs.dtype)], 1)


class _SMPLFamilyLBSLayer:
    """Shared PoseInversion-compatible LBS wrapper for native SMPL-family rigs."""

    NATIVE_UNIT = Unit.METERS

    def __init__(self, data_root, *, device=None, mode: str = "warp",
                 output_unit=Unit.METERS) -> None:
        # ``device`` is upstream's torch device: accepted and ignored.
        if mode not in ("warp", "dense"):
            raise ValueError(f"mode must be 'warp' or 'dense', got {mode!r}")
        self.data_root = Path(data_root)
        self.mode = mode
        self.output_unit = _coerce_unit(output_unit)
        self._native_to_output_scale = (
            self.NATIVE_UNIT.meters_per_unit / self.output_unit.meters_per_unit)
        self.low_lod = False
        self.nv_lod_mid_to_low = None
        self.root_joint_idx = 0
        self.excluded_vert_ids = np.zeros((0,), np.int64)

    # Subclasses set: num_identity_coeffs, _v_template, _shapedirs,
    # _J_regressor, skinning_weights, joint_parent_ids (root 0), faces,
    # posedirs, rig_data; then call `_finish_init()`.
    def _finish_init(self) -> None:
        parents = np.asarray(self.joint_parent_ids, np.int64).copy()
        parents[0] = -1                                   # SOMA-JAX root convention
        self._parents_int = parents
        self._levels = compute_skeleton_levels(parents)
        if self.mode == "warp":
            idx, val = topk_skinning(np.asarray(self.skinning_weights, np.float32), 8)
            self._weight_indices, self._weight_values = jnp.asarray(idx), jnp.asarray(val)
        else:
            self._weight_indices = self._weight_values = None
        self._default_identity = self.prepare_identity(None)
        self.bind_shape = self._default_identity.rest_shape[0]
        self.bind_pose_world = self._default_identity.bind_transforms_world[0]
        self.t_pose_world = self.bind_pose_world
        self.rig_data["bind_shape"] = np.asarray(self.bind_shape)

    @property
    def num_joints(self) -> int:
        return int(len(self.joint_parent_ids))

    def _shape_native(self, identity_coeffs):
        raise NotImplementedError

    def _shape(self, identity_coeffs):
        coeffs = _identity_coeffs(identity_coeffs, num_coeffs=self.num_identity_coeffs,
                                  dtype=self._v_template.dtype)
        rest_shape, joints = self._shape_native(coeffs)
        scale = self._native_to_output_scale
        return rest_shape * scale, joints * scale

    @staticmethod
    def _make_bind_world(joints: jnp.ndarray) -> jnp.ndarray:
        B, J, _ = joints.shape
        bind = jnp.broadcast_to(jnp.eye(4, dtype=joints.dtype), (B, J, 4, 4))
        return bind.at[:, :, :3, 3].set(joints)

    def _pose_corrective_offsets(self, rotations: jnp.ndarray) -> jnp.ndarray:
        B = rotations.shape[0]
        pose_feature = (rotations[:, 1:] - jnp.eye(3, dtype=rotations.dtype)).reshape(B, -1)
        posedirs = jnp.asarray(self.posedirs, rotations.dtype)
        if posedirs.ndim == 3:
            posedirs = jnp.transpose(posedirs, (2, 0, 1)).reshape(posedirs.shape[2], -1)
        elif posedirs.ndim != 2:
            raise ValueError(f"Expected posedirs to be rank 2 or 3, got shape {posedirs.shape}.")
        if posedirs.shape[0] != pose_feature.shape[1] and posedirs.shape[1] == pose_feature.shape[1]:
            posedirs = posedirs.T
        if posedirs.shape[0] != pose_feature.shape[1]:
            raise ValueError(
                "SMPL-family posedirs do not match the pose feature width: "
                f"{posedirs.shape[0]} vs {pose_feature.shape[1]}.")
        return (pose_feature @ posedirs).reshape(B, -1, 3) * self._native_to_output_scale

    def prepare_identity(self, identity_coeffs=None, scale_params=None,
                         repose_to_bind_pose: bool = True, kwargs=None) -> SMPLFamilyIdentity:
        """Shape the model; bind = regressed joints with identity rotations."""
        del scale_params, repose_to_bind_pose, kwargs
        rest_shape, joints = self._shape(identity_coeffs)
        return SMPLFamilyIdentity(rest_shape, self._make_bind_world(joints))

    def pose(self, poses, identity: Optional[SMPLFamilyIdentity] = None, pose2rot: bool = True,
             apply_correctives: bool = True, absolute_pose: bool = False,
             global_translation=None, fk_only: bool = False) -> dict:
        """LBS pose. ``identity`` defaults to the neutral one prepared at init.

        The bind orientation is the identity, so ``absolute_pose`` changes
        nothing — as upstream, where the joint orient is the identity too.
        """
        del absolute_pose
        if identity is None:
            identity = self._default_identity
        poses = jnp.asarray(poses)
        B = poses.shape[0]
        if pose2rot:
            rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(B, self.num_joints, 3, 3)
        else:
            rotations = poses.reshape(B, self.num_joints, 3, 3)
        rotations = rotations.astype(self._v_template.dtype)
        if global_translation is None:
            global_translation = jnp.zeros((B, 3), rotations.dtype)
        global_translation = jnp.broadcast_to(jnp.asarray(global_translation, rotations.dtype),
                                              (B, 3))
        rest = identity.rest_shape
        if apply_correctives and not fk_only:
            rest = rest + self._pose_corrective_offsets(rotations)
        verts, T_world = pose_from_bind(
            identity.bind_transforms_world, rest, jnp.asarray(self.skinning_weights),
            self._levels, self._parents_int, rotations, global_translation, hips_idx=0,
            weight_values=self._weight_values, weight_indices=self._weight_indices,
            skip_lbs=fk_only)
        if fk_only:
            return {"joints": T_world[:, :, :3, 3], "transforms": T_world}
        return {"vertices": verts, "joints": T_world[:, :, :3, 3], "transforms": T_world}

    def forward(self, poses, identity_coeffs=None, pose2rot: bool = True,
                apply_correctives: bool = True, absolute_pose: bool = False,
                global_translation=None) -> dict:
        """Upstream ``forward``: prepare the identity, then pose it."""
        return self.pose(poses, self.prepare_identity(identity_coeffs), pose2rot=pose2rot,
                         apply_correctives=apply_correctives, absolute_pose=absolute_pose,
                         global_translation=global_translation)

    __call__ = forward


class _SMPLFamilyRigLayer(_SMPLFamilyLBSLayer):
    _MODEL_TYPE = ""
    _JOINT_NAMES: list = []

    def __init__(self, data_root, *, device=None, mode: str = "warp", output_unit=Unit.METERS,
                 gender: str = "neutral", model_path=None, **model_kwargs: Any) -> None:
        super().__init__(data_root, device=device, mode=mode, output_unit=output_unit)
        self.model_type = self._MODEL_TYPE
        self.model_spec = self.model_type
        self.gender = gender.lower()
        self.topology_family = "body"
        self.identity_model_type = f"{self.model_type}_native"
        self.identity_model_kwargs = {"model_type": self.model_type, "gender": self.gender}
        self.rig_data = {"joint_names": self._JOINT_NAMES}
        self.default_skin_mesh_name = self.model_type
        model_dir = self.data_root / self.model_type.upper()
        self.base_mesh_path = model_dir / "base_body.obj"
        self.wrap_mesh_path = model_dir / "SOMA_wrap.obj"
        self.model_path = self._resolve_model_path(model_path)
        num_betas = int(model_kwargs.pop("num_betas", 10))
        t = load_smpl_family_model(self.model_path, model_type=self.model_type,
                                   num_betas=num_betas)
        if len(t["parents"]) != len(self._JOINT_NAMES):
            raise ValueError(
                f"Expected {len(self._JOINT_NAMES)} {self.model_type.upper()} joints, "
                f"got {len(t['parents'])}.")
        self.num_identity_coeffs = int(t["shapedirs"].shape[2])
        self._v_template = jnp.asarray(t["v_template"])
        self._shapedirs = jnp.asarray(t["shapedirs"])
        self._J_regressor = jnp.asarray(t["J_regressor"])
        self.skinning_weights = jnp.asarray(t["lbs_weights"])
        self.joint_parent_ids = np.asarray(t["parents"], np.int64)
        self.faces = jnp.asarray(t["faces"])
        self.posedirs = jnp.asarray(t["posedirs"])
        self._finish_init()

    def _resolve_model_path(self, model_path) -> Path:
        if model_path is not None:
            return Path(model_path)
        model_dir = self.data_root / self.model_type.upper()
        prefix, gender = self.model_type.upper(), self.gender.upper()
        for suffix in ("npz", "pkl"):
            candidate = model_dir / f"{prefix}_{gender}.{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find {prefix}_{gender}.npz or {prefix}_{gender}.pkl in {model_dir}.")

    def _shape_native(self, identity_coeffs):
        blend = jnp.einsum("bk,vdk->bvd", identity_coeffs, self._shapedirs)
        verts = self._v_template[None] + blend
        joints = jnp.einsum("jv,bvd->bjd", self._J_regressor, verts)
        return verts, joints


class SMPLLayer(_SMPLFamilyRigLayer):
    """SMPL LBS rig adapter implementing the PoseInversion layer contract."""

    _MODEL_TYPE = "smpl"
    _JOINT_NAMES = SMPL_JOINT_NAMES


class SMPLXLayer(_SMPLFamilyRigLayer):
    """SMPL-X LBS rig adapter implementing the PoseInversion layer contract."""

    _MODEL_TYPE = "smplx"
    _JOINT_NAMES = SMPLX_JOINT_NAMES


def create_smpl_family_layer(model: str, data_root, *, device=None, mode: str = "warp",
                             output_unit=Unit.METERS, **kwargs: Any) -> _SMPLFamilyLBSLayer:
    """Create an SMPL-family rig layer from a compact model spec.

    ``"smpl"``, ``"smplx"`` / ``"smpl-x"``, ``"mano-left"`` / ``"mano-right"``
    (and the aliases upstream accepts).
    """
    spec = model.lower().replace("_", "-")
    if spec in {"smpl", "body-smpl"}:
        return SMPLLayer(data_root, device=device, mode=mode, output_unit=output_unit, **kwargs)
    if spec in {"smplx", "smpl-x", "body-smplx", "body-smpl-x"}:
        return SMPLXLayer(data_root, device=device, mode=mode, output_unit=output_unit,
                          **kwargs)
    if spec in {"mano-left", "left-mano", "mano:l", "mano-l"}:
        from ..hand.mano import MANOLayer
        return MANOLayer(data_root, "left", device, mode=mode, output_unit=output_unit)
    if spec in {"mano-right", "right-mano", "mano:r", "mano-r"}:
        from ..hand.mano import MANOLayer
        return MANOLayer(data_root, "right", device, mode=mode, output_unit=output_unit)
    raise ValueError(
        f"Unsupported SMPL-family model {model!r}. Use 'smpl', 'smplx', "
        "'mano-left', or 'mano-right'.")
