"""SMPL-family pose transfer helpers.

Upstream: ``soma/smpl/transfer.py`` (SOMA-X v0.3.3). The source rig is posed,
its mesh bridged onto the target topology, and :class:`SOMAPoseInversion`
recovers the target layer's absolute local rotations and root translation.

SOMA-JAX layers do not cache an identity: ``prepare_identity`` returns it (or,
for :class:`~soma_jax.SOMALayer`, the forward prepares it). The helpers here
therefore carry the prepared identity from upstream's ``prepare_identity`` call
to its ``pose`` call explicitly; the sequence and arguments are upstream's.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from ..units import Unit

__all__ = [
    "SMPLFamilyPoseTransferResult",
    "SMPLFamilyTopologyBridge",
    "transfer_smpl_family_pose_parameters",
]


@dataclass
class SMPLFamilyPoseTransferResult:
    """Result of fitting target pose parameters to a source rig animation.

    From :func:`transfer_smpl_family_pose_parameters`, ``rotations`` are the
    target layer's absolute local rotation matrices (T, J, 3, 3), as upstream.
    The SOMA-JAX :func:`~soma_jax.smpl.transfer_pose_between_layers` stores
    local axis-angle (T, J, 3) there instead.
    """

    rotations: jnp.ndarray
    root_translation: jnp.ndarray          # (T, 3)
    per_vertex_error: jnp.ndarray          # (T, V_target)
    source_vertices: jnp.ndarray           # (T, V_source, 3)
    fit_vertices: jnp.ndarray              # (T, V_target, 3) source mesh in target topology
    reconstructed_vertices: jnp.ndarray    # (T, V_target, 3) target rig at `rotations`


def _layer_unit(layer: Any) -> Unit:
    unit = getattr(layer, "output_unit", Unit.METERS)
    if isinstance(unit, Unit):
        return unit
    return Unit.from_name(unit)


def _unit_scale(source_layer: Any, target_layer: Any) -> float:
    return _layer_unit(source_layer).meters_per_unit / _layer_unit(target_layer).meters_per_unit


def _load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    mesh = trimesh.load(Path(path), maintain_order=True, process=False)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)


def _num_vertices(layer: Any) -> int:
    """Upstream's ``layer.bind_shape.shape[0]`` (``v_template`` on SOMALayer)."""
    bind_shape = getattr(layer, "bind_shape", None)
    if bind_shape is None:
        bind_shape = layer.v_template
    return int(np.shape(bind_shape)[0])


def _num_identity_coeffs_for_layer(layer: Any) -> int:
    value = getattr(layer, "num_identity_coeffs", None)
    if value is not None:
        return int(value)
    identity_model = getattr(layer, "identity_model", None)
    if identity_model is not None:
        value = getattr(identity_model, "num_identity_coeffs", None)
        if value is not None:
            return int(value)
    value = getattr(layer, "num_shape_components", None)
    if value is not None:
        return int(value)
    return 0


def _layer_identity_coeffs(layer: Any, values, *, batch_size: Optional[int] = None) -> jnp.ndarray:
    """Identity coefficients shaped to the layer: zeros when absent, rows
    broadcast to ``batch_size``, width truncated or zero-padded."""
    num_coeffs = _num_identity_coeffs_for_layer(layer)
    if values is None:
        rows = 1 if batch_size is None else batch_size
        return jnp.zeros((rows, num_coeffs), jnp.float32)
    coeffs = jnp.asarray(values, jnp.float32)
    if coeffs.ndim == 1:
        coeffs = coeffs[None]
    if coeffs.ndim != 2:
        raise ValueError(f"Expected identity coefficients with shape (B, C), got {coeffs.shape}.")
    if batch_size is not None and coeffs.shape[0] == 1 and batch_size > 1:
        coeffs = jnp.broadcast_to(coeffs, (batch_size, coeffs.shape[1]))
    if coeffs.shape[1] > num_coeffs:
        return coeffs[:, :num_coeffs]
    if coeffs.shape[1] < num_coeffs:
        coeffs = jnp.pad(coeffs, ((0, 0), (0, num_coeffs - coeffs.shape[1])))
    return coeffs


def _adapt_identity_coeffs(values, target_layer: Any) -> jnp.ndarray:
    return _layer_identity_coeffs(target_layer, values)


def _pose_batch_size(poses, pose2rot: bool) -> int:
    if pose2rot:
        return 1 if poses.ndim == 2 else int(poses.shape[0])
    if poses.ndim == 3 and poses.shape[-2:] == (3, 3):
        return 1
    return int(poses.shape[0])


def _with_pose_batch(poses, pose2rot: bool):
    if pose2rot:
        return poses[None] if poses.ndim == 2 else poses
    if poses.ndim == 3 and poses.shape[-2:] == (3, 3):
        return poses[None]
    return poses


class SMPLFamilyTopologyBridge:
    """Map posed vertices between SOMA and native SMPL-family topologies.

    Upstream builds it from two layers, ``SMPLFamilyTopologyBridge(source_layer,
    target_layer)``, and routes ``source -> canonical -> target`` through two
    ``BarycentricInterpolator`` embeddings, *canonical* being SOMA topology:

    ==============================  =========================================
    upstream                        embedding (``BarycentricInterpolator``)
    ==============================  =========================================
    ``source_to_canonical``         ``(source_base_v, source_base_f, source_wrap_v)``
    ``canonical_to_target``         ``(target_wrap_v, target_wrap_f, target_base_v)``
    ==============================  =========================================

    Read those as "embed the third argument in the mesh given by the first two,
    then drive it with the deformed first mesh". Layers sharing a
    ``model_spec`` need only the unit scale; a SOMA source (no
    ``topology_family``) is already canonical, so only the second stage runs.

    SOMA-JAX extra: the two arguments may instead be model specs
    (``"smpl"``, ``"smplx"``, ``"mhr"``, ...) keyed into :attr:`ASSETS`, with
    ``scale`` and ``asset_dir`` given explicitly — the form
    :func:`~soma_jax.smpl.transfer_pose_between_layers` uses.
    """

    #: ``model_spec`` -> (base mesh, SOMA-wrap mesh) relative asset paths, for
    #: the spec form.
    ASSETS = {
        "smpl": ("SMPL/base_body.obj", "SMPL/SOMA_wrap.obj"),
        "smplh": ("SMPL/base_body.obj", "SMPL/SOMA_wrap.obj"),
        "smplx": ("SMPLX/base_body.obj", "SMPLX/SOMA_wrap.obj"),
        "anny": ("Anny/base_body.obj", "Anny/SOMA_wrap.obj"),
        "mhr": ("MHR/base_body_lod1.obj", "MHR/SOMA_wrap_lod1.obj"),
        "garment": ("GarmentMeasurements/mean.obj",
                    "GarmentMeasurements/SOMA_wrap.obj"),
    }

    def __init__(self, source_layer: Any, target_layer: Any, *,
                 scale: Optional[float] = None, asset_dir: str | Path | None = None) -> None:
        self.source_to_canonical = None
        self.canonical_to_target = None
        if isinstance(source_layer, str) or isinstance(target_layer, str):
            self._init_from_specs(str(source_layer), str(target_layer),
                                  1.0 if scale is None else float(scale), asset_dir)
            return
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.scale = _unit_scale(source_layer, target_layer) if scale is None else float(scale)
        self.direct = self._can_use_direct_topology(source_layer, target_layer)
        if self.direct:
            return

        source_family = getattr(source_layer, "topology_family", None)
        target_family = getattr(target_layer, "topology_family", None)
        target_base = getattr(target_layer, "base_mesh_path", None)
        target_wrap = getattr(target_layer, "wrap_mesh_path", None)

        if (source_family is None and target_family in {"body", "hand"}
                and target_base and target_wrap):
            target_base_v, _ = _load_mesh(target_base)
            target_wrap_v, target_wrap_f = _load_mesh(target_wrap)
            source_num_verts = _num_vertices(source_layer)
            if source_num_verts != target_wrap_v.shape[0]:
                raise ValueError(
                    "SOMA-to-SMPL-family topology bridge requires matching SOMA wrap topology. "
                    f"Got {source_num_verts} source vertices, expected {target_wrap_v.shape[0]}.")
            self.canonical_to_target = self._embedding(target_wrap_v, target_wrap_f,
                                                       target_base_v)
            return

        if source_family != target_family:
            raise ValueError(
                f"No registered SMPL-family topology bridge from {source_family!r} "
                f"to {target_family!r}.")
        source_base = getattr(source_layer, "base_mesh_path", None)
        source_wrap = getattr(source_layer, "wrap_mesh_path", None)
        if not all((source_base, source_wrap, target_base, target_wrap)):
            raise ValueError(
                "Both SMPL-family layers must define base_mesh_path and wrap_mesh_path.")
        source_base_v, source_base_f = _load_mesh(source_base)
        source_wrap_v, _ = _load_mesh(source_wrap)
        target_base_v, _ = _load_mesh(target_base)
        target_wrap_v, target_wrap_f = _load_mesh(target_wrap)
        self.source_to_canonical = self._embedding(source_base_v, source_base_f, source_wrap_v)
        self.canonical_to_target = self._embedding(target_wrap_v, target_wrap_f, target_base_v)

    def _init_from_specs(self, source_spec: str, target_spec: str, scale: float,
                         asset_dir) -> None:
        self.source_spec = source_spec.lower()
        self.target_spec = target_spec.lower()
        self.scale = scale
        self.direct = self.source_spec == self.target_spec
        if self.direct:
            return
        src_base_v, src_base_f = self._spec_mesh(self.source_spec, 0, asset_dir)
        src_wrap_v, _ = self._spec_mesh(self.source_spec, 1, asset_dir)
        tgt_base_v, _ = self._spec_mesh(self.target_spec, 0, asset_dir)
        tgt_wrap_v, tgt_wrap_f = self._spec_mesh(self.target_spec, 1, asset_dir)
        if src_wrap_v.shape[0] != tgt_wrap_v.shape[0]:
            raise ValueError(
                "SMPL-family topology bridge requires a shared SOMA wrap topology. "
                f"{self.source_spec} wrap has {src_wrap_v.shape[0]} vertices, "
                f"{self.target_spec} wrap has {tgt_wrap_v.shape[0]}.")
        self.source_to_canonical = self._embedding(src_base_v, src_base_f, src_wrap_v)
        self.canonical_to_target = self._embedding(tgt_wrap_v, tgt_wrap_f, tgt_base_v)

    @staticmethod
    def _can_use_direct_topology(source_layer: Any, target_layer: Any) -> bool:
        source_spec = getattr(source_layer, "model_spec", None)
        target_spec = getattr(target_layer, "model_spec", None)
        return source_spec is not None and source_spec == target_spec

    @staticmethod
    def _embedding(src_v: np.ndarray, src_f: np.ndarray, query_v: np.ndarray) -> tuple:
        """Upstream ``BarycentricInterpolator(src_v, src_f, query_v)``."""
        from ..geometry.barycentric_interp import compute_barycentric_coords
        face_ids, bary = compute_barycentric_coords(query_v, src_v, src_f)
        return (np.asarray(src_f, np.int32), np.asarray(face_ids, np.int32),
                np.asarray(bary, np.float32))

    @classmethod
    def _spec_mesh(cls, spec: str, which: int, asset_dir):
        """Load ``base_body``/``SOMA_wrap`` for a model spec as (verts, faces)."""
        try:
            rel = cls.ASSETS[spec][which]
        except KeyError:
            raise ValueError(
                f"No registered SMPL-family topology assets for {spec!r}. "
                f"Known: {sorted(cls.ASSETS)}.") from None
        if asset_dir is not None:
            path = Path(asset_dir) / rel
            if not path.exists():
                raise FileNotFoundError(f"{path} not found (asset_dir={asset_dir})")
        else:
            from ..assets import resolve
            path = resolve(rel)
        return _load_mesh(path)

    @staticmethod
    def _interpolate(vertices: jnp.ndarray, embedding: tuple) -> jnp.ndarray:
        from ..geometry.barycentric_interp import barycentric_interpolate
        faces, face_ids, bary = embedding
        return barycentric_interpolate(vertices, jnp.asarray(faces), jnp.asarray(face_ids),
                                       jnp.asarray(bary))

    def __call__(self, vertices: jnp.ndarray) -> jnp.ndarray:
        """Map (..., V_source, 3) posed vertices to (..., V_target, 3), unit-scaled."""
        v = jnp.asarray(vertices)
        added = v.ndim == 2
        if added:
            v = v[None]
        if not self.direct:
            if self.source_to_canonical is not None:
                v = self._interpolate(v, self.source_to_canonical)
            v = self._interpolate(v, self.canonical_to_target)
        out = v * self.scale
        return out[0] if added else out

    forward = __call__


def _is_soma_body_layer(layer: Any) -> bool:
    from ..body.soma import SOMALayer
    return isinstance(layer, SOMALayer)


@dataclass
class _SOMABodyIdentity:
    """Upstream ``SOMALayer.prepare_identity`` arguments, applied at pose time."""

    identity_coeffs: jnp.ndarray
    scale_params: Optional[jnp.ndarray] = None
    global_scale: Any = 1.0
    kwargs: Optional[dict] = None
    repose_to_bind_pose: bool = True


_SOMA_BODY_PREPARE_ARGS = ("scale_params", "repose_to_bind_pose", "global_scale", "kwargs")


def _prepare_layer_identity(layer: Any, identity_coeffs, prepare_kwargs: Optional[dict]):
    """Upstream ``_prepare_layer_identity``: the kwargs the layer's
    ``prepare_identity`` accepts are passed on; the prepared identity is
    returned for :func:`_pose_layer`."""
    kwargs = dict(prepare_kwargs or {})
    if _is_soma_body_layer(layer):
        accepted = {k: v for k, v in kwargs.items() if k in _SOMA_BODY_PREPARE_ARGS}
        return _SOMABodyIdentity(identity_coeffs, **accepted)
    signature = inspect.signature(layer.prepare_identity).parameters
    accepted = {k: v for k, v in kwargs.items() if k in signature}
    return layer.prepare_identity(identity_coeffs, **accepted)


def _output_dict(out) -> dict:
    if isinstance(out, dict):
        return out
    return {key: getattr(out, key) for key in ("vertices", "joints", "transforms")
            if hasattr(out, key)}


def _pose_layer(layer: Any, identity, poses, root_translation, *, pose2rot: bool,
                absolute_pose: bool, extra_kwargs: Optional[dict]) -> dict:
    """Upstream ``_pose_layer``: pose the prepared identity; the root translation
    goes to ``global_translation`` (the SOMA body layer's ``transl``)."""
    extra = dict(extra_kwargs or {})
    if _is_soma_body_layer(layer):
        from ..types import SOMAParams
        apply_correctives = extra.pop("apply_correctives", None)
        reference_pose = extra.pop("reference_pose", None)
        if extra:
            raise TypeError(f"Unsupported SOMALayer pose kwargs: {sorted(extra)}")
        poses = jnp.asarray(poses, jnp.float32)
        if pose2rot and poses.ndim == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)
        params = SOMAParams(poses=poses, transl=jnp.asarray(root_translation, jnp.float32),
                            identity_coeffs=identity.identity_coeffs,
                            scale_params=identity.scale_params)
        out = layer(params, apply_correctives=apply_correctives, absolute_pose=absolute_pose,
                    global_scale=identity.global_scale, reference_pose=reference_pose,
                    kwargs=identity.kwargs, repose_to_bind_pose=identity.repose_to_bind_pose)
        return _output_dict(out)
    pose_params = inspect.signature(layer.pose).parameters
    kwargs: dict[str, Any] = {"pose2rot": pose2rot, "absolute_pose": absolute_pose}
    if "global_translation" in pose_params:
        kwargs["global_translation"] = root_translation
    elif "transl" in pose_params:
        kwargs["transl"] = root_translation
    else:
        raise TypeError(
            f"{type(layer).__name__}.pose() must accept either global_translation or transl.")
    kwargs.update(extra)
    return _output_dict(layer.pose(poses, identity, **kwargs))


def transfer_smpl_family_pose_parameters(
    source_layer: Any,
    target_layer: Any,
    source_poses,
    *,
    source_identity_coeffs=None,
    target_identity_coeffs=None,
    source_root_translation=None,
    source_pose2rot: bool = False,
    source_absolute_pose: bool = True,
    source_prepare_kwargs: Optional[dict] = None,
    target_prepare_kwargs: Optional[dict] = None,
    source_pose_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
    topology_bridge: Optional[SMPLFamilyTopologyBridge] = None,
) -> SMPLFamilyPoseTransferResult:
    """Transfer source pose parameters into a target SMPL-family rig layer.

    The source rig is evaluated first, its posed mesh is bridged to the target
    topology if needed, and :class:`~soma_jax.fitting.pose_inversion.PoseInversion`
    then recovers the target layer's absolute local rotations and root
    translation.
    """
    from ..fitting.pose_inversion import SOMAPoseInversion

    source_poses = _with_pose_batch(jnp.asarray(source_poses, jnp.float32), source_pose2rot)
    batch_size = _pose_batch_size(source_poses, source_pose2rot)

    source_identity = _layer_identity_coeffs(source_layer, source_identity_coeffs,
                                             batch_size=None)
    if target_identity_coeffs is None:
        target_identity = _adapt_identity_coeffs(source_identity, target_layer)
    else:
        target_identity = _layer_identity_coeffs(target_layer, target_identity_coeffs)

    source_prepared = _prepare_layer_identity(source_layer, source_identity,
                                              source_prepare_kwargs)
    _prepare_layer_identity(target_layer, target_identity, target_prepare_kwargs)

    if source_root_translation is None:
        source_root_translation = jnp.zeros((batch_size, 3), jnp.float32)
    else:
        source_root_translation = jnp.asarray(source_root_translation, jnp.float32)
        if source_root_translation.ndim == 1:
            source_root_translation = source_root_translation[None]
        if source_root_translation.shape[0] == 1 and batch_size > 1:
            source_root_translation = jnp.broadcast_to(source_root_translation, (batch_size, 3))

    source_out = _pose_layer(source_layer, source_prepared, source_poses,
                             source_root_translation, pose2rot=source_pose2rot,
                             absolute_pose=source_absolute_pose,
                             extra_kwargs=source_pose_kwargs)
    source_vertices = source_out["vertices"]
    if topology_bridge is None:
        topology_bridge = SMPLFamilyTopologyBridge(source_layer, target_layer)
    fit_vertices = topology_bridge(source_vertices)

    inv = SOMAPoseInversion(target_layer, low_lod=False)
    inv.prepare_identity(target_identity)
    result = inv.fit(fit_vertices, **dict(fit_kwargs or {}))

    # Upstream reposes with the identity the inversion prepared last (the
    # layer's cache), i.e. the target identity without ``target_prepare_kwargs``.
    recon = _pose_layer(target_layer, _prepare_layer_identity(target_layer, target_identity, None),
                        result["rotations"], result["root_translation"], pose2rot=False,
                        absolute_pose=True,
                        extra_kwargs={"apply_correctives": False})["vertices"]

    return SMPLFamilyPoseTransferResult(
        rotations=result["rotations"],
        root_translation=result["root_translation"],
        per_vertex_error=result["per_vertex_error"],
        source_vertices=source_vertices,
        fit_vertices=fit_vertices,
        reconstructed_vertices=recon,
    )
