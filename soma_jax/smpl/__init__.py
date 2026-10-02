"""SMPL-family rig layers and pose transfer (JAX port of ``soma.smpl``).

Upstream: ``soma/smpl/__init__.py``, ``soma/smpl/transfer.py``.

**Ported** (:mod:`soma_jax.smpl.transfer`, re-exported here):
``SMPLFamilyPoseTransferResult``, ``SMPLFamilyTopologyBridge`` and
``transfer_smpl_family_pose_parameters`` — pose the source rig, bridge its mesh
onto the target topology, recover the target's absolute local rotations with
:class:`~soma_jax.fitting.pose_inversion.PoseInversion`.

**Ported** (:mod:`soma_jax.smpl.layers`): the native SMPL-family rig layers
``SMPLLayer`` / ``SMPLXLayer``, ``create_smpl_family_layer`` (which also builds
SOMA-X v0.3's ``MANOLayer``), ``load_smpl_family_model`` and the joint-name
lists. They implement upstream's PoseInversion layer contract; like every
SOMA-JAX layer, ``prepare_identity`` returns the identity instead of caching it.

**SOMA-JAX extras:**

* :class:`BarycentricBridge` — a one-stage helper for callers that want
  SOMA-topology output. It is *not* upstream's bridge: its second stage embeds
  the target wrap in the canonical mesh rather than the target base mesh, so it
  stops at SOMA topology.
* :func:`transfer_pose_between_layers` — retargets between
  :mod:`soma_jax.body_models` layers (identity as ``betas``) through the spec
  form of :class:`SMPLFamilyTopologyBridge` and the lightweight
  :class:`soma_jax.PoseInversion`, returning local axis-angle. Use
  :func:`transfer_smpl_family_pose_parameters` for upstream's behaviour.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import numpy as np
import jax.numpy as jnp

from .transfer import (  # noqa: E402
    SMPLFamilyPoseTransferResult,
    SMPLFamilyTopologyBridge,
    transfer_smpl_family_pose_parameters,
)
from .layers import (  # noqa: E402
    SMPL_JOINT_NAMES,
    SMPLX_JOINT_NAMES,
    SMPLFamilyIdentity,
    SMPLLayer,
    SMPLXLayer,
    create_smpl_family_layer,
    load_smpl_family_model,
)

__all__ = [
    "BarycentricBridge",
    "SMPLFamilyIdentity",
    "SMPLFamilyPoseTransferResult",
    "SMPLFamilyTopologyBridge",
    "SMPLLayer",
    "SMPLXLayer",
    "SMPL_JOINT_NAMES",
    "SMPLX_JOINT_NAMES",
    "create_smpl_family_layer",
    "load_smpl_family_model",
    "transfer_pose_between_layers",
    "transfer_smpl_family_pose_parameters",
]


class BarycentricBridge:
    """Transfer per-frame vertices from one SMPL-family topology to another.

    Mirrors the simpler "source -> wrap (target topology) -> target_base"
    direction used inside SOMA-X. Build the bridge once on rest meshes, then
    call it on any (T, V_source, 3) sequence to get (T, V_target, 3).
    """

    def __init__(
        self,
        source_rest_verts: np.ndarray,
        source_faces: np.ndarray,
        target_wrap_verts: np.ndarray,
        scale: float = 1.0,
        canonical_rest_verts: np.ndarray | None = None,
        canonical_faces: np.ndarray | None = None,
        direct: bool = False,
    ):
        """Args:
        source_rest_verts: (V_s, 3) source model rest verts.
        source_faces:      (F_s, 3) source model faces.
        target_wrap_verts: (V_t, 3) target topology embedded in the source
            mesh's coordinate frame at rest — comes from the SOMA-X
            ``<MODEL>/SOMA_wrap.obj`` shipped with the HF assets.
        scale: optional uniform unit conversion (e.g. cm -> m).
        """
        # Optional second stage: when the source and target do not share a
        # topology, upstream routes source -> canonical -> target through two
        # interpolators (`SMPLFamilyTopologyBridge`). Passing
        # `canonical_rest_verts`/`canonical_faces` enables the same here.
        self._direct = bool(direct)
        if self._direct:
            # Same topology on both sides: upstream returns `vertices * scale`
            # without building any interpolator.
            self.face_ids = None
            self.bary = None
            self.source_faces = np.asarray(source_faces, dtype=np.int32)
            self.scale = float(scale)
            self._stage1 = None
            return

        self._stage1 = None
        if canonical_rest_verts is not None and canonical_faces is not None:
            from ..geometry.barycentric_interp import compute_barycentric_coords
            f1, b1 = compute_barycentric_coords(
                np.asarray(canonical_rest_verts, np.float32),
                source_rest_verts.astype(np.float32),
                source_faces.astype(np.int32),
            )
            self._stage1 = (np.asarray(source_faces, np.int32),
                            np.asarray(f1, np.int32), np.asarray(b1, np.float32))
            # Stage 2 then maps the canonical mesh onto the target wrap.
            source_rest_verts = np.asarray(canonical_rest_verts, np.float32)
            source_faces = np.asarray(canonical_faces, np.int32)

        # Self-contained: `tools/` is a repo-local scripts directory and is NOT
        # packaged (pyproject includes only `soma_jax*`), so importing from it
        # made this class unusable from an installed wheel. The geometry helper
        # below ships with the package and falls back to a brute-force nearest
        # -triangle search when trimesh/rtree are unavailable.
        from ..geometry.barycentric_interp import compute_barycentric_coords
        face_ids, bary = compute_barycentric_coords(
            target_wrap_verts.astype(np.float32),
            source_rest_verts.astype(np.float32),
            source_faces.astype(np.int32),
        )
        self.face_ids = np.asarray(face_ids, dtype=np.int32)   # (V_t,)
        self.bary = np.asarray(bary, dtype=np.float32)         # (V_t, 4) tet coords
        self.source_faces = np.asarray(source_faces, dtype=np.int32)
        self.scale = float(scale)

    @staticmethod
    def can_use_direct_topology(source_spec, target_spec) -> bool:
        """Whether transfer reduces to a unit rescale — upstream's predicate.

        Port of ``SMPLFamilyTopologyBridge._can_use_direct_topology``: when both
        layers declare the same ``model_spec`` the meshes share a topology, so
        no barycentric step is needed and only the unit scale applies.

        Args:
            source_spec, target_spec: the two layers' ``model_spec`` values.

        Returns:
            True when the source can be passed through with only a scale.
        """
        return source_spec is not None and source_spec == target_spec

    @staticmethod
    def unit_scale(source_unit, target_unit) -> float:
        """Source-to-target unit ratio — upstream's ``_unit_scale``."""
        return float(source_unit.meters_per_unit / target_unit.meters_per_unit)

    def __call__(self, posed_seq: np.ndarray) -> np.ndarray:
        """Apply the bridge to a posed source-mesh sequence.

        Args:
            posed_seq: (..., V_source, 3) — works for (T, V, 3) and (V, 3).
        Returns:
            (..., V_target, 3) in target topology, scaled.
        """
        verts = np.asarray(posed_seq)
        added_T = verts.ndim == 2
        if added_T:
            verts = verts[None]
        if self._direct:
            out = verts * self.scale
            return out[0] if added_T else out
        from ..geometry.barycentric_interp import barycentric_interpolate
        if self._stage1 is not None:
            sf, fid, bary = self._stage1
            verts = np.asarray(barycentric_interpolate(
                jnp.asarray(verts), jnp.asarray(sf), jnp.asarray(fid), jnp.asarray(bary)))
        out = np.asarray(barycentric_interpolate(
            jnp.asarray(verts), jnp.asarray(self.source_faces),
            jnp.asarray(self.face_ids), jnp.asarray(self.bary),
        ))
        out = out * self.scale
        return out[0] if added_T else out


# Joint layout of each SMPL-family params NamedTuple, as
# ``field -> (first joint index, joint count)``. Used to split a flat
# ``(T, J, 3)`` axis-angle array into the per-field arguments our body models
# take, mirroring the ``poses`` argument upstream's layers accept directly.
_POSE_FIELD_LAYOUT = {
    "SMPLParams": {"body_pose": (1, None)},
    "SMPLHParams": {
        "body_pose": (1, 21), "left_hand_pose": (22, 15), "right_hand_pose": (37, 15),
    },
    "SMPLXParams": {
        "body_pose": (1, 21), "jaw_pose": (22, 1), "leye_pose": (23, 1),
        "reye_pose": (24, 1), "left_hand_pose": (25, 15), "right_hand_pose": (40, 15),
    },
    "AnnyParams": {"body_pose": (1, None)},
}


def _params_class(layer: Any):
    """The params NamedTuple a body-model layer expects."""
    import importlib
    name = type(layer).__name__.replace("Model", "Params")
    module = importlib.import_module(type(layer).__module__)
    cls = getattr(module, name, None)
    if cls is None:
        raise TypeError(
            f"Cannot determine the params type for {type(layer).__name__}; "
            f"expected {name} in {type(layer).__module__}."
        )
    return cls


def _build_params(layer: Any, poses_aa: jnp.ndarray, betas: jnp.ndarray,
                  transl: jnp.ndarray):
    """Assemble a params NamedTuple from a flat (T, J, 3) axis-angle array.

    Fields the layer declares but the pose array does not cover (``expression``
    and the like) are zero-filled, which is what upstream's layers do for the
    equivalent unset inputs.
    """
    cls = _params_class(layer)
    layout = _POSE_FIELD_LAYOUT.get(cls.__name__)
    if layout is None:
        raise TypeError(f"No pose layout registered for {cls.__name__}.")

    T = poses_aa.shape[0]
    kwargs: dict[str, Any] = {
        "betas": betas,
        "global_orient": poses_aa[:, 0],
        "transl": transl,
    }
    for fieldname, (start, count) in layout.items():
        n = poses_aa.shape[1] - start if count is None else count
        kwargs[fieldname] = poses_aa[:, start:start + n].reshape(T, n * 3)

    for fieldname in cls._fields:
        if fieldname in kwargs:
            continue
        if fieldname == "expression":
            width = int(getattr(layer, "num_expression_coeffs", 10))
        else:
            width = 3
        kwargs[fieldname] = jnp.zeros((T, width), dtype=poses_aa.dtype)
    return cls(**kwargs)


def _rest_state(layer: Any, betas: jnp.ndarray):
    """(rest_verts, rest_joints) for one identity — the layer's zero pose."""
    J = layer.num_joints
    zero = jnp.zeros((1, J, 3), dtype=jnp.float32)
    params = _build_params(layer, zero, betas[:1], jnp.zeros((1, 3), jnp.float32))
    out = layer(params)
    return out.v_shaped[0], jnp.einsum("jv,vd->jd", layer.J_regressor, out.v_shaped[0])


def transfer_pose_between_layers(
    source_layer: Any,
    target_layer: Any,
    source_poses: jnp.ndarray,
    *,
    source_betas: Optional[jnp.ndarray] = None,
    target_betas: Optional[jnp.ndarray] = None,
    source_transl: Optional[jnp.ndarray] = None,
    source_spec: Optional[str] = None,
    target_spec: Optional[str] = None,
    asset_dir: str | Path | None = None,
    topology_bridge: "SMPLFamilyTopologyBridge | None" = None,
    unit_scale: float = 1.0,
    refine_iters: int = 0,
    fit_mode: str = "combined",
) -> SMPLFamilyPoseTransferResult:
    """Retarget a SMPL-family motion clip onto another SMPL-family rig.

    SOMA-JAX extra modelled on ``soma.smpl.transfer.
    transfer_smpl_family_pose_parameters`` (ported as
    :func:`transfer_smpl_family_pose_parameters`) for :mod:`soma_jax.body_models`
    layers. The four stages, in order:

    1. **Forward the source rig** at ``source_poses`` -> ``source_vertices``.
    2. **Bridge to the target topology** via
       :class:`SMPLFamilyTopologyBridge` (source -> SOMA wrap -> target)
       -> ``fit_vertices``.
    3. **Invert** the bridged mesh against the target rig with
       :class:`~soma_jax.PoseInversion` -> absolute rotations and root
       translation.
    4. **Forward the target rig** at the recovered rotations ->
       ``reconstructed_vertices``.

    Differences from upstream, both structural rather than numerical:

    * Upstream's ``SMPLLayer``/``SMPLXLayer`` are SOMA-style
      ``BatchedSkinning`` rigs with a ``.pose()`` method and an internal
      identity model. This port drives :mod:`soma_jax.body_models`, whose
      layers take a params NamedTuple, so identity arrives as ``betas``
      instead of ``identity_coeffs`` and the pose split is done by
      :func:`_build_params`.
    * ``PoseInversion`` recovers **absolute** (world) rotations; they are
      converted to the layers' local axis-angle convention before step 4.

    Args:
        source_layer, target_layer: :mod:`soma_jax.body_models` layers.
        source_poses: (T, J_source, 3) axis-angle, joint 0 the root.
        source_betas: (T, K) or (K,) source shape; zeros when omitted.
        target_betas: (K,) target shape; zeros when omitted. Upstream adapts
            the source coefficients when the target has none — matched here by
            truncating/zero-padding ``source_betas`` to the target width.
        source_transl: (T, 3) root translation; zeros when omitted.
        source_spec, target_spec: model identifiers for the bridge assets
            (``"smpl"``, ``"smplx"``, ``"mhr"``, ...). Inferred from the layer
            class name when omitted.
        asset_dir: root holding the ``<MODEL>/`` packs; defaults to the
            resolver.
        topology_bridge: prebuilt bridge, to amortise its cost across clips.
        unit_scale: source-to-target unit ratio (upstream's ``_unit_scale``).
        refine_iters: autograd refinement steps inside the inversion.
        fit_mode: ``"analytical"``, ``"autograd"`` or ``"combined"``.

    Returns:
        :class:`SMPLFamilyPoseTransferResult`.
    """
    from ..geometry.transforms import rotmat_to_axis_angle
    from ..pose_inversion_lite import PoseInversion

    poses = jnp.asarray(source_poses, dtype=jnp.float32)
    if poses.ndim == 2:
        poses = poses[None]
    T = poses.shape[0]

    def _widen(betas, width, n):
        if betas is None:
            return jnp.zeros((n, width), jnp.float32)
        b = jnp.asarray(betas, jnp.float32)
        if b.ndim == 1:
            b = b[None]
        if b.shape[0] == 1 and n > 1:
            b = jnp.broadcast_to(b, (n, b.shape[1]))
        if b.shape[1] < width:
            b = jnp.pad(b, ((0, 0), (0, width - b.shape[1])))
        return b[:, :width]

    src_betas = _widen(source_betas, source_layer.num_betas, T)
    # Upstream reuses the source identity for the target when none is given.
    tgt_betas = _widen(
        source_betas if target_betas is None else target_betas,
        target_layer.num_betas, 1)
    transl = (jnp.zeros((T, 3), jnp.float32) if source_transl is None
              else jnp.broadcast_to(jnp.asarray(source_transl, jnp.float32).reshape(-1, 3),
                                    (T, 3)))

    # 1. Forward the source rig.
    source_vertices = source_layer(
        _build_params(source_layer, poses, src_betas, transl)).vertices

    # 2. Bridge into the target topology.
    def _spec_of(layer):
        return type(layer).__name__.replace("Model", "").lower()
    if topology_bridge is None:
        topology_bridge = SMPLFamilyTopologyBridge(
            source_spec or _spec_of(source_layer),
            target_spec or _spec_of(target_layer),
            scale=unit_scale, asset_dir=asset_dir,
        )
    fit_vertices = topology_bridge(source_vertices)

    # 3. Invert against the target rig.
    rest_verts, rest_joints = _rest_state(target_layer, tgt_betas)
    inv = PoseInversion(
        rest_verts=rest_verts, weights=target_layer.weights,
        rest_joints=rest_joints, parents=target_layer._parents_np,
    )
    world_R = inv.fit(fit_vertices, mode=fit_mode, num_refine_iters=refine_iters)

    # PoseInversion returns absolute rotations; the layers pose from local
    # axis-angle, so compose out each joint's parent: local = R_parent^T @ R.
    parents = np.asarray(target_layer._parents_np)
    local_R = [world_R[:, 0]]
    for j in range(1, len(parents)):
        pj = int(parents[j])
        local_R.append(jnp.einsum("bij,bjk->bik",
                                  jnp.swapaxes(world_R[:, pj], -1, -2), world_R[:, j]))
    local_aa = rotmat_to_axis_angle(jnp.stack(local_R, axis=1))

    # Root translation: the offset that best aligns the reconstruction with the
    # bridged mesh, upstream's ``_update_root_translation`` reduced to its
    # unweighted form (all vertices count equally here).
    recon_at_origin = target_layer(_build_params(
        target_layer, local_aa, jnp.broadcast_to(tgt_betas, (T, tgt_betas.shape[1])),
        jnp.zeros((T, 3), jnp.float32))).vertices
    root_translation = (fit_vertices.mean(axis=1) - recon_at_origin.mean(axis=1))

    # 4. Forward the target rig at the recovered pose.
    reconstructed = target_layer(_build_params(
        target_layer, local_aa,
        jnp.broadcast_to(tgt_betas, (T, tgt_betas.shape[1])), root_translation,
    )).vertices

    return SMPLFamilyPoseTransferResult(
        rotations=local_aa,
        root_translation=root_translation,
        per_vertex_error=jnp.linalg.norm(reconstructed - fit_vertices, axis=-1),
        source_vertices=source_vertices,
        fit_vertices=fit_vertices,
        reconstructed_vertices=reconstructed,
    )
