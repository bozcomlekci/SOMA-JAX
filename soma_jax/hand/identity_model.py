"""Hand identity-model backends for SOMA-JAX.

Upstream: ``soma/hand/identity_model.py`` (SOMA-X v0.3.0).

Three backends produce a wrist-local rest shape on the SOMA-hand topology:

* :class:`SOMAHandIdentityModel` — the hand shape PCA stored in ``SOMAHand.npz``
  (20 components; the right hand mirrors the left across X).
* :class:`MANOHandIdentityModel` — MANO shape betas, transferred onto the
  SOMA-hand topology by barycentric interpolation (MANO model files are licensed
  separately and user-supplied).
* :class:`MHRHandIdentityModel` — the full-body MHR model with only its 5 hand
  identity dims (40-44) and 26 per-hand scales exposed; the SOMA-hand vertices are
  extracted after transfer to the full SOMA topology.

Faithful port. Differences are mechanical: upstream keeps a ``_last_wrist_pos``
attribute between ``get_rest_shape`` and ``forward``; here the wrist travels as a
return value so the models stay stateless (and ``jit``-friendly). The MHR backend
evaluates MHR with :class:`~soma_jax.body_models.mhr_native.MHRNativeModel` (the
weights lifted out of the same TorchScript archive) instead of TorchScript.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from ..geometry.barycentric_interp import barycentric_interpolate, compute_barycentric_coords
from ..units import Unit
from ._smpl_family_loader import load_mano_pkl

__all__ = [
    "BaseHandIdentityModel",
    "MANOHandIdentityModel",
    "MHRHandIdentityModel",
    "SOMAHandIdentityModel",
]


# Shared with the body backends, as upstream keeps them in the base module.
from ..identity_model import _PERM_PARITY, CoordAxis  # noqa: E402,F401


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    mesh = trimesh.load(str(path), maintain_order=True, process=False)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)


class _HandIdentityBase:
    """Shared plumbing of upstream's ``BaseIdentityModel`` that the hand uses:
    unit conversion, the native-axis remap and optional barycentric transfer."""

    NATIVE_UNIT: Unit
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    #: Names of the per-identity scale controls in ``scale_params`` order.
    scale_param_names: tuple = ()

    def __init__(self, data_root, low_lod: bool = False, output_unit: Unit = Unit.METERS):
        self.data_root = Path(data_root)
        self.low_lod = bool(low_lod)
        self.output_unit = output_unit
        self._unit_conversion = self.NATIVE_UNIT.meters_per_unit / output_unit.meters_per_unit
        self._interp = None       # (src_faces, face_ids, bary) once set up

    @property
    def num_identity_coeffs(self) -> int:
        raise NotImplementedError

    @property
    def num_scale_params(self) -> Optional[int]:
        """Scale parameters ``get_rest_shape`` expects, or ``None`` if unused."""
        return None

    def _setup_topology_transfer(self, V_source, F_source, V_soma) -> None:
        """Upstream ``_setup_topology_transfer``: barycentric, no Laplacian blend."""
        # float32 in, as upstream builds its interpolators from float32 tensors.
        face_ids, bary = compute_barycentric_coords(
            np.asarray(V_soma, np.float32), np.asarray(V_source, np.float32),
            np.asarray(F_source, np.int64))
        self._interp = (jnp.asarray(np.asarray(F_source, np.int32)),
                        jnp.asarray(face_ids), jnp.asarray(bary, jnp.float32))

    def identity_model_to_soma(self, identity_rest_shape: jnp.ndarray) -> jnp.ndarray:
        """Source topology -> SOMA topology (no-op when no transfer is set up)."""
        if self._interp is None:
            return identity_rest_shape
        faces, face_ids, bary = self._interp
        return barycentric_interpolate(identity_rest_shape, faces, face_ids, bary)

    def _apply_coord_transform(self, verts: jnp.ndarray) -> jnp.ndarray:
        """Reorder/negate axes from the native convention to SOMA (Y+ up, Z+ fwd)."""
        if self.NATIVE_UP == CoordAxis.Y and self.NATIVE_FORWARD == CoordAxis.Z:
            return verts
        up_idx, up_sign = self.NATIVE_UP
        fwd_idx, fwd_sign = self.NATIVE_FORWARD
        right_idx = 3 - up_idx - fwd_idx
        right_sign = _PERM_PARITY[(right_idx, up_idx, fwd_idx)] * up_sign * fwd_sign
        return jnp.concatenate([
            verts[..., right_idx:right_idx + 1] * right_sign,
            verts[..., up_idx:up_idx + 1] * up_sign,
            verts[..., fwd_idx:fwd_idx + 1] * fwd_sign,
        ], axis=-1)

    def _finish(self, result: jnp.ndarray, global_scale) -> jnp.ndarray:
        if self._unit_conversion != 1.0:
            result = result * self._unit_conversion
        if isinstance(global_scale, (jnp.ndarray, np.ndarray)):
            result = result * jnp.asarray(global_scale).reshape(-1, 1, 1)
        elif global_scale != 1.0:
            result = result * global_scale
        return result


class BaseHandIdentityModel(_HandIdentityBase):
    """Base class for hand identity models (upstream ``BaseHandIdentityModel``).

    - ``hand_type`` ("left" or "right"); per-hand assets are loaded
      independently, so no X-flip is needed.
    - Wrist centering happens in :meth:`__call__`, **after** topology transfer,
      so the transfer sees verts in the same frame as the reference meshes.

    Subclasses implement :meth:`_get_shaped_verts` returning verts + wrist
    position in the model's native frame.
    """

    def __init__(self, data_root, low_lod: bool = False, device=None, hand_type: str = "left",
                 output_unit: Unit = Unit.METERS):
        # ``device`` keeps upstream's positional order; recorded, not used.
        super().__init__(data_root, low_lod, output_unit)
        self.device = device
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got '{hand_type}'")
        self.hand_type = hand_type

    def _get_shaped_verts(self, identity_coeffs, scale_params=None):
        raise NotImplementedError

    def get_rest_shape(self, identity_coeffs: jnp.ndarray,
                       scale_params: Optional[jnp.ndarray] = None,
                       kwargs: Optional[Mapping[str, Any]] = None) -> jnp.ndarray:
        """Shaped verts in the native frame (same as ``base_body.obj``)."""
        return self._get_shaped_verts(identity_coeffs, scale_params)[0]

    def __call__(self, identity_coeffs: jnp.ndarray,
                 scale_params: Optional[jnp.ndarray] = None,
                 kwargs: Optional[Mapping[str, Any]] = None,
                 global_scale=1.0) -> jnp.ndarray:
        """Shape -> topology transfer -> wrist center -> units -> global scale."""
        v_shaped, wrist_pos = self._get_shaped_verts(jnp.asarray(identity_coeffs), scale_params)
        result = self._apply_coord_transform(self.identity_model_to_soma(v_shaped))
        wrist_pos = self._apply_coord_transform(wrist_pos[:, None, :])[:, 0, :]
        result = result - wrist_pos[:, None, :]
        return self._finish(result, global_scale)

    forward = __call__


class MANOHandIdentityModel(BaseHandIdentityModel):
    """MANO hand shape model producing SOMAHand-topology rest shapes.

    Loads raw MANO ``v_template`` + ``shapedirs`` (MANO's native frame, metres),
    shapes them with betas, centres at the wrist joint, and transfers to the
    SOMAHand topology via barycentric interpolation.

    Assets in ``data_root / "MANO"`` (per hand): ``MANO_{LEFT,RIGHT}.pkl`` (the
    licensed MANO model, or pass ``model_path``), ``base_hand_{left,right}.obj``
    and ``SOMA_wrap_{left,right}.obj`` (shipped).
    """

    NATIVE_UNIT = Unit.METERS
    _wrist_joint_index = 0

    def __init__(self, data_root, low_lod: bool = False, device=None, hand_type: str = "left",
                 output_unit: Unit = Unit.METERS, model_path=None):
        super().__init__(data_root, low_lod, device, hand_type, output_unit)
        mano = load_mano_pkl(self.data_root, hand_type, model_path=model_path)
        self._v_template = jnp.asarray(mano["v_template"])                  # (778, 3)
        self._shapedirs = jnp.asarray(mano["shapedirs"])                    # (778, 3, 10)
        self._num_betas = int(mano["shapedirs"].shape[2])
        self._J_regressor = jnp.asarray(mano["J_regressor"])                # (16, 778)

        mano_dir = self.data_root / "MANO"
        V_base, F_base = _load_obj(mano_dir / f"base_hand_{hand_type}.obj")
        V_wrap, _ = _load_obj(mano_dir / f"SOMA_wrap_{hand_type}.obj")
        # SOMA vertices with no real MANO correspondence (Laplacian-blended
        # wrist boundary verts far from the MANO surface) — excluded from pose
        # inversion by the layer. Upstream's exact criterion.
        import trimesh
        base = trimesh.Trimesh(vertices=V_base, faces=F_base, process=False)
        _, dists, _ = base.nearest.on_surface(V_wrap)
        threshold = max(float(np.median(dists)) * 5, 0.002)
        self.no_correspondence_ids = np.where(dists > threshold)[0].astype(np.int64)
        self._setup_topology_transfer(V_base, F_base, V_wrap)

    @property
    def num_identity_coeffs(self) -> int:
        return self._num_betas

    def _get_shaped_verts(self, identity_coeffs, scale_params=None):
        """(v_shaped (B, 778, 3), wrist_pos (B, 3)) in MANO native metres."""
        blend = jnp.einsum("bk,vdk->bvd", identity_coeffs, self._shapedirs)
        v_shaped = self._v_template[None] + blend
        wrist_pos = jnp.einsum("jv,bvd->bjd", self._J_regressor, v_shaped)[
            :, self._wrist_joint_index]
        return v_shaped, wrist_pos


class MHRHandIdentityModel(BaseHandIdentityModel):
    """MHR hand identity model for SOMAHand.

    Runs the full-body MHR model with hand-only parameters exposed, then
    extracts hand vertices from the SOMA-topology output. Exposes 5 identity
    shape coefficients (MHR dims 40-44) and 26 named scale parameters per hand
    (overall hand scale + 25 per-finger segment lengths, offsets and null
    transforms).
    """

    NATIVE_UNIT = Unit.CENTIMETERS

    # Per-hand scale parameter names and their indices into the 68-dim MHR scale vector.
    _SCALE_LAYOUT_RIGHT = [
        (8, "scale_r_hands"),
        (18, "scale_r_index1_length"), (19, "scale_r_middle1_length"),
        (20, "scale_r_ring1_length"), (21, "scale_r_pinky1_length"),
        (22, "scale_r_thumb1_length"),
        (23, "scale_r_index1_offset"), (24, "scale_r_middle1_offset"),
        (25, "scale_r_ring1_offset"), (26, "scale_r_pinky1_offset"),
        (27, "scale_r_thumb1_offset"),
        (28, "scale_r_index2_length"), (29, "scale_r_middle2_length"),
        (30, "scale_r_ring2_length"), (31, "scale_r_pinky2_length"),
        (32, "scale_r_thumb2_length"),
        (33, "scale_r_index3_length"), (34, "scale_r_middle3_length"),
        (35, "scale_r_ring3_length"), (36, "scale_r_pinky3_length"),
        (37, "scale_r_thumb3_length"),
        (38, "scale_r_index_null_tx"), (39, "scale_r_middle_null_tx"),
        (40, "scale_r_ring_null_tx"), (41, "scale_r_pinky_null_tx"),
        (42, "scale_r_thumb_null_tx"),
    ]
    _SCALE_LAYOUT_LEFT = [
        (9, "scale_l_hands"),
        (43, "scale_l_index1_length"), (44, "scale_l_middle1_length"),
        (45, "scale_l_ring1_length"), (46, "scale_l_pinky1_length"),
        (47, "scale_l_thumb1_length"),
        (48, "scale_l_index1_offset"), (49, "scale_l_middle1_offset"),
        (50, "scale_l_ring1_offset"), (51, "scale_l_pinky1_offset"),
        (52, "scale_l_thumb1_offset"),
        (53, "scale_l_index2_length"), (54, "scale_l_middle2_length"),
        (55, "scale_l_ring2_length"), (56, "scale_l_pinky2_length"),
        (57, "scale_l_thumb2_length"),
        (58, "scale_l_index3_length"), (59, "scale_l_middle3_length"),
        (60, "scale_l_ring3_length"), (61, "scale_l_pinky3_length"),
        (62, "scale_l_thumb3_length"),
        (63, "scale_l_index_null_tx"), (64, "scale_l_middle_null_tx"),
        (65, "scale_l_ring_null_tx"), (66, "scale_l_pinky_null_tx"),
        (67, "scale_l_thumb_null_tx"),
    ]

    # MHR identity shape: dims 40-44 are hand shape (5 dims out of 45).
    _MHR_HAND_IDENTITY_OFFSET = 40
    _MHR_IDENTITY_DIM = 45

    def __init__(self, data_root, low_lod: bool = False, device=None, hand_type: str = "left",
                 output_unit: Unit = Unit.METERS, mhr_model=None):
        """
        Args:
            mhr_model: an existing :class:`~soma_jax.body_models.mhr_native.MHRNativeModel`
                to share; lifted from ``data_root/MHR/mhr_model_lod1.pt`` when
                omitted (needs ``torch`` once, to read the archive).
        """
        super().__init__(data_root, low_lod, device, hand_type, output_unit)
        layout = self._SCALE_LAYOUT_LEFT if hand_type == "left" else self._SCALE_LAYOUT_RIGHT
        self._scale_mhr_indices = np.asarray([idx for idx, _ in layout], np.int64)
        self.scale_param_names = tuple(name for _, name in layout)

        hand_data = np.load(self.data_root / "SOMAHand.npz", allow_pickle=False)
        if mhr_model is None:
            from ..body_models.mhr_native import MHRNativeModel
            mhr_model = MHRNativeModel.from_torchscript(self.data_root / "MHR" / "mhr_model_lod1.pt")
        self._mhr_model = mhr_model

        # Full-body MHR -> SOMA topology transfer. Upstream calls
        # `_setup_topology_transfer_with_blending(..., None)`: with no excluded
        # vertices that is plain barycentric interpolation.
        V_mhr, F_mhr = _load_obj(self.data_root / "MHR" / "base_body_lod1.obj")
        V_soma, _ = _load_obj(self.data_root / "MHR" / "SOMA_wrap_lod1.obj")
        self._setup_topology_transfer(V_mhr, F_mhr, V_soma)

        self._hand_vert_ids = jnp.asarray(hand_data[f"{hand_type}_vert_ids"].astype(np.int64))
        boundary_local = hand_data[f"{hand_type}_boundary_loop"]
        self._wrist_boundary_global = jnp.asarray(
            hand_data[f"{hand_type}_vert_ids"][boundary_local].astype(np.int64))

    @property
    def num_identity_coeffs(self) -> int:
        return 5

    @property
    def num_scale_params(self) -> Optional[int]:
        return 26

    def _full_body_mhr_verts(self, identity_coeffs, scale_params):
        B = identity_coeffs.shape[0]
        full_identity = jnp.pad(
            identity_coeffs,
            ((0, 0), (self._MHR_HAND_IDENTITY_OFFSET,
                      self._MHR_IDENTITY_DIM - self._MHR_HAND_IDENTITY_OFFSET - 5)))
        full_scale = jnp.zeros((B, 68), identity_coeffs.dtype)
        if scale_params is not None:
            full_scale = full_scale.at[:, self._scale_mhr_indices].set(jnp.asarray(scale_params))
        model_params = jnp.concatenate([jnp.zeros((B, 136), identity_coeffs.dtype), full_scale], 1)
        verts, _ = self._mhr_model(full_identity, model_params,
                                   jnp.zeros((B, 72), identity_coeffs.dtype))
        return verts                                                          # (B, V_mhr, 3) cm

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        """Full-body MHR vertices (native cm), as upstream's ``get_rest_shape``."""
        return self._full_body_mhr_verts(jnp.asarray(identity_coeffs), scale_params)

    def __call__(self, identity_coeffs, scale_params=None, kwargs=None, global_scale=1.0):
        """Shape -> topology transfer -> hand extraction -> wrist center -> units."""
        full_soma = self._apply_coord_transform(
            self.identity_model_to_soma(self.get_rest_shape(identity_coeffs, scale_params)))
        wrist_pos = full_soma[:, self._wrist_boundary_global].mean(axis=1)
        result = full_soma[:, self._hand_vert_ids] - wrist_pos[:, None, :]
        return self._finish(result, global_scale)

    forward = __call__


class SOMAHandIdentityModel(_HandIdentityBase):
    """Hand-specific PCA identity model.

    Loads the left-hand PCA from ``SOMAHand.npz``. For the right hand, mirrors
    the left PCA by negating the X component of the mean and shapedirs.
    """

    NATIVE_UNIT = Unit.CENTIMETERS

    def __init__(self, data_root, low_lod: bool = False, device=None, *, hand_map, hand_type,
                 output_unit: Unit = Unit.METERS):
        # ``device`` keeps upstream's positional order; recorded, not used.
        super().__init__(data_root, low_lod, output_unit)
        self.device = device
        required = ("left_mean", "left_shapedirs", "left_eigenvalues")
        missing = [k for k in required if k not in hand_map]
        if missing:
            raise KeyError(
                f"SOMAHand.npz is missing required hand PCA arrays: {', '.join(missing)}")
        mean = np.asarray(hand_map["left_mean"], np.float64)            # (Vh, 3) wrist-local cm
        sd = np.asarray(hand_map["left_shapedirs"], np.float64)         # (K, Vh*3)
        eigenvalues = np.asarray(hand_map["left_eigenvalues"])          # (K,)
        if hand_type == "right":
            mean = mean.copy()
            mean[:, 0] *= -1
            K, Vh = sd.shape[0], mean.shape[0]
            sd = sd.reshape(K, Vh, 3).copy()
            sd[:, :, 0] *= -1
            sd = sd.reshape(K, Vh * 3)
        self.pca_mean = jnp.asarray(mean.ravel().astype(np.float32))
        self.pca_matrix = jnp.asarray(sd.astype(np.float32))
        self.eigenvalues = jnp.asarray(eigenvalues.astype(np.float32))

    @property
    def num_identity_coeffs(self) -> int:
        return int(self.eigenvalues.shape[0])

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        """(B, Vh, 3) wrist-local vertices in native SOMA centimetres."""
        identity_coeffs = jnp.asarray(identity_coeffs)
        weighted = identity_coeffs * jnp.sqrt(self.eigenvalues)
        shape = weighted @ self.pca_matrix + self.pca_mean[None]
        return shape.reshape(identity_coeffs.shape[0], -1, 3)

    def __call__(self, identity_coeffs, scale_params=None, kwargs=None, global_scale=1.0):
        """Upstream ``BaseIdentityModel.forward``: rest shape -> units -> scale."""
        result = self._apply_coord_transform(
            self.identity_model_to_soma(self.get_rest_shape(identity_coeffs, scale_params)))
        return self._finish(result, global_scale)

    forward = __call__
