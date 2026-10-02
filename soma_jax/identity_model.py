"""Identity-model base class and coordinate conventions.

Upstream: ``soma/identity_model.py`` (SOMA-X v0.3.3).
    :class:`CoordAxis` and :class:`BaseIdentityModel` — native units inside,
    ``output_unit`` out — which the data-root backends in
    :mod:`soma_jax.body.identity_model` build on. As upstream, the body
    backends (``SOMAIdentityModel``, ``MHRIdentityModel``, ``AnnyIdentityModel``,
    ``SMPLIdentityModel``, ``GarmentMeasurementIdentityModel``, the
    ``SMPLSimplified`` / ``AnnySimplified`` wrappers and
    ``create_identity_model``) also resolve here, lazily, for legacy
    ``soma.identity_model.<backend>`` references. Upstream's
    ``NonPersistentModuleWrapper`` hides torch module weights from
    ``state_dict``; JAX modules have no state dict, so it has no counterpart.

SOMA-JAX's own pack-based backends (``create_identity_model(type, soma_data,
model_data)``) live in :mod:`soma_jax.identity_packs`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from .geometry.barycentric_interp import barycentric_interpolate, compute_barycentric_coords
from .units import Unit


class CoordAxis:
    """Named axis constants for declaring a model's native coordinate convention.

    Each constant is an ``(axis_index, sign)`` tuple (0=X, 1=Y, 2=Z). SOMA
    standard: Y+ up (``Y``), Z+ forward (``Z``).
    """

    X = (0, +1)
    Y = (1, +1)
    Z = (2, +1)
    NEG_X = (0, -1)
    NEG_Y = (1, -1)
    NEG_Z = (2, -1)


#: Parity of each (right, up, forward) axis permutation, so the derived right
#: axis sign keeps the remap a proper rotation (upstream ``_PERM_PARITY``).
_PERM_PARITY = {
    (0, 1, 2): +1, (1, 2, 0): +1, (2, 0, 1): +1,
    (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1,
}


def apply_coord_transform(verts, native_up, native_forward):
    """Reorder/negate axes from a native convention to SOMA (Y+ up, Z+ forward).

    Upstream ``BaseIdentityModel._apply_coord_transform``.
    """
    if native_up == CoordAxis.Y and native_forward == CoordAxis.Z:
        return verts
    up_idx, up_sign = native_up
    fwd_idx, fwd_sign = native_forward
    right_idx = 3 - up_idx - fwd_idx
    right_sign = _PERM_PARITY[(right_idx, up_idx, fwd_idx)] * up_sign * fwd_sign
    return jnp.concatenate([
        verts[..., right_idx:right_idx + 1] * right_sign,
        verts[..., up_idx:up_idx + 1] * up_sign,
        verts[..., fwd_idx:fwd_idx + 1] * fwd_sign,
    ], axis=-1)


class BaseIdentityModel(ABC):
    """Upstream ``BaseIdentityModel``: native units inside, ``output_unit`` out.

    Subclasses declare ``NATIVE_UNIT`` (and, when not SOMA's Y-up / Z-forward,
    ``NATIVE_UP`` / ``NATIVE_FORWARD``) and implement :meth:`get_rest_shape`.
    """

    NATIVE_UNIT: Unit
    NATIVE_UP: tuple = CoordAxis.Y
    NATIVE_FORWARD: tuple = CoordAxis.Z
    #: Names of the per-identity scale controls, in ``scale_params`` order.
    scale_param_names: tuple = ()

    def __init__(self, data_root, low_lod: bool, device=None,
                 output_unit: Unit = Unit.METERS, *,
                 nv_lod_mid_to_low=None, soma_low_lod_faces=None):
        if not isinstance(getattr(self, "NATIVE_UNIT", None), Unit):
            raise TypeError(
                f"{type(self).__name__} must define a NATIVE_UNIT class attribute "
                "(a Unit enum member)")
        self.data_root = Path(data_root)
        self.low_lod = bool(low_lod)
        self.device = device
        self.output_unit = output_unit
        self._unit_conversion = self.NATIVE_UNIT.meters_per_unit / output_unit.meters_per_unit
        self._nv_lod_mid_to_low = (None if nv_lod_mid_to_low is None
                                   else np.asarray(nv_lod_mid_to_low, np.int64))
        self._soma_low_lod_faces = (None if soma_low_lod_faces is None
                                    else np.asarray(soma_low_lod_faces, np.int64))
        self._interp = None            # (src_faces, face_ids, bary) once set up
        self._laplacian_mesh = None

    def _apply_soma_lod(self, V_soma, F_soma=None):
        """Subset SOMA-topology vertices (and swap in the LOD faces) for low LOD."""
        if self._nv_lod_mid_to_low is None:
            return V_soma, F_soma
        V_low = np.asarray(V_soma)[self._nv_lod_mid_to_low]
        F_low = self._soma_low_lod_faces if F_soma is not None else None
        return V_low, F_low

    @property
    @abstractmethod
    def num_identity_coeffs(self) -> int:
        """Number of identity coefficients :meth:`get_rest_shape` expects."""

    @property
    def num_scale_params(self) -> Optional[int]:
        """Number of scale parameters :meth:`get_rest_shape` expects, or ``None``."""
        return None

    @abstractmethod
    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None) -> jnp.ndarray:
        """The rest shape in the backend's native topology, frame and unit."""

    def _setup_topology_transfer(self, V_source, F_source, V_soma) -> None:
        """Barycentric transfer without blending (the source has inner-face geometry)."""
        # float32 in, as upstream builds its interpolators from float32 tensors.
        face_ids, bary = compute_barycentric_coords(
            np.asarray(V_soma, np.float32), np.asarray(V_source, np.float32),
            np.asarray(F_source, np.int64))
        self._interp = (jnp.asarray(np.asarray(F_source, np.int32)),
                        jnp.asarray(face_ids), jnp.asarray(bary, jnp.float32))
        self._laplacian_mesh = None

    def _setup_topology_transfer_with_blending(self, V_source, F_source, V_soma, F_soma,
                                               vertex_ids_to_exclude) -> None:
        """Barycentric transfer, then Laplacian-solve the excluded vertices.

        The excluded vertices (eye bags, mouth bag) have no counterpart on the
        source mesh; they are re-solved to keep the SOMA wrap's Laplacian
        coordinates. The wrap itself — native units, this LOD — is the
        reference mesh, as upstream's ``LaplacianMesh(V_soma, F_soma, ...)``.
        """
        from .geometry.laplacian import LaplacianMesh
        self._setup_topology_transfer(V_source, F_source, V_soma)
        if vertex_ids_to_exclude is None or len(vertex_ids_to_exclude) == 0:
            return
        mask_anchors = np.ones(np.asarray(V_soma).shape[0], dtype=bool)
        mask_anchors[np.asarray(vertex_ids_to_exclude, np.int64)] = False
        self._laplacian_mesh = LaplacianMesh(np.asarray(V_soma, np.float64),
                                             np.asarray(F_soma, np.int64), mask_anchors)

    def identity_model_to_soma(self, identity_rest_shape: jnp.ndarray) -> jnp.ndarray:
        """Source topology -> SOMA topology (with the optional Laplacian blend)."""
        if self._interp is None:
            return identity_rest_shape
        faces, face_ids, bary = self._interp
        soma_verts = barycentric_interpolate(identity_rest_shape, faces, face_ids, bary)
        if self._laplacian_mesh is not None:
            soma_verts = self._laplacian_mesh.solve(soma_verts)
        return soma_verts

    def _apply_coord_transform(self, verts: jnp.ndarray) -> jnp.ndarray:
        return apply_coord_transform(verts, self.NATIVE_UP, self.NATIVE_FORWARD)

    def forward(self, identity_coeffs, scale_params=None,
                kwargs: Optional[Mapping[str, Any]] = None,
                global_scale=1.0) -> jnp.ndarray:
        """A SOMA-topology rest shape in ``output_unit`` (upstream ``forward``)."""
        if not isinstance(identity_coeffs, dict):      # Anny also takes label dicts
            identity_coeffs = jnp.asarray(identity_coeffs)
        rest = self.get_rest_shape(identity_coeffs, scale_params, kwargs)
        result = self._apply_coord_transform(self.identity_model_to_soma(rest))
        if self._unit_conversion != 1.0:
            result = result * self._unit_conversion
        gs = jnp.asarray(global_scale, dtype=result.dtype)
        if gs.ndim == 0:
            return result * gs
        return result * gs.reshape(-1, 1, 1)

    __call__ = forward


_BODY_BACKEND_EXPORTS = frozenset({
    "AnnyIdentityModel",
    "AnnySimplified",
    "GarmentMeasurementIdentityModel",
    "MHRIdentityModel",
    "SMPLIdentityModel",
    "SMPLSimplified",
    "SOMAIdentityModel",
    "create_identity_model",
})


def __getattr__(name: str):
    """Resolve legacy ``identity_model.<body backend>`` references lazily, as upstream."""
    if name in _BODY_BACKEND_EXPORTS:
        from .body import identity_model as _body_identity_model

        return getattr(_body_identity_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseIdentityModel",
    "CoordAxis",
    *sorted(_BODY_BACKEND_EXPORTS),
]
