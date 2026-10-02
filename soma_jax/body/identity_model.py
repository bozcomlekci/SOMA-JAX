"""Full-body identity-model backends built from ``data_root``, as upstream builds them.

Upstream: ``soma/body/identity_model.py`` (SOMA native, MHR, Anny, SMPL family,
GarmentMeasurements) on top of ``soma/identity_model.py``'s ``BaseIdentityModel``.

Each backend is constructed from the asset directory exactly as upstream's is:
the native model, then a correspondence from the backend's own mesh
(``<pack>/base_body.obj``) onto its SOMA wrap (``<pack>/SOMA_wrap.obj``) with
upstream's four-coordinate tetrahedral barycentric embedding, and — for the
backends whose meshes lack eye bags and a mouth bag — a Laplacian blend of those
vertices against that same wrap. ``forward`` returns SOMA-topology vertices in
``output_unit`` (global scale applied), like upstream's ``forward``.

This is the faithful construction :meth:`soma_jax.SOMALayer.from_upstream_assets`
uses for non-default backends; ``BaseIdentityModel`` lives in
:mod:`soma_jax.identity_model`, as upstream's does. :mod:`soma_jax.identity_packs`
keeps SOMA-JAX's own pack-based backends (``create_identity_model(type,
soma_data, model_data)``, fed by ``tools/pipeline/build_identity_packs.py``),
which predate this module.

Differences are mechanical:

* JAX arrays instead of ``nn.Module`` buffers; ``device`` is accepted and ignored.
* MHR runs :class:`~soma_jax.body_models.mhr_native.MHRNativeModel`, the weights
  lifted out of the same TorchScript archive upstream loads.
* Anny evaluates ``template + blendshapes · coeffs`` in JAX
  (:class:`~soma_jax.body_models.anny_native.AnnyNativeModel`); the phenotype ->
  coefficient step stays inside the ``anny`` package, as upstream's does.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from ..identity_model import BaseIdentityModel, CoordAxis, apply_coord_transform  # noqa: F401
from ..units import Unit

logger = logging.getLogger(__name__)

__all__ = [
    "BaseIdentityModel",
    "AnnySimplified",
    "SMPLSimplified",
    "AnnyIdentityModel",
    "GarmentMeasurementIdentityModel",
    "MHRIdentityModel",
    "SMPLIdentityModel",
    "SOMAIdentityModel",
    "create_identity_model",
]


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``trimesh.load(path, maintain_order=True, process=False)``, as upstream."""
    import trimesh
    mesh = trimesh.load(str(path), maintain_order=True, process=False)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)




class MHRIdentityModel(BaseIdentityModel):
    """MHR: 45 identity coefficients and 68 body-part scales (centimetres, Y-up)."""

    NATIVE_UNIT = Unit.CENTIMETERS

    @property
    def num_identity_coeffs(self) -> int:
        return 45

    @property
    def num_scale_params(self) -> Optional[int]:
        # 68 body-part scales; the MHR model takes 136 pose + 68 scale parameters.
        return 68

    def __init__(self, data_root, low_lod, device=None, *, vertex_ids_to_exclude=None,
                 mhr_model=None, **kwargs):
        """
        Args:
            mhr_model: an existing :class:`~soma_jax.body_models.mhr_native.MHRNativeModel`
                to share (SOMA-JAX extra); lifted from ``MHR/mhr_model_{lod}.pt``
                (``lod6`` at low LOD, else ``lod1``) when omitted, which needs torch.
        """
        super().__init__(data_root, low_lod, device, **kwargs)
        lod = "lod1" if not self.low_lod else "lod6"
        if mhr_model is None:
            from ..body_models.mhr_native import MHRNativeModel
            mhr_model = MHRNativeModel.from_torchscript(self.data_root / "MHR" / f"mhr_model_{lod}.pt")
        self.identity_model = mhr_model
        V_mhr, F_mhr = _load_obj(self.data_root / "MHR" / f"base_body_{lod}.obj")
        V_soma, F_soma = _load_obj(self.data_root / "MHR" / "SOMA_wrap_lod1.obj")
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(V_mhr, F_mhr, V_soma, F_soma,
                                                    vertex_ids_to_exclude)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None) -> jnp.ndarray:
        """Rest shape in centimetres. ``scale_params`` (B, 68) is required.

        ``kwargs["bone_length_flexibles"]`` (B, 6) is written into
        ``pose_params[130:136]`` as upstream does.
        """
        if scale_params is None:
            raise AssertionError("scale_params is required for MHR")
        blf = None if kwargs is None else kwargs.get("bone_length_flexibles")
        return self.identity_model.get_rest_shape(identity_coeffs, scale_params,
                                                  bone_length_flexibles=blf)


class SMPLSimplified:
    """Minimal SMPL-family shape model for identity-only forward passes (upstream's)."""

    def __init__(self, model_data: dict, device=None):
        self.device = device
        self.faces = model_data["faces"]
        self.v_template = jnp.asarray(model_data["v_template"], jnp.float32)
        self.shape_dirs = jnp.asarray(model_data["shapedirs"], jnp.float32)
        self.num_betas = int(self.shape_dirs.shape[2])

    def forward(self, betas=None) -> jnp.ndarray:
        """Template plus shape blend: ``v_template + einsum(betas, shapedirs)``."""
        return self.v_template + jnp.einsum("bl,mkl->bmk", jnp.asarray(betas), self.shape_dirs)

    __call__ = forward


class AnnySimplified:
    """Wrapper around Anny that simplifies the forward pass (upstream's).

    Keeps the six SOMA phenotype labels and the local changes SOMA drives
    (facial detail is ignored), and evaluates the rest shape in JAX through
    :class:`~soma_jax.body_models.anny_native.AnnyNativeModel`; the phenotype ->
    blendshape-coefficient step stays inside the ``anny`` package, as upstream's.
    """

    def __init__(self, anny_model, device=None):
        from ..body_models.anny_native import AnnyNativeModel
        self.device = device
        self.native = AnnyNativeModel.from_anny(anny_model)
        self.anny_model = self.native._anny
        self.phenotype_labels = list(self.native.phenotype_labels)
        self.local_change_labels = list(self.native.local_change_labels)
        self.ignore_change_labels = list(self.native.ignored_change_labels)

    def forward(self, phenotype_kwargs=None, local_changes_kwargs=None) -> jnp.ndarray:
        """Phenotype values (a dict, or a (B, 6) array) and optional local
        changes -> rest vertices."""
        return self.native.get_rest_shape(phenotype_kwargs, local_changes=local_changes_kwargs)

    __call__ = forward


class AnnyIdentityModel(BaseIdentityModel):
    """Anny: six phenotype values plus local changes (metres, Z-up, -Y forward)."""

    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Z
    NATIVE_FORWARD = CoordAxis.NEG_Y

    @property
    def num_identity_coeffs(self) -> int:
        return len(self.identity_model.phenotype_labels)

    @property
    def num_scale_params(self) -> Optional[int]:
        return len(self.identity_model.local_change_labels)

    def __init__(self, data_root, low_lod, device=None, *, vertex_ids_to_exclude=None,
                 anny_model=None, **kwargs):
        # Anny's mesh has a mouth bag and eye bags, so nothing is excluded.
        super().__init__(data_root, low_lod, device, **kwargs)
        from ..body_models.anny_native import AnnyNativeModel
        self.identity_model = AnnyNativeModel.from_anny(anny_model)
        V_anny, F_anny = _load_obj(self.data_root / "Anny" / "base_body.obj")
        V_soma, _ = _load_obj(self.data_root / "Anny" / "SOMA_wrap.obj")
        V_soma, _ = self._apply_soma_lod(V_soma)
        self._setup_topology_transfer(V_anny, F_anny, V_soma)
        self.scale_param_names = tuple(self.identity_model.local_change_labels)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None) -> jnp.ndarray:
        return self.identity_model.get_rest_shape(identity_coeffs, local_changes=scale_params)


class SMPLIdentityModel(BaseIdentityModel):
    """SMPL / SMPL-H / SMPL-X shape betas (metres, Y-up)."""

    NATIVE_UNIT = Unit.METERS

    @property
    def num_identity_coeffs(self) -> int:
        return self.identity_model.num_betas

    def __init__(self, data_root, low_lod, device=None, model_type: str = "smpl", *,
                 vertex_ids_to_exclude=None, gender: str = "neutral", model_path=None,
                 num_betas: int = 10, **kwargs):
        super().__init__(data_root, low_lod, device, **kwargs)
        from ..smpl.layers import load_smpl_family_model
        imt = model_type
        if model_path is not None:
            model_path = Path(model_path).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"SMPL model not found at '{model_path}'")
            logger.info("Loading %s model from %s", imt.upper(), model_path)
        else:
            model_dir = self.data_root / imt.upper()
            npz = model_dir / f"{imt.upper()}_{gender.upper()}.npz"
            pkl = model_dir / f"{imt.upper()}_{gender.upper()}.pkl"
            if npz.exists():
                model_path = npz
                logger.info("Loading %s model from %s", imt.upper(), npz)
            elif pkl.exists():
                model_path = pkl
                logger.info("Loading %s model from %s", imt.upper(), pkl)
            else:
                raise FileNotFoundError(
                    f"Neither {npz} nor {pkl} found. Cannot load {imt.upper()} model.\n"
                    "Pass model_path via identity_model_kwargs, or place the file in "
                    f"<data_root>/{imt.upper()}/.")
        model_data = load_smpl_family_model(model_path, model_type=imt, num_betas=int(num_betas))
        self.identity_model = SMPLSimplified(model_data, self.device)
        self.faces = np.asarray(model_data["faces"])
        V_smpl, F_smpl = _load_obj(self.data_root / imt.upper() / "base_body.obj")
        V_soma, F_soma = _load_obj(self.data_root / imt.upper() / "SOMA_wrap.obj")
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(V_smpl, F_smpl, V_soma, F_soma,
                                                    vertex_ids_to_exclude)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None) -> jnp.ndarray:
        return self.identity_model(jnp.asarray(identity_coeffs, jnp.float32))


class GarmentMeasurementIdentityModel(BaseIdentityModel):
    """GarmentMeasurements PCA (``point.npz``; metres, Y-up)."""

    NATIVE_UNIT = Unit.METERS

    @property
    def num_identity_coeffs(self) -> int:
        return int(self.eigenvalues.shape[0])

    def __init__(self, data_root, low_lod, device=None, *, vertex_ids_to_exclude=None,
                 **kwargs):
        super().__init__(data_root, low_lod, device, **kwargs)
        self.pca_npz_file = self.data_root / "GarmentMeasurements" / "point.npz"
        data = np.load(self.pca_npz_file, allow_pickle=False)
        self.pca_matrix = jnp.asarray(data["pca_matrix"], jnp.float32)
        self.pca_mean = jnp.asarray(data["pca_mean"], jnp.float32)
        self.eigenvalues = jnp.asarray(data["eigenvalues"], jnp.float32)
        _, F_garment = _load_obj(self.data_root / "GarmentMeasurements" / "mean.obj")
        V_garment = np.asarray(data["pca_mean"], np.float32).reshape(-1, 3)
        V_soma, F_soma = _load_obj(self.data_root / "GarmentMeasurements" / "SOMA_wrap.obj")
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(V_garment, F_garment, V_soma, F_soma,
                                                    vertex_ids_to_exclude)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None) -> jnp.ndarray:
        c = jnp.asarray(identity_coeffs, jnp.float32)
        shape = self.pca_mean[None] + (c * jnp.sqrt(self.eigenvalues)) @ self.pca_matrix.T
        return shape.reshape(c.shape[0], -1, 3)


class SOMAIdentityModel(BaseIdentityModel):
    """SOMA's own shape PCA from ``SOMA_neutral.npz`` (centimetres, Y-up)."""

    NATIVE_UNIT = Unit.CENTIMETERS

    @property
    def num_identity_coeffs(self) -> int:
        return int(self.eigenvalues.shape[0])

    def __init__(self, data_root, low_lod, device=None, *, vertex_ids_to_exclude=None,
                 **kwargs):
        super().__init__(data_root, low_lod, device, **kwargs)
        self.pca_npz_file = self.data_root / "SOMA_neutral.npz"
        data = np.load(self.pca_npz_file, allow_pickle=False)
        mean = np.asarray(data["mean"])
        shapedirs = np.asarray(data["shapedirs"]).reshape(data["shapedirs"].shape[0], -1, 3)
        self._pca_is_lod_subset = self._nv_lod_mid_to_low is not None
        if self._pca_is_lod_subset:
            mean = mean[self._nv_lod_mid_to_low]
            shapedirs = shapedirs[:, self._nv_lod_mid_to_low, :]
        self.pca_matrix = jnp.asarray(shapedirs.reshape(shapedirs.shape[0], -1).T, jnp.float32)
        self.pca_mean = jnp.asarray(mean.reshape(-1), jnp.float32)
        self.eigenvalues = jnp.asarray(data["eigenvalues"], jnp.float32)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None) -> jnp.ndarray:
        c = jnp.asarray(identity_coeffs, jnp.float32)
        shape = self.pca_mean[None] + (c * jnp.sqrt(self.eigenvalues)) @ self.pca_matrix.T
        return shape.reshape(c.shape[0], -1, 3)


def create_identity_model(identity_model_type: str, data_root, low_lod: bool, device=None,
                          output_unit: Unit = Unit.METERS, **kwargs: Any) -> BaseIdentityModel:
    """Build a full-body identity backend from ``data_root`` (upstream's factory).

    Args:
        identity_model_type: ``"soma"``, ``"mhr"``, ``"anny"``, ``"smpl"``,
            ``"smplh"``, ``"smplx"`` or ``"garment"``.
        data_root: the asset directory (e.g. :func:`soma_jax.assets.data_root`).
        low_lod: evaluate on the low LOD (MHR switches to its ``lod6`` model).
        output_unit: unit of :meth:`BaseIdentityModel.forward`'s output.
        **kwargs: ``nv_lod_mid_to_low``, ``soma_low_lod_faces``,
            ``vertex_ids_to_exclude`` and the backend's own options
            (``model_path``, ``gender``, ``num_betas`` for the SMPL family).
    """
    t = identity_model_type.lower()
    if t == "soma":
        return SOMAIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    if t == "mhr":
        return MHRIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    if t == "anny":
        return AnnyIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    if t in ("smplx", "smplh", "smpl"):
        return SMPLIdentityModel(data_root, low_lod, device, model_type=t,
                                 output_unit=output_unit, **kwargs)
    if t == "garment":
        return GarmentMeasurementIdentityModel(data_root, low_lod, device,
                                               output_unit=output_unit, **kwargs)
    raise ValueError(f"Invalid identity model: {identity_model_type}")
