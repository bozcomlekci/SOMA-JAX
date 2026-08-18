"""Identity models for SOMA-JAX.

Each identity model converts model-specific shape coefficients into
SOMA-topology vertices via:
  1. rest shape generation in the native model space
  2. topology transfer to SOMA topology (barycentric interpolation)
  3. optional Laplacian blending of inner faces
  4. coordinate transform (axis reordering to Y+ up, Z+ forward)
  5. unit conversion to meters

Supported models:
  - SMPLIdentityModel  (SMPL / SMPL-X / SMPL-H)
  - MHRIdentityModel   (MHR high-fidelity model, centimeters, body-part scales)
  - AnnyIdentityModel  (children's model, meters, Z-up / -Y-forward)
  - SOMAIdentityModel  (SOMA's own 128-coeff PCA model)
  - GarmentMeasurementIdentityModel  (CAESARS garment-measurement PCA)

Upstream: ``soma/identity_model.py``
    Mixed. `SOMAIdentityModel` (PCA with sqrt-eigenvalue scaling) is a faithful
    port. MHR / Anny / SMPL / Garment are **simplified substitutes** — linear
    PCA or bare topology transfer — not upstream's TorchScript MHR model,
    Anny phenotype logic, or eigenvalue-scaled Garment basis. The Laplacian
    inner-face blend matches upstream's conditions and free/anchor set, but
    only MHR currently invokes it; upstream also blends SMPL and Garment.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import jax.numpy as jnp

from .units import Unit
from .geometry.barycentric_interp import barycentric_interpolate, compute_barycentric_coords
from .geometry.laplacian import LaplacianMesh, laplacian_solve


class BaseIdentityModel(ABC):
    """Abstract base class for SOMA identity models."""

    #: Native unit of the model (before conversion to meters)
    native_unit: Unit = Unit.METERS

    def __init__(self, soma_data: dict, model_data: dict | None = None):
        """
        Args:
            soma_data: dict with SOMA topology data (vertices, faces, joints, etc.)
            model_data: dict with identity-model-specific parameters (loaded from NPZ/PKL).
        """
        self.soma_v_template = jnp.array(soma_data["v_template"], dtype=jnp.float32)
        self.soma_faces = np.array(soma_data["faces"], dtype=np.int32)
        self.soma_J_regressor = jnp.array(soma_data["J_regressor"], dtype=jnp.float32)
        self.soma_parents = np.array(soma_data["parents"], dtype=np.int32)
        self.joint_names: list[str] = list(soma_data.get("joint_names", []))

        # Inner-face geometry (eye bags + mouth bag) that source meshes do not
        # carry. Upstream passes exactly these to the identity model as
        # `vertex_ids_to_exclude`, and Laplacian-blends them after topology
        # transfer (soma/identity_model.py `_setup_topology_transfer_with_blending`).
        excl: list[int] = []
        for seg in ("segment_eye_bags", "segment_mouth_bag"):
            if seg in soma_data:
                excl.extend(np.asarray(soma_data[seg]).astype(int).ravel().tolist())
        self.soma_inner_face_ids = np.unique(np.asarray(excl, dtype=np.int64)) if excl else None
        self._laplacian_mesh = None

        if model_data is not None:
            self._load_model(model_data)

    @abstractmethod
    def _load_model(self, model_data: dict) -> None:
        """Load model-specific parameters."""

    @abstractmethod
    def get_rest_shape(
        self, identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Generate rest-pose vertices in native model topology and units.

        Args:
            identity_coeffs: (B, C) model-specific shape coefficients.

        Returns:
            (B, V_src, 3) rest-pose vertices in native model space.
        """

    @abstractmethod
    def identity_model_to_soma(
        self,
        src_vertices: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Transfer vertices from native topology to SOMA topology.

        Args:
            src_vertices: (B, V_src, 3) source model vertices.
            scale_params: optional (B, S) body-part scale parameters.

        Returns:
            (B, V_soma, 3) vertices in SOMA topology, SOMA units (meters, Y-up).
        """

    def forward(
        self,
        identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Full identity pipeline: coefficients → SOMA vertices + joint positions.

        Args:
            identity_coeffs: (B, C) shape coefficients.
            scale_params: optional (B, S) scale parameters.

        Returns:
            Tuple of:
            - (B, V_soma, 3) SOMA-topology rest vertices (meters, Y-up, Z-forward)
            - (B, J, 3) rest joint positions.
        """
        # Upstream hands scale_params to the *identity model* (MHR consumes
        # them inside its own forward); backends that ignore them apply the
        # per-part fallback in identity_model_to_soma instead.
        src_verts = self.get_rest_shape(identity_coeffs, scale_params)
        soma_verts = self.identity_model_to_soma(src_verts, scale_params)
        joints = jnp.einsum("jv,bvd->bjd", self.soma_J_regressor, soma_verts)
        return soma_verts, joints


    def _laplacian_free_ids(self) -> Optional[np.ndarray]:
        """Vertices the Laplacian solve fills in (upstream `vertex_ids_to_exclude`).

        The model archive may override; otherwise the SOMA rig's inner-face
        segments are used, matching upstream. ``None`` disables blending.
        """
        ids = self._laplacian_constrained_ids
        if ids is None:
            ids = self.soma_inner_face_ids
        if ids is None:
            return None
        ids = np.unique(np.asarray(ids).astype(np.int64).ravel())
        return ids if ids.size else None

    def _apply_laplacian_blend(self, soma_verts: jnp.ndarray) -> jnp.ndarray:
        """Laplacian-blend the inner-face vertices — upstream's blending step.

        Barycentric transfer cannot produce eye-bag / mouth-bag vertices,
        because the source mesh has no geometry there. Upstream re-solves them
        so they keep the SOMA template's Laplacian coordinates while joining
        smoothly onto the surrounding transferred surface
        (`soma/identity_model.py`: ``self._laplacian_mesh.solve(soma_verts)``).

        The :class:`~soma_jax.geometry.laplacian.LaplacianMesh` is built on
        first use and cached — assembly and factorisation depend only on the
        constant SOMA topology, and the per-call solve is pure JAX.

        Args:
            soma_verts: (B, V, 3) or (V, 3) interpolated SOMA vertices.

        Returns:
            Same shape, inner-face vertices re-solved.
        """
        free_ids = self._laplacian_free_ids()
        if free_ids is None:
            return soma_verts
        if self._laplacian_mesh is None:
            mask_anchors = np.ones(int(self.soma_v_template.shape[0]), dtype=bool)
            mask_anchors[free_ids] = False
            # UNITS: the blend runs on `soma_verts` in the *backend's native
            # unit* (upstream keeps every transfer in native units and converts
            # only at the output boundary), while `soma_v_template` is stored in
            # metres. The reference mesh sets ``btilde = L_U @ V_ref``, which is
            # an absolute quantity, so a metre template against centimetre
            # anchors makes the right-hand side 100x too small and drags the
            # solved vertices toward the origin. Convert the template into the
            # same unit as the vertices being solved. (The cotangent weights
            # themselves are scale-invariant, so only btilde is affected.)
            v_ref = np.asarray(self.soma_v_template) / self.native_unit.meters_per_unit
            self._laplacian_mesh = LaplacianMesh(
                v_ref, self.soma_faces, mask_anchors,
            )
        return self._laplacian_mesh.solve(soma_verts)


class SMPLIdentityModel(BaseIdentityModel):
    """Identity model backed by SMPL/SMPL-X/SMPL-H.

    Native space: Y-up, meters (same as SOMA), no topology transfer needed
    when SOMA topology matches SMPL (use barycentric transfer otherwise).
    """

    native_unit = Unit.METERS

    def _load_model(self, model_data: dict) -> None:
        self.v_template = jnp.array(model_data["v_template"], dtype=jnp.float32)
        self.shapedirs = jnp.array(model_data["shapedirs"], dtype=jnp.float32)  # (V, 3, K)
        self.src_faces = np.array(model_data["faces"], dtype=np.int32)
        n_betas = self.shapedirs.shape[-1]
        self.n_betas = n_betas

        # Precompute barycentric coords for topology transfer
        if "bary_face_ids" in model_data and "bary_coords" in model_data:
            self._face_ids = np.array(model_data["bary_face_ids"], dtype=np.int32)
            self._bary_coords = jnp.array(model_data["bary_coords"], dtype=jnp.float32)
        else:
            self._face_ids = None
            self._bary_coords = None

    def get_rest_shape(
        self, identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Pad or truncate betas to match shapedirs
        B = identity_coeffs.shape[0]
        k = min(identity_coeffs.shape[1], self.n_betas)
        betas = jnp.zeros((B, self.n_betas), dtype=jnp.float32)
        betas = betas.at[:, :k].set(identity_coeffs[:, :k])
        # Shape blend: (B, V, 3)
        return self.v_template[None] + jnp.einsum("vcp,bp->bvc", self.shapedirs, betas)

    def identity_model_to_soma(
        self,
        src_vertices: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self._face_ids is None:
            # Same topology: return as-is (assumes SOMA ≡ SMPL in this mode)
            return src_vertices
        soma_verts = barycentric_interpolate(
            src_vertices, jnp.array(self.src_faces), self._face_ids, self._bary_coords
        )
        # Upstream routes SMPL through `_setup_topology_transfer_with_blending`,
        # so the inner-face vertices the source mesh lacks are re-solved rather
        # than left as barycentric interpolation produced them.
        if self._laplacian_free_ids() is not None:
            soma_verts = self._apply_laplacian_blend(soma_verts)
        return soma_verts


class MHRIdentityModel(BaseIdentityModel):
    """MHR high-fidelity identity model.

    Native: centimeters, supports per-body-part scale parameters.
    Topology transfer via barycentric interpolation.

    .. note::
        Inner-face vertices (eye bags, mouth bag) are Laplacian-blended after
        transfer, as upstream does — see :meth:`_apply_laplacian_blend`.
    """

    native_unit = Unit.CENTIMETERS

    def _load_model(self, model_data: dict) -> None:
        self.v_template = np.array(model_data["v_template"], dtype=np.float32)
        self.shapedirs = jnp.array(model_data["shapedirs"], dtype=jnp.float32)
        self.src_faces = np.array(model_data["faces"], dtype=np.int32)
        self.n_betas = self.shapedirs.shape[-1]

        # Scale parameters: per-body-part multiplicative scale
        self.part_vertex_ids: dict[str, np.ndarray] = model_data.get("part_vertex_ids", {})
        self.n_scale_params = len(self.part_vertex_ids)

        if "bary_face_ids" in model_data and "bary_coords" in model_data:
            self._face_ids = np.array(model_data["bary_face_ids"], dtype=np.int32)
            self._bary_coords = jnp.array(model_data["bary_coords"], dtype=jnp.float32)
            self._laplacian_constrained_ids = model_data.get("laplacian_constrained_ids")
        else:
            self._face_ids = None
            self._bary_coords = None
            self._laplacian_constrained_ids = None

    #: Attached by :meth:`attach_native_archive`; ``None`` uses the PCA fallback.
    _native = None

    def attach_native_archive(self, path=None, *, from_npz: bool = False) -> None:
        """Switch this backend onto the real MHR forward.

        Upstream's ``MHRIdentityModel`` evaluates the TorchScript archive
        ``MHR/mhr_model_lod1.pt`` — a 127-joint Momentum rig with linear
        identity blendshapes, a parameter transform, pose correctives and LBS.
        Attaching it here replaces the PCA fallback below with that forward, so
        ``get_rest_shape`` reproduces upstream to ~1e-3 cm.

        Args:
            path: the archive (``.pt``), or a ``.npz`` written by
                :meth:`~soma_jax.body_models.mhr_native.MHRNativeModel.to_npz`.
                Resolved from the asset search path when omitted.
            from_npz: read ``path`` as a lifted-weights ``.npz`` instead of
                TorchScript, which avoids needing ``torch`` at all.
        """
        from .body_models.mhr_native import MHRNativeModel
        self._native = (MHRNativeModel.from_npz(path) if from_npz
                        else MHRNativeModel.from_torchscript(path))

    @property
    def uses_native_archive(self) -> bool:
        """Whether the faithful TorchScript-derived forward is in use."""
        return self._native is not None

    def get_rest_shape(
        self, identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Rest shape in centimetres — upstream's ``MHRIdentityModel.get_rest_shape``.

        With a native archive attached this is upstream's call verbatim: zero
        pose, the requested body-part scales, zero face expression, first return
        value. Without one it falls back to a linear-PCA approximation, which is
        *not* upstream's model — the scales are then applied per-part in
        :meth:`identity_model_to_soma` rather than inside the rig.
        """
        if self._native is not None:
            n_scale = 68
            if scale_params is None:
                scale_params = jnp.zeros((identity_coeffs.shape[0], n_scale), jnp.float32)
            return self._native.get_rest_shape(identity_coeffs, scale_params)

        B = identity_coeffs.shape[0]
        k = min(identity_coeffs.shape[1], self.n_betas)
        betas = jnp.zeros((B, self.n_betas), dtype=jnp.float32)
        betas = betas.at[:, :k].set(identity_coeffs[:, :k])
        return jnp.array(self.v_template)[None] + jnp.einsum("vcp,bp->bvc", self.shapedirs, betas)

    def _apply_scale_params(
        self, vertices: jnp.ndarray, scale_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply per-body-part multiplicative scale factors."""
        for i, (part_name, part_ids) in enumerate(self.part_vertex_ids.items()):
            if i >= scale_params.shape[-1]:
                break
            scale = scale_params[:, i : i + 1, None]  # (B, 1, 1)
            ids = jnp.array(part_ids)
            vertices = vertices.at[:, ids, :].mul(scale)
        return vertices

    def identity_model_to_soma(
        self,
        src_vertices: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # The native archive already consumed scale_params inside its own
        # forward (they drive joint scales, which LBS then propagates), exactly
        # as upstream does. Re-applying the per-part multiplier here would scale
        # twice. The fallback path has no other place to put them.
        if scale_params is not None and self._native is None:
            src_vertices = self._apply_scale_params(src_vertices, scale_params)

        if self._face_ids is not None:
            soma_verts = barycentric_interpolate(
                src_vertices, jnp.array(self.src_faces), self._face_ids, self._bary_coords
            )
            # Apply Laplacian blending if inner-face constraints are provided.
            # This smooths the transition between barycentric-interpolated
            # surface vertices and the inner-face vertices.
            # Upstream blends whenever the SOMA rig declares inner-face
            # geometry; `laplacian_constrained_ids` in the model archive is
            # honoured as an explicit override.
            if self._laplacian_free_ids() is not None:
                soma_verts = self._apply_laplacian_blend(soma_verts)
        else:
            soma_verts = src_vertices

        # Convert from centimeters to meters
        soma_verts = soma_verts * self.native_unit.meters_per_unit
        return soma_verts

class AnnyIdentityModel(BaseIdentityModel):
    """Anny children's body model.

    Native: meters, Z-up / -Y-forward coordinate system.
    Requires coordinate reordering to SOMA convention (Y-up, Z-forward).
    """

    native_unit = Unit.METERS

    # Coordinate mapping from Anny (Z-up, -Y-fwd) to SOMA (Y-up, Z-fwd)
    # Anny: X=right, Y=backward, Z=up
    # SOMA: X=right, Y=up, Z=forward
    _COORD_PERM = (0, 2, 1)   # X, Z, Y
    _COORD_SIGN = (1, 1, -1)  # X, Z, -Y

    def _load_model(self, model_data: dict) -> None:
        self.v_template = jnp.array(model_data["v_template"], dtype=jnp.float32)
        self.shapedirs = jnp.array(model_data["shapedirs"], dtype=jnp.float32)
        self.src_faces = np.array(model_data["faces"], dtype=np.int32)
        self.n_betas = self.shapedirs.shape[-1]

        if "bary_face_ids" in model_data and "bary_coords" in model_data:
            self._face_ids = np.array(model_data["bary_face_ids"], dtype=np.int32)
            self._bary_coords = jnp.array(model_data["bary_coords"], dtype=jnp.float32)
        else:
            self._face_ids = None
            self._bary_coords = None

    #: Attached by :meth:`attach_native_model`; ``None`` uses the PCA fallback.
    _native = None

    def attach_native_model(self, anny_model=None, *, npz_path=None) -> None:
        """Switch this backend onto the real Anny model.

        Upstream ``AnnyIdentityModel`` imports the ``anny`` package and calls it
        (``anny.create_fullbody_model(...)``), so using the package *is* the
        faithful behaviour — it is not vendored anywhere. Install it with
        ``pip install soma-jax[anny]``.

        The phenotype -> blendshape-coefficient step stays inside ``anny``
        (it is not affine, so there is no matrix to lift); the
        coefficient -> vertex step runs in JAX and is exact. See
        :mod:`soma_jax.body_models.anny_native`.

        Args:
            anny_model: an existing ``anny`` model, or ``None`` to build one
                with upstream's arguments.
            npz_path: load a cached basis instead (no ``anny``/torch needed);
                phenotype mapping is then unavailable.
        """
        from .body_models.anny_native import AnnyNativeModel
        self._native = (AnnyNativeModel.from_npz(npz_path) if npz_path is not None
                        else AnnyNativeModel.from_anny(anny_model))

    @property
    def uses_native_model(self) -> bool:
        """Whether the real Anny model is in use rather than the PCA stand-in."""
        return self._native is not None

    def get_rest_shape(
        self, identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self._native is not None:
            # Upstream's AnnySimplified.forward: phenotypes -> coeffs -> vertices.
            return self._native.get_rest_shape(identity_coeffs)

        B = identity_coeffs.shape[0]
        k = min(identity_coeffs.shape[1], self.n_betas)
        betas = jnp.zeros((B, self.n_betas), dtype=jnp.float32)
        betas = betas.at[:, :k].set(identity_coeffs[:, :k])
        return self.v_template[None] + jnp.einsum("vcp,bp->bvc", self.shapedirs, betas)

    def _apply_coord_transform(self, vertices: jnp.ndarray) -> jnp.ndarray:
        """Reorder axes from Anny convention to SOMA convention."""
        perm = list(self._COORD_PERM)
        sign = jnp.array(self._COORD_SIGN, dtype=vertices.dtype)
        return vertices[..., perm] * sign

    def identity_model_to_soma(
        self,
        src_vertices: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        src_vertices = self._apply_coord_transform(src_vertices)

        if self._face_ids is not None:
            return barycentric_interpolate(
                src_vertices, jnp.array(self.src_faces), self._face_ids, self._bary_coords
            )
        return src_vertices


class SOMAIdentityModel(BaseIdentityModel):
    """SOMA's own PCA identity model (128 principal components).

    Native: Y-up, meters — same convention as SOMA, no coordinate transform needed.
    """

    native_unit = Unit.METERS

    def __init__(self, soma_data: dict, model_data: dict | None = None):
        super().__init__(soma_data, model_data)
        if model_data is None:
            self._load_model(soma_data)

    def _load_model(self, model_data: dict) -> None:
        self.v_template = jnp.array(model_data["v_template"], dtype=jnp.float32)
        self.shapedirs = jnp.array(model_data["shapedirs"], dtype=jnp.float32)  # (V, 3, 128)
        self.n_betas = self.shapedirs.shape[-1]  # 128
        # SOMA-X weights the PCA coefficients by sqrt(eigenvalues) before the
        # basis matmul (soma/identity_model.py::SOMAIdentityModel.get_rest_shape:
        # ``weighted_coeffs = identity_coeffs * torch.sqrt(self.eigenvalues)``),
        # so unit-variance coefficients map to anatomically plausible shapes.
        # Assets converted with the current tooling carry ``eigenvalues``; older
        # ones don't, in which case coefficients are interpreted as raw PCA
        # weights (pre-scaled by the caller).
        if "eigenvalues" in model_data:
            eig = np.asarray(model_data["eigenvalues"], dtype=np.float32).reshape(-1)
            self.sqrt_eigenvalues = jnp.asarray(np.sqrt(eig))
        else:
            self.sqrt_eigenvalues = None

    def get_rest_shape(
        self, identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        B = identity_coeffs.shape[0]
        k = min(identity_coeffs.shape[1], self.n_betas)
        betas = jnp.zeros((B, self.n_betas), dtype=jnp.float32)
        betas = betas.at[:, :k].set(identity_coeffs[:, :k])
        if self.sqrt_eigenvalues is not None:
            betas = betas * self.sqrt_eigenvalues[None, :]     # SOMA-X coeff scaling
        return self.v_template[None] + jnp.einsum("vcp,bp->bvc", self.shapedirs, betas)

    def identity_model_to_soma(
        self,
        src_vertices: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Already in SOMA topology and coordinate system
        return src_vertices


class GarmentMeasurementIdentityModel(BaseIdentityModel):
    """CAESARS garment-measurement PCA identity model.

    PCA coefficients derived from physical body measurements for garment fitting.
    Native: meters, Y-up (same as SOMA).
    """

    native_unit = Unit.METERS

    def _load_model(self, model_data: dict) -> None:
        self.v_template = jnp.array(model_data["v_template"], dtype=jnp.float32)
        self.shapedirs = jnp.array(model_data["shapedirs"], dtype=jnp.float32)
        self.src_faces = np.array(model_data["faces"], dtype=np.int32)
        self.n_betas = self.shapedirs.shape[-1]

        # Upstream weights the coefficients by sqrt(eigenvalues) before the PCA
        # matmul (`identity_coeffs * sqrt(eigenvalues) @ pca_matrix.T`), so a
        # coefficient of 1.0 means "one standard deviation". Without it the
        # same coefficients produce a different body. Archives that predate the
        # field fall back to unscaled coefficients.
        self.sqrt_eigenvalues = (
            jnp.sqrt(jnp.asarray(model_data["eigenvalues"], dtype=jnp.float32))
            if "eigenvalues" in model_data else None
        )

        if "bary_face_ids" in model_data and "bary_coords" in model_data:
            self._face_ids = np.array(model_data["bary_face_ids"], dtype=np.int32)
            self._bary_coords = jnp.array(model_data["bary_coords"], dtype=jnp.float32)
            self._laplacian_constrained_ids = model_data.get("laplacian_constrained_ids")
            self._soma_constrained_ids = model_data.get("soma_constrained_ids")
        else:
            self._face_ids = None
            self._bary_coords = None
            self._laplacian_constrained_ids = None
            self._soma_constrained_ids = None

    def get_rest_shape(
        self, identity_coeffs: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        B = identity_coeffs.shape[0]
        k = min(identity_coeffs.shape[1], self.n_betas)
        betas = jnp.zeros((B, self.n_betas), dtype=jnp.float32)
        betas = betas.at[:, :k].set(identity_coeffs[:, :k])
        if self.sqrt_eigenvalues is not None:
            betas = betas * self.sqrt_eigenvalues[None, :self.n_betas]
        return self.v_template[None] + jnp.einsum("vcp,bp->bvc", self.shapedirs, betas)

    def identity_model_to_soma(
        self,
        src_vertices: jnp.ndarray,
        scale_params: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self._face_ids is not None:
            soma_verts = barycentric_interpolate(
                src_vertices, jnp.array(self.src_faces), self._face_ids, self._bary_coords
            )
            # Upstream blends the inner face for Garment too.
            if self._laplacian_free_ids() is not None:
                soma_verts = self._apply_laplacian_blend(soma_verts)
        else:
            soma_verts = src_vertices
        return soma_verts


# Registry of identity model types
_IDENTITY_MODEL_REGISTRY: dict[str, type[BaseIdentityModel]] = {
    "smpl": SMPLIdentityModel,
    "smplx": SMPLIdentityModel,
    "smplh": SMPLIdentityModel,
    "mhr": MHRIdentityModel,
    "anny": AnnyIdentityModel,
    "soma": SOMAIdentityModel,
    "soma_shape": SOMAIdentityModel,
    "garment_measurement": GarmentMeasurementIdentityModel,
}


def create_identity_model(
    model_type: str,
    soma_data: dict,
    model_data: dict | None = None,
) -> BaseIdentityModel:
    """Factory function for identity models.

    Args:
        model_type: one of 'smpl', 'smplx', 'smplh', 'mhr', 'anny', 'soma',
                    'soma_shape', 'garment_measurement'.
        soma_data: SOMA topology data dict.
        model_data: model-specific parameter dict.

    Returns:
        Instantiated identity model.
    """
    key = model_type.lower().strip()
    if key not in _IDENTITY_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown identity model type: {model_type!r}. "
            f"Valid: {sorted(_IDENTITY_MODEL_REGISTRY.keys())}"
        )
    return _IDENTITY_MODEL_REGISTRY[key](soma_data, model_data)
