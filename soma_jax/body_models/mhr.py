"""MHR (Meta Human Rig) body model in JAX.

MHR is a high-fidelity body model with per-body-part scaling.
Native units are centimeters, axis convention same as SOMA (Y-up).

Per-body-part scale parameters allow finer-grained control over body
proportions than SMPL's PCA-only shape model.

Note: this implementation provides a generic MHR body model. Users with
actual MHR model files should adapt the loading logic to their data format.

Upstream: none — SOMA-JAX-only.
    Standalone MHR forward pass.
"""
from __future__ import annotations
from typing import NamedTuple, Optional
import numpy as np
import jax
import jax.numpy as jnp

from ._base import BaseBodyModel, BodyModelOutput
from ..geometry.transforms import axis_angle_to_rotmat


class MHRParams(NamedTuple):
    """MHR pose, shape, and scale parameters."""
    betas: jnp.ndarray              # (B, num_betas) PCA shape
    body_pose: jnp.ndarray          # (B, (J-1)*3) axis-angle
    global_orient: jnp.ndarray      # (B, 3)
    transl: jnp.ndarray             # (B, 3)
    scale_params: jnp.ndarray       # (B, num_scale_params) per-body-part scale


class MHRModel(BaseBodyModel):
    """MHR high-fidelity body model with body-part scale parameters."""

    def __init__(
        self,
        v_template: np.ndarray,
        shapedirs: np.ndarray,
        posedirs: Optional[np.ndarray],
        J_regressor: np.ndarray,
        parents: np.ndarray,
        weights: np.ndarray,
        faces: np.ndarray,
        part_vertex_ids: Optional[dict[str, np.ndarray]] = None,
        num_betas: int = 10,
    ):
        super().__init__(
            v_template, shapedirs, posedirs, J_regressor,
            parents, weights, faces, num_betas,
        )
        self.part_vertex_ids = part_vertex_ids or {}
        self.num_scale_params = len(self.part_vertex_ids)

    @classmethod
    def load(cls, path: str, num_betas: int = 10) -> "MHRModel":
        """Load an MHR model from an NPZ file."""
        from .model_io import load_smpl_data
        data = load_smpl_data(path)
        return cls(
            v_template=data["v_template"],
            shapedirs=data["shapedirs"],
            posedirs=data.get("posedirs"),
            J_regressor=data["J_regressor"],
            parents=data["parents"],
            weights=data["weights"],
            faces=data["faces"],
            part_vertex_ids=data.get("part_vertex_ids", {}),
            num_betas=num_betas,
        )

    def _vertex_blend_shapes(self, params: MHRParams) -> jnp.ndarray:
        """MHR: shape blend shapes + per-body-part scaling."""
        v = self.v_template[None] + jnp.einsum(
            "vcp,bp->bvc", self.shapedirs, params.betas
        )
        # Apply per-body-part scale (multiplicative)
        v = self._apply_part_scaling(v, params.scale_params)
        return v

    def _apply_part_scaling(
        self, vertices: jnp.ndarray, scale_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Multiply vertices belonging to each body part by its scale parameter."""
        if scale_params.shape[-1] == 0:
            return vertices
        for i, (_, vid_arr) in enumerate(self.part_vertex_ids.items()):
            if i >= scale_params.shape[-1]:
                break
            ids = jnp.asarray(vid_arr, dtype=jnp.int32)
            scale = scale_params[:, i : i + 1, None]   # (B, 1, 1)
            vertices = vertices.at[:, ids, :].mul(scale)
        return vertices

    def _build_rotmats(self, params: MHRParams, B: int) -> jnp.ndarray:
        n_body = self.num_joints - 1
        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]
        body_pose_flat = params.body_pose.reshape(B * n_body, 3)
        R_body = jax.vmap(axis_angle_to_rotmat)(body_pose_flat).reshape(
            B, n_body, 3, 3
        )
        return jnp.concatenate([R_root, R_body], axis=1)

    @staticmethod
    def make_params(
        batch_size: int = 1,
        num_betas: int = 10,
        num_joints: int = 24,
        num_scale_params: int = 0,
        **overrides,
    ) -> MHRParams:
        n_body = num_joints - 1
        defaults = dict(
            betas=jnp.zeros((batch_size, num_betas)),
            body_pose=jnp.zeros((batch_size, n_body * 3)),
            global_orient=jnp.zeros((batch_size, 3)),
            transl=jnp.zeros((batch_size, 3)),
            # Default scale_params is 1.0 (no scaling), filled with ones
            scale_params=jnp.ones((batch_size, num_scale_params)),
        )
        defaults.update(overrides)
        return MHRParams(**defaults)
