"""Anny: children's body model in JAX.

Anny is designed for children's body proportions. Native coordinate system
is Z-up, -Y-forward (vs SOMA's Y-up, Z-forward).

The forward pass first runs SMPL-style LBS, then applies the coordinate
transform when output is consumed by SOMA pipelines.

Upstream: none — SOMA-JAX-only.
    Standalone Anny forward pass.
"""
from __future__ import annotations
from typing import NamedTuple, Optional
import numpy as np
import jax
import jax.numpy as jnp

from ._base import BaseBodyModel, BodyModelOutput
from ..geometry.transforms import axis_angle_to_rotmat


# Anny → SOMA axis remapping
# Anny: X=right, Y=backward, Z=up  →  SOMA: X=right, Y=up, Z=forward
# Permutation (axes): (0, 2, 1)  (X→X, Z→Y, Y→Z)
# Signs:              (1, 1, -1) (negate Y because Anny Y=back, SOMA Z=forward)
_COORD_PERM = (0, 2, 1)
_COORD_SIGN = (1, 1, -1)


class AnnyParams(NamedTuple):
    betas: jnp.ndarray           # (B, num_betas)
    body_pose: jnp.ndarray       # (B, (J-1)*3)
    global_orient: jnp.ndarray   # (B, 3)
    transl: jnp.ndarray          # (B, 3)


def to_soma_coords(vertices: jnp.ndarray) -> jnp.ndarray:
    """Convert vertices from Anny convention (Z-up, -Y-fwd) to SOMA (Y-up, Z-fwd)."""
    perm = list(_COORD_PERM)
    sign = jnp.array(_COORD_SIGN, dtype=vertices.dtype)
    return vertices[..., perm] * sign


def from_soma_coords(vertices: jnp.ndarray) -> jnp.ndarray:
    """Inverse of to_soma_coords."""
    # Inverse permutation of (0,2,1) is (0,2,1) (its own inverse)
    perm = list(_COORD_PERM)
    sign = jnp.array(_COORD_SIGN, dtype=vertices.dtype)
    v = vertices * sign
    return v[..., perm]


class AnnyModel(BaseBodyModel):
    """Anny children's body model."""

    def __init__(
        self,
        v_template: np.ndarray,
        shapedirs: np.ndarray,
        posedirs: Optional[np.ndarray],
        J_regressor: np.ndarray,
        parents: np.ndarray,
        weights: np.ndarray,
        faces: np.ndarray,
        num_betas: int = 10,
        output_soma_coords: bool = True,
    ):
        super().__init__(
            v_template, shapedirs, posedirs, J_regressor,
            parents, weights, faces, num_betas,
        )
        self.output_soma_coords = output_soma_coords

    @classmethod
    def load(cls, path: str, num_betas: int = 10) -> "AnnyModel":
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
            num_betas=num_betas,
        )

    def _vertex_blend_shapes(self, params: AnnyParams) -> jnp.ndarray:
        return self.v_template[None] + jnp.einsum(
            "vcp,bp->bvc", self.shapedirs, params.betas
        )

    def _build_rotmats(self, params: AnnyParams, B: int) -> jnp.ndarray:
        n_body = self.num_joints - 1
        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]
        body_pose_flat = params.body_pose.reshape(B * n_body, 3)
        R_body = jax.vmap(axis_angle_to_rotmat)(body_pose_flat).reshape(
            B, n_body, 3, 3
        )
        return jnp.concatenate([R_root, R_body], axis=1)

    def forward(self, params: AnnyParams) -> BodyModelOutput:
        out = super().forward(params)
        if self.output_soma_coords:
            out = BodyModelOutput(
                vertices=to_soma_coords(out.vertices),
                joints=to_soma_coords(out.joints),
                v_shaped=to_soma_coords(out.v_shaped),
            )
        return out

    @staticmethod
    def make_params(
        batch_size: int = 1,
        num_betas: int = 10,
        num_joints: int = 24,
        **overrides,
    ) -> AnnyParams:
        n_body = num_joints - 1
        defaults = dict(
            betas=jnp.zeros((batch_size, num_betas)),
            body_pose=jnp.zeros((batch_size, n_body * 3)),
            global_orient=jnp.zeros((batch_size, 3)),
            transl=jnp.zeros((batch_size, 3)),
        )
        defaults.update(overrides)
        return AnnyParams(**defaults)
