"""SMPL: Skinned Multi-Person Linear Model in JAX.

Standard SMPL model with 6890 vertices and 24 joints.

References:
    Loper et al. (2015). SMPL: A Skinned Multi-Person Linear Model.
    ACM Trans. Graphics (SIGGRAPH Asia).

Upstream: ``soma/smpl/__init__.py (SMPLLayer)``
    Corresponding core math to upstream's SMPL-family forward
    (`soma/smpl/__init__.py`, `SMPLLayer`), with a different API and feature
    surface. Not parity-tested against it.
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from ._base import BaseBodyModel, BodyModelOutput
from ..geometry.transforms import axis_angle_to_rotmat


class SMPLParams(NamedTuple):
    """SMPL pose & shape parameters."""
    betas: jnp.ndarray           # (B, num_betas)
    body_pose: jnp.ndarray       # (B, (J-1)*3) axis-angle for non-root joints
    global_orient: jnp.ndarray   # (B, 3) root rotation axis-angle
    transl: jnp.ndarray          # (B, 3) root translation


class SMPLModel(BaseBodyModel):
    """SMPL body model.

    Joint layout (24 joints):
        0: pelvis        1: l_hip     2: r_hip      3: spine1
        4: l_knee        5: r_knee    6: spine2     7: l_ankle
        8: r_ankle       9: spine3   10: l_foot    11: r_foot
       12: neck         13: l_collar 14: r_collar  15: head
       16: l_shoulder   17: r_shoulder 18: l_elbow 19: r_elbow
       20: l_wrist      21: r_wrist  22: l_hand    23: r_hand
    """

    @classmethod
    def load(cls, path: str, num_betas: int = 10) -> "SMPLModel":
        """Load an SMPL model from a .pkl or .npz file.

        Args:
            path: path to the SMPL model file (e.g. SMPL_NEUTRAL.pkl).
            num_betas: number of shape coefficients to keep.

        Returns:
            Instantiated SMPLModel.
        """
        from .model_io import load_smpl_data
        data = load_smpl_data(path)
        return cls(
            v_template=data["v_template"],
            shapedirs=data["shapedirs"],
            posedirs=data["posedirs"],
            J_regressor=data["J_regressor"],
            parents=data["parents"],
            weights=data["weights"],
            faces=data["faces"],
            num_betas=num_betas,
        )

    def _vertex_blend_shapes(self, params: SMPLParams) -> jnp.ndarray:
        """SMPL: shape blend shapes only."""
        return self.v_template[None] + jnp.einsum(
            "vcp,bp->bvc", self.shapedirs, params.betas
        )

    def _build_rotmats(self, params: SMPLParams, B: int) -> jnp.ndarray:
        """Build (B, J=24, 3, 3) from global orient + (B, 69) body pose."""
        n_body = self.num_joints - 1   # 23 non-root joints
        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]

        body_pose_flat = params.body_pose.reshape(B * n_body, 3)
        R_body = jax.vmap(axis_angle_to_rotmat)(body_pose_flat).reshape(
            B, n_body, 3, 3
        )
        return jnp.concatenate([R_root, R_body], axis=1)

    @staticmethod
    def make_params(
        betas: jnp.ndarray | None = None,
        body_pose: jnp.ndarray | None = None,
        global_orient: jnp.ndarray | None = None,
        transl: jnp.ndarray | None = None,
        batch_size: int = 1,
        num_betas: int = 10,
        num_joints: int = 24,
    ) -> SMPLParams:
        """Create SMPLParams with zero-defaults for missing fields."""
        n_body = num_joints - 1
        return SMPLParams(
            betas=betas if betas is not None else jnp.zeros((batch_size, num_betas)),
            body_pose=body_pose if body_pose is not None else jnp.zeros((batch_size, n_body * 3)),
            global_orient=global_orient if global_orient is not None else jnp.zeros((batch_size, 3)),
            transl=transl if transl is not None else jnp.zeros((batch_size, 3)),
        )


# Standard SMPL joint names
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
