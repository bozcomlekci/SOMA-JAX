"""SMPL-H: SMPL with articulated hands (MANO hands).

SMPL-H adds 30 hand joints (15 per hand) to the SMPL body, for a total of 52 joints.
No face joints or expressions (those are SMPL-X).

References:
    Romero, Tzionas, Black (2017). Embodied Hands: Modeling and Capturing Hands
    and Bodies Together. ACM Trans. Graphics (SIGGRAPH Asia).

Upstream: none — SOMA-JAX-only.
    Standalone SMPL-H forward pass.
"""
from __future__ import annotations
from typing import NamedTuple, Optional
import numpy as np
import jax
import jax.numpy as jnp

from ._base import BaseBodyModel, BodyModelOutput
from ..geometry.transforms import axis_angle_to_rotmat


_NUM_BODY_JOINTS = 21    # SMPL body without hands/head (root + 21 = 22)
_NUM_HAND_JOINTS = 15


class SMPLHParams(NamedTuple):
    """SMPL-H pose & shape parameters."""
    betas: jnp.ndarray              # (B, num_betas)
    body_pose: jnp.ndarray          # (B, 21*3) axis-angle
    global_orient: jnp.ndarray      # (B, 3)
    transl: jnp.ndarray             # (B, 3)
    left_hand_pose: jnp.ndarray     # (B, 15*3)
    right_hand_pose: jnp.ndarray    # (B, 15*3)


class SMPLHModel(BaseBodyModel):
    """SMPL-H: SMPL body + MANO hands."""

    def __init__(
        self,
        v_template: np.ndarray,
        shapedirs: np.ndarray,
        posedirs: np.ndarray,
        J_regressor: np.ndarray,
        parents: np.ndarray,
        weights: np.ndarray,
        faces: np.ndarray,
        num_betas: int = 10,
        flat_hand_mean: bool = True,
        hand_pose_mean_l: Optional[np.ndarray] = None,
        hand_pose_mean_r: Optional[np.ndarray] = None,
    ):
        super().__init__(
            v_template, shapedirs, posedirs, J_regressor,
            parents, weights, faces, num_betas,
        )
        # Hand-pose offset. Zero under the flat-hand convention (default), which
        # is what the SOMA `pose_hand` parameters assume; set
        # flat_hand_mean=False to add the model's MANO mean hand pose instead
        # (the PyTorch `smplx` package default).
        self.flat_hand_mean = bool(flat_hand_mean)
        zero = np.zeros(_NUM_HAND_JOINTS * 3, dtype=np.float32)
        hand_l = zero if (flat_hand_mean or hand_pose_mean_l is None) else np.asarray(hand_pose_mean_l).reshape(-1)
        hand_r = zero if (flat_hand_mean or hand_pose_mean_r is None) else np.asarray(hand_pose_mean_r).reshape(-1)
        self.hand_pose_mean_l = jnp.array(hand_l, dtype=jnp.float32)
        self.hand_pose_mean_r = jnp.array(hand_r, dtype=jnp.float32)

    @classmethod
    def load(cls, path: str, num_betas: int = 10,
             flat_hand_mean: bool = True) -> "SMPLHModel":
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
            flat_hand_mean=flat_hand_mean,
            hand_pose_mean_l=data.get("hands_meanl"),
            hand_pose_mean_r=data.get("hands_meanr"),
        )

    def _vertex_blend_shapes(self, params: SMPLHParams) -> jnp.ndarray:
        return self.v_template[None] + jnp.einsum(
            "vcp,bp->bvc", self.shapedirs, params.betas
        )

    def _build_rotmats(self, params: SMPLHParams, B: int) -> jnp.ndarray:
        """Build (B, J, 3, 3) — root + 21 body + 15 left hand + 15 right hand."""
        def aa_block(aa_flat: jnp.ndarray, n: int) -> jnp.ndarray:
            return jax.vmap(axis_angle_to_rotmat)(aa_flat.reshape(B * n, 3)).reshape(
                B, n, 3, 3
            )

        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]
        R_body = aa_block(params.body_pose, _NUM_BODY_JOINTS)
        R_lhand = aa_block(
            params.left_hand_pose + self.hand_pose_mean_l[None], _NUM_HAND_JOINTS,
        )
        R_rhand = aa_block(
            params.right_hand_pose + self.hand_pose_mean_r[None], _NUM_HAND_JOINTS,
        )
        R_all = jnp.concatenate([R_root, R_body, R_lhand, R_rhand], axis=1)

        J = self.num_joints
        if R_all.shape[1] < J:
            pad = jnp.broadcast_to(
                jnp.eye(3, dtype=jnp.float32)[None, None],
                (B, J - R_all.shape[1], 3, 3),
            )
            R_all = jnp.concatenate([R_all, pad], axis=1)
        elif R_all.shape[1] > J:
            R_all = R_all[:, :J]
        return R_all

    @staticmethod
    def make_params(
        batch_size: int = 1,
        num_betas: int = 10,
        num_body_joints: int = _NUM_BODY_JOINTS,
        num_hand_joints: int = _NUM_HAND_JOINTS,
        **overrides,
    ) -> SMPLHParams:
        defaults = dict(
            betas=jnp.zeros((batch_size, num_betas)),
            body_pose=jnp.zeros((batch_size, num_body_joints * 3)),
            global_orient=jnp.zeros((batch_size, 3)),
            transl=jnp.zeros((batch_size, 3)),
            left_hand_pose=jnp.zeros((batch_size, num_hand_joints * 3)),
            right_hand_pose=jnp.zeros((batch_size, num_hand_joints * 3)),
        )
        defaults.update(overrides)
        return SMPLHParams(**defaults)


SMPLH_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]
