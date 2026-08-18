"""SMPL-X: Expressive Human Body Model in JAX.

SMPL-X extends SMPL with face joints (jaw, eyes), hand joints (45 each),
and expression blend shapes for facial expressions.

Joint layout (55 joints):
    0:        pelvis
    1-21:     body (21 joints)
    22:       jaw
    23-24:    left_eye, right_eye
    25-39:    left hand (15 joints)
    40-54:    right hand (15 joints)
Total parameters: 10475 vertices, 55 joints.

References:
    Pavlakos et al. (2019). Expressive Body Capture: 3D Hands, Face, and Body
    from a Single Image. CVPR.

Upstream: ``soma/smpl/__init__.py (SMPLXLayer)``
    Corresponding core math to upstream's SMPL-family forward
    (`soma/smpl/__init__.py`, `SMPLXLayer`), with a different API and feature
    surface. Not parity-tested against it.
"""
from __future__ import annotations
from typing import NamedTuple, Optional
import numpy as np
import jax
import jax.numpy as jnp

from ._base import BaseBodyModel, BodyModelOutput
from ..geometry.transforms import axis_angle_to_rotmat


_NUM_BODY_JOINTS = 21
_NUM_FACE_JOINTS = 3   # jaw, left_eye, right_eye
_NUM_HAND_JOINTS = 15


class SMPLXParams(NamedTuple):
    """SMPL-X pose, shape, and expression parameters."""
    betas: jnp.ndarray              # (B, num_betas) shape
    body_pose: jnp.ndarray          # (B, 21*3) axis-angle body
    global_orient: jnp.ndarray      # (B, 3) root
    transl: jnp.ndarray             # (B, 3)
    expression: jnp.ndarray         # (B, num_expression_coeffs)
    jaw_pose: jnp.ndarray           # (B, 3)
    leye_pose: jnp.ndarray          # (B, 3)
    reye_pose: jnp.ndarray          # (B, 3)
    left_hand_pose: jnp.ndarray     # (B, 15*3) full or (B, num_pca) PCA
    right_hand_pose: jnp.ndarray    # (B, 15*3)


class SMPLXModel(BaseBodyModel):
    """SMPL-X expressive body model.

    Hand-pose convention (``flat_hand_mean``):
        SMPL-X model files ship a MANO mean hand pose (``hands_meanl`` /
        ``hands_meanr``).  With ``flat_hand_mean=True`` (the default)
        ``left_hand_pose`` / ``right_hand_pose`` are absolute axis-angle poses
        and zeros give a flat, open hand — the SOMA ``pose_hand`` convention,
        whose field is fed in directly, and the convention
        ``smpl_jax.SMPLXModel`` uses.  With
        ``flat_hand_mean=False`` the model's mean hand pose is added on top
        (the PyTorch ``smplx`` package default).  The two differ by up to
        ~7.6 cm in vertex position, so the flag must match whatever produced
        the pose parameters.
    """

    def __init__(
        self,
        v_template: np.ndarray,
        shapedirs: np.ndarray,
        exprdirs: np.ndarray,
        posedirs: np.ndarray,
        J_regressor: np.ndarray,
        parents: np.ndarray,
        weights: np.ndarray,
        faces: np.ndarray,
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
        flat_hand_mean: bool = True,
        hand_pose_mean_l: Optional[np.ndarray] = None,
        hand_pose_mean_r: Optional[np.ndarray] = None,
    ):
        super().__init__(
            v_template, shapedirs, posedirs, J_regressor,
            parents, weights, faces, num_betas,
        )
        self.num_expression_coeffs = num_expression_coeffs
        if exprdirs is None:
            raise ValueError("SMPL-X requires expression blend shapes (exprdirs).")
        self.exprdirs = jnp.array(
            exprdirs[..., :num_expression_coeffs], dtype=jnp.float32
        )
        # Hand-pose offset. Zero under the flat-hand convention (default) or
        # when the model file carries no mean; see the class docstring.
        self.flat_hand_mean = bool(flat_hand_mean)
        zero = np.zeros(_NUM_HAND_JOINTS * 3, dtype=np.float32)
        hand_l = zero if (flat_hand_mean or hand_pose_mean_l is None) else np.asarray(hand_pose_mean_l).reshape(-1)
        hand_r = zero if (flat_hand_mean or hand_pose_mean_r is None) else np.asarray(hand_pose_mean_r).reshape(-1)
        self.hand_pose_mean_l = jnp.array(hand_l, dtype=jnp.float32)
        self.hand_pose_mean_r = jnp.array(hand_r, dtype=jnp.float32)

    @classmethod
    def load(
        cls,
        path: str,
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
        flat_hand_mean: bool = True,
    ) -> "SMPLXModel":
        """Load an SMPL-X model from .pkl or .npz.

        Args:
            path: path to the SMPL-X model file.
            num_betas: number of shape components to keep.
            num_expression_coeffs: number of expression components to keep.
            flat_hand_mean: hand-pose convention; see the class docstring.
        """
        from .model_io import load_smpl_data
        data = load_smpl_data(path)
        if data.get("exprdirs") is None:
            raise ValueError(
                f"{path} has no expression blend shapes; use SMPLModel for vanilla SMPL."
            )
        return cls(
            v_template=data["v_template"],
            shapedirs=data["shapedirs"],
            exprdirs=data["exprdirs"],
            posedirs=data["posedirs"],
            J_regressor=data["J_regressor"],
            parents=data["parents"],
            weights=data["weights"],
            faces=data["faces"],
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
            flat_hand_mean=flat_hand_mean,
            hand_pose_mean_l=data.get("hands_meanl"),
            hand_pose_mean_r=data.get("hands_meanr"),
        )

    def _vertex_blend_shapes(self, params: SMPLXParams) -> jnp.ndarray:
        """SMPL-X: shape + expression blend shapes."""
        v = self.v_template[None] + jnp.einsum(
            "vcp,bp->bvc", self.shapedirs, params.betas
        )
        v = v + jnp.einsum("vcp,bp->bvc", self.exprdirs, params.expression)
        return v

    def _build_rotmats(self, params: SMPLXParams, B: int) -> jnp.ndarray:
        """Build (B, J=55, 3, 3) joint rotations from full SMPL-X pose set."""
        def aa_block(aa_flat: jnp.ndarray, n: int) -> jnp.ndarray:
            return jax.vmap(axis_angle_to_rotmat)(aa_flat.reshape(B * n, 3)).reshape(
                B, n, 3, 3
            )

        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]
        R_body = aa_block(params.body_pose, _NUM_BODY_JOINTS)
        R_jaw = jax.vmap(axis_angle_to_rotmat)(params.jaw_pose)[:, None]
        R_leye = jax.vmap(axis_angle_to_rotmat)(params.leye_pose)[:, None]
        R_reye = jax.vmap(axis_angle_to_rotmat)(params.reye_pose)[:, None]
        R_face = jnp.concatenate([R_jaw, R_leye, R_reye], axis=1)

        # Hand poses (assumed full axis-angle per joint, not PCA)
        R_lhand = aa_block(
            params.left_hand_pose + self.hand_pose_mean_l[None], _NUM_HAND_JOINTS,
        )
        R_rhand = aa_block(
            params.right_hand_pose + self.hand_pose_mean_r[None], _NUM_HAND_JOINTS,
        )

        R_all = jnp.concatenate(
            [R_root, R_body, R_face, R_lhand, R_rhand], axis=1
        )

        # Pad/truncate to match num_joints
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
        num_expression_coeffs: int = 10,
        num_body_joints: int = _NUM_BODY_JOINTS,
        num_hand_joints: int = _NUM_HAND_JOINTS,
        **overrides,
    ) -> SMPLXParams:
        """Create SMPLXParams with zero-defaults for missing fields."""
        defaults = dict(
            betas=jnp.zeros((batch_size, num_betas)),
            body_pose=jnp.zeros((batch_size, num_body_joints * 3)),
            global_orient=jnp.zeros((batch_size, 3)),
            transl=jnp.zeros((batch_size, 3)),
            expression=jnp.zeros((batch_size, num_expression_coeffs)),
            jaw_pose=jnp.zeros((batch_size, 3)),
            leye_pose=jnp.zeros((batch_size, 3)),
            reye_pose=jnp.zeros((batch_size, 3)),
            left_hand_pose=jnp.zeros((batch_size, num_hand_joints * 3)),
            right_hand_pose=jnp.zeros((batch_size, num_hand_joints * 3)),
        )
        defaults.update(overrides)
        return SMPLXParams(**defaults)


SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
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
