"""Abstract base class for all parametric body models in SOMA-JAX.

The forward pass is the standard 7-step LBS pipeline:
    1. Shape blend shapes: V_template + sum(beta_k * shapedirs_k)
    2. (optional) Expression blend shapes: V + sum(expr_k * exprdirs_k)
    3. Joint regression: J = J_regressor @ V_shaped
    4. Build rotation matrices from pose parameters
    5. Forward kinematics: per-joint world transforms
    6. Pose blend shapes: V += sum((R_j - I) * posedirs_j)
    7. LBS: V_posed = sum_j(weights_j @ T_j @ V_shaped)

All subclasses share this pipeline but differ in:
    - _vertex_blend_shapes (may add expression on top of shape)
    - _build_rotmats (different joint hierarchies / pose parameterizations)

Upstream: none — SOMA-JAX-only.
    Standalone parametric body models; upstream only loads SMPL-family assets, it does not implement their forward pass.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from ..geometry.transforms import axis_angle_to_rotmat
from ..geometry.lbs import forward_kinematics, lbs_transforms, lbs


def shape_blend_shapes(
    v_template: jnp.ndarray,
    shapedirs: jnp.ndarray,
    betas: jnp.ndarray,
) -> jnp.ndarray:
    """Apply shape blend shapes: V = V_template + sum_k(beta_k * shapedirs_k).

    Args:
        v_template: (V, 3) rest template.
        shapedirs: (V, 3, K) PCA shape basis.
        betas: (B, K) shape coefficients.

    Returns:
        (B, V, 3) shaped vertices.
    """
    return v_template[None] + jnp.einsum("vcp,bp->bvc", shapedirs, betas)


def pose_blend_shapes(
    rotmats: jnp.ndarray,
    posedirs: jnp.ndarray,
) -> jnp.ndarray:
    """Apply pose blend shapes: corrective deformations from joint rotations.

    For each non-root joint, the deviation of its rotation from identity
    drives a linear corrective offset for the vertices.

    Args:
        rotmats: (B, J, 3, 3) joint rotation matrices.
        posedirs: (V*3, P) flat pose-corrective basis. P = (J-1) * 9 typically.

    Returns:
        (B, V, 3) pose-corrective vertex offsets.
    """
    B, J = rotmats.shape[:2]
    P = posedirs.shape[1]
    V3 = posedirs.shape[0]
    V = V3 // 3
    I = jnp.eye(3, dtype=rotmats.dtype)
    # Skip root (joint 0); flatten each (3x3-I) to 9 features
    pose_feat = (rotmats[:, 1:] - I).reshape(B, -1)[:, :P]   # (B, P)
    return (pose_feat @ posedirs.T).reshape(B, V, 3)


class BodyModelOutput(NamedTuple):
    vertices: jnp.ndarray    # (B, V, 3) posed vertices
    joints: jnp.ndarray      # (B, J, 3) posed joint positions
    v_shaped: jnp.ndarray    # (B, V, 3) rest-shaped vertices (no pose)


class BaseBodyModel(ABC):
    """Abstract base class for parametric body models.

    Provides the common 7-step LBS forward pass. Subclasses customize:
      - _vertex_blend_shapes: apply shape (+ expression / blend shapes)
      - _build_rotmats: assemble (B, J, 3, 3) rotation matrices from
                        pose parameters with model-specific structure
    """

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
    ):
        self.num_joints = int(weights.shape[1])
        self.num_betas = num_betas
        self._parents_np = np.asarray(parents, dtype=np.int32)
        self._faces_np = np.asarray(faces, dtype=np.int32)

        self.v_template = jnp.array(v_template, dtype=jnp.float32)
        self.shapedirs = jnp.array(shapedirs[..., :num_betas], dtype=jnp.float32)
        self.posedirs = (
            jnp.array(posedirs, dtype=jnp.float32) if posedirs is not None else None
        )
        self.J_regressor = jnp.array(
            J_regressor[: self.num_joints], dtype=jnp.float32
        )
        self.parents = jnp.array(self._parents_np)
        self.weights = jnp.array(weights, dtype=jnp.float32)
        self.faces = jnp.array(self._faces_np)

    @property
    def num_vertices(self) -> int:
        return int(self.v_template.shape[0])

    # ----- Subclass-specific -------------------------------------------------
    @abstractmethod
    def _vertex_blend_shapes(self, params: Any) -> jnp.ndarray:
        """Apply shape (and possibly expression) blend shapes."""

    @abstractmethod
    def _build_rotmats(self, params: Any, B: int) -> jnp.ndarray:
        """Build (B, J, 3, 3) joint rotation matrices from pose params."""

    # ----- Main forward pass -------------------------------------------------
    def forward(self, params: Any) -> BodyModelOutput:
        """Standard 7-step LBS forward pass.

        Args:
            params: model-specific NamedTuple of pose + shape parameters.

        Returns:
            BodyModelOutput (vertices, joints, v_shaped).
        """
        unbatched = params.betas.ndim == 1
        if unbatched:
            params = jax.tree_util.tree_map(lambda x: x[None] if hasattr(x, "ndim") else x, params)

        B = params.betas.shape[0]

        # 1. Shape blend shapes (and expression for SMPL-X)
        v_shaped = self._vertex_blend_shapes(params)                # (B, V, 3)

        # 2. Joint regression in shaped space
        joints = jnp.einsum("jv,bvd->bjd", self.J_regressor, v_shaped)  # (B, J, 3)

        # 3. Build rotation matrices
        rotmats = self._build_rotmats(params, B)                    # (B, J, 3, 3)

        # 4. Forward kinematics (vmap over batch)
        parents_np = self._parents_np
        G = jax.vmap(
            lambda R, j: forward_kinematics(R, j, parents_np)
        )(rotmats, joints)                                          # (B, J, 4, 4)

        # 5. Pose blend shapes — kept separate from v_shaped so the returned
        #    v_shaped stays the pose-independent rest shape it is documented as.
        if self.posedirs is not None:
            pose_corr = pose_blend_shapes(rotmats, self.posedirs)    # (B, V, 3)
        else:
            pose_corr = jnp.zeros_like(v_shaped)

        # 6. LBS
        bone_T = lbs_transforms(G, joints)                          # (B, J, 3, 4)
        v_posed = lbs(v_shaped, pose_corr, bone_T, self.weights)    # (B, V, 3)

        # 7. Apply translation
        v_posed = v_posed + params.transl[:, None, :]
        posed_joints = G[:, :, :3, 3] + params.transl[:, None, :]

        out = BodyModelOutput(vertices=v_posed, joints=posed_joints, v_shaped=v_shaped)
        if unbatched:
            out = BodyModelOutput(
                vertices=out.vertices[0],
                joints=out.joints[0],
                v_shaped=out.v_shaped[0],
            )
        return out

    def __call__(self, params: Any) -> BodyModelOutput:
        return self.forward(params)
