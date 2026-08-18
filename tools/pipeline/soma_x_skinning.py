"""SOMA-X BatchedSkinning ported to JAX/numpy — line-by-line faithful.

soma_jax's simplified LBS (position-only FK) does not handle the SOMA rig's
full joint bind orientations, so external motion sources that respect those
orientations (BVH clips, SOMA-X poses with `absolute_pose=False`)
produce twisted limbs. This module mirrors the SOMA-X PyTorch
``BatchedSkinning.pose()`` path exactly:

    R_oriented = orient_parent_T @ R_in @ orient                 # joint-orient
    local_t    = bind_local_translations, hips slot replaced     # root motion
    T_local    = SE3(R_oriented, local_t)
    T_world    = level-order FK of T_local
    T_bone     = T_world @ inverse_bind_transform                # per joint
    verts      = Σ_j W[v,j] · (T_bone_j ⊙ bind_shape[v])         # LBS

Reference: third_party/SOMA-X/soma/geometry/{batched_skinning,rig_utils,lbs,transforms}.py
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# SE3 primitives (mirror soma/geometry/transforms.py)
# ---------------------------------------------------------------------------
def SE3_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(..., 3, 3) + (..., 3) -> (..., 4, 4)."""
    out = np.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype)
    out[..., :3, :3] = R
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def SE3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert SE(3) transforms."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_T = np.swapaxes(R, -2, -1)
    t_new = -(R_T @ t)
    return SE3_from_Rt(R_T, t_new[..., 0])


# ---------------------------------------------------------------------------
# Skeleton helpers (mirror soma/geometry/rig_utils.py)
# ---------------------------------------------------------------------------
def joint_world_to_local(world_T: np.ndarray, parents) -> tuple[np.ndarray, np.ndarray]:
    """world transforms -> local transforms (parent-relative). Returns (local, inv_world)."""
    inv_world = SE3_inverse(world_T)
    parents = np.asarray(parents).astype(int)
    parents = np.where(parents < 0, 0, parents)
    local_T = inv_world[parents] @ world_T
    return local_T, inv_world


def compute_skeleton_levels(parents) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group joints by BFS depth. Returns [(joint_ids, parent_ids)] per level."""
    parents = np.asarray(parents).astype(int)
    J = len(parents)
    depth = [0] * J
    for j in range(1, J):
        p = parents[j]
        if p < 0 or p == j:
            depth[j] = 0
        else:
            depth[j] = depth[p] + 1
    by_level: dict[int, list[int]] = {}
    for j, d in enumerate(depth):
        by_level.setdefault(d, []).append(j)
    levels = []
    for d in sorted(by_level):
        joints = np.array(by_level[d], dtype=np.int64)
        pars = np.array([parents[j] if (parents[j] >= 0 and parents[j] != j) else j for j in joints],
                        dtype=np.int64)
        levels.append((joints, pars))
    return levels


def joint_local_to_world_levelorder(local_T: np.ndarray, levels) -> np.ndarray:
    """Level-order FK: world[j] = world[parent] @ local[j]. local_T: (..., J, 4, 4)."""
    world = local_T.copy()
    for joint_ids, parent_ids in levels[1:]:
        world[..., joint_ids, :, :] = world[..., parent_ids, :, :] @ local_T[..., joint_ids, :, :]
    return world


def precompute_joint_orient(joint_orient_world: np.ndarray, parents) -> tuple[np.ndarray, np.ndarray]:
    """SOMA-X's exact rule (rig_utils.py:151).

    orient[j] = world bind rotation of joint j; orient_parent_T[j] = orient[parent].T.
    """
    orient = joint_orient_world[..., :3, :3].astype(np.float32)
    parents = np.asarray(parents).astype(int)
    orient_parent_T = np.swapaxes(orient[np.where(parents < 0, 0, parents)], -2, -1)
    return orient, orient_parent_T


def apply_joint_orient_local(local_R: np.ndarray, orient: np.ndarray, orient_parent_T: np.ndarray) -> np.ndarray:
    """R_out[j] = orient_parent_T[j] @ R_in[j] @ orient[j]   (rig_utils.py:167)."""
    return orient_parent_T @ local_R @ orient


# ---------------------------------------------------------------------------
# LBS (mirror soma/geometry/lbs.py)
# ---------------------------------------------------------------------------
def lbs_dense(bind_shape: np.ndarray, skinning_weights: np.ndarray, bone_T: np.ndarray) -> np.ndarray:
    """Dense LBS. bind_shape (V,3) or (B,V,3); W (V,J); bone_T (B,J,4,4) or (J,4,4)."""
    R = bone_T[..., :3, :3]                   # (..., J, 3, 3)
    t = bone_T[..., :3, 3]                    # (..., J, 3)
    tv = np.einsum("...jmk,...vk->...jvm", R, bind_shape) + t[..., None, :]
    return np.einsum("vj,...jvm->...vm", skinning_weights, tv)


# ---------------------------------------------------------------------------
# Faithful SOMA-X skinning class
# ---------------------------------------------------------------------------
class SomaXSkinning:
    """Drop-in port of soma/geometry/batched_skinning.py: BatchedSkinning (dense).

    Args:
        parents:                 (J,) parent indices
        skinning_weights:        (V, J)
        bind_world_transforms:   (J, 4, 4)  in METERS (rig translations cm -> m if needed)
        bind_shape:              (V, 3)  in METERS
        joint_orient_world:      (J, 4, 4) or (J, 3, 3) — typically the rig's `t_pose_world`
        hips_joint:              joint index whose translation gets replaced by `hips_translations`
                                 (1 = "Hips" in SOMA's 78-joint skeleton)
    """

    def __init__(self, parents, skinning_weights, bind_world_transforms,
                 bind_shape, joint_orient_world=None, hips_joint=1):
        self.parents = np.asarray(parents).astype(int)
        self.J = len(self.parents)
        self.weights = np.asarray(skinning_weights, dtype=np.float32)
        self.bind_world = np.asarray(bind_world_transforms, dtype=np.float32)
        bind_local, inv_world = joint_world_to_local(self.bind_world, self.parents)
        self.bind_local_t = bind_local[..., :3, 3].astype(np.float32)
        self.inverse_bind = inv_world.astype(np.float32)
        self.bind_shape = np.asarray(bind_shape, dtype=np.float32)
        self.levels = compute_skeleton_levels(self.parents)
        self.hips = int(hips_joint)
        self.orient = self.orient_parent_T = None
        if joint_orient_world is not None:
            self.orient, self.orient_parent_T = precompute_joint_orient(
                np.asarray(joint_orient_world, dtype=np.float32), self.parents)

    def rebind(self, bind_world_transforms, bind_shape):
        """Swap the bind pose for a different identity / scale."""
        self.bind_world = np.asarray(bind_world_transforms, dtype=np.float32)
        bind_local, inv_world = joint_world_to_local(self.bind_world, self.parents)
        self.bind_local_t = bind_local[..., :3, 3].astype(np.float32)
        self.inverse_bind = inv_world.astype(np.float32)
        self.bind_shape = np.asarray(bind_shape, dtype=np.float32)

    def pose(self, local_R: np.ndarray, hips_translation: np.ndarray = None,
             absolute_pose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """SOMA-X BatchedSkinning.pose, unbatched.

        local_R: (J, 3, 3) per-joint rotations (relative-to-T-pose if absolute_pose=False).
        hips_translation: (3,) world position to place the Hips joint at (default: bind position).
        Returns (posed_verts (V,3), T_world (J,4,4)).
        """
        R = local_R.astype(np.float32)
        if self.orient is not None and not absolute_pose:
            R = apply_joint_orient_local(R, self.orient, self.orient_parent_T)
        local_t = self.bind_local_t.copy()
        if hips_translation is not None:
            local_t[self.hips] = np.asarray(hips_translation, dtype=np.float32)
        T_local = SE3_from_Rt(R, local_t)
        T_world = joint_local_to_world_levelorder(T_local, self.levels)
        bone_T = T_world @ self.inverse_bind
        verts = lbs_dense(self.bind_shape, self.weights, bone_T)
        return verts, T_world
