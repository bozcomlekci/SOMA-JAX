"""Geometry utilities for SOMA-JAX.

Upstream: ``soma/geometry/__init__.py``
    Re-export shim; each submodule names its own upstream counterpart.
"""
from .transforms import (
    axis_angle_to_rotmat,
    rotmat_to_axis_angle,
    euler_xyz_to_rotmat,
    rotmat_to_euler_xyz,
    quaternion_xyzw_to_rotmat,
    rotmat_to_6d,
    rotation_6d_to_rotmat,
    safe_normalize,
    kabsch,
    newton_schulz,
    se3_from_rt,
    se3_inverse,
    rodrigues_rotation,
    compute_covariance,
    align_vectors,
    rotation_from_covariance,
    regularize_covariance_with_reference,
    rotation_matrices_are_valid,
    quaternion_half_angle_xyzw,
)
from .lbs import (
    forward_kinematics,
    lbs_transforms,
    lbs,
    lbs_sparse,
    compute_skeleton_levels,
    fk_levelorder,
)
from .barycentric_interp import (
    compute_barycentric_coords,
    barycentric_interpolate,
)
from .laplacian import (
    laplacian_solve,
)
from .skeleton_transfer import (
    fit_joint_positions,
    PoseMirror,
    SkeletonTransfer,
)
from .interpolate import RadialBasisFunction
from .rig_utils import (
    get_joint_children_ids,
    get_joint_descendents,
    get_joint_subtree,
    get_body_part_vertex_ids,
    joint_world_to_local,
    joint_local_to_world,
    precompute_joint_orient,
    infer_joint_orient_from_rest,
    PoseMirrorSOMA,
    PoseMirrorMHR,
    apply_joint_orient_local,
    remove_joint_orient_local,
    compute_bone_lengths,
)
from .batched_skinning import (
    BatchedSkinning,
    pose_from_bind,
    topk_skinning,
)
from .chamfer import (
    chamfer_distance,
    chamfer_distance_batched,
    nearest_neighbor_indices,
)

__all__ = [
    # Transforms
    "axis_angle_to_rotmat",
    "rotmat_to_axis_angle",
    "euler_xyz_to_rotmat",
    "rotmat_to_euler_xyz",
    "quaternion_xyzw_to_rotmat",
    "rotmat_to_6d",
    "rotation_6d_to_rotmat",
    "safe_normalize",
    "kabsch",
    "newton_schulz",
    "se3_from_rt",
    "se3_inverse",
    "rodrigues_rotation",
    "compute_covariance",
    "align_vectors",
    "rotation_from_covariance",
    "regularize_covariance_with_reference",
    "rotation_matrices_are_valid",
    "quaternion_half_angle_xyzw",
    # LBS / FK
    "forward_kinematics",
    "lbs_transforms",
    "lbs",
    "lbs_sparse",
    "compute_skeleton_levels",
    "fk_levelorder",
    # Topology transfer
    "compute_barycentric_coords",
    "barycentric_interpolate",
    "laplacian_solve",
    # Skeleton transfer
    "fit_joint_positions",
    "PoseMirror",
    "SkeletonTransfer",
    "RadialBasisFunction",
    # Rig utilities
    "get_joint_children_ids",
    "get_joint_descendents",
    "get_joint_subtree",
    "get_body_part_vertex_ids",
    "joint_world_to_local",
    "joint_local_to_world",
    "precompute_joint_orient",
    "infer_joint_orient_from_rest",
    "PoseMirrorSOMA",
    "PoseMirrorMHR",
    "apply_joint_orient_local",
    "remove_joint_orient_local",
    "compute_bone_lengths",
    # Batched skinning
    "BatchedSkinning",
    "pose_from_bind",
    "topk_skinning",
    # Chamfer
    "chamfer_distance",
    "chamfer_distance_batched",
    "nearest_neighbor_indices",
]
