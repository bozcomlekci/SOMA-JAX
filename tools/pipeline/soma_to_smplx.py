"""SOMA poses -> SMPL/SMPL-X poses, for visualizing SOMA motions in the
SMPL family.

The body articulation is just per-joint world rotations on shared anatomy: SOMA
and SMPL/SMPL-X joint sub-skeletons are subsets of the same human kinematic tree
(Hips/pelvis, Spine, Chest, Shoulders, Arms, ...). The clean transfer is name-
based — match each SMPL/SMPL-X joint to its SOMA counterpart, read SOMA's posed
*world rotation* (from the SOMA-X FK we already run), then re-express it as
SMPL/SMPL-X parent-relative:

    R_local_smplx[j] = R_world_soma[map_to_soma[parent(j)]].T @ R_world_soma[map_to_soma[j]]

This avoids the mesh-based inverse-LBS step entirely (which under-recovers on
small limb regions). It mirrors SOMA-X's `smpl2soma` flow in spirit: SOMA-X
uses PoseInversion only because SMPL and SOMA topologies differ; for transfering
*rotations* between shared anatomical joints, a direct mapping is exact.
"""
from __future__ import annotations
import numpy as np


# SMPL-X joint index → SOMA joint name (body subset; 0-21 are pelvis + 21 body joints).
# SMPL-X joint order from smpl_jax / SMPL-X spec:
#   0 pelvis, 1-2 hips, 3 spine1, 4-5 knees, 6 spine2, 7-8 ankles, 9 spine3,
#   10-11 feet (toes), 12 neck, 13-14 collars, 15 head, 16-17 shoulders,
#   18-19 elbows, 20-21 wrists, then 22 jaw, 23-24 eyes, 25-54 hands.
_SMPLX_TO_SOMA = {
    # Body (0-21)
    0:  "Hips",
    1:  "LeftLeg",     2:  "RightLeg",
    3:  "Spine1",
    4:  "LeftShin",    5:  "RightShin",
    6:  "Spine2",
    7:  "LeftFoot",    8:  "RightFoot",
    9:  "Chest",
    10: "LeftToeBase", 11: "RightToeBase",
    12: "Neck1",
    13: "LeftShoulder", 14: "RightShoulder",
    15: "Head",
    16: "LeftArm",     17: "RightArm",
    18: "LeftForeArm", 19: "RightForeArm",
    20: "LeftHand",    21: "RightHand",
    # Face (22-24)
    22: "Jaw",
    23: "LeftEye",     24: "RightEye",
    # Left hand (25-39): SMPL-X uses MANO's 3-joint chain per finger; we map
    # to SOMA's first three finger segments (the "End" tip joint is skinning-
    # only with no body-pose channel).
    25: "LeftHandIndex1",  26: "LeftHandIndex2",  27: "LeftHandIndex3",
    28: "LeftHandMiddle1", 29: "LeftHandMiddle2", 30: "LeftHandMiddle3",
    31: "LeftHandPinky1",  32: "LeftHandPinky2",  33: "LeftHandPinky3",
    34: "LeftHandRing1",   35: "LeftHandRing2",   36: "LeftHandRing3",
    37: "LeftHandThumb1",  38: "LeftHandThumb2",  39: "LeftHandThumb3",
    # Right hand (40-54)
    40: "RightHandIndex1",  41: "RightHandIndex2",  42: "RightHandIndex3",
    43: "RightHandMiddle1", 44: "RightHandMiddle2", 45: "RightHandMiddle3",
    46: "RightHandPinky1",  47: "RightHandPinky2",  48: "RightHandPinky3",
    49: "RightHandRing1",   50: "RightHandRing2",   51: "RightHandRing3",
    52: "RightHandThumb1",  53: "RightHandThumb2",  54: "RightHandThumb3",
}


def soma_world_to_smplx_local(R_world_soma_seq, soma_joint_names, smplx_parents,
                              soma_orient_world=None, n_smplx_joints=55):
    """Convert per-frame SOMA world rotations (from SOMA-X FK, T_world[:,:3,:3])
    into SMPL-X parent-relative local rotations via name mapping.

    The math: SOMA bakes per-joint bind orientations `orient[j]` into world
    rotations (W_soma[j] = R_pose_world[j] @ orient[j]); SMPL-X has identity
    bind orientations (W_smplx[j] = R_pose_world[j]). So the transfer is:

        R_pose_world_soma[j] = W_soma[j] @ orient_soma[j].T          (strip bind)
        W_smplx[k]           = R_pose_world_soma[map[k]]             (rename)
        R_local_smplx[k]     = W_smplx[parent].T @ W_smplx[k]        (re-localize)

    Without the `orient_soma.T` strip, SOMA's bind orientation gets re-applied
    on top of SMPL-X's already-identity bind, producing flipped/twisted bodies.

    R_world_soma_seq: (T, J_soma, 3, 3) world rotations of every SOMA joint per frame.
    soma_joint_names: list of SOMA joint names (length J_soma).
    smplx_parents:    (J_smplx,) SMPL-X parent indices (root: -1 or self).
    soma_orient_world: (J_soma, 3, 3) SOMA bind world orientations
        (= t_pose_world[:, :3, :3]). If None, treat SOMA bind as identity
        (incorrect but kept as a fallback).
    Returns:          (T, J_smplx, 3, 3) parent-relative local rotations.
    """
    Tn = R_world_soma_seq.shape[0]
    name_to_idx = {n: i for i, n in enumerate(soma_joint_names)}

    # Build the SMPL-X-joint → SOMA-joint index mapping. If a SMPL-X joint has
    # no matching SOMA name (older rig revisions, or face joints absent from
    # the BVH skeleton), fall back to the nearest body parent so the chain
    # rotation stays sensible (wrist for fingers, head for eyes/jaw).
    smplx_to_soma = np.full(n_smplx_joints, -1, dtype=np.int64)
    for j_smplx in range(n_smplx_joints):
        nm = _SMPLX_TO_SOMA.get(j_smplx)
        if nm and nm in name_to_idx:
            smplx_to_soma[j_smplx] = name_to_idx[nm]
    fallback = {**{j: "LeftHand"  for j in range(25, 40)},
                **{j: "RightHand" for j in range(40, 55)},
                22: "Head", 23: "Head", 24: "Head"}
    for j, anchor in fallback.items():
        if smplx_to_soma[j] < 0 and anchor in name_to_idx:
            smplx_to_soma[j] = name_to_idx[anchor]

    # Strip SOMA's per-joint bind orientation: R_pose_world[j] = W[j] @ orient[j].T
    R_pose_world_soma = R_world_soma_seq.copy()
    if soma_orient_world is not None:
        soma_orient_world = np.asarray(soma_orient_world, dtype=np.float32)
        orient_T = np.swapaxes(soma_orient_world, -2, -1)               # (J_soma, 3, 3)
        R_pose_world_soma = R_world_soma_seq @ orient_T[None]            # (T, J_soma, 3, 3)

    # SMPL-X bind orientation is identity per joint, so its world pose-rotation
    # equals SOMA's pose-only world rotation at the mapped joint.
    R_world_smplx = R_pose_world_soma[:, smplx_to_soma]                  # (T, J_smplx, 3, 3)
    smplx_parents = np.asarray(smplx_parents).astype(int)

    R_local = np.empty_like(R_world_smplx)
    for t in range(Tn):
        for j in range(n_smplx_joints):
            p = int(smplx_parents[j])
            if p < 0 or p == j:
                R_local[t, j] = R_world_smplx[t, j]
            else:
                R_local[t, j] = R_world_smplx[t, p].T @ R_world_smplx[t, j]
    return R_local


def smplx_poses_from_soma_world(R_world_soma_seq, soma_joint_names, smplx_parents,
                                soma_orient_world=None):
    """Convenience: returns axis-angle pose dict directly from SOMA world rotations.

    soma_orient_world: (J_soma, 3, 3) SOMA bind world orientations (rig
        t_pose_world[:, :3, :3]). Required for correct global orientation —
        without it, SOMA's bind rotation gets baked into SMPL-X's pelvis and
        the body comes out flipped.

    Output dict keys match SMPLXParams field structure:
      global_orient (T,3), body_pose (T,63), jaw_pose (T,3), leye_pose, reye_pose,
      left_hand_pose (T,45), right_hand_pose (T,45).
    """
    import jax, jax.numpy as jnp
    from soma_jax.geometry.transforms import rotmat_to_axis_angle

    R_local = soma_world_to_smplx_local(
        R_world_soma_seq, soma_joint_names, smplx_parents,
        soma_orient_world=soma_orient_world)
    aa = np.asarray(jax.vmap(jax.vmap(rotmat_to_axis_angle))(jnp.asarray(R_local)))   # (T, J, 3)
    T = aa.shape[0]
    return {
        "global_orient": aa[:, 0],
        "body_pose": aa[:, 1:22].reshape(T, 63),
        "jaw_pose": aa[:, 22],
        "leye_pose": aa[:, 23],
        "reye_pose": aa[:, 24],
        "left_hand_pose": aa[:, 25:40].reshape(T, 45),
        "right_hand_pose": aa[:, 40:55].reshape(T, 45),
    }


