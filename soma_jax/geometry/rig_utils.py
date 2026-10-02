"""Rig and joint topology utilities for SOMA-JAX.

Provides helpers for skeleton manipulation:
  - World↔local joint transform conversion
  - Joint orient precomputation and application
  - Joint hierarchy traversal (children, descendants)
  - Body-part vertex grouping

Upstream: ``soma/geometry/rig_utils.py``
    Faithful port of every public helper under upstream's names
    (``PoseMirror_SOMA`` / ``PoseMirror_MHR`` are aliases of
    ``PoseMirrorSOMA`` / ``PoseMirrorMHR``). ``apply_joint_orient_local`` and
    ``remove_joint_orient_local`` also accept parent indices in place of
    upstream's precomputed ``orient_parent_T``. SOMA-JAX extras:
    ``get_joint_subtree``, ``group_body_part_vertex_ids``,
    ``infer_joint_orient_from_rest``, ``compute_bone_lengths``; the
    rotations-only ``PoseMirror`` lives in ``skeleton_transfer.py``.
"""
from __future__ import annotations
import logging
import numpy as np
import jax
import jax.numpy as jnp
from .transforms import se3_inverse

logger = logging.getLogger(__name__)


def get_joint_children_ids(joint_parent_ids) -> list[list[int]]:
    """Each joint's immediate children, as upstream's list of lists.

    Upstream ``rig_utils.get_joint_children_ids`` scans joints ``1..J-1``, so
    the self-parented root (SOMA's joint 0) is never its own child. A negative
    parent (an SMPL-style root) is skipped here, where upstream's Python
    indexing would file the joint under the last joint.

    Args:
        joint_parent_ids: (J,) parent indices.

    Returns:
        ``children[j]`` — the children of joint ``j``.
    """
    parents = [int(p) for p in np.asarray(joint_parent_ids).reshape(-1)]
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for i in range(1, len(parents)):
        if 0 <= parents[i] != i:
            children[parents[i]].append(i)
    return children


def get_joint_descendents(joint_parent_ids, joint_id: int) -> list[int]:
    """All descendants of a joint (excluding it), in upstream's depth-first pre-order.

    Args:
        joint_parent_ids: (J,) parent indices.
        joint_id: the joint whose descendants to list.
    """
    children = get_joint_children_ids(joint_parent_ids)
    result: list[int] = []
    stack = list(reversed(children[joint_id]))
    while stack:
        j = stack.pop()
        result.append(j)
        stack.extend(reversed(children[j]))
    return result


def get_joint_subtree(parents: np.ndarray, root_joint: int) -> list[int]:
    """Return the joint plus all its descendants (joint subtree)."""
    return [root_joint] + get_joint_descendents(parents, root_joint)


def get_body_part_vertex_ids(
    skinning_weights,
    joint_parent_ids,
    root_joint_id: int,
    include_root: bool = True,
    weight_threshold: float = 0.01,
) -> list[int]:
    """Vertex IDs influenced by a body part: a root joint and its descendants.

    Upstream ``rig_utils.get_body_part_vertex_ids``; see :func:`body_part_vertex_ids`.
    """
    return body_part_vertex_ids(skinning_weights, joint_parent_ids, root_joint_id,
                                include_root=include_root, weight_threshold=weight_threshold)


def group_body_part_vertex_ids(
    weights: np.ndarray,
    joint_groups: dict[str, list[int]],
    threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """Group vertices by body part based on skinning weight influence (SOMA-JAX extra).

    Args:
        weights: (V, J) skinning weight matrix.
        joint_groups: dict mapping part name → list of joint indices.
        threshold: vertex assigned to a part if any of its joints has weight >= threshold.

    Returns:
        Dict {part_name: vertex_ids} with vertex indices for each body part.
    """
    result: dict[str, np.ndarray] = {}
    for part_name, joint_ids in joint_groups.items():
        part_weights = weights[:, joint_ids].sum(axis=1)
        vertex_ids = np.where(part_weights >= threshold)[0]
        result[part_name] = vertex_ids.astype(np.int32)
    return result


def body_part_vertex_ids(
    skinning_weights: np.ndarray,
    parents: np.ndarray,
    root_joint_id: int,
    include_root: bool = True,
    weight_threshold: float = 0.01,
) -> list[int]:
    """Vertices influenced by a joint's subtree — SOMA-X's
    ``rig_utils.get_body_part_vertex_ids`` (also available under that name).

    Walks the joint hierarchy from ``root_joint_id`` and unions the influence
    masks of every descendant, which is what the pose-inversion vertex
    weighting expects.

    Args:
        skinning_weights: (V, J) dense skinning weights.
        parents: (J,) parent indices.
        root_joint_id: joint whose subtree defines the body part.
        include_root: include vertices influenced by ``root_joint_id`` itself.
        weight_threshold: minimum weight for a vertex to count as influenced.

    Returns:
        Sorted list of vertex indices.
    """
    W = np.asarray(skinning_weights)
    joints = get_joint_descendents(np.asarray(parents), root_joint_id)
    if include_root:
        joints = [root_joint_id] + joints
    mask = np.zeros(W.shape[0], dtype=bool)
    for j in joints:
        mask |= W[:, j] > weight_threshold
    return np.where(mask)[0].tolist()


def joint_world_to_local(
    joint_world_transforms: jnp.ndarray,
    joint_parent_ids: np.ndarray,
    return_inverse: bool = False,
):
    """Convert global (world) joint transforms to local (parent-relative).

    For each joint j: T_local[j] = T_world[parent[j]]^-1 @ T_world[j]
    For root joints: T_local[j] = T_world[j]

    A joint counts as a root when ``joint_parent_ids[j] < 0`` (SMPL convention) **or**
    ``joint_parent_ids[j] == j`` (SOMA's own rig self-parents joint 0). SOMA-X handles
    the self-parented form explicitly; treating it as an ordinary joint would
    return identity for the root and break the world→local→world round trip.

    Args:
        joint_world_transforms: (..., J, 4, 4) transforms or (..., J, 3, 3)
            rotations, as upstream accepts.
        joint_parent_ids: (J,) parent indices.
        return_inverse: also return every joint's inverse world transform,
            as upstream's ``return_inverse=True``.

    Returns:
        Local transforms, same shape as ``joint_world_transforms`` — and, with
        ``return_inverse``, the inverse world transforms too.
    """
    parents_arr = np.asarray(joint_parent_ids)
    is_root = (parents_arr < 0) | (parents_arr == np.arange(len(parents_arr)))
    safe_parents = np.maximum(parents_arr, 0)

    parent_world = joint_world_transforms[..., safe_parents, :, :]   # (..., J, M, M)
    if joint_world_transforms.shape[-2:] == (3, 3):
        parent_inv = jnp.swapaxes(parent_world, -2, -1)
    elif joint_world_transforms.shape[-2:] == (4, 4):
        parent_inv = se3_inverse(parent_world)
    else:
        raise ValueError(
            "Expected joint_world_transforms to have shape (...,4,4) or (...,3,3); "
            f"got {joint_world_transforms.shape}")
    local = jnp.einsum("...jik,...jkl->...jil", parent_inv, joint_world_transforms)

    # Root joints keep their world transforms
    root_mask = jnp.asarray(is_root, dtype=local.dtype)[:, None, None]  # (J, 1, 1)
    local = jnp.where(root_mask > 0.5, joint_world_transforms, local)
    if return_inverse:
        inverse = (jnp.swapaxes(joint_world_transforms, -2, -1) if joint_world_transforms.shape[-1] == 3
                   else se3_inverse(joint_world_transforms))
        return local, inverse
    return local


def joint_local_to_world(
    joint_local_transforms: jnp.ndarray,
    joint_parent_ids: np.ndarray,
) -> jnp.ndarray:
    """Convert local (parent-relative) joint transforms to world.

    Sequential FK along the kinematic chain.

    Args:
        joint_local_transforms: (..., J, 4, 4) transforms or (..., J, 3, 3)
            rotations — upstream accepts ``(J, M, M)`` or ``(B, J, M, M)``.
        joint_parent_ids: (J,) parent indices; a root is ``< 0`` or its own parent.

    Returns:
        World transforms, same shape as ``joint_local_transforms``.
    """
    joint_local_transforms = jnp.asarray(joint_local_transforms)
    M = joint_local_transforms.shape[-1]
    if joint_local_transforms.shape[-2:] not in ((3, 3), (4, 4)):
        raise ValueError(
            "Expected joint_local_transforms to have shape (...,4,4) or (...,3,3); "
            f"got {joint_local_transforms.shape}")
    parents_arr = jnp.asarray(joint_parent_ids)
    safe_parents = jnp.maximum(parents_arr, 0)
    J = joint_local_transforms.shape[-3]
    local_j = jnp.moveaxis(joint_local_transforms, -3, 0)            # (J, ..., M, M)
    eye = jnp.broadcast_to(jnp.eye(M, dtype=joint_local_transforms.dtype), local_j.shape[1:])
    G = jnp.zeros_like(local_j)

    def step(G, i):
        # Root = parent < 0 (SMPL) or self-parented (SOMA); mirrors
        # joint_world_to_local so the two are exact inverses.
        is_root = (parents_arr[i] < 0) | (parents_arr[i] == i)
        parent_T = jnp.where(is_root, eye, G[safe_parents[i]])
        return G.at[i].set(parent_T @ local_j[i]), None

    G, _ = jax.lax.scan(step, G, jnp.arange(J))
    return jnp.moveaxis(G, 0, -3)


def infer_joint_orient_from_rest(
    rest_joints: jnp.ndarray,
    parents: np.ndarray,
) -> jnp.ndarray:
    """Precompute per-joint orient (T-pose) rotation from rest joint positions.

    The joint orient aligns each joint's local frame to point along the bone
    from parent to child. For joints without children, identity is used.

    Args:
        rest_joints: (J, 3) rest joint positions.
        parents: (J,) parent indices.

    Returns:
        (J, 3, 3) joint orient rotation matrices.
    """
    J = rest_joints.shape[0]
    children = get_joint_children_ids(np.asarray(parents))

    orients = []
    for j in range(J):
        child_ids = children[j]
        if len(child_ids) == 0:
            orients.append(jnp.eye(3))
            continue
        # Use first child for bone direction
        c = child_ids[0]
        bone_dir = rest_joints[c] - rest_joints[j]
        bone_dir = bone_dir / (jnp.linalg.norm(bone_dir) + 1e-8)
        # Build orthonormal frame with Y-axis along bone
        up = jnp.where(jnp.abs(bone_dir[1]) > 0.99,
                       jnp.array([1.0, 0.0, 0.0]),
                       jnp.array([0.0, 1.0, 0.0]))
        x_axis = jnp.cross(up, bone_dir)
        x_axis = x_axis / (jnp.linalg.norm(x_axis) + 1e-8)
        z_axis = jnp.cross(x_axis, bone_dir)
        R = jnp.stack([x_axis, bone_dir, z_axis], axis=-1)
        orients.append(R)

    return jnp.stack(orients, axis=0)


def _orient_pair(orient, orient_parent_T):
    """``(orient, orient_parent_T)`` from upstream's pair or from parent indices."""
    orient = jnp.asarray(orient)[..., :3, :3]
    if np.ndim(orient_parent_T) == 1:
        return precompute_joint_orient(orient, orient_parent_T)
    return orient, jnp.asarray(orient_parent_T)


def apply_joint_orient_local(
    local_rotations: jnp.ndarray,
    orient: jnp.ndarray,
    orient_parent_T,
) -> jnp.ndarray:
    """Apply joint orient as a per-joint local operation (no FK loop).

    Upstream ``rig_utils.apply_joint_orient_local``::

        R_out[j] = orient_parent_T[j] @ R_in[j] @ orient[j]

    equivalent to rotating each world transform by its joint's orient. The
    third argument is upstream's ``orient_parent_T`` from
    :func:`precompute_joint_orient`, or — SOMA-JAX convenience — the (J,)
    parent indices to precompute it from. It is required, as upstream's is.

    Args:
        local_rotations: (..., J, 3, 3) T-pose-relative local rotations.
        orient: (J, 3, 3) per-joint world orientation (``orient`` of the pair).
        orient_parent_T: (J, 3, 3) parents' transposed orients, or (J,) parents.

    Returns:
        (..., J, 3, 3) oriented local rotations.
    """
    orient, orient_parent_T = _orient_pair(orient, orient_parent_T)
    return orient_parent_T @ local_rotations @ orient


def remove_joint_orient_local(
    local_rotations: jnp.ndarray,
    orient: jnp.ndarray,
    orient_parent_T,
) -> jnp.ndarray:
    """Remove joint orient — inverse of :func:`apply_joint_orient_local`.

    Upstream ``rig_utils.remove_joint_orient_local``::

        R_rel[j] = orient_parent_T[j]^T @ R_abs[j] @ orient[j]^T

    converting absolute local rotations (PoseInversion output) back to the
    T-pose-relative convention. Arguments as for
    :func:`apply_joint_orient_local`.
    """
    orient, orient_parent_T = _orient_pair(orient, orient_parent_T)
    return jnp.swapaxes(orient_parent_T, -2, -1) @ local_rotations @ jnp.swapaxes(orient, -2, -1)


def compute_bone_lengths(
    joints: jnp.ndarray,
    parents: np.ndarray,
) -> jnp.ndarray:
    """Compute the length of each bone (distance from joint to its parent).

    Args:
        joints: (..., J, 3) joint positions.
        parents: (J,) parent indices.

    Returns:
        (..., J) bone lengths; root joints have length 0.
    """
    parents_arr = np.asarray(parents)
    safe_parents = np.maximum(parents_arr, 0)
    parent_pos = joints[..., safe_parents, :]
    diff = joints - parent_pos
    lengths = jnp.linalg.norm(diff, axis=-1)
    is_root = jnp.asarray(parents_arr < 0)
    return jnp.where(is_root, 0.0, lengths)


def precompute_joint_orient(joint_orient, joint_parent_ids):
    """Split authored joint orientations for :func:`apply_joint_orient_local`.

    Faithful port of upstream ``soma.geometry.rig_utils.precompute_joint_orient``:
    it *consumes* authored world-space orientations and returns the pair the
    apply function needs. It does **not** infer orientation from geometry — for
    that see :func:`infer_joint_orient_from_rest`, which is SOMA-JAX-only and
    was previously (confusingly) exported under this name.

    Args:
        joint_orient: (J, 3, 3) or (J, 4, 4) world-space orientation per joint.
        joint_parent_ids: (J,) parent indices.

    Returns:
        ``(orient, orient_parent_T)``, both (J, 3, 3).
    """
    orient = jnp.asarray(joint_orient)[..., :3, :3]
    parents = np.asarray(joint_parent_ids, dtype=np.int64)
    # A self-parented or negative root index must select itself, matching how
    # upstream indexes with the raw parent array on the stock rig.
    safe = np.where(parents < 0, np.arange(len(parents)), parents)
    orient_parent_T = jnp.swapaxes(orient[safe], -2, -1)
    return orient, orient_parent_T


class PoseMirrorSOMA:
    """Mirror world-space SOMA poses across the sagittal (YZ) plane.

    Faithful port of upstream ``soma.geometry.rig_utils.PoseMirror_SOMA``. Note
    this operates on full world **4x4 transforms** — positions included —
    unlike :class:`~soma_jax.geometry.skeleton_transfer.PoseMirror`, which
    mirrors rotation matrices only.

    Rig assumptions (upstream's): world up +Y, forward +Z, local +X points
    along the bone toward the child, and symmetric joints are named
    ``Left*`` / ``Right*``.

    The mirror is three steps::

        swap Left/Right joints  ->  diag(-1,1,1,1) @ T  ->  T @ local_adjust

    The left multiply reflects across YZ; the per-joint right multiply restores
    a right-handed frame and realigns the bone axis, using upstream's three
    cases: limbs ``diag(-1,-1,-1,1)``, centre ``diag(1,1,-1,1)``, root
    ``diag(-1,1,1,1)``.
    """

    def __init__(self, joint_names, root_name: str = "Root"):
        """
        Args:
            joint_names: (J,) joint names, ``Left*`` / ``Right*`` for symmetric pairs.
            root_name: name of the root joint, which gets its own correction.
        """
        names = [str(n) for n in joint_names]
        self.joint_names = names
        self.num_joints = len(names)

        perm = list(range(self.num_joints))
        left_idx, right_idx, center_idx = [], [], []
        root_index = -1
        lookup = {n: i for i, n in enumerate(names)}
        for i, name in enumerate(names):
            if name == root_name:
                root_index = i
            elif name.startswith("Left"):
                # Upstream registers the limb correction only once a Right mate
                # is found, and pairs both directions at the same time. An
                # unmatched Left* joint therefore keeps an identity adjust
                # rather than being treated as a limb.
                j = lookup.get("Right" + name[len("Left"):])
                if j is not None:
                    perm[i] = j
                    perm[j] = i
                    left_idx.append(i)
            elif name.startswith("Right"):
                right_idx.append(i)
            else:
                center_idx.append(i)

        self.perm = np.asarray(perm, dtype=np.int64)
        self.global_ref = jnp.asarray(np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32))

        adjust = np.tile(np.eye(4, dtype=np.float32), (self.num_joints, 1, 1))
        adjust[left_idx + right_idx] = np.diag([-1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        adjust[center_idx] = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        if root_index != -1:
            adjust[root_index] = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        else:
            logger.warning(
                "Root joint '%s' not found in joint list. Root rotation fix not applied.",
                root_name)
        self.local_adjust = jnp.asarray(adjust)

    def __call__(self, pose_world: jnp.ndarray) -> jnp.ndarray:
        """Mirror world transforms.

        Args:
            pose_world: (..., J, 4, 4) world-space joint transforms.

        Returns:
            (..., J, 4, 4) mirrored transforms.
        """
        T = jnp.asarray(pose_world)
        if T.shape[-2:] != (4, 4) or T.shape[-3] != self.num_joints:
            raise ValueError(
                f"Expected (..., {self.num_joints}, 4, 4), got {T.shape}")
        T = T[..., self.perm, :, :]
        T = self.global_ref @ T
        return T @ self.local_adjust


_DEFAULT_MHR_NEGATE_PARAMS = frozenset(
    [
        "head_lean",
        "head_twist",
        "neck_lean",
        "neck_twist",
        "root_ry",
        "root_rz",
        "root_tx",
        "spine0_rx_flexible",
        "spine0_ry_flexible",
        "spine1_rx_flexible",
        "spine1_ry_flexible",
        "spine2_rx_flexible",
        "spine2_ry_flexible",
        "spine3_rx_flexible",
        "spine3_ry_flexible",
        "spine_lean0",
        "spine_lean1",
        "spine_twist0",
        "spine_twist1",
    ]
)


class PoseMirrorMHR:
    """Mirror native-MHR parameter vectors — port of upstream ``PoseMirror_MHR``.

    MHR poses are a flat parameter vector, not transforms, so mirroring is a
    permutation (``l_`` <-> ``r_``, ``scale_l_`` <-> ``scale_r_``) followed by
    negating the parameters whose sign flips across the sagittal plane.

    Signs align with the *destination* index: the sign for parameter ``i`` is
    applied after data has been swapped into ``i``, matching upstream.
    """

    def __init__(self, param_names, negate_params=_DEFAULT_MHR_NEGATE_PARAMS):
        """
        Args:
            param_names: (N,) MHR parameter names.
            negate_params: names whose value negates under mirroring.
        """
        names = [str(n) for n in param_names]
        self.param_names = names
        self.num_params = len(names)
        lookup = {n: i for i, n in enumerate(names)}

        perm = list(range(self.num_params))
        signs = [1.0] * self.num_params
        for i, name in enumerate(names):
            mirror = None
            if name.startswith("scale_l_"):
                mirror = "scale_r_" + name[8:]
            elif name.startswith("scale_r_"):
                mirror = "scale_l_" + name[8:]
            elif name.startswith("l_"):
                mirror = "r_" + name[2:]
            elif name.startswith("r_"):
                mirror = "l_" + name[2:]
            if mirror is not None and mirror in lookup:
                perm[i] = lookup[mirror]
            if name in negate_params:
                signs[i] = -1.0

        self.perm = np.asarray(perm, dtype=np.int64)
        self.signs = jnp.asarray(np.asarray(signs, dtype=np.float32))

    def __call__(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            params: (..., N) MHR parameter vectors.

        Returns:
            (..., N) mirrored parameters.
        """
        p = jnp.asarray(params)
        if p.shape[-1] != self.num_params:
            raise ValueError(f"Expected (..., {self.num_params}), got {p.shape}")
        return p[..., self.perm] * self.signs


# ---------------------------------------------------------------------------
# Upstream names (``soma.geometry.rig_utils``)
# ---------------------------------------------------------------------------

#: ``(joint_ids, parent_ids)`` index pairs, one per tree depth.
SkeletonLevels = list[tuple[np.ndarray, np.ndarray]]

PoseMirror_SOMA = PoseMirrorSOMA
PoseMirror_MHR = PoseMirrorMHR


def compute_skeleton_levels(joint_parent_ids, device=None) -> SkeletonLevels:
    """Group joints by tree depth for level-order forward kinematics.

    Upstream ``rig_utils.compute_skeleton_levels``: depths follow
    ``depth[i] = depth[parent[i]] + 1`` over joints ``1..J-1`` (parents precede
    children), and level ``d`` is the pair of int64 arrays ``(joint_ids,
    parent_ids)``. ``device`` is accepted for signature compatibility. (The
    list-of-joint-lists :func:`soma_jax.geometry.lbs.compute_skeleton_levels`
    is the SOMA-JAX FK helper.)
    """
    parent_ids = [int(p) for p in np.asarray(joint_parent_ids).reshape(-1)]
    num_joints = len(parent_ids)
    depth = [0] * num_joints
    for i in range(1, num_joints):
        depth[i] = depth[parent_ids[i]] + 1
    max_depth = max(depth) if num_joints > 0 else 0
    levels = []
    for d in range(max_depth + 1):
        jids = [i for i in range(num_joints) if depth[i] == d]
        levels.append((np.asarray(jids, np.int64),
                       np.asarray([parent_ids[i] for i in jids], np.int64)))
    return levels


def joint_local_to_world_levelorder(joint_local_transforms, levels: SkeletonLevels) -> jnp.ndarray:
    """Level-order forward kinematics — upstream ``joint_local_to_world_levelorder``.

    Equivalent to :func:`joint_local_to_world`, composing every joint of a
    depth level in one batched matmul.

    Args:
        joint_local_transforms: (J, M, M) or (B, J, M, M) with M = 3 or 4.
        levels: output of :func:`compute_skeleton_levels`.

    Returns:
        World transforms, same shape as the input.
    """
    local = jnp.asarray(joint_local_transforms)
    added_batch = local.ndim == 3
    if added_batch:
        local = local[None]
    world = local
    for joint_ids, parent_ids in levels[1:]:
        world = world.at[:, joint_ids].set(world[:, parent_ids] @ local[:, joint_ids])
    return world[0] if added_batch else world
