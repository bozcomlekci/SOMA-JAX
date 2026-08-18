"""Rig and joint topology utilities for SOMA-JAX.

Provides helpers for skeleton manipulation:
  - World↔local joint transform conversion
  - Joint orient precomputation and application
  - Joint hierarchy traversal (children, descendants)
  - Body-part vertex grouping

Upstream: ``soma/geometry/rig_utils.py``
    World<->local conversion and joint-hierarchy queries are a faithful port.
    Two name collisions: `precompute_joint_orient` here infers a frame from
    joint positions, where upstream's same-named function consumes authored
    orientation matrices; and `PoseMirror` (in skeleton_transfer.py) mirrors
    rotations only, where upstream's `PoseMirror_SOMA` mirrors full world
    transforms including positions. `PoseMirror_MHR` is not ported.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from .transforms import se3_from_rt, se3_inverse


def get_joint_children_ids(parents: np.ndarray) -> dict[int, list[int]]:
    """Return a dict mapping each joint to the list of its immediate children.

    Args:
        parents: (J,) integer parent indices; root has parent < 0.

    Returns:
        Dict {joint_id: [child_ids]} with all children for each joint.
    """
    J = len(parents)
    children: dict[int, list[int]] = {j: [] for j in range(J)}
    for j in range(J):
        p = int(parents[j])
        # A self-parented root (SOMA's joint 0) is not its own child — listing
        # it would make subtree walks such as `get_joint_descendents` recurse
        # forever. Upstream sidesteps this by starting the scan at index 1.
        if p >= 0 and p != j:
            children[p].append(j)
    return children


def get_joint_descendents(parents: np.ndarray, root_joint: int) -> list[int]:
    """Return all descendants of a joint (excluding the joint itself).

    Args:
        parents: (J,) parent indices.
        root_joint: index of the joint whose descendants we want.

    Returns:
        List of descendant joint indices in BFS order.
    """
    children = get_joint_children_ids(parents)
    result: list[int] = []
    queue = list(children[root_joint])
    while queue:
        j = queue.pop(0)
        result.append(j)
        queue.extend(children[j])
    return result


def get_joint_subtree(parents: np.ndarray, root_joint: int) -> list[int]:
    """Return the joint plus all its descendants (joint subtree)."""
    return [root_joint] + get_joint_descendents(parents, root_joint)


def get_body_part_vertex_ids(
    weights: np.ndarray,
    joint_groups: dict[str, list[int]],
    threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """Group vertices by body part based on skinning weight influence.

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
    ``rig_utils.get_body_part_vertex_ids``.

    Differs from :func:`get_body_part_vertex_ids` (which groups by a
    ``{name: joint_ids}`` dict): this walks the joint hierarchy from
    ``root_joint_id`` and unions the influence masks of every descendant, which
    is what the pose-inversion vertex weighting expects.

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
    world_transforms: jnp.ndarray,
    parents: np.ndarray,
) -> jnp.ndarray:
    """Convert global (world) joint transforms to local (parent-relative).

    For each joint j: T_local[j] = T_world[parent[j]]^-1 @ T_world[j]
    For root joints: T_local[j] = T_world[j]

    A joint counts as a root when ``parents[j] < 0`` (SMPL convention) **or**
    ``parents[j] == j`` (SOMA's own rig self-parents joint 0). SOMA-X handles
    the self-parented form explicitly; treating it as an ordinary joint would
    return identity for the root and break the world→local→world round trip.

    Args:
        world_transforms: (..., J, 4, 4) global joint transforms.
        parents: (J,) parent indices.

    Returns:
        (..., J, 4, 4) local joint transforms.
    """
    J = world_transforms.shape[-3]
    parents_arr = np.asarray(parents)
    is_root = (parents_arr < 0) | (parents_arr == np.arange(len(parents_arr)))
    safe_parents = np.maximum(parents_arr, 0)

    parent_world = world_transforms[..., safe_parents, :, :]   # (..., J, 4, 4)
    parent_inv = se3_inverse(parent_world)                      # (..., J, 4, 4)
    local = jnp.einsum("...jik,...jkl->...jil", parent_inv, world_transforms)

    # Root joints keep their world transforms
    root_mask = jnp.asarray(is_root, dtype=local.dtype)[:, None, None]  # (J, 1, 1)
    local = jnp.where(root_mask > 0.5, world_transforms, local)
    return local


def joint_local_to_world(
    local_transforms: jnp.ndarray,
    parents: np.ndarray,
) -> jnp.ndarray:
    """Convert local (parent-relative) joint transforms to world.

    Sequential FK along the kinematic chain.

    Args:
        local_transforms: (J, 4, 4) local joint transforms.
        parents: (J,) parent indices.

    Returns:
        (J, 4, 4) world (global) joint transforms.
    """
    parents_arr = jnp.asarray(parents)
    safe_parents = jnp.maximum(parents_arr, 0)
    J = local_transforms.shape[0]
    G = jnp.eye(4, dtype=local_transforms.dtype)[None].repeat(J, axis=0)

    def step(G, i):
        p = safe_parents[i]
        # Root = parent < 0 (SMPL) or self-parented (SOMA); mirrors
        # joint_world_to_local so the two are exact inverses.
        is_root = (parents_arr[i] < 0) | (parents_arr[i] == i)
        parent_T = jnp.where(is_root, jnp.eye(4, dtype=G.dtype), G[p])
        world_T = parent_T @ local_transforms[i]
        return G.at[i].set(world_T), None

    G, _ = jax.lax.scan(step, G, jnp.arange(J))
    return G


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


def apply_joint_orient_local(
    local_rotmats: jnp.ndarray,
    joint_orient: jnp.ndarray,
    parents: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Apply joint-orient remap to T-pose-relative local rotations.

    Matches NVlabs/SOMA-X's `rig_utils.apply_joint_orient_local`:

        R_out[j] = orient[parent[j]].T @ R_in[j] @ orient[j]

    i.e. conjugate by the PARENT's bind orient on the left and SELF's bind
    orient on the right. This is what takes a rotation expressed in the
    joint's own bind frame ("rotate the elbow by X around its bone axis")
    and rewrites it in the parent's bind frame so the standard FK chain
    (which composes in parent space) produces the intended pose.

    For backward compatibility, if `parents` is None we fall back to the
    legacy formula `R_out = orient @ R @ orient.T` (a same-joint conjugation,
    which is wrong whenever the parent's bind differs from the joint's bind).
    Callers should always pass `parents`.

    Args:
        local_rotmats: (..., J, 3, 3) T-pose-relative local rotations.
        joint_orient:  (J, 3, 3) per-joint world bind orientation.
        parents:       (J,) parent indices, root encoded as <0 or self.

    Returns:
        (..., J, 3, 3) rotations remapped to the parent-relative skinning frame.
    """
    if parents is None:
        return jnp.einsum("jrs,...jsk,jkt->...jrt",
                          joint_orient, local_rotmats,
                          jnp.swapaxes(joint_orient, -2, -1))
    parents_np = np.asarray(parents).astype(int)
    safe_parents = np.where(parents_np < 0, np.arange(len(parents_np)), parents_np)
    # Root: orient_parent = identity (root has no bound parent in world).
    is_root = (parents_np < 0) | (parents_np == np.arange(len(parents_np)))
    orient_parent_T = jnp.swapaxes(joint_orient[safe_parents], -2, -1)
    # Replace root rows with identity so we don't conjugate by self.T at root.
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=joint_orient.dtype), orient_parent_T.shape)
    orient_parent_T = jnp.where(is_root[:, None, None], eye3, orient_parent_T)
    return jnp.einsum("jrs,...jsk,jkt->...jrt",
                      orient_parent_T, local_rotmats, joint_orient)


def remove_joint_orient_local(
    local_rotmats: jnp.ndarray,
    joint_orient: jnp.ndarray,
    parents: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Inverse of `apply_joint_orient_local` — SOMA-X's reverse remap.

        R_in[j] = orient[parent[j]] @ R_out[j] @ orient[j].T

    Used by pose-inversion + smpl2soma to convert absolute skinning frames
    back to T-pose-relative locals for export.

    Args:
        local_rotmats: (..., J, 3, 3) joint-orient-aligned rotations.
        joint_orient: (J, 3, 3) joint orient matrices.
        parents:       (J,) parent indices.

    Returns:
        (..., J, 3, 3) local rotations with joint orient removed.
    """
    if parents is None:
        return jnp.einsum("jrs,...jst,jtk->...jrk",
                          jnp.swapaxes(joint_orient, -2, -1), local_rotmats,
                          joint_orient)
    parents_np = np.asarray(parents).astype(int)
    safe_parents = np.where(parents_np < 0, np.arange(len(parents_np)), parents_np)
    is_root = (parents_np < 0) | (parents_np == np.arange(len(parents_np)))
    orient_parent = joint_orient[safe_parents]
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=joint_orient.dtype), orient_parent.shape)
    orient_parent = jnp.where(is_root[:, None, None], eye3, orient_parent)
    return jnp.einsum("jrs,...jst,jtk->...jrk",
                      orient_parent, local_rotmats,
                      jnp.swapaxes(joint_orient, -2, -1))


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
