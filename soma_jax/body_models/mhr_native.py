"""MHR (Meta Human Rig) native forward pass in JAX.

Upstream: ``soma/identity_model.py`` (``MHRIdentityModel``) delegates the whole
forward to the **TorchScript** archive ``assets/MHR/mhr_model_lod1.pt``:

.. code-block:: python

    identity_rest_shape, _ = self.identity_model(
        identity_coeffs,                                   # (B, 45)
        torch.cat([pose_params, scale_params], dim=1),      # (B, 136 + 68)
        face_expr_params,                                   # (B, 72)
    )

This module reimplements that archive's forward in JAX so the MHR backend does
not need ``torch`` at inference time. The archive is *not* opaque — it exposes
55 named tensors plus readable TorchScript source per submodule — so every step
below is a direct transcription, and :func:`from_torchscript` lifts the weights
out once.

The archive's forward, transcribed
==================================

.. code-block:: text

    identity_rest_pose = blend_shape(identity_coeffs)
    joint_parameters   = parameter_transform(cat[model_parameters, zeros(45)])
    skel_state         = joint_parameters_to_skeleton_state(joint_parameters)
    face_expressions   = face_expressions_model(face_expr_coeffs)
    unposed            = identity_rest_pose + face_expressions
    if apply_correctives:
        unposed       += pose_correctives_model(joint_parameters)
    verts              = skin_points(skel_state, unposed)
    return (verts, skel_state)

``MHRIdentityModel.get_rest_shape`` takes the **first** return value, so the
"rest shape" is the fully skinned mesh at zero pose with the requested body-part
scales — not the raw blend shape.

Conventions, established empirically against the archive
========================================================

============================  =================================================
quantity                      convention
============================  =================================================
quaternions                   **xyzw** (``joint_prerotations[0] == [0,0,0,1]``)
``euler_xyz_to_quaternion``   ``qz(ez) * qy(ey) * qx(ex)``, i.e. ``R = Rz·Ry·Rx``
joint scale                   ``exp(p * ln2) == 2**p``
skeleton state                ``(tx,ty,tz, qx,qy,qz,qw, s)`` per joint, 8 wide
joint parameters              ``(..., 127, 7)`` = 3 translation, 3 euler, 1 scale
============================  =================================================

Native unit is **centimetres** and the native frame is Y-up / Z-forward, matching
``MHRIdentityModel.NATIVE_UNIT`` / ``NATIVE_UP`` / ``NATIVE_FORWARD``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "MHRNativeModel",
    "euler_xyz_to_quaternion_xyzw",
    "quaternion_multiply_xyzw",
    "quaternion_rotate",
    "skel_state_compose",
]

#: The archive scales joints as ``exp(p * ln2)``; kept at the archive's own
#: float32 literal so the port reproduces its rounding.
_LN2 = 0.69314718246459961

#: Joint parameters are 7 per joint: 3 translation, 3 Euler XYZ, 1 log2 scale.
_PARAMS_PER_JOINT = 7

#: ``_pose_features_from_joint_params`` drops the first two joints before
#: building the 6D feature (``joint_parameters[:, 2:, 3:6]``).
_POSE_FEATURE_SKIP_JOINTS = 2


def euler_xyz_to_quaternion_xyzw(euler: jnp.ndarray) -> jnp.ndarray:
    """Port of ``pymomentum.quaternion.euler_xyz_to_quaternion``.

    Composes as ``qz(ez) * qy(ey) * qx(ex)`` — verified against the archive on
    mixed angles, where the opposite order (``qx*qy*qz``) disagrees in the first
    component by ~0.19.

    Args:
        euler: (..., 3) Euler XYZ angles in radians.
    Returns:
        (..., 4) xyzw quaternions.
    """
    half = euler * 0.5
    c = jnp.cos(half)
    s = jnp.sin(half)
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    # Expanded product of qz * qy * qx.
    return jnp.stack([
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ], axis=-1)


def quaternion_multiply_xyzw(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of xyzw quaternions — ``multiply_assume_normalized``."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quaternion_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate ``v`` (..., 3) by xyzw quaternion ``q`` (..., 4)."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * jnp.cross(qv, v)
    return v + qw * t + jnp.cross(qv, t)


def skel_state_compose(parent: jnp.ndarray, child: jnp.ndarray) -> jnp.ndarray:
    """Compose two skeleton states ``(t, q, s)``.

    Port of the composition inside
    ``pymomentum.backend.skel_state_backend``: the child's translation is scaled
    and rotated into the parent's frame, rotations multiply, scales multiply.

    Args:
        parent, child: (..., 8) states, ``(tx,ty,tz, qx,qy,qz,qw, s)``.
    Returns:
        (..., 8) composed state.
    """
    pt, pq, ps = parent[..., :3], parent[..., 3:7], parent[..., 7:8]
    ct, cq, cs = child[..., :3], child[..., 3:7], child[..., 7:8]
    t = pt + ps * quaternion_rotate(pq, ct)
    q = quaternion_multiply_xyzw(pq, cq)
    return jnp.concatenate([t, q, ps * cs], axis=-1)


def _skel_state_apply(state: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """Transform points by a skeleton state: ``t + s * (q ⊗ p)``."""
    t, q, s = state[..., :3], state[..., 3:7], state[..., 7:8]
    return t + s * quaternion_rotate(q, points)


class MHRNativeModel:
    """Pure-JAX evaluation of ``mhr_model_lod1.pt``.

    Construct with :meth:`from_torchscript` (needs ``torch`` once) or
    :meth:`from_npz` (no ``torch``). :meth:`to_npz` writes the lifted weights so
    the ``torch`` dependency is build-time only.
    """

    #: Every array the forward needs, in the order ``to_npz``/``from_npz`` use.
    FIELDS = (
        "shape_vectors", "base_shape", "expr_shape_vectors",
        "parameter_transform", "joint_translation_offsets", "joint_prerotations",
        "joint_parents", "inverse_bind_pose",
        "skin_indices", "skin_weights", "vert_indices",
        "pc_sparse_indices", "pc_sparse_weight", "pc_linear_weight",
    )

    def __init__(self, **arrays: Any):
        missing = [f for f in self.FIELDS if f not in arrays]
        if missing:
            raise ValueError(f"MHRNativeModel missing arrays: {missing}")
        for name in self.FIELDS:
            value = arrays[name]
            if name in ("joint_parents", "skin_indices", "vert_indices",
                        "pc_sparse_indices"):
                setattr(self, name, np.asarray(value))
            else:
                setattr(self, name, jnp.asarray(value, dtype=jnp.float32))

        self.num_joints = int(self.joint_translation_offsets.shape[0])
        self.num_vertices = int(self.base_shape.shape[0])
        self.num_identity_coeffs = int(self.shape_vectors.shape[0])
        self.num_expression_coeffs = int(self.expr_shape_vectors.shape[0])
        # The transform consumes model parameters plus a zero identity block.
        self.num_model_parameters = (
            int(self.parameter_transform.shape[1]) - self.num_identity_coeffs)
        self.pc_hidden = int(arrays.get("pc_sparse_shape", (3000, 750))[0])
        self.pc_in = int(arrays.get("pc_sparse_shape", (3000, 750))[1])

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_torchscript(cls, path: str | Path | None = None) -> "MHRNativeModel":
        """Lift the weights out of the TorchScript archive.

        Args:
            path: ``mhr_model_lod1.pt``; resolved from the asset search path
                when omitted.
        """
        import torch
        if path is None:
            from ..assets import resolve
            path = resolve("MHR/mhr_model_lod1.pt")
        module = torch.jit.load(str(path), map_location="cpu")
        sd = {k: v.numpy() for k, v in module.state_dict().items()}
        c = "character_torch."
        sparse_shape = tuple(
            int(x) for x in
            module.pose_correctives_model.pose_dirs_predictor._modules["0"].sparse_shape)
        return cls(
            shape_vectors=sd[c + "blend_shape.shape_vectors"],
            base_shape=sd[c + "blend_shape.base_shape"],
            expr_shape_vectors=sd["face_expressions_model.shape_vectors"],
            parameter_transform=sd[c + "parameter_transform.parameter_transform"],
            joint_translation_offsets=sd[c + "skeleton.joint_translation_offsets"],
            joint_prerotations=sd[c + "skeleton.joint_prerotations"],
            joint_parents=sd[c + "skeleton.joint_parents"].astype(np.int32),
            inverse_bind_pose=sd[c + "linear_blend_skinning.inverse_bind_pose"],
            skin_indices=sd[c + "linear_blend_skinning.skin_indices_flattened"].astype(np.int32),
            skin_weights=sd[c + "linear_blend_skinning.skin_weights_flattened"],
            vert_indices=sd[c + "linear_blend_skinning.vert_indices_flattened"].astype(np.int32),
            pc_sparse_indices=sd["pose_correctives_model.pose_dirs_predictor.0.sparse_indices"],
            pc_sparse_weight=sd["pose_correctives_model.pose_dirs_predictor.0.sparse_weight"],
            pc_linear_weight=sd["pose_correctives_model.pose_dirs_predictor.2.weight"],
            pc_sparse_shape=sparse_shape,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "MHRNativeModel":
        data = np.load(path, allow_pickle=False)
        arrays = {k: data[k] for k in cls.FIELDS}
        if "pc_sparse_shape" in data:
            arrays["pc_sparse_shape"] = tuple(int(x) for x in data["pc_sparse_shape"])
        return cls(**arrays)

    def to_npz(self, path: str | Path) -> None:
        """Write the lifted weights so inference needs no ``torch``."""
        out = {name: np.asarray(getattr(self, name)) for name in self.FIELDS}
        out["pc_sparse_shape"] = np.asarray([self.pc_hidden, self.pc_in], np.int64)
        np.savez_compressed(path, **out)

    # ---- forward stages --------------------------------------------------
    def blend_shape(self, identity_coeffs: jnp.ndarray) -> jnp.ndarray:
        """``einsum("nvd,...n->...vd", shape_vectors, coeffs) + base_shape``."""
        return jnp.einsum("nvd,...n->...vd", self.shape_vectors, identity_coeffs) \
            + self.base_shape

    def face_expressions(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Face-expression blend shapes; no base term, matching the archive."""
        return jnp.einsum("nvd,...n->...vd", self.expr_shape_vectors, coeffs)

    def model_parameters_to_joint_parameters(self, mp: jnp.ndarray) -> jnp.ndarray:
        """``einsum("dn,...n->...d", parameter_transform, model_parameters)``.

        The archive appends a zero identity block to the model parameters before
        the transform (``cat[model_parameters, zeros_like(identity_coeffs)]``),
        so ``mp`` here is the full 249-wide vector.
        """
        return jnp.einsum("dn,...n->...d", self.parameter_transform, mp)

    def local_skeleton_state(self, joint_parameters: jnp.ndarray) -> jnp.ndarray:
        """Per-joint local ``(t, q, s)`` — ``joint_parameters_to_local_skeleton_state``."""
        jp = joint_parameters.reshape(
            joint_parameters.shape[:-1] + (self.num_joints, _PARAMS_PER_JOINT))
        t = jp[..., :3] + self.joint_translation_offsets
        q = quaternion_multiply_xyzw(
            self.joint_prerotations, euler_xyz_to_quaternion_xyzw(jp[..., 3:6]))
        s = jnp.exp(jp[..., 6:7] * _LN2)
        return jnp.concatenate([t, q, s], axis=-1)

    def global_skeleton_state(self, local: jnp.ndarray) -> jnp.ndarray:
        """Compose local states down the hierarchy.

        Port of ``global_skel_state_from_local_skel_state``. The archive drives
        this from a precomputed ``pmi`` traversal; the parent array gives the
        same result because MHR's joints are stored parent-before-child (checked
        in ``tests/test_mhr_native.py``).
        """
        parents = self.joint_parents
        out = [None] * self.num_joints
        for j in range(self.num_joints):
            p = int(parents[j])
            child = local[..., j, :]
            out[j] = child if p < 0 or p == j else skel_state_compose(out[p], child)
        return jnp.stack(out, axis=-2)

    def pose_correctives(self, joint_parameters: jnp.ndarray) -> jnp.ndarray:
        """Pose-dependent vertex offsets — ``pose_correctives_model.forward``.

        Feature: the 6D form of every joint's Euler angles except the first two,
        with 1 subtracted from components 0 and 4 (the two diagonal entries), so
        the feature vanishes at zero pose. Then a sparse 750->3000 layer, ReLU,
        and a dense 3000->3V layer with no bias.
        """
        jp = joint_parameters.reshape(
            joint_parameters.shape[:-1] + (self.num_joints, _PARAMS_PER_JOINT))
        euler = jp[..., _POSE_FEATURE_SKIP_JOINTS:, 3:6]
        feat = _batch_6d_from_xyz(euler)
        feat = feat.at[..., 0].add(-1.0).at[..., 4].add(-1.0)
        feat = feat.reshape(feat.shape[:-2] + (-1,))

        rows, cols = self.pc_sparse_indices
        # Sparse (3000, 750) @ featᵀ, expressed as a scatter-add over nonzeros.
        contrib = feat[..., cols] * self.pc_sparse_weight
        hidden = jnp.zeros(feat.shape[:-1] + (self.pc_hidden,), feat.dtype)
        hidden = hidden.at[..., rows].add(contrib)
        hidden = jax.nn.relu(hidden)
        flat = hidden @ self.pc_linear_weight.T
        return flat.reshape(flat.shape[:-1] + (self.num_vertices, 3))

    def skin(self, skel_state: jnp.ndarray, rest_vertices: jnp.ndarray) -> jnp.ndarray:
        """Linear blend skinning from a skeleton state.

        Port of ``skin_points_from_skel_state``: each of the flattened
        (vertex, joint, weight) triples contributes
        ``w * apply(skel_state[j] ∘ inverse_bind_pose[j], v)``.
        """
        bone = skel_state_compose(skel_state, self.inverse_bind_pose)
        picked = bone[..., self.skin_indices, :]                 # (..., N, 8)
        verts = rest_vertices[..., self.vert_indices, :]         # (..., N, 3)
        contrib = _skel_state_apply(picked, verts) * self.skin_weights[:, None]
        out = jnp.zeros(rest_vertices.shape, contrib.dtype)
        return out.at[..., self.vert_indices, :].add(contrib)

    # ---- full forward ----------------------------------------------------
    def __call__(
        self,
        identity_coeffs: jnp.ndarray,
        model_parameters: jnp.ndarray,
        face_expr_coeffs: Optional[jnp.ndarray] = None,
        apply_correctives: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate the archive's forward.

        Args:
            identity_coeffs: (B, 45) shape coefficients.
            model_parameters: (B, 204) = 136 pose + 68 body-part scale, exactly
                what ``MHRIdentityModel.get_rest_shape`` concatenates.
            face_expr_coeffs: (B, 72); zeros when omitted, as SOMA passes.
            apply_correctives: include the pose-corrective offsets.

        Returns:
            ``(vertices, skel_state)`` — vertices in **centimetres**.
        """
        identity_coeffs = jnp.atleast_2d(jnp.asarray(identity_coeffs, jnp.float32))
        model_parameters = jnp.atleast_2d(jnp.asarray(model_parameters, jnp.float32))
        B = identity_coeffs.shape[0]
        if face_expr_coeffs is None:
            face_expr_coeffs = jnp.zeros((B, self.num_expression_coeffs), jnp.float32)
        face_expr_coeffs = jnp.atleast_2d(jnp.asarray(face_expr_coeffs, jnp.float32))

        rest = self.blend_shape(identity_coeffs)
        # The archive appends a zero identity block before the transform.
        mp_full = jnp.concatenate(
            [model_parameters, jnp.zeros_like(identity_coeffs)], axis=-1)
        joint_parameters = self.model_parameters_to_joint_parameters(mp_full)
        skel_state = self.global_skeleton_state(
            self.local_skeleton_state(joint_parameters))

        unposed = rest + self.face_expressions(face_expr_coeffs)
        if apply_correctives:
            unposed = unposed + self.pose_correctives(joint_parameters)
        return self.skin(skel_state, unposed), skel_state

    def get_rest_shape(
        self,
        identity_coeffs: jnp.ndarray,
        scale_params: jnp.ndarray,
        bone_length_flexibles: Optional[jnp.ndarray] = None,
        apply_correctives: bool = True,
    ) -> jnp.ndarray:
        """Rest shape in centimetres — ``MHRIdentityModel.get_rest_shape``.

        Args:
            identity_coeffs: (B, 45).
            scale_params: (B, 68) body-part scales; upstream asserts these are
                supplied for MHR.
            bone_length_flexibles: optional (B, 6) written into
                ``pose_params[130:136]`` exactly as upstream does — spine, neck,
                shoulder, arm, hip and leg bone lengths, identity-like but
                carried in MHR's pose vector.
            apply_correctives: include pose correctives (zero pose makes the
                feature vanish, so this is a no-op at the default pose).
        """
        identity_coeffs = jnp.atleast_2d(jnp.asarray(identity_coeffs, jnp.float32))
        scale_params = jnp.atleast_2d(jnp.asarray(scale_params, jnp.float32))
        B = identity_coeffs.shape[0]
        n_pose = self.num_model_parameters - scale_params.shape[1]
        pose = jnp.zeros((B, n_pose), jnp.float32)
        if bone_length_flexibles is not None:
            blf = jnp.atleast_2d(jnp.asarray(bone_length_flexibles, jnp.float32))
            pose = pose.at[:, n_pose - blf.shape[1]:].set(blf)
        vertices, _ = self(
            identity_coeffs, jnp.concatenate([pose, scale_params], axis=-1),
            apply_correctives=apply_correctives)
        return vertices


def _batch_6d_from_xyz(euler: jnp.ndarray) -> jnp.ndarray:
    """6D rotation feature from Euler XYZ — ``pymomentum.mhr.utils.batch6DFromXYZ``.

    The archive subtracts 1 from feature components 0 and 4, which pins them to
    the two diagonal entries ``R00`` and ``R11``. Both a column-major and a
    row-major flatten of the first two columns/rows satisfy that, so the layout
    is fixed by end-to-end comparison against the archive
    (``tests/test_mhr_native.py::test_pose_correctives_matches_archive``):
    the first two **columns**, stacked column-major.
    """
    q = euler_xyz_to_quaternion_xyzw(euler)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Columns 0 and 1 of the rotation matrix for an xyzw quaternion.
    c0 = jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    c1 = jnp.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], -1)
    return jnp.concatenate([c0, c1], axis=-1)
