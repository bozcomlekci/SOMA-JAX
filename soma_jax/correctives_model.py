"""Pose-corrective MLP for SOMA-JAX, faithful to SOMA-X v0.2.1.

Architecture (matches third_party/SOMA-X/soma/correctives_model.py)::

    # Bind-pose relative rotation: R_local = R_bind.T @ R_pose
    x = bindpose.T @ pose_rotmats               # (B, J, 3, 3)
    x[..., 0, 0] -= 1                            # subtract identity diagonal
    x[..., 1, 1] -= 1
    feat = x[..., :, :2].reshape(B, J*6)         # first two columns -> 6D
    W1   = self.W1 * M1_prior                    # (D=J*6, K=J*C)
    W2   = self.W2 * M2_prior                    # (K, 3V)
    z    = relu(feat @ W1)
    if use_tanh: z = tanh(z)
    y    = z @ W2                                # (B, 3V)
    out  = y.reshape(B, V, 3)

Where masks M1=(J,J) and M2=(J,V) are repeat-interleaved into the (D,K) and
(K,3V) shapes that the matmuls expect.

Trained checkpoints distributed with SOMA-X (HuggingFace ``nvidia/SOMA-X``)
ship as PyTorch ``.pt`` files; convert them to ``.npz`` via
``tools/convert/convert_correctives_pt_to_npz.py`` and pass the resulting path to
``SOMALayer.load(correctives_path=...)``.

Upstream: ``soma/correctives_model.py``
    Faithful port of that code. Pose-corrective MLP; equinox Module instead of torch.nn.Module.
"""
from __future__ import annotations
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class CorrectivesMLP(eqx.Module):
    """Pose-corrective displacement predictor (SOMA-X-compatible).

    Args:
        n_joints: number of joints J (78 for SOMA).
        n_vertices: number of mesh vertices V (18056 for SOMA).
        cors_per_joint: per-joint correctives C (24 in trained checkpoint).
        bindpose: (J, 3, 3) bind-pose rotation matrices for input remapping.
        W1: (J*6, J*C) layer-1 weights.
        W2: (J*C, 3V) layer-2 weights.
        M1_mask: (J, J) sparse joint-to-joint anatomical mask.
        M2_mask: (J, V) sparse joint-to-vertex anatomical mask.
        use_tanh: apply tanh after relu on hidden activations. Defaults to
            True, matching upstream's constructor and its checkpoint
            fallback; a checkpoint trained with tanh evaluated without it
            produces silently wrong offsets.
    """

    n_joints: int = eqx.field(static=True)
    n_vertices: int = eqx.field(static=True)
    cors_per_joint: int = eqx.field(static=True)
    use_tanh: bool = eqx.field(static=True)

    bindpose: jnp.ndarray            # (J, 3, 3)
    W1: jnp.ndarray                  # (D=J*6, K=J*C)
    W2: jnp.ndarray                  # (K, 3*V)
    M1_prior: Optional[jnp.ndarray]  # (D, K) -- expanded from M1_mask (J,J)
    M2_prior: Optional[jnp.ndarray]  # (K, 3*V) -- expanded from M2_mask (J,V)

    def __init__(
        self,
        n_joints: int,
        n_vertices: int,
        cors_per_joint: int = 24,
        bindpose: Optional[np.ndarray] = None,
        W1: Optional[np.ndarray] = None,
        W2: Optional[np.ndarray] = None,
        M1_mask: Optional[np.ndarray] = None,
        M2_mask: Optional[np.ndarray] = None,
        use_tanh: bool = True,
        key: Optional[jax.Array] = None,
    ):
        self.n_joints = int(n_joints)
        self.n_vertices = int(n_vertices)
        self.cors_per_joint = int(cors_per_joint)
        self.use_tanh = bool(use_tanh)

        # Architectural dims:
        #   D = J * 6              # input feature length (6D rotation per joint)
        #   K = J * cors_per_joint # corrective basis count
        # For the v0.2.1 trained checkpoint: J=78, cors_per_joint=24 → K=1872.
        D = self.n_joints * 6
        K = self.n_joints * self.cors_per_joint

        # Bindpose buffer: per-joint world bind rotation. The forward pass
        # left-multiplies by `bindpose.T`, so passing identity reduces the
        # input feature to the raw 6D rotation. The trained checkpoint
        # supplies its own bindpose (the SOMA bind-pose orientation), so the
        # identity default is only useful for an untrained-from-scratch model.
        if bindpose is None:
            bindpose = np.broadcast_to(np.eye(3, dtype=np.float32),
                                       (self.n_joints, 3, 3)).copy()
        self.bindpose = jnp.asarray(bindpose, dtype=jnp.float32)

        # W1 (D, K) is Xavier-uniform initialized when not provided. This
        # gives a sensible scale for an untrained model; the trained
        # checkpoint always provides W1 explicitly.
        if W1 is None:
            if key is None:
                key = jax.random.PRNGKey(0)
            bound = float(np.sqrt(6.0 / (D + K)))
            self.W1 = jax.random.uniform(key, (D, K), minval=-bound, maxval=bound)
        else:
            self.W1 = jnp.asarray(W1, dtype=jnp.float32)

        # W2 (K, 3V) defaults to ZERO so an unloaded model produces zero
        # correctives — i.e. acts as a no-op layer. This matches SOMA-X's
        # behaviour and lets callers safely skip the checkpoint without
        # corrupting their LBS output.
        if W2 is None:
            self.W2 = jnp.zeros((K, 3 * self.n_vertices), dtype=jnp.float32)
        else:
            self.W2 = jnp.asarray(W2, dtype=jnp.float32)

        # M1 mask: (J, J) joint-to-joint anatomical adjacency. Expanded to
        # the (D=J*6, K=J*C) shape by repeat-interleaving the rows by 6
        # (one feature axis per input column) and the cols by `cors_per_joint`
        # (one column per corrective basis vector at that joint).
        if M1_mask is not None:
            m1 = np.asarray(M1_mask, dtype=np.float32)
            assert m1.shape == (self.n_joints, self.n_joints), \
                f"M1_mask must be (J,J)=({self.n_joints},{self.n_joints}), got {m1.shape}"
            prior = np.repeat(np.repeat(m1, 6, axis=0), self.cors_per_joint, axis=1)
            self.M1_prior = jnp.asarray(prior, dtype=jnp.float32)
        else:
            self.M1_prior = None

        # M2 mask: (J, V) joint-to-vertex anatomical influence. Expanded to
        # (K=J*C, 3V) by repeat-interleaving rows by `cors_per_joint` (each
        # joint contributes C basis rows) and cols by 3 (xyz per vertex).
        if M2_mask is not None:
            m2 = np.asarray(M2_mask, dtype=np.float32)
            assert m2.shape == (self.n_joints, self.n_vertices), \
                f"M2_mask must be (J,V)=({self.n_joints},{self.n_vertices}), got {m2.shape}"
            prior = np.repeat(np.repeat(m2, self.cors_per_joint, axis=0), 3, axis=1)
            self.M2_prior = jnp.asarray(prior, dtype=jnp.float32)
        else:
            self.M2_prior = None

    def __call__(self, rotmats: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Predict pose correctives.

        Args:
            rotmats: (..., J, 3, 3) absolute joint rotations (joint-orient applied).
            training: unused (kept for API compat).

        Returns:
            (..., V, 3) vertex displacements in meters.
        """
        batch_shape = rotmats.shape[:-3]
        J = self.n_joints

        # Step 1 — bindpose-relative input feature.
        # Compute R_local = bindpose.T @ R_pose; subtract the identity on the
        # first two diagonal entries so the feature is zero at the bind pose.
        # The einsum "jba,...jbc->...jac" is a batched per-joint matmul
        # equivalent to `bindpose[j].T @ R_pose[j]` for each joint j.
        x = jnp.einsum("jba,...jbc->...jac", self.bindpose, rotmats[..., :3, :3])
        x = x.at[..., 0, 0].add(-1.0)
        x = x.at[..., 1, 1].add(-1.0)
        # 6D rotation feature: first two columns flattened.
        feat = x[..., :, :2].reshape(batch_shape + (J * 6,))

        # Step 2 — apply anatomical masks. Each mask is a binary multiplier
        # over the corresponding weight matrix; locations where the mask is 0
        # contribute nothing to the layer's output regardless of W1 / W2
        # values. The masks enforce spatial locality (a finger joint cannot
        # influence vertex positions on the opposite leg).
        W1 = self.W1 * self.M1_prior if self.M1_prior is not None else self.W1
        W2 = self.W2 * self.M2_prior if self.M2_prior is not None else self.W2

        # Step 3 — two-layer MLP: feat (B, D) @ W1 (D, K) -> z (B, K),
        # ReLU + optional tanh, then z @ W2 (K, 3V) -> y (B, 3V).
        z = feat @ W1
        z = jax.nn.relu(z)
        if self.use_tanh:
            z = jnp.tanh(z)
        y = z @ W2

        # Output: per-vertex displacement in the model's output unit (meters
        # for the trained checkpoint shipped with v0.2.1).
        return y.reshape(batch_shape + (self.n_vertices, 3))

    def save_checkpoint(self, path: str) -> None:
        """Save weights to a numpy .npz that :py:meth:`load_checkpoint` reads."""
        arrays = {
            "C_max":    np.int32(self.cors_per_joint),
            "use_tanh": np.bool_(self.use_tanh),
            "bindpose": np.array(self.bindpose),
            "W1":       np.array(self.W1),
            "W2":       np.array(self.W2),
        }
        if self.M1_prior is not None:
            # Round-trip back to (J,J) by sampling one cell per (joint_a, joint_b)
            # block: rows were repeat-interleaved by 6, cols by cors_per_joint.
            m1 = np.array(self.M1_prior)[::6, ::self.cors_per_joint]
            arrays["M1_mask"] = m1
        if self.M2_prior is not None:
            # Inverse of the (J,V)->(K,3V) expansion: rows by cors_per_joint, cols by 3.
            m2 = np.array(self.M2_prior)[::self.cors_per_joint, ::3]
            arrays["M2_mask"] = m2
        np.savez(path, **arrays)

    @classmethod
    def load_checkpoint(cls, path: str, v_index_map=None) -> "CorrectivesMLP":
        """Load a checkpoint produced by ``tools/convert/convert_correctives_pt_to_npz.py``
        (or by :py:meth:`save_checkpoint`).

        Args:
            path: checkpoint path.
            v_index_map: optional vertex subset to slice the output layer onto,
                e.g. ``lod_mid_to_low`` for a low-LOD layer. Mirrors upstream's
                ``correctives_vertex_index_map``.
        """
        data = np.load(path, allow_pickle=False)
        bindpose = np.asarray(data["bindpose"], dtype=np.float32)
        W1 = np.asarray(data["W1"], dtype=np.float32)
        W2 = np.asarray(data["W2"], dtype=np.float32)
        M1 = np.asarray(data["M1_mask"], dtype=np.float32) if "M1_mask" in data.files else None
        M2 = np.asarray(data["M2_mask"], dtype=np.float32) if "M2_mask" in data.files else None
        C  = int(np.asarray(data["C_max"]).item())
        ut = bool(np.asarray(data["use_tanh"]).item()) if "use_tanh" in data.files else True
        J = bindpose.shape[0]

        if v_index_map is not None:
            # Slice the output layer onto a vertex subset (upstream's
            # `_slice_checkpoint_tensors` with `v_index_map`). W2 stores xyz
            # interleaved per vertex, so columns are gathered as 3*v + {0,1,2};
            # M2's mask is per-vertex. Without this a low-LOD layer gets
            # 18056 offsets for a 4505-vertex rest shape.
            v_idx = np.asarray(v_index_map, dtype=np.int64).ravel()
            if v_idx.size and (v_idx.min() < 0 or v_idx.max() >= W2.shape[1] // 3):
                raise ValueError(
                    f"v_index_map out of range for a {W2.shape[1] // 3}-vertex checkpoint.")
            col = (v_idx[:, None] * 3 + np.arange(3)).reshape(-1)
            W2 = W2[:, col]
            if M2 is not None:
                M2 = M2[:, v_idx]

        V = W2.shape[1] // 3
        return cls(
            n_joints=J, n_vertices=V, cors_per_joint=C,
            bindpose=bindpose, W1=W1, W2=W2,
            M1_mask=M1, M2_mask=M2, use_tanh=ut,
        )
