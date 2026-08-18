"""Pure-JAX port of NVlabs/SOMA-X's *Unified Pose Correctives* (Beta).

SOMA-X applies a single shared MLP (`correctives_model.pt`) that predicts
per-vertex pose-dependent offsets in the SOMA topology, added to the rest shape
BEFORE linear blend skinning. This fixes the volume-collapse / LBS artifacts that
otherwise appear at bent joints (elbows, armpits, hips, knees) — and because the
SOMA skeleton is shared, the same correctives work for *every* identity model
(SOMA, MHR, Anny, GarmentMeasurement).

The reference forward (soma/correctives_model.py)::

    x = bindpose.T @ correctives_input            # correctives_input = apply_joint_orient_local(rel)
    x[:, 0, 0] -= 1; x[:, 1, 1] -= 1              # zero at rest
    feat = x[..., :2].reshape(6J)
    z = relu(feat @ W1)                           # W1 = W1_raw * M1_prior
    y = z @ W2                                    # W2 = (W2_raw * M2_prior) * cm->m
    out = y.reshape(V, 3)

The per-joint world bind orientation `orient` is recovered from the checkpoint's
`bindpose` by FK integration (orient[j] = orient[parent] @ bindpose[j]); this
guarantees the rest pose maps to zero correctives without external skeleton data.
Only `correctives_model.pt` reading uses torch (one-off); evaluation is pure JAX.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

_NATIVE_TO_METERS = 0.01  # checkpoint stores displacements in centimeters


def load_correctives(pt_path: str, parents: np.ndarray, n_lod: int | None = None) -> dict:
    """Read correctives_model.pt and return JAX arrays for the forward pass.

    n_lod: if set, slice the per-vertex output to the first n_lod vertices
        (low-LOD; the SOMA mid mesh is coarse-first, so low-LOD = verts[:n_lod]).
    """
    import torch
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    C = int(d["C_max"])
    I = 6
    bindpose = d["bindpose"].numpy().astype(np.float32)          # (J, 3, 3)
    J = bindpose.shape[0]
    W1 = d["W1"].to_dense().numpy().astype(np.float32)           # (6J, K)
    W2 = d["W2"].to_dense().numpy().astype(np.float32)           # (K, 3V)
    M1 = d["M1_mask"].to_dense().numpy().astype(np.float32)      # (J, J)
    M2 = d["M2_mask"].to_dense().numpy().astype(np.float32)      # (J, V)

    # Expand masks the SOMA-X way and bake them (+ unit scale) into the weights.
    M1_prior = np.repeat(np.repeat(M1, I, axis=0), C, axis=1)    # (6J, J*C)
    M2_prior = np.repeat(np.repeat(M2, C, axis=0), 3, axis=1)    # (J*C, 3V)
    W1_eff = W1 * M1_prior
    W2_eff = (W2 * M2_prior) * _NATIVE_TO_METERS
    if n_lod is not None:
        W2_eff = W2_eff[:, : n_lod * 3]                          # low-LOD vertex subset

    # Recover per-joint world bind orientation by FK over bindpose.
    parents = np.asarray(parents).astype(int)
    orient = np.zeros((J, 3, 3), np.float32)
    for j in range(J):
        p = parents[j]
        is_root = p < 0 or p == j      # SOMA root is self-parented (parents[0] == 0)
        orient[j] = bindpose[j] if is_root else orient[p] @ bindpose[j]
    orient_parent = np.stack(
        [np.eye(3, dtype=np.float32) if (parents[j] < 0 or parents[j] == j) else orient[parents[j]]
         for j in range(J)], axis=0)

    return {
        "bindpose": jnp.asarray(bindpose),
        "orient": jnp.asarray(orient),
        "orient_parent_T": jnp.asarray(np.transpose(orient_parent, (0, 2, 1))),
        "W1": jnp.asarray(W1_eff),
        "W2": jnp.asarray(W2_eff),
        "n_vertices": W2_eff.shape[1] // 3,
    }


def corrective_offsets(local_rotmats: jnp.ndarray, c: dict, absolute_pose: bool = False) -> jnp.ndarray:
    """Per-vertex pose correctives for one frame.

    Args:
        local_rotmats: (J, 3, 3) rotations.
        c: dict from load_correctives.
        absolute_pose: if True, ``local_rotmats`` are already in the absolute
            skinning frame (e.g. BVH-style) — skip the joint-orient remap.
            If False, ``local_rotmats`` are T-pose-relative (SOMA-X default).
    Returns:
        (V, 3) corrective offsets in meters.
    """
    # correctives_input matches SOMA-X: poses_rot if absolute_pose else apply_joint_orient_local(poses_rot)
    if absolute_pose:
        cin = local_rotmats
    else:
        cin = c["orient_parent_T"] @ local_rotmats @ c["orient"]      # (J, 3, 3)
    x = jnp.swapaxes(c["bindpose"], -2, -1) @ cin                 # bindpose^T @ cin
    x = x.at[:, 0, 0].add(-1.0)
    x = x.at[:, 1, 1].add(-1.0)
    feat = x[:, :, :2].reshape(-1)                                # (6J,)
    z = jax.nn.relu(feat @ c["W1"])                              # (K,)
    y = z @ c["W2"]                                              # (3V,)
    return y.reshape(c["n_vertices"], 3)
