"""Pose the SOMA body over the shared motion using SOMA-JAX (JAX + Warp hybrid).

Two jobs:
  1. RENDER: pose the clip's frames (fixed neutral identity, skeleton fit once,
     LBS-only) -> posed vertex + joint sequences for the GIF. Matches SOMA-X to
     ~cm.
  2. SPEED: the reported relative speed is the LARGE-BATCH (batch 2048) FULL
     forward throughput — identity blend + Warp-accelerated SkeletonTransfer.fit
     + FK/LBS — the honest long-term-scaling regime. Batch-1 over-credits JAX's
     launch overhead; at large batch the Warp-hybrid skeleton transfer (Warp
     svd3 Kabsch) is what keeps SOMA-JAX ahead (pure-JAX SVD loses at scale).
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--hf", default=str(REPO / "assets" / "hf"))
    p.add_argument("--bench-batch", type=int, default=2048,
                   help="batch size for the reported large-batch throughput")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    # FAIRNESS: SOMA-X (torch, allow_tf32=False; Warp scalar kernels) runs in
    # full float32. XLA defaults float32 matmuls to TF32 on Ampere+ GPUs, which
    # would give the JAX side a ~1.5-1.7x precision edge — so the teaser speedup
    # would not be a like-for-like comparison. Force float32 to match SOMA-X.
    jax.config.update("jax_default_matmul_precision", "highest")
    from scipy.sparse import csc_matrix
    from soma_jax.geometry.skeleton_transfer import SkeletonTransfer
    from soma_jax.geometry.batched_skinning import pose_from_bind, topk_skinning
    from soma_jax.geometry.lbs import compute_skeleton_levels
    from soma_jax.geometry.rig_utils import apply_joint_orient_local

    rig = dict(np.load(Path(args.hf) / "SOMA_neutral.npz", allow_pickle=False))
    bind_shape = np.asarray(rig["bind_shape"], np.float32)
    bind_world = np.asarray(rig["bind_pose_world"], np.float32)
    parents = rig["joint_parent_ids"].astype(np.int64).copy(); parents[0] = -1
    weights = np.asarray(csc_matrix(
        (rig["skinning_weights_data"], rig["skinning_weights_indices"],
         rig["skinning_weights_indptr"]),
        shape=tuple(rig["skinning_weights_shape"])).todense(), np.float32)
    t_pose = np.asarray(rig["t_pose_world"], np.float32)
    mean = np.asarray(rig["mean"], np.float32).reshape(-1)
    shapedirs = np.asarray(rig["shapedirs"], np.float32)
    eig = np.asarray(rig["eigenvalues"], np.float32)
    facial = np.concatenate([rig["segment_eye_bags"], rig["segment_mouth_bag"]])
    V, J, K = bind_shape.shape[0], bind_world.shape[0], eig.shape[0]

    m = np.load(args.motion)
    rotmats = m["rotmats"].astype(np.float32)                     # (T,78,3,3) absolute
    trans_np = m["trans"].astype(np.float32) * 100.0             # m -> cm
    absolute = bool(m["absolute"])
    T = rotmats.shape[0]

    # JAX + Warp hybrid skeleton transfer (Warp svd3 Kabsch).
    st = SkeletonTransfer(parents, bind_world, bind_shape, weights, rotation_method="auto",
                          vertex_ids_to_exclude=facial.tolist(), rotation_backend="warp")
    levels = compute_skeleton_levels(parents)
    w_idx_np, w_val_np = topk_skinning(weights, 8)
    w_idx, w_val = jnp.asarray(w_idx_np), jnp.asarray(w_val_np)
    weights_j = jnp.asarray(weights)
    orient = jnp.asarray(t_pose[:, :3, :3])
    mean_j, sd_j = jnp.asarray(mean), jnp.asarray(shapedirs)
    sqrt_eig = jnp.sqrt(jnp.asarray(eig))

    # ---------- 1) render: pose the clip (neutral identity, fit once) ----------
    rest0 = jnp.asarray(mean.reshape(V, 3))[None]
    bind_T0 = st.fit(rest0)
    bind_Tb = jnp.broadcast_to(bind_T0, (T,) + bind_T0.shape[1:])
    restb = jnp.broadcast_to(rest0, (T, V, 3))

    @jax.jit
    def pose_clip(R, hips):
        Ro = R if absolute else apply_joint_orient_local(R, orient, parents)
        posed, Tw = pose_from_bind(bind_Tb, restb, weights_j, levels, parents,
                                   Ro, hips, hips_idx=1, weight_values=w_val, weight_indices=w_idx)
        return posed * 0.01, Tw[:, :, :3, 3] * 0.01

    v, jt = pose_clip(jnp.asarray(rotmats), jnp.asarray(trans_np))
    verts = np.asarray(v); joints = np.asarray(jt)

    # ---------- 2) speed: large-batch FULL forward throughput (batch 2048) ----------
    B = args.bench_batch
    rng = np.random.default_rng(0)
    coeffs = jnp.asarray(rng.standard_normal((B, K)).astype(np.float32))
    Rb = jnp.asarray(np.tile(np.eye(3, dtype=np.float32), (B, J, 1, 1)))
    hipsb = jnp.zeros((B, 3), np.float32)

    @jax.jit
    def full_forward(coeffs, R, hips):
        rest = (mean_j[None] + (coeffs * sqrt_eig) @ sd_j).reshape(coeffs.shape[0], V, 3)
        bT = st.fit(rest)
        posed, _ = pose_from_bind(bT, rest, weights_j, levels, parents, R, hips,
                                  hips_idx=1, weight_values=w_val, weight_indices=w_idx)
        return posed

    for _ in range(3):
        full_forward(coeffs, Rb, hipsb).block_until_ready()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        full_forward(coeffs, Rb, hipsb).block_until_ready()
        ts.append(time.perf_counter() - t0)
    total = float(np.median(ts))
    fps = B / total

    faces = np.asarray(rig["triangles"]).astype(np.int32)
    parents_out = parents.copy(); parents_out[0] = 0
    np.savez(args.out, verts=verts, faces=faces, fps=fps, batch=B, bench_total_s=total,
             joints=joints, joint_names=np.asarray([str(n) for n in rig["joint_names"]]),
             parents=parents_out, duration_s=float(m["duration_s"]), play_fps=float(m["play_fps"]),
             label="SOMA-JAX (JAX + Warp)")
    print(f"[soma_jax] render T={T}  |  full-forward B={B}: {total*1e3:.2f} ms "
          f"=> {fps:.0f} meshes/s  wrote {args.out}")


if __name__ == "__main__":
    main()
