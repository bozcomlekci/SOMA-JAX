"""Pose the SOMA body over the shared motion using SOMA-X (PyTorch + Warp).

Two jobs:
  1. RENDER: pose the clip's frames (identity prepared once, LBS-only) -> posed
     vertex + joint sequences for the GIF.
  2. SPEED: the reported relative speed is the LARGE-BATCH (batch 2048) FULL
     forward throughput — prepare_identity (SkeletonTransfer.fit) + pose — the
     honest long-term-scaling regime (batch-1 is dominated by Warp kernel-launch
     latency). apply_correctives=False to match the JAX side.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--hf", default=str(REPO / "assets" / "hf"))
    p.add_argument("--bench-batch", type=int, default=2048)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from soma import SOMALayer

    m = np.load(args.motion)
    rotmats77 = m["rotmats"].astype(np.float32)[:, 1:, :, :]     # drop Root (identity)
    trans_np = m["trans"].astype(np.float32)
    absolute = bool(m["absolute"])
    T = rotmats77.shape[0]

    layer = SOMALayer(data_root=str(args.hf), device="cuda:0",
                      identity_model_type="soma", mode="warp",
                      correctives_model_path=None,
                      enable_procedural_transforms=False).to("cuda:0")
    layer.eval()
    K = layer.num_shape_components

    # ---------- 1) render: pose the clip (neutral identity, prepared once) ----------
    layer.prepare_identity(torch.zeros(1, K, device="cuda:0"), repose_to_bind_pose=False)
    poses_t = torch.from_numpy(rotmats77).to("cuda:0")
    transl_t = torch.from_numpy(trans_np).to("cuda:0")
    with torch.no_grad():
        out = layer.pose(poses_t, transl=transl_t, pose2rot=False,
                         absolute_pose=absolute, apply_correctives=False)
    verts = out.vertices.detach().cpu().numpy()
    joints = out.transforms[:, :, :3, 3].detach().cpu().numpy()

    # ---------- 2) speed: large-batch FULL forward throughput (batch 2048) ----------
    B = args.bench_batch
    rng = np.random.default_rng(0)
    ident = torch.from_numpy(rng.standard_normal((B, K)).astype(np.float32)).to("cuda:0")
    Rb = torch.from_numpy(np.tile(np.eye(3, dtype=np.float32), (B, 77, 1, 1))).to("cuda:0")
    trb = torch.zeros(B, 3, device="cuda:0")

    def full_forward():
        layer.prepare_identity(ident, repose_to_bind_pose=False)
        return layer.pose(Rb, transl=trb, pose2rot=False, absolute_pose=True,
                          apply_correctives=False)

    for _ in range(3):
        with torch.no_grad():
            full_forward()
    torch.cuda.synchronize()
    ts = np.empty(20)
    for i in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            full_forward()
        torch.cuda.synchronize(); ts[i] = time.perf_counter() - t0
    total = float(np.median(ts))
    fps = B / total

    faces = layer.faces.detach().cpu().numpy().astype(np.int32)
    rig = np.load(Path(args.hf) / "SOMA_neutral.npz")
    names = [str(n) for n in rig["joint_names"]]
    parents = rig["joint_parent_ids"].astype(int).copy(); parents[0] = 0
    np.savez(args.out, verts=verts, faces=faces, fps=fps, batch=B, bench_total_s=total,
             joints=joints, joint_names=np.asarray(names), parents=parents,
             duration_s=float(m["duration_s"]), play_fps=float(m["play_fps"]),
             label="SOMA-X (PyTorch + Warp)")
    print(f"[soma_x] render T={T}  |  full-forward B={B}: {total*1e3:.2f} ms "
          f"=> {fps:.0f} meshes/s  wrote {args.out}")


if __name__ == "__main__":
    main()
