"""Build low-LOD SOMA assets (SOMA-X `low_lod=True` equivalent).

The SOMA mid mesh is ordered coarse-first: the low-LOD mesh is the first
`N_low` vertices (lod_mid_to_low == arange(N_low)) with `triangles_low` faces.
This tool slices every asset to the low-LOD vertex set and refits the joint
regressor, producing a drop-in low-LOD SOMA model + identity packs.

Outputs (default assets/lowlod/):
    SOMA_neutral.npz            low-LOD rig (v_template, weights, shapedirs, faces, J_regressor)
    identity_{mhr,anny,garment}.npz   identity packs with bary sliced to low-LOD SOMA verts

Usage::
    python tools/build_lowlod.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from build_soma_rig import build_regressor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Resolved through soma_jax.assets rather than hard-coded: the NVIDIA source
    # assets live in assets/third_party/ or the vendored submodule depending on
    # how they were obtained, and resolve() knows both.
    from soma_jax.assets import resolve
    p.add_argument("--hf", default=str(resolve("SOMA_neutral.npz", required=False) or ""))
    p.add_argument("--soma", default="SOMA_neutral.npz")
    p.add_argument("--identity-dir", default="assets/identity")
    p.add_argument("--out-dir", default="assets/lowlod")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    hf = dict(np.load(args.hf, allow_pickle=True))
    n_low = int(hf["triangles_low"].max()) + 1          # 4505
    lod = np.arange(n_low)                               # lod_mid_to_low == arange
    faces_low = hf["triangles_low"].astype(np.int32)
    print(f"Low-LOD: {n_low} verts, {faces_low.shape[0]} faces")

    # ---- low-LOD SOMA rig ----
    full = dict(np.load(args.soma, allow_pickle=True))
    from scipy.sparse import csc_matrix
    W_full = csc_matrix((hf["skinning_weights_data"], hf["skinning_weights_indices"],
                         hf["skinning_weights_indptr"]), shape=tuple(hf["skinning_weights_shape"])).toarray()
    bind = hf["bind_shape"].astype(np.float64)
    joints_t = hf["bind_pose_world"][:, :3, 3].astype(np.float64)
    parents = hf["joint_parent_ids"].astype(int).copy(); parents[0] = 0
    J = W_full.shape[1]
    children = {j: [k for k in range(J) if parents[k] == j and k != j] for j in range(J)}
    # Refit regressor on the low-LOD vertex subset.
    Jreg_low = build_regressor(bind[:n_low], joints_t, W_full[:n_low], parents, children)
    err = np.linalg.norm(Jreg_low @ bind[:n_low] - joints_t, axis=1)
    real = ~np.array(["End" in str(n) or "Eye" in str(n) for n in hf["joint_names"]])
    print(f"  low-LOD J_regressor fit (body joints): max {err[real].max():.3f} cm")

    soma_low = dict(full)
    soma_low["v_template"] = full["v_template"][:n_low].astype(np.float32)
    soma_low["weights"] = full["weights"][:n_low].astype(np.float32)
    soma_low["shapedirs"] = full["shapedirs"][:n_low].astype(np.float32)
    soma_low["faces"] = faces_low
    soma_low["J_regressor"] = Jreg_low.astype(np.float32)
    np.savez(out / "SOMA_neutral.npz", **soma_low)
    print(f"  wrote {out/'SOMA_neutral.npz'}")

    # ---- low-LOD identity packs (slice bary to low-LOD SOMA verts) ----
    for name in ["mhr", "anny", "garment"]:
        src = Path(args.identity_dir) / f"identity_{name}.npz"
        if not src.exists():
            print(f"  [{name}] {src} missing — skipping")
            continue
        pk = dict(np.load(src, allow_pickle=False))
        pk["bary_face_ids"] = pk["bary_face_ids"][:n_low]
        pk["bary_coords"] = pk["bary_coords"][:n_low]
        np.savez(out / f"identity_{name}.npz", **pk)
        print(f"  wrote {out/f'identity_{name}.npz'}")

    # Record the LOD size for downstream slicing (correctives, MHR bary).
    np.save(out / "n_low.npy", np.array(n_low))
    print(f"Done. n_low={n_low}")


if __name__ == "__main__":
    main()
