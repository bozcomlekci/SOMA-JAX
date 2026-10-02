"""Build low-LOD SOMA assets (SOMA-X `low_lod=True` equivalent).

The SOMA mid mesh is ordered coarse-first: the low-LOD mesh is the first
`N_low` vertices (lod_mid_to_low == arange(N_low)) with `triangles_low` faces.
This tool slices every asset to the low-LOD vertex set and refits the joint
regressor, producing a drop-in low-LOD SOMA model + identity packs.

Outputs (default assets/lowlod/):
    SOMA_neutral.npz            low-LOD rig (v_template, weights, shapedirs, faces, J_regressor)
    identity_{mhr,anny,garment}.npz   identity packs with bary sliced to low-LOD SOMA verts

Usage::
    python tools/pipeline/build_lowlod.py
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
    p.add_argument("--hf", default=str(resolve("SOMA_neutral.npz", required=False) or ""),
                   help="upstream SOMA_neutral.npz (lod_mid_to_low, triangles_low)")
    p.add_argument("--soma", default="assets/SOMA_neutral_fixed.npz",
                   help="SOMA-JAX runtime archive (tools/pipeline/build_soma_rig.py)")
    p.add_argument("--identity-dir", default="assets/identity")
    p.add_argument("--out-dir", default="assets/lowlod")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    hf = dict(np.load(args.hf, allow_pickle=True))
    lod = np.asarray(hf["lod_mid_to_low"], np.int64)     # == arange(4505) on the shipped mesh
    n_low = int(lod.shape[0])
    faces_low = hf["triangles_low"].astype(np.int32)
    print(f"Low-LOD: {n_low} verts, {faces_low.shape[0]} faces")

    # ---- low-LOD SOMA rig ----
    # The rig comes from the runtime archive (itself derived from
    # SOMA_template_rig.usda): since SOMA-X v0.3 SOMA_neutral.npz has none.
    from soma_jax.rig_build import load_public_rig
    full = dict(np.load(args.soma, allow_pickle=True))
    rig = load_public_rig(args.soma)
    W_full = rig["weights"].astype(np.float64)
    bind = rig["bind_shape"].astype(np.float64)
    joints_t = rig["bind_pose_world"][:, :3, 3].astype(np.float64)
    parents = rig["parents"].astype(int).copy(); parents[0] = 0
    J = W_full.shape[1]
    children = {j: [k for k in range(J) if parents[k] == j and k != j] for j in range(J)}
    # Refit regressor on the low-LOD vertex subset.
    Jreg_low = build_regressor(bind[lod], joints_t, W_full[lod], parents, children)
    err = np.linalg.norm(Jreg_low @ bind[lod] - joints_t, axis=1)
    real = ~np.array(["End" in str(n) or "Eye" in str(n) for n in rig["joint_names"]])
    print(f"  low-LOD J_regressor fit (body joints): max {err[real].max():.3f} cm")

    # The same slice `SOMALayer.load(lod="low")` applies (upstream's
    # `lod="low"`): every per-vertex array, `triangles_low` faces and the
    # facial segments remapped into the subset — then the refitted regressor.
    from soma_jax.body.soma import _slice_rig_to_low_lod
    full.setdefault("lod_mid_to_low", lod)
    full.setdefault("triangles_low", faces_low)
    soma_low = _slice_rig_to_low_lod(full)
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
        pk["bary_face_ids"] = pk["bary_face_ids"][lod]
        pk["bary_coords"] = pk["bary_coords"][lod]
        np.savez(out / f"identity_{name}.npz", **pk)
        print(f"  wrote {out/f'identity_{name}.npz'}")

    # Record the LOD size for downstream slicing (correctives, MHR bary).
    np.save(out / "n_low.npy", np.array(n_low))
    print(f"Done. n_low={n_low}")


if __name__ == "__main__":
    main()
