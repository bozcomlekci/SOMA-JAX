"""Convert NVlabs/SOMA-X HuggingFace assets into soma_jax identity-model .npz packs.

Source: huggingface.co/nvidia/SOMA-X
Per-backend assets: base_body.obj, SOMA_wrap.obj, + optional shape basis (mhr_model_lod1.pt, point.npz)

Output: one .npz per backend with the keys soma_jax's identity_model.py expects:
    v_template      (V_src, 3)
    shapedirs       (V_src, 3, K)
    faces           (F_src, 3)
    bary_face_ids   (V_canonical,)
    bary_coords     (V_canonical, 3)

The wrap .obj is loaded the same way SOMA-X loads it (`maintain_order=True,
process=False`). Per investigation of NVlabs/SOMA-X (soma/identity_model.py:274-287,
soma/geometry/barycentric_interp.py:106-181), the wrap mesh's vertex row order
IS the canonical SOMA vertex order — the LBS pipeline never reads the wrap obj's
faces, only its positions, and consumes the canonical skinning weights from
SOMA_neutral.npz which assume that same row order.

Usage::

    python tools/build_identity_packs.py \\
        --hf-dir assets/third_party --output-dir assets/identity
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import trimesh


# Vertex positions are written in the backend's NATIVE unit; soma_jax's identity
# models multiply by `native_unit.meters_per_unit` at runtime (identity_model.py:217)
# so we should not pre-scale here.


def load_obj_soma_x_style(path: str | Path):
    """Load .obj the same way NVlabs/SOMA-X does, to preserve canonical ordering."""
    return trimesh.load(str(path), process=False, maintain_order=True, force="mesh")


def compute_bary(V_src: np.ndarray, F_src: np.ndarray, V_dst: np.ndarray):
    """Compute (face_ids, bary_coords) for each V_dst onto (V_src, F_src).

    Uses trimesh closest-point queries (3D triangle barycentric — the 3-component
    variant; SOMA-X uses a 4-component tetrahedral version for tiny off-surface
    corrections, but the 3D version is correct for points on the source surface,
    which is what the wrap obj provides by construction).
    """
    src_mesh = trimesh.Trimesh(vertices=V_src, faces=F_src, process=False)
    closest, _, face_ids = trimesh.proximity.closest_point(src_mesh, V_dst)
    tris = V_src[F_src[face_ids]]  # (V_dst, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(tris, closest)
    return face_ids.astype(np.int32), bary.astype(np.float32)


def build_mhr_pack(hf_dir: Path, out_path: Path):
    import torch

    src = load_obj_soma_x_style(hf_dir / "MHR" / "base_body_lod1.obj")
    wrap = load_obj_soma_x_style(hf_dir / "MHR" / "SOMA_wrap_lod1.obj")
    V_src = np.asarray(src.vertices, dtype=np.float32)
    F_src = np.asarray(src.faces, dtype=np.int32)
    V_wrap = np.asarray(wrap.vertices, dtype=np.float32)

    # Shape basis from the TorchScript blob: (K, V, 3) -> (V, 3, K)
    sm = torch.jit.load(str(hf_dir / "MHR" / "mhr_model_lod1.pt"), map_location="cpu")
    sd = sm.state_dict()
    shape_vectors = sd["character_torch.blend_shape.shape_vectors"].numpy()      # (45, 18439, 3)
    shapedirs = shape_vectors.transpose(1, 2, 0).astype(np.float32)              # (V, 3, K)

    face_ids, bary = compute_bary(V_src, F_src, V_wrap)
    print(f"  MHR: V_src={V_src.shape[0]} V_wrap={V_wrap.shape[0]} K={shapedirs.shape[-1]}")
    np.savez(out_path,
             v_template=V_src, shapedirs=shapedirs, faces=F_src,
             bary_face_ids=face_ids, bary_coords=bary)


def _invert_anny_coord_transform(v: np.ndarray) -> np.ndarray:
    """Pre-invert soma_jax's AnnyIdentityModel coord transform.

    AnnyIdentityModel.identity_model_to_soma applies a Z-up->Y-up reorder
    (perm (0,2,1), sign (1,1,-1)) assuming the Anny source is Z-up. The HF
    `Anny/base_body.obj` is already Y-up, so we pre-apply the inverse here; the
    runtime transform then restores the correct Y-up orientation. The barycentric
    indices/weights are frame-independent, so they are unaffected.
    """
    return v[:, (0, 2, 1)] * np.array([1.0, -1.0, 1.0], dtype=v.dtype)


def build_anny_pack(hf_dir: Path, out_path: Path):
    src = load_obj_soma_x_style(hf_dir / "Anny" / "base_body.obj")
    wrap = load_obj_soma_x_style(hf_dir / "Anny" / "SOMA_wrap.obj")
    V_src = np.asarray(src.vertices, dtype=np.float32)
    F_src = np.asarray(src.faces, dtype=np.int32)
    V_wrap = np.asarray(wrap.vertices, dtype=np.float32)

    # Anny has no shape PCA on HF — single fixed identity. Use a 1-component
    # zero basis so identity_coeffs is a no-op.
    shapedirs = np.zeros((V_src.shape[0], 3, 1), dtype=np.float32)

    # bary indices/weights are frame-independent; compute in the HF frame.
    face_ids, bary = compute_bary(V_src, F_src, V_wrap)
    # Store v_template pre-inverted so the runtime coord transform restores Y-up.
    V_src_packed = _invert_anny_coord_transform(V_src)
    print(f"  Anny: V_src={V_src.shape[0]} V_wrap={V_wrap.shape[0]} (no PCA, coord pre-inverted)")
    np.savez(out_path,
             v_template=V_src_packed, shapedirs=shapedirs, faces=F_src,
             bary_face_ids=face_ids, bary_coords=bary)


def build_garment_pack(hf_dir: Path, out_path: Path):
    src = load_obj_soma_x_style(hf_dir / "GarmentMeasurements" / "mean.obj")
    wrap = load_obj_soma_x_style(hf_dir / "GarmentMeasurements" / "SOMA_wrap.obj")
    V_src = np.asarray(src.vertices, dtype=np.float32)
    F_src = np.asarray(src.faces, dtype=np.int32)
    V_wrap = np.asarray(wrap.vertices, dtype=np.float32)

    pd = np.load(hf_dir / "GarmentMeasurements" / "point.npz")
    pca_matrix = pd["pca_matrix"]      # (V*3, K)
    V_count = V_src.shape[0]
    shapedirs = pca_matrix.reshape(V_count, 3, -1).astype(np.float32)  # (V, 3, K)

    face_ids, bary = compute_bary(V_src, F_src, V_wrap)
    print(f"  Garment: V_src={V_src.shape[0]} V_wrap={V_wrap.shape[0]} K={shapedirs.shape[-1]}")
    np.savez(out_path,
             v_template=V_src, shapedirs=shapedirs, faces=F_src,
             bary_face_ids=face_ids, bary_coords=bary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dir", default=None,
                   help="Directory containing the HuggingFace assets (downloaded from nvidia/SOMA-X)")
    p.add_argument("--output-dir", default="assets/identity",
                   help="Directory to write the converted .npz files into")
    p.add_argument("--backends", nargs="+", default=["mhr", "anny", "garment"],
                   choices=["mhr", "anny", "garment"])
    args = p.parse_args()

    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "mhr":     ("identity_mhr.npz",     build_mhr_pack),
        "anny":    ("identity_anny.npz",    build_anny_pack),
        "garment": ("identity_garment.npz", build_garment_pack),
    }
    for name in args.backends:
        out_path = out_dir / builders[name][0]
        print(f"\nBuilding {name} → {out_path}")
        builders[name][1](hf_dir, out_path)

    print(f"\nDone. Identity packs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
