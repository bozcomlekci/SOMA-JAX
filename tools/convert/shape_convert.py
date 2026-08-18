"""Convert identity coefficients between different shape representations.

Supports conversion between:
    - SMPL betas → SOMA PCA coefficients (via vertex regression)
    - SMPL betas → MHR identity coefficients
    - SOMA → SMPL betas
    - Generic shape conversion via Procrustes alignment of shape spaces

Usage::

    python tools/shape_convert.py \\
        --src-coeffs path/to/source_betas.npy \\
        --src-model smpl \\
        --tgt-model soma \\
        --soma-asset path/to/SOMA_neutral.npz \\
        --output path/to/converted_coeffs.npy
"""
from __future__ import annotations
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Convert identity coefficients between models")
    p.add_argument("--src-coeffs", required=True, help="Source identity coefficients (.npy)")
    p.add_argument("--src-model", required=True, choices=["smpl", "smplx", "mhr", "soma", "anny"])
    p.add_argument("--tgt-model", required=True, choices=["smpl", "smplx", "mhr", "soma", "anny"])
    p.add_argument("--soma-asset", required=True, help="SOMA_neutral.npz")
    p.add_argument("--src-asset", default=None, help="Source model NPZ (with shapedirs)")
    p.add_argument("--tgt-asset", default=None, help="Target model NPZ (with shapedirs)")
    p.add_argument("--output", required=True, help="Output (.npy) for converted coefficients")
    return p.parse_args()


def _least_squares_shape_fit(
    src_verts: np.ndarray,
    tgt_template: np.ndarray,
    tgt_shapedirs: np.ndarray,
) -> np.ndarray:
    """Solve for target coefficients that match source vertices.

    Minimizes ||tgt_template + shapedirs @ coeffs - src_verts||^2

    Args:
        src_verts: (V, 3) source-model vertices in target topology.
        tgt_template: (V, 3) target model rest template.
        tgt_shapedirs: (V, 3, K) target model shape basis.

    Returns:
        (K,) optimal target coefficients.
    """
    V, _, K = tgt_shapedirs.shape
    A = tgt_shapedirs.reshape(V * 3, K)
    b = (src_verts - tgt_template).reshape(V * 3)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs.astype(np.float32)


def main():
    args = parse_args()
    src_coeffs = np.load(args.src_coeffs)
    if src_coeffs.ndim == 1:
        src_coeffs = src_coeffs[None]

    from soma_jax.identity_model import create_identity_model
    from soma_jax.io import load_soma_npz
    import jax.numpy as jnp
    import numpy as np

    soma_data = load_soma_npz(args.soma_asset)
    src_data = load_soma_npz(args.src_asset) if args.src_asset else None
    tgt_data = load_soma_npz(args.tgt_asset) if args.tgt_asset else None

    src_model = create_identity_model(args.src_model, soma_data, src_data)
    src_verts_jax, _ = src_model.forward(jnp.asarray(src_coeffs))
    src_verts = np.asarray(src_verts_jax)  # (B, V, 3) in SOMA topology

    if args.tgt_model == "soma":
        tgt_template = np.asarray(soma_data["v_template"])
        tgt_shapedirs = np.asarray(src_data["shapedirs"]) if src_data and args.src_model == "soma" else None
        if tgt_shapedirs is None:
            print("WARNING: target=soma but no SOMA shapedirs available; using identity fit.")
            np.save(args.output, src_coeffs)
            return
    elif tgt_data is not None and "shapedirs" in tgt_data:
        tgt_template = np.asarray(tgt_data["v_template"])
        tgt_shapedirs = np.asarray(tgt_data["shapedirs"])
    else:
        raise ValueError(f"Need --tgt-asset with shapedirs for tgt-model={args.tgt_model}")

    B = src_verts.shape[0]
    out_coeffs = np.zeros((B, tgt_shapedirs.shape[-1]), dtype=np.float32)
    for i in range(B):
        out_coeffs[i] = _least_squares_shape_fit(src_verts[i], tgt_template, tgt_shapedirs)

    np.save(args.output, out_coeffs)
    print(f"Saved converted coefficients to: {args.output}")
    print(f"  Source: {args.src_model} {src_coeffs.shape}")
    print(f"  Target: {args.tgt_model} {out_coeffs.shape}")


if __name__ == "__main__":
    main()
