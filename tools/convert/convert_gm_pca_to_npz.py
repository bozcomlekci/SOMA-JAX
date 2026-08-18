"""Convert GarmentMeasurement PCA model from raw matrices to SOMA NPZ format.

The GarmentMeasurement model is typically stored as separate matrices:
    - v_template: (V, 3) rest vertices
    - shapedirs: (V, 3, K) PCA basis vectors
    - mean: (K,) mean shape coefficients (optional)

This tool packages these into a single NPZ file ready for use with
GarmentMeasurementIdentityModel.

Usage::

    python tools/convert_gm_pca_to_npz.py \\
        --v-template path/to/v_template.npy \\
        --shapedirs path/to/shapedirs.npy \\
        --faces path/to/faces.npy \\
        --output path/to/garment_measurement.npz
"""
from __future__ import annotations
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Convert GM PCA matrices to SOMA NPZ")
    p.add_argument("--v-template", required=True, help="Rest template (.npy, V x 3)")
    p.add_argument("--shapedirs", required=True, help="PCA basis (.npy, V x 3 x K)")
    p.add_argument("--faces", required=True, help="Triangle faces (.npy, F x 3)")
    p.add_argument("--mean", default=None, help="Mean coefficients (.npy, K)")
    p.add_argument("--bary-face-ids", default=None, help="Optional barycentric face IDs")
    p.add_argument("--bary-coords", default=None, help="Optional barycentric coords")
    p.add_argument("--output", required=True, help="Output NPZ file path")
    return p.parse_args()


def main():
    args = parse_args()
    v_template = np.load(args.v_template).astype(np.float32)
    shapedirs = np.load(args.shapedirs).astype(np.float32)
    faces = np.load(args.faces).astype(np.int32)

    V = v_template.shape[0]
    if v_template.shape != (V, 3):
        raise ValueError(f"v_template must be (V, 3), got {v_template.shape}")

    # Reshape shapedirs if flat (V*3, K)
    if shapedirs.ndim == 2:
        K = shapedirs.shape[1]
        shapedirs = shapedirs.reshape(V, 3, K)
    elif shapedirs.shape != (V, 3, shapedirs.shape[-1]):
        raise ValueError(f"shapedirs must be (V, 3, K), got {shapedirs.shape}")

    arrays = dict(
        v_template=v_template,
        shapedirs=shapedirs,
        faces=faces,
    )

    if args.mean is not None:
        arrays["mean"] = np.load(args.mean).astype(np.float32)
    if args.bary_face_ids is not None:
        arrays["bary_face_ids"] = np.load(args.bary_face_ids).astype(np.int32)
    if args.bary_coords is not None:
        arrays["bary_coords"] = np.load(args.bary_coords).astype(np.float32)

    np.savez_compressed(args.output, **arrays)
    print(f"Saved garment measurement model to: {args.output}")
    print(f"  Vertices: {V}, Faces: {faces.shape[0]}, PCA components: {shapedirs.shape[-1]}")


if __name__ == "__main__":
    main()
