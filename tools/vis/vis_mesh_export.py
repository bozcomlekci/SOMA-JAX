"""Export SOMA / body model meshes to OBJ or PLY format for external viewing.

Usage::

    # Export a single rest mesh (no pose) from SOMA NPZ:
    python tools/vis_mesh_export.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --output mesh.obj

    # Export a posed mesh sequence:
    python tools/vis_mesh_export.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --animation path/to/anim.soma.npz \\
        --output-dir frames/ \\
        --format ply
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Export meshes to OBJ/PLY")
    p.add_argument("--soma-model", required=True, help="SOMA_neutral.npz")
    p.add_argument("--animation", default=None, help="Optional animation NPZ to pose")
    p.add_argument("--output", default=None, help="Single mesh output path")
    p.add_argument("--output-dir", default=None, help="Directory for animation frames")
    p.add_argument("--format", default="obj", choices=["obj", "ply"], help="Output format")
    p.add_argument("--frame", type=int, default=0, help="If --animation, frame index to export")
    p.add_argument("--all-frames", action="store_true", help="Export all animation frames")
    p.add_argument("--identity-model", default="soma", help="Identity model type")
    p.add_argument("--identity-coeffs", default=None, help=".npy file with shape coeffs")
    return p.parse_args()


def write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a simple OBJ file. Faces are 0-indexed in input, 1-indexed in OBJ."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_ply(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary PLY file."""
    nv = len(vertices)
    nf = len(faces)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {nv}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {nf}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(vertices.astype(np.float32).tobytes())
        for tri in faces:
            f.write(np.array([3], dtype=np.uint8).tobytes())
            f.write(tri.astype(np.int32).tobytes())


def write_mesh(path: str, vertices: np.ndarray, faces: np.ndarray, fmt: str) -> None:
    if fmt == "obj":
        write_obj(path, vertices, faces)
    elif fmt == "ply":
        write_ply(path, vertices, faces)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main():
    args = parse_args()
    import jax.numpy as jnp
    from soma_jax import SOMALayer, SOMAParams, load_soma_npz

    if args.identity_coeffs:
        identity_coeffs = np.load(args.identity_coeffs).astype(np.float32)
        if identity_coeffs.ndim == 1:
            identity_coeffs = identity_coeffs[None]
    else:
        identity_coeffs = np.zeros((1, 10), dtype=np.float32)

    layer = SOMALayer.load(args.soma_model, identity_model_type=args.identity_model)
    faces = np.array(layer.faces)

    if args.animation is None:
        # Just output the rest mesh
        rest_verts, _ = layer.prepare_identity(jnp.asarray(identity_coeffs), repose_to_bind_pose=False, skeleton_fit="linear")
        verts = np.array(rest_verts[0])
        out_path = args.output or f"rest_mesh.{args.format}"
        write_mesh(out_path, verts, faces, args.format)
        print(f"Wrote rest mesh: {out_path}")
        print(f"  Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}")
        return

    # Pose with animation
    anim = load_soma_npz(args.animation)
    poses = np.asarray(anim["poses"], dtype=np.float32)
    transl = np.asarray(anim["transl"], dtype=np.float32)
    if "identity_coeffs" in anim:
        identity_coeffs = np.asarray(anim["identity_coeffs"], dtype=np.float32)
        if identity_coeffs.ndim == 1:
            identity_coeffs = identity_coeffs[None]

    N = poses.shape[0]
    frame_idxs = range(N) if args.all_frames else [args.frame]

    if args.all_frames:
        if not args.output_dir:
            raise ValueError("--output-dir required with --all-frames")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.output).parent if args.output else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in frame_idxs:
        # Broadcast identity_coeffs to per-frame if it's (1, C)
        coeffs = identity_coeffs[i] if identity_coeffs.shape[0] > 1 else identity_coeffs[0]
        params = SOMAParams(
            poses=jnp.asarray(poses[i]),
            transl=jnp.asarray(transl[i]),
            identity_coeffs=jnp.asarray(coeffs),
        )
        out = layer(params)
        verts = np.array(out.vertices)
        if args.all_frames:
            out_path = out_dir / f"frame_{i:04d}.{args.format}"
        else:
            out_path = args.output or f"frame_{i}.{args.format}"
        write_mesh(str(out_path), verts, faces, args.format)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
