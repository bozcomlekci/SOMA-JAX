"""PyRender-based 3D mesh visualization for SOMA-JAX.

Renders a SOMA mesh (rest or posed) as a static PNG image or interactive viewer.
Requires `pyrender` and `trimesh`: ``pip install pyrender trimesh``

Usage::

    # Static render of rest mesh
    python tools/vis_pyrender.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --output rest.png

    # Static render of a specific frame
    python tools/vis_pyrender.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --animation path/to/anim.soma.npz \\
        --frame 0 --output frame0.png

    # Interactive viewer
    python tools/vis_pyrender.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --animation path/to/anim.soma.npz \\
        --interactive
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="PyRender visualization for SOMA-JAX")
    p.add_argument("--soma-model", required=True, help="SOMA_neutral.npz")
    p.add_argument("--animation", default=None, help="Optional animation NPZ")
    p.add_argument("--output", default=None, help="Output image path (PNG)")
    p.add_argument("--frame", type=int, default=0, help="Animation frame index")
    p.add_argument("--width", type=int, default=800, help="Render width")
    p.add_argument("--height", type=int, default=800, help="Render height")
    p.add_argument("--interactive", action="store_true", help="Open interactive viewer")
    p.add_argument("--identity-model", default="soma", help="Identity model type")
    p.add_argument("--identity-coeffs", default=None, help=".npy file with shape coeffs")
    p.add_argument(
        "--color", nargs=3, type=float, default=[0.7, 0.7, 0.85],
        help="RGB diffuse color (0-1)",
    )
    p.add_argument("--no-smooth", action="store_true", help="Use flat shading")
    p.add_argument(
        "--camera-distance", type=float, default=3.0,
        help="Camera distance from mesh center (meters)",
    )
    return p.parse_args()


def _check_deps():
    try:
        import trimesh   # noqa: F401
        import pyrender  # noqa: F401
    except ImportError:
        print("ERROR: pyrender and trimesh are required.")
        print("  pip install pyrender trimesh")
        sys.exit(1)


def _compute_mesh(args, jnp, layer, SOMAParams):
    """Run SOMA forward to compute (vertices, faces, joints)."""
    if args.identity_coeffs:
        identity_coeffs = np.load(args.identity_coeffs).astype(np.float32)
        if identity_coeffs.ndim == 1:
            identity_coeffs = identity_coeffs[None]
    else:
        identity_coeffs = np.zeros((1, 10), dtype=np.float32)

    if args.animation is None:
        from soma_jax import load_soma_npz
        rest_verts, joints = layer.prepare_identity(jnp.asarray(identity_coeffs), repose_to_bind_pose=False, skeleton_fit="linear")
        return np.array(rest_verts[0]), np.array(layer.faces), np.array(joints[0])

    from soma_jax import load_soma_npz
    anim = load_soma_npz(args.animation)
    poses = np.asarray(anim["poses"], dtype=np.float32)
    transl = np.asarray(anim["transl"], dtype=np.float32)
    if "identity_coeffs" in anim:
        identity_coeffs = np.asarray(anim["identity_coeffs"], dtype=np.float32)
        if identity_coeffs.ndim == 1:
            identity_coeffs = identity_coeffs[None]
    coeffs = identity_coeffs[args.frame] if identity_coeffs.shape[0] > 1 else identity_coeffs[0]

    params = SOMAParams(
        poses=jnp.asarray(poses[args.frame]),
        transl=jnp.asarray(transl[args.frame]),
        identity_coeffs=jnp.asarray(coeffs),
    )
    out = layer(params)
    return np.array(out.vertices), np.array(layer.faces), np.array(out.joints)


def main():
    args = parse_args()
    _check_deps()

    import trimesh
    import pyrender
    import jax.numpy as jnp
    from soma_jax import SOMALayer, SOMAParams

    layer = SOMALayer.load(args.soma_model, identity_model_type=args.identity_model)
    vertices, faces, joints = _compute_mesh(args, jnp, layer, SOMAParams)

    # Build trimesh
    tm = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False,
    )
    if not args.no_smooth:
        # Compute vertex normals for smooth shading
        tm.vertex_normals  # populates _cache

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*args.color, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=not args.no_smooth)

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh)

    # Camera positioned to look at mesh center
    center = vertices.mean(axis=0)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = center + np.array([0.0, 0.2, args.camera_distance])
    scene.add(cam, pose=cam_pose)

    # Two directional lights
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = center + np.array([2.0, 2.0, 2.0])
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = center + np.array([-2.0, 1.0, 1.5])
    scene.add(light, pose=light_pose1)
    scene.add(light, pose=light_pose2)

    if args.interactive:
        pyrender.Viewer(scene, use_raymond_lighting=True)
    else:
        if args.output is None:
            args.output = "soma_render.png"
        renderer = pyrender.OffscreenRenderer(args.width, args.height)
        color, _ = renderer.render(scene)
        try:
            from PIL import Image
            Image.fromarray(color).save(args.output)
        except ImportError:
            import imageio.v2 as imageio
            imageio.imwrite(args.output, color)
        renderer.delete()
        print(f"Saved render: {args.output}")
        print(f"  Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")


if __name__ == "__main__":
    main()
