"""MHR shape to SMPL shape converter.

Upstream: ``tools/shape_convert.py`` (SOMA-X v0.3.3). :class:`ShapeTransfer`
converts MHR shape parameters to SMPL betas through SOMA: the MHR rest shape
(SOMA topology) is skeleton-fitted, posed into the SMPL template's T-pose,
carried onto SMPL topology through the SMPL ``SOMA_wrap``, and the betas are
solved by least squares against the SMPL shape basis. The script renders random
smooth MHR identities next to their SMPL conversions.

Upstream v0.3.3 raises in :meth:`ShapeTransfer.forward` with its default
procedural layer: the pose-into-SMPL step builds a ``BatchedSkinning`` from the
layer's ``joint_parent_ids`` / ``skinning_weights`` (the 110-joint twist rig)
around the 78-joint ``skeleton_transfer.fit`` binds. The port skins on the
public 78-joint rig those binds belong to (upstream's
``enable_procedural_transforms=False`` legacy rig).

JAX port: ``--device`` is accepted and ignored; ``--seed`` (SOMA-JAX extra)
seeds the random identities, which upstream draws unseeded. ``--data-root`` is
accepted as an alias of upstream's ``--data_root``.

Usage::

    python tools/convert/shape_convert.py --sequence-length 300 --output-dir out/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def get_smooth_noise(T, dim, rng=None, num_keyframes=None, mode="normal"):
    """(T, dim) noise linearly interpolated between random keyframes
    (``F.interpolate(mode="linear", align_corners=True)``)."""
    rng = np.random.default_rng() if rng is None else rng
    if num_keyframes is None:
        num_keyframes = max(3, T // 30)
    if mode == "normal":
        keyframes = rng.standard_normal((dim, num_keyframes))
    elif mode == "uniform":
        keyframes = rng.random((dim, num_keyframes))
    else:
        raise ValueError(f"Unknown noise mode {mode!r}")
    x = np.linspace(0.0, num_keyframes - 1, T)
    xp = np.arange(num_keyframes)
    return np.stack([np.interp(x, xp, k) for k in keyframes], axis=1).astype(np.float32)


class ShapeTransfer:
    """MHR shape to SMPL shape converter."""

    def __init__(self, data_root, device="cuda"):
        import trimesh

        from soma_jax import SOMALayer

        del device
        self.data_root = Path(data_root)
        self.mhr_soma = SOMALayer.from_upstream_assets(identity_model_type="mhr",
                                                       data_root=str(self.data_root))
        self.smpl_soma = SOMALayer.from_upstream_assets(identity_model_type="smpl",
                                                        data_root=str(self.data_root))
        self.soma_to_smpl = self.get_soma_to_smpl_interpolator()

        smpl_rest_mesh = trimesh.load(self.data_root / "SMPL" / "base_body.obj",
                                      maintain_order=True, process=False)
        smpl_rest_shape = np.asarray(smpl_rest_mesh.vertices, np.float32)[None]
        self.smpl_rest_shape_soma = self.smpl_soma.identity_model.identity_model_to_soma(
            smpl_rest_shape)
        self.posed_world_smpl_tpose = self.mhr_soma.skeleton_transfer.fit(
            self.smpl_rest_shape_soma)

    def get_soma_to_smpl_interpolator(self):
        """Upstream ``BarycentricInterpolator(V_soma, F_soma, V_smpl)``: the SMPL
        base mesh embedded in the SOMA-topology wrap of SMPL."""
        import trimesh

        from soma_jax.geometry.barycentric_interp import compute_barycentric_coords
        mesh_smpl = trimesh.load(self.data_root / "SMPL" / "base_body.obj",
                                 maintain_order=True, process=False)
        mesh_soma = trimesh.load(self.data_root / "SMPL" / "SOMA_wrap.obj",
                                 maintain_order=True, process=False)
        V_smpl = np.asarray(mesh_smpl.vertices, np.float32)
        V_soma = np.asarray(mesh_soma.vertices, np.float32)
        F_soma = np.asarray(mesh_soma.faces, np.int32)
        face_ids, bary = compute_barycentric_coords(V_smpl, V_soma, F_soma)
        return F_soma, np.asarray(face_ids, np.int32), np.asarray(bary, np.float32)

    def _soma_to_smpl(self, vertices):
        import jax.numpy as jnp

        from soma_jax.geometry.barycentric_interp import barycentric_interpolate
        faces, face_ids, bary = self.soma_to_smpl
        return barycentric_interpolate(jnp.asarray(vertices), jnp.asarray(faces),
                                       jnp.asarray(face_ids), jnp.asarray(bary))

    def forward(self, identity_coeffs, scale_params):
        """MHR ``(identity_coeffs, scale_params)`` -> SMPL betas ``(B, 10)``."""
        import jax.numpy as jnp

        from soma_jax.geometry.batched_skinning import pose_from_bind, topk_skinning
        from soma_jax.geometry.lbs import compute_skeleton_levels

        identity_coeffs = jnp.asarray(identity_coeffs, jnp.float32)
        batch_size = identity_coeffs.shape[0]

        # 1. MHR rest shape (SOMA topology).
        mhr_rest_shape_soma = self.mhr_soma.identity_model(
            identity_coeffs, jnp.asarray(scale_params, jnp.float32))
        # 2. MHR bind: the skeleton fitted to it (public 78-joint rig).
        posed_world_mhr = self.mhr_soma.skeleton_transfer.fit(mhr_rest_shape_soma)

        # 3. Skin the MHR rest shape into the SMPL rest pose: identity local
        # rotations relative to the SMPL T-pose orient, hips at the SMPL hips.
        parents = np.asarray(self.mhr_soma._parents_np, np.int64)
        weights = np.asarray(self.mhr_soma.public_skinning_weights(), np.float32)
        indices, values = topk_skinning(weights, 8)
        orient = self.posed_world_smpl_tpose[0, :, :3, :3]
        # Upstream's precompute_joint_orient: orient[parent].T, the
        # self-parented root included.
        orient_parent_T = jnp.swapaxes(orient[parents], -2, -1)
        pose_rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32),
                                          (batch_size, len(parents), 3, 3))
        local = jnp.einsum("jrs,bjsk,jkt->bjrt", orient_parent_T, pose_rotations, orient)
        pose_translations = jnp.broadcast_to(self.posed_world_smpl_tpose[:, 1, :3, 3],
                                             (batch_size, 3))
        vertices, _ = pose_from_bind(
            posed_world_mhr, mhr_rest_shape_soma, jnp.asarray(weights),
            compute_skeleton_levels(parents), parents, local, pose_translations, hips_idx=1,
            weight_values=jnp.asarray(values), weight_indices=jnp.asarray(indices))

        # 4. SMPL topology.
        mhr_vertices_smpl = self._soma_to_smpl(vertices)

        # 5. Solve the betas against the SMPL shape basis.
        smpl_model = self.smpl_soma.identity_model.identity_model      # SMPLSimplified
        shape_dirs = smpl_model.shape_dirs
        B = (mhr_vertices_smpl - smpl_model.v_template[None]).reshape(batch_size, -1)
        A = shape_dirs.reshape(-1, shape_dirs.shape[-1])
        return jnp.linalg.lstsq(A, B.T)[0].T

    __call__ = forward


def main():
    import imageio

    import jax.numpy as jnp
    from tqdm import tqdm

    from soma_jax.assets import data_root as default_data_root
    from soma_jax.types import SOMAParams
    from vis_pyrender import MeshRenderer, look_at, set_pyopengl_platform

    parser = argparse.ArgumentParser(description="Shape transfer.")
    parser.add_argument("--data_root", "--data-root", dest="data_root", type=str, default=None,
                        help="Path to the data root (default: soma_jax.assets).")
    parser.add_argument("--device", default="cuda:0",
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--output-dir", default="out/")
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--pyopengl-platform", default="osmesa")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random identities (SOMA-JAX extra; upstream is "
                             "unseeded).")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    set_pyopengl_platform(args.pyopengl_platform)

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    shape_transfer = ShapeTransfer(data_root)
    T = args.sequence_length
    rng = np.random.default_rng(args.seed)

    mhr_im = shape_transfer.mhr_soma.identity_model
    identity_coeffs = jnp.asarray(get_smooth_noise(T, mhr_im.num_identity_coeffs, rng))
    scale_params = jnp.asarray(
        get_smooth_noise(T, mhr_im.num_scale_params, rng, mode="normal") * 0.2)
    zero_pose = jnp.zeros((T, 77, 3), jnp.float32)
    zero_transl = jnp.zeros((T, 3), jnp.float32)

    betas = shape_transfer(identity_coeffs, scale_params)

    smpl_vertices = shape_transfer.smpl_soma(
        SOMAParams(poses=zero_pose, transl=zero_transl, identity_coeffs=betas)).vertices
    mhr_vertices = shape_transfer.mhr_soma(
        SOMAParams(poses=zero_pose, transl=zero_transl, identity_coeffs=identity_coeffs,
                   scale_params=scale_params)).vertices
    smpl_vertices = np.asarray(smpl_vertices)
    mhr_vertices = np.asarray(mhr_vertices)

    logger.info("Rendering videos...")
    colors = {
        "mhr": (0.98, 0.65, 0.15, 1.0),
        "anny": (0.25, 0.75, 1.0, 1.0),
        "smpl": (0.55, 0.15, 0.85, 1.0),
    }

    def save_video(frames, path, fps=30):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=fps)
        logger.info(f"Saved {path}")

    renderer = MeshRenderer(image_size=args.image_size, light_intensity=5)
    cam_pose = look_at(eye=np.array([0.0, 0.0, 6.0]), target=np.array([0.0, 0.0, 0.0]),
                       up=np.array([0.0, 1.0, 0.0]))
    light_dir = np.array([0.0, -0.5, -1.0])
    faces = np.asarray(shape_transfer.mhr_soma.faces)
    render_kw = dict(cam_pose=cam_pose, light_dir=light_dir, metallic=0.0, roughness=0.5,
                     base_color_factor=[0.9, 0.9, 0.9, 1.0])
    frames = []
    for t in tqdm(range(T)):
        mhr_img = renderer.render(mhr_vertices[t], faces, mesh_color=colors["mhr"], **render_kw)
        smpl_img = renderer.render(smpl_vertices[t], faces, mesh_color=colors["smpl"],
                                   **render_kw)
        merged_img = (0.5 * mhr_img + 0.5 * smpl_img).astype(np.uint8)
        img = np.concatenate([mhr_img, merged_img, smpl_img], axis=1)
        frames.append(img[..., ::-1])
    renderer.delete()
    save_video(frames, Path(args.output_dir) / "shape_transfer.mp4")


if __name__ == "__main__":
    main()
