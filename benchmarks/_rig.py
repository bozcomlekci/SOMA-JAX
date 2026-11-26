"""The rig both sides of the benchmark skin with.

The SOMA-X side times ``SOMALayer(enable_procedural_transforms=False)``, whose
78-joint public rig comes from ``SOMA_template_rig.usda``: since SOMA-X v0.3,
``SOMA_neutral.npz`` holds shape and topology data only. The JAX pipelines read
the same rig through :func:`soma_jax.rig_build.load_public_rig` (bind transforms
and parents identical to upstream's ``rig_data``, pruned weights to 6e-8) and
the same shape data from the npz, so both sides skin the same rig with the same
FLOPs and produce the same meshes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_SHAPE_KEYS = ("mean", "shapedirs", "eigenvalues", "segment_eye_bags", "segment_mouth_bag",
               "triangles")


def public_rig(asset_dir) -> dict:
    """Upstream's public rig (centimetres) plus the core asset's shape data.

    Args:
        asset_dir: an upstream-layout asset directory holding ``SOMA_neutral.npz``
            and ``SOMA_template_rig.usda`` (``soma_jax.assets.data_root()``).

    Returns:
        ``joint_names``, ``parents`` (root self-parented), dense ``weights``,
        ``bind_pose_world`` / ``bind_pose_local`` / ``t_pose_world`` /
        ``t_pose_local``, ``bind_shape``, and the npz's ``mean``, ``shapedirs``,
        ``eigenvalues``, ``segment_eye_bags``, ``segment_mouth_bag``, ``triangles``.
    """
    from soma_jax.rig_build import load_public_rig

    asset_dir = Path(asset_dir)
    rig = load_public_rig(asset_dir / "SOMA_neutral.npz",
                          usd_path=asset_dir / "SOMA_template_rig.usda")
    with np.load(asset_dir / "SOMA_neutral.npz", allow_pickle=False) as core:
        rig.update({key: np.asarray(core[key]) for key in _SHAPE_KEYS})
    return rig
