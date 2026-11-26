"""Helpers for tool-side SOMA rig asset loading.

Upstream: ``tools/soma_rig_assets.py`` (SOMA-X). ``load_public_mid_soma_rig``
returns the core ``SOMA_neutral.npz`` arrays plus the public (78-joint)
mid-LOD rig in the npz schema — ``joint_names``, ``joint_parent_ids`` (root
self-parented), bind/T-pose world and local transforms, ``bind_shape`` and CSC
``skinning_weights_*`` — derived from ``SOMA_template_rig.usda`` as upstream
derives it (the v0.3 npz ships no rig of its own).

The derivation goes through :func:`soma_jax.rig_build.load_public_rig`, the
port of upstream's template read + ``derive_soma_rig_without_procedural_joints``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SOMA_TEMPLATE_RIG_FILENAME = "SOMA_template_rig.usda"


def load_public_mid_soma_rig(data_root: str | Path, *,
                             template_rig_path: str | Path | None = None) -> dict[str, Any]:
    """Load SOMA core arrays plus public mid-LOD rig arrays for tools."""
    data_root = Path(data_root)
    template_rig_path = (data_root / SOMA_TEMPLATE_RIG_FILENAME if template_rig_path is None
                         else Path(template_rig_path))
    return dict(_load_public_mid_soma_rig_cached(str(data_root.resolve()),
                                                 str(template_rig_path.resolve())))


@lru_cache(maxsize=4)
def _load_public_mid_soma_rig_cached(data_root_str: str, template_rig_path_str: str):
    from scipy.sparse import csc_matrix

    from soma_jax.io import missing_soma_neutral_rig_keys
    from soma_jax.rig_build import load_public_rig

    data_root = Path(data_root_str)
    template_rig_path = Path(template_rig_path_str)
    core_asset = data_root / "SOMA_neutral.npz"
    if not core_asset.exists():
        raise FileNotFoundError(
            f"Core asset not found: {core_asset}\nRun 'git lfs pull' to fetch LFS-tracked files.")

    rig_data = dict(np.load(core_asset, allow_pickle=False))
    if template_rig_path.exists():
        pub = load_public_rig(core_asset, template_rig_path)
        weights = csc_matrix(np.asarray(pub["weights"], np.float32))
        rig_data.update(
            joint_names=np.asarray(pub["joint_names"]),
            joint_parent_ids=np.asarray(pub["parents"], np.int32),
            bind_pose_world=np.asarray(pub["bind_pose_world"], np.float32),
            bind_pose_local=np.asarray(pub["bind_pose_local"], np.float32),
            t_pose_world=np.asarray(pub["t_pose_world"], np.float32),
            t_pose_local=np.asarray(pub["t_pose_local"], np.float32),
            bind_shape=np.asarray(pub["bind_shape"], np.float32),
            skinning_weights_data=weights.data.astype(np.float32),
            skinning_weights_indices=weights.indices.astype(np.int32),
            skinning_weights_indptr=weights.indptr.astype(np.int32),
            skinning_weights_shape=np.asarray(weights.shape, np.int32),
        )
    else:
        missing = missing_soma_neutral_rig_keys(rig_data)
        if missing:
            raise FileNotFoundError(
                f"Template rig asset not found: {template_rig_path}. "
                f"Core asset '{core_asset}' is a slim SOMA_neutral.npz and no longer contains "
                f"rig fields: {', '.join(missing)}. Install "
                f"'{SOMA_TEMPLATE_RIG_FILENAME}' next to the core asset.")
    return rig_data
