"""MANO asset loading helpers for hand-only SOMA-JAX code.

Upstream: ``soma/hand/_smpl_family_loader.py`` (SOMA-X v0.3.0).

MANO's ``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl`` are licensed separately and are not
shipped by upstream or here; place them under ``data_root/MANO/`` or pass
``model_path``. Pickles are read with SOMA-JAX's chumpy-free unpickler
(:mod:`soma_jax.body_models.model_io`), so the ``chumpy`` package is not needed —
upstream installs a chumpy stub for the same purpose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..body_models.model_io import _load_npz, _load_pickle, parent_ids_from_kintree


def _read_model_file(model_path: Path) -> dict[str, Any]:
    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(str(model_path))
    if suffix == ".pkl":
        return _load_pickle(str(model_path))
    raise ValueError(f"Unsupported SMPL-family model file extension: {model_path.suffix!r}.")


def _get_required(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"SMPL-family model is missing required key; tried {keys}.")


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "toarray"):          # scipy sparse (MANO's J_regressor)
        value = value.toarray()
    return np.asarray(value, dtype=dtype)


def load_mano_pkl(
    data_root: str | Path,
    hand_type: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load MANO pickle data as plain NumPy arrays."""
    if hand_type not in ("left", "right"):
        raise ValueError(f"hand_type must be 'left' or 'right', got {hand_type!r}.")
    if model_path is None:
        pkl_path = Path(data_root) / "MANO" / f"MANO_{hand_type.upper()}.pkl"
    else:
        pkl_path = Path(model_path).expanduser()
        if not pkl_path.is_file():
            raise FileNotFoundError(f"MANO model not found at '{pkl_path}'")
    data = _read_model_file(pkl_path)
    return {
        "v_template": _to_numpy(_get_required(data, "v_template"), np.float32),
        "shapedirs": _to_numpy(_get_required(data, "shapedirs"), np.float32),
        "J_regressor": _to_numpy(_get_required(data, "J_regressor"), np.float32),
        "weights": _to_numpy(_get_required(data, "weights"), np.float32),
        "kintree_table": _to_numpy(_get_required(data, "kintree_table"), np.int64),
        "faces": _to_numpy(_get_required(data, "f"), np.int64),
        "posedirs": _to_numpy(_get_required(data, "posedirs"), np.float32),
        "hands_mean": _to_numpy(_get_required(data, "hands_mean"), np.float32),
        "hands_components": _to_numpy(_get_required(data, "hands_components"), np.float32),
    }


def mano_parent_ids(kintree_table: np.ndarray) -> list[int]:
    """Convert a MANO kintree table into parent column indices (root = 0).

    Upstream's convention: the root's parent is itself (index 0).
    """
    parents = parent_ids_from_kintree(kintree_table).astype(np.int64)
    parents[0] = 0
    return parents.tolist()
