"""I/O for parametric body model files (.pkl and .npz).

Supports loading SMPL, SMPL-X, SMPL-H, MHR, and Anny model files in their
common storage formats. Standardizes the returned data to a common dict
shape regardless of input format.

The returned dict has these keys:
    v_template:  (V, 3) rest template
    shapedirs:   (V, 3, K) PCA shape basis
    posedirs:    (V*3, P) flat pose-corrective basis (or None)
    exprdirs:    (V, 3, E) expression basis (SMPL-X only; None otherwise)
    J_regressor: (J, V) joint regressor
    parents:     (J,) parent indices
    weights:     (V, J) skinning weights
    faces:       (F, 3) triangle faces
    hands_meanl: (45,) MANO left hand pose mean (SMPL-X/SMPL-H; None otherwise)
    hands_meanr: (45,) MANO right hand pose mean

Upstream: ``soma/_smpl_family_loader.py``
    Faithful port of that code. .pkl/.npz loading for the SMPL family.
"""
from __future__ import annotations
import pickle
from typing import Any
import numpy as np


# Official SMPL-X files ship shape and expression blend shapes concatenated
# into a single `shapedirs` block: the first 300 columns are the identity
# (beta) basis and everything after is the expression basis. Those files carry
# no `expr_dirs` key, so the split has to be applied on load.
_SMPLX_SHAPE_COMPONENTS = 300


def _get(raw: Any, key: str, default=None):
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


class _ChArray:
    """Chumpy-free stand-in for a pickled ``chumpy.Ch`` array.

    Classic SMPL ``.pkl`` files serialize their arrays as ``chumpy`` objects
    whose concrete values live under ``state['x']``. Unpickling those normally
    imports chumpy — which the repo deliberately does not depend on (and which
    is broken on numpy >= 1.24, where ``np.bool`` was removed). This stub
    captures the pickled state and exposes the underlying ndarray via the array
    protocol, so the loader's ``np.asarray(...)`` calls work with no chumpy.
    """

    def __setstate__(self, state):
        self._arr = np.asarray(state["x"] if isinstance(state, dict) else state)

    def __array__(self, dtype=None):
        return np.asarray(self._arr, dtype=dtype)


class _ChumpyFreeUnpickler(pickle.Unpickler):
    """Unpickler that swaps every ``chumpy`` class for :class:`_ChArray`, so SMPL
    ``.pkl`` files load without importing the chumpy package."""

    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _ChArray
        return super().find_class(module, name)


def _load_pickle(path: str) -> Any:
    """Load a pickled SMPL/SMPL-X/SMPL-H file (Python-2 ``latin1`` encoding),
    converting any chumpy arrays to plain ndarrays so the repo never needs the
    chumpy package."""
    with open(path, "rb") as f:
        return _ChumpyFreeUnpickler(f, encoding="latin1").load()


def _load_npz(path: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def parent_ids_from_kintree(kintree_table) -> np.ndarray:
    """Convert an SMPL-family ``kintree_table`` into parent **column indices**.

    Port of upstream ``soma._smpl_family_loader.parent_ids_from_kintree``. Row 0
    holds parent *joint ids* and row 1 the child *joint ids*; the parent of a
    joint is the column whose child id matches. Taking row 0 verbatim is only
    correct when the child ids happen to be ``0..J-1`` — true of the stock SMPL
    and SMPL-X files, and silently wrong for any file that numbers joints
    differently.

    The root's sentinel (commonly ``J``, ``-1`` or a self-reference) is
    normalised to ``-1``.

    Args:
        kintree_table: (2, J) array-like.

    Returns:
        (J,) int32 parent column indices, root = -1.
    """
    kintree = np.asarray(kintree_table, dtype=np.int64)
    if kintree.ndim != 2 or kintree.shape[0] != 2:
        raise ValueError(f"Expected kintree_table shape (2, J), got {kintree.shape}.")
    child_ids = kintree[1]
    id_to_col = {int(j): i for i, j in enumerate(child_ids)}
    parents = np.full(child_ids.shape[0], -1, dtype=np.int32)
    for idx in range(1, child_ids.shape[0]):
        pid = int(kintree[0, idx])
        if pid not in id_to_col:
            raise ValueError(
                f"Parent joint id {pid} (column {idx}) is not among the kintree children.")
        parents[idx] = id_to_col[pid]
    parents[0] = -1
    return parents

def load_smpl_data(path: str) -> dict:
    """Load an SMPL-family model file (.pkl or .npz) into a standard dict.

    Args:
        path: path to the model file.

    Returns:
        Dict with standardized keys. See module docstring.
    """
    path_lower = path.lower()
    if path_lower.endswith(".pkl") or path_lower.endswith(".pickle"):
        raw = _load_pickle(path)
    elif path_lower.endswith(".npz"):
        raw = _load_npz(path)
    else:
        # Try pickle first, then npz
        try:
            raw = _load_pickle(path)
        except Exception:
            raw = _load_npz(path)

    v_template = np.asarray(_get(raw, "v_template"), dtype=np.float32)
    V = v_template.shape[0]

    # Shape directions: may be (V, 3, K) or flat (V*3, K)
    shapedirs = np.asarray(_get(raw, "shapedirs"), dtype=np.float32)
    if shapedirs.ndim == 2:
        shapedirs = shapedirs.reshape(V, 3, -1)

    # Pose directions: may be (V, 3, P), (V*3, P) or (P, V*3)
    posedirs_raw = _get(raw, "posedirs")
    posedirs: np.ndarray | None
    if posedirs_raw is not None:
        posedirs_raw = np.asarray(posedirs_raw, dtype=np.float32)
        if posedirs_raw.ndim == 3:
            posedirs = posedirs_raw.reshape(V * 3, -1)
        elif posedirs_raw.shape[0] == V * 3:
            posedirs = posedirs_raw
        else:
            posedirs = posedirs_raw.T
    else:
        posedirs = None

    # Joint regressor: may be sparse (scipy) or dense
    J_reg_raw = _get(raw, "J_regressor")
    if J_reg_raw is None:
        # Some MHR files use 'J_regressor_h' or similar
        J_reg_raw = _get(raw, "J_regressor_prior")
    if hasattr(J_reg_raw, "todense"):
        J_regressor = np.asarray(J_reg_raw.todense(), dtype=np.float32)
    elif hasattr(J_reg_raw, "toarray"):
        J_regressor = np.asarray(J_reg_raw.toarray(), dtype=np.float32)
    else:
        J_regressor = np.asarray(J_reg_raw, dtype=np.float32)

    # Kintree / parents
    kintree = _get(raw, "kintree_table")
    if kintree is not None:
        parents = parent_ids_from_kintree(kintree)
    else:
        parents = np.asarray(_get(raw, "parents", []), dtype=np.int32)

    weights = np.asarray(_get(raw, "weights"), dtype=np.float32)
    faces = np.asarray(_get(raw, "f", _get(raw, "faces")), dtype=np.int32)

    # Expression blend shapes (SMPL-X only)
    exprdirs_raw = _get(raw, "expr_dirs", _get(raw, "exprdirs"))
    exprdirs: np.ndarray | None
    if exprdirs_raw is not None:
        exprdirs = np.asarray(exprdirs_raw, dtype=np.float32)
        if exprdirs.ndim == 2:
            exprdirs = exprdirs.reshape(V, 3, -1)
    elif shapedirs.shape[-1] > _SMPLX_SHAPE_COMPONENTS:
        # Official SMPL-X layout: split the packed basis into shape + expression.
        exprdirs = shapedirs[..., _SMPLX_SHAPE_COMPONENTS:]
        shapedirs = shapedirs[..., :_SMPLX_SHAPE_COMPONENTS]
    else:
        exprdirs = None

    # MANO hand pose means (SMPL-H / SMPL-X)
    hands_meanl_raw = _get(raw, "hands_meanl")
    hands_meanr_raw = _get(raw, "hands_meanr")
    hands_meanl = (
        np.asarray(hands_meanl_raw, dtype=np.float32).reshape(-1)
        if hands_meanl_raw is not None else None
    )
    hands_meanr = (
        np.asarray(hands_meanr_raw, dtype=np.float32).reshape(-1)
        if hands_meanr_raw is not None else None
    )

    # Optional MHR body-part vertex IDs
    part_vertex_ids: dict[str, np.ndarray] = {}
    pvids_raw = _get(raw, "part_vertex_ids")
    if pvids_raw is not None:
        if isinstance(pvids_raw, dict):
            part_vertex_ids = {k: np.asarray(v, dtype=np.int32) for k, v in pvids_raw.items()}

    return dict(
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        exprdirs=exprdirs,
        J_regressor=J_regressor,
        parents=parents,
        weights=weights,
        faces=faces,
        hands_meanl=hands_meanl,
        hands_meanr=hands_meanr,
        part_vertex_ids=part_vertex_ids,
    )


def save_smpl_data(path: str, data: dict) -> None:
    """Save standardized body model data to an NPZ file.

    Args:
        path: output file path.
        data: dict from load_smpl_data (or compatible).
    """
    arrays = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, dict):
            # Skip nested dicts (part_vertex_ids) — write as separate items
            for sk, sv in v.items():
                arrays[f"{k}__{sk}"] = sv
        else:
            arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)
