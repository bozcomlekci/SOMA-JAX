"""SMPL-family asset loading helpers.

Upstream: ``soma/_smpl_family_loader.py`` (SOMA-X v0.3.3).

``load_smpl_family_model`` (ported in :mod:`soma_jax.smpl.layers`) reads
SMPL / SMPL-H / SMPL-X files into plain NumPy arrays, and
``parent_ids_from_kintree`` converts their kintree tables. SOMA-JAX unpickles
the files with a chumpy-free unpickler (:mod:`soma_jax.body_models.model_io`),
so nothing here needs to be called first; :func:`ensure_chumpy_compat`
installs upstream's process-wide shims for callers that ``pickle.load`` the
legacy files themselves.
"""
from __future__ import annotations

import inspect
import sys
import types
from collections import namedtuple
from typing import Any

import numpy as np

from .body_models.model_io import parent_ids_from_kintree as _parent_ids_root_minus_one
from .smpl.layers import load_smpl_family_model

__all__ = ["ensure_chumpy_compat", "load_smpl_family_model", "parent_ids_from_kintree"]


def parent_ids_from_kintree(kintree_table) -> np.ndarray:
    """Convert an SMPL-family kintree table into parent column indices.

    Upstream's convention: int64, the root (column 0) its own parent. (The
    SOMA-JAX helper in :mod:`soma_jax.body_models.model_io` marks it ``-1``.)
    """
    parents = _parent_ids_root_minus_one(kintree_table).astype(np.int64)
    parents[0] = 0
    return parents


def ensure_chumpy_compat() -> None:
    """Install compatibility shims needed by legacy Chumpy-backed model pickles."""
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec

    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    try:
        import chumpy  # noqa: F401
    except ModuleNotFoundError:
        _install_chumpy_pickle_stub()


def _install_chumpy_pickle_stub() -> None:
    """Install the small Chumpy subset needed to unpickle legacy model shapedirs."""

    class Ch:
        def __setstate__(self, state: dict[str, Any]) -> None:
            self.__dict__.update(state)

        @property
        def r(self) -> np.ndarray:
            return np.asarray(self)

        @property
        def shape(self) -> tuple[int, ...]:
            return self.r.shape

        def __array__(self, dtype=None) -> np.ndarray:
            if not hasattr(self, "x"):
                raise TypeError("Unsupported Chumpy pickle object without 'x' data.")
            array = np.asarray(self.x)
            if dtype is not None:
                array = array.astype(dtype, copy=False)
            return array

    class Select(Ch):
        def __array__(self, dtype=None) -> np.ndarray:
            source = np.asarray(self.a).reshape(-1)
            result = source[np.asarray(self.idxs, dtype=np.int64)]
            if getattr(self, "preferred_shape", None) is not None:
                result = result.reshape(self.preferred_shape)
            if dtype is not None:
                result = result.astype(dtype, copy=False)
            return result

    Ch.__module__ = "chumpy.ch"
    Select.__module__ = "chumpy.reordering"

    chumpy_mod = types.ModuleType("chumpy")
    ch_mod = types.ModuleType("chumpy.ch")
    reordering_mod = types.ModuleType("chumpy.reordering")
    ch_mod.Ch = Ch
    reordering_mod.Select = Select
    chumpy_mod.Ch = Ch
    chumpy_mod.ch = ch_mod
    chumpy_mod.reordering = reordering_mod

    sys.modules.setdefault("chumpy", chumpy_mod)
    sys.modules.setdefault("chumpy.ch", ch_mod)
    sys.modules.setdefault("chumpy.reordering", reordering_mod)
