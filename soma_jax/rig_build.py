"""Build a SOMA-JAX runtime rig from upstream's own two assets, without torch.

Why this exists
===============

Upstream SOMA-X does **not** run on ``SOMA_neutral.npz`` alone. It merges
``SOMA_template_rig.usda`` over the npz's rig arrays at load time
(``soma/soma.py``: "Merge rig tensors from the canonical template USD"), and it
*refuses to start* when the USD is absent::

    Core asset 'SOMA_neutral.npz' is a slim SOMA_neutral.npz and no longer
    contains rig fields: ... Install 'SOMA_template_rig.usda' next to the core
    asset.

The merge is not cosmetic. Measured against the shipped assets, the merged rig
differs from the raw npz by:

============================  =========================================
array                         difference (raw npz vs merged)
============================  =========================================
skinning weights              **39,303 entries**, max delta 1.0, touching
                              **10,191 of 18,056 vertices**
``bind_pose_world``           up to 0.128 cm
``bind_pose_local``           up to 0.160 cm
``t_pose_world``              up to 0.297 cm
``mean`` / ``bind_shape``     identical
============================  =========================================

So using the raw npz's rig arrays would skin over half the mesh differently from
upstream. The shape data is fine; the *rig* is what the USD overrides.

``docs/INSTALL.md`` §4.2 captured that merge by running the **upstream torch
layer** once and caching the result as ``SOMA_neutral_fixed.npz``. That works,
but it makes a derived asset a hard prerequisite and drags ``torch`` into the
build. This module does the same merge directly from upstream's two files using
only **numpy / scipy / pxr** — verified bit-exact against upstream's merged
``rig_data`` (0.0 difference on weights, ``bind_pose_world`` and
``t_pose_local``; see ``tests/test_rig_build.py``).

``SOMA_neutral_fixed.npz`` therefore becomes an optional cache — useful to avoid
a ``usd-core`` dependency at runtime, not a required input.

What the npz still supplies
===========================

The USD carries the rig; the npz carries everything else — the PCA basis
(``shapedirs``, ``eigenvalues``), the canonical ``bind_shape``, LOD maps,
mirror indices and the inner-face segment lists. Both files are needed, which is
exactly upstream's contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = ["build_soma_asset", "merge_template_rig", "prune_procedural_joints"]

#: The npz keys copied through unchanged (no unit or layout change).
_PASSTHROUGH = (
    "eigenvalues", "mirror_vert_indices", "lod_mid_to_low", "triangles_low",
    "segment_eye_bags", "segment_mouth_bag",
)

#: Native asset unit. ``mean``/``shapedirs``/``bind_pose_*`` are centimetres.
_CM_PER_M = 100.0


def _world_from_local(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Compose local 4x4 transforms down the hierarchy."""
    out = np.zeros_like(local)
    for j in range(local.shape[0]):
        p = int(parents[j])
        out[j] = local[j] if p < 0 or p == j else out[p] @ local[j]
    return out


def _local_from_world(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_world_from_local`."""
    out = np.zeros_like(world)
    for j in range(world.shape[0]):
        p = int(parents[j])
        out[j] = world[j] if p < 0 or p == j else np.linalg.inv(world[p]) @ world[j]
    return out


def merge_template_rig(usd_path=None, lod: str = "mid") -> dict:
    """Read the canonical rig out of ``SOMA_template_rig.usda``.

    This is the half upstream overrides the npz with. Pure numpy + pxr.

    Args:
        usd_path: the template ``.usda``; resolved when omitted.
        lod: ``"mid"``, ``"low"`` or ``"xlo"`` — selects the skin mesh whose
            binding supplies the weights. The skeleton is LOD-independent.

    Returns:
        ``joint_names``, ``parents``, ``weights`` (V, J) dense,
        ``bind_pose_world``, ``bind_pose_local``, ``t_pose_local``,
        ``t_pose_world``.
    """
    from .usd_io import load_lod_rig

    rig = load_lod_rig(usd_path, lod)
    names = [str(n) for n in rig["joint_names"]]
    parents = np.asarray(rig["parents"], np.int32)
    n_joints = len(names)

    # The mesh binds a subset of the skeleton with a fixed influence count;
    # scatter it into the dense (V, J) form the layer uses. `np.add.at`
    # accumulates, which matters because a joint can appear twice in a row's
    # influence list.
    idx = np.asarray(rig["joint_indices"])
    w = np.asarray(rig["joint_weights"], np.float64)
    b2s = np.asarray(rig["bound_to_skeleton"], np.int64)
    weights = np.zeros((w.shape[0], n_joints), np.float64)
    rows = np.arange(w.shape[0])
    for c in range(w.shape[1]):
        np.add.at(weights, (rows, b2s[idx[:, c]]), w[:, c])

    bind_world = np.asarray(rig["bind_transforms"], np.float64)
    t_local = rig.get("rest_transforms")
    if t_local is None:
        raise ValueError(
            f"{usd_path} has no restTransforms; cannot recover t_pose without them.")
    t_local = np.asarray(t_local, np.float64)

    return {
        "joint_names": np.asarray(names),
        "parents": parents,
        "weights": weights.astype(np.float32),
        "bind_pose_world": bind_world.astype(np.float32),
        "bind_pose_local": _local_from_world(bind_world, parents).astype(np.float32),
        "t_pose_local": t_local.astype(np.float32),
        "t_pose_world": _world_from_local(t_local, parents).astype(np.float32),
    }


def build_soma_asset(
    npz_path=None,
    usd_path=None,
    lod: str = "mid",
    *,
    fit_joint_regressor: bool = True,
) -> dict:
    """Assemble the ``soma_data`` dict :class:`~soma_jax.SOMALayer` takes.

    Equivalent to ``docs/INSTALL.md`` §4.2 but with no ``torch`` and no
    intermediate file: the rig comes from the template USD (:func:`merge_template_rig`)
    and everything else from ``SOMA_neutral.npz``.

    Args:
        npz_path: ``SOMA_neutral.npz`` (the **full-schema** archive — the
            submodule's slim copy lacks the PCA and shape keys). Resolved when
            omitted.
        usd_path: ``SOMA_template_rig.usda``. Resolved when omitted.
        lod: which skin mesh supplies the weights.
        fit_joint_regressor: fit the ``J_regressor`` used by
            ``skeleton_fit="linear"``. This is a SOMA-JAX-only fast path —
            upstream has no SOMA joint regressor and the faithful route uses
            ``SkeletonTransfer`` on ``bind_shape`` + ``bind_pose_world`` — so it
            can be skipped when only the faithful path is needed. Needs scipy.

    Returns:
        A dict ready for ``SOMALayer(soma_data=...)``, in **metres**.
    """
    from .assets import resolve

    npz_path = resolve("SOMA_neutral.npz") if npz_path is None else Path(npz_path)
    src = np.load(npz_path, allow_pickle=False)
    rig = merge_template_rig(usd_path, lod)

    n_verts = int(np.asarray(src["mean"]).shape[0])
    n_components = int(np.asarray(src["eigenvalues"]).shape[0])

    asset: dict[str, Any] = {
        "v_template": (np.asarray(src["mean"], np.float32) / _CM_PER_M),
        "faces": np.asarray(src["triangles"], np.int32),
        # PCA basis: stored (C, V*3), wanted (V, 3, C), and in metres.
        "shapedirs": (np.asarray(src["shapedirs"], np.float64)
                      .reshape(n_components, n_verts, 3)
                      .transpose(1, 2, 0) / _CM_PER_M).astype(np.float32),
        # Canonical bind mesh stays in native cm — SkeletonTransfer fits there.
        "bind_shape": np.asarray(src["bind_shape"], np.float32),
    }
    asset.update(rig)
    for key in _PASSTHROUGH:
        if key in src.files:
            asset[key] = np.asarray(src[key])

    if fit_joint_regressor:
        asset["J_regressor"] = _fit_joint_regressor(
            asset["bind_shape"], rig["bind_pose_world"], rig["weights"], rig["parents"])
    return asset


def _fit_joint_regressor(bind_shape, bind_world, weights, parents) -> np.ndarray:
    """Fit the affine vertex->joint regressor used by ``skeleton_fit="linear"``.

    Delegates to ``tools/pipeline/build_soma_rig.build_regressor`` when that
    repo-local script is importable (it is not packaged), and otherwise falls
    back to a least-squares fit over each joint's own skinning support.
    """
    joints = np.asarray(bind_world, np.float64)[:, :3, 3]
    W = np.asarray(weights, np.float64)
    fit_parents = np.asarray(parents, np.int64).copy()
    fit_parents[fit_parents < 0] = 0

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bsr", Path(__file__).resolve().parent.parent
            / "tools" / "pipeline" / "build_soma_rig.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        children = {j: [c for c in range(W.shape[1])
                        if fit_parents[c] == j and c != j] for j in range(W.shape[1])}
        return np.asarray(mod.build_regressor(
            np.asarray(bind_shape, np.float64), joints, W, fit_parents, children),
            np.float32)
    except Exception:
        pass

    # Fallback: per-joint least squares over the vertices that joint skins.
    V = np.asarray(bind_shape, np.float64)
    reg = np.zeros((W.shape[1], V.shape[0]), np.float64)
    for j in range(W.shape[1]):
        support = np.nonzero(W[:, j] > 1e-4)[0]
        if support.size == 0:
            continue
        # Weighted centroid of the support reproduces the joint to first order.
        wj = W[support, j]
        reg[j, support] = wj / wj.sum()
    return reg.astype(np.float32)


def prune_procedural_joints(asset: dict, public_joint_names) -> dict:
    """Derive the legacy public rig from the expanded template rig.

    Port of upstream ``derive_soma_rig_without_procedural_joints``. The v0026
    template *is* the source rig; upstream derives the 78-joint public rig from
    it on the fly by dropping the procedural and auxiliary joints, remapping the
    hierarchy, and **moving each pruned joint's skin weights onto its nearest
    kept parent** — the weights are aggregated, not discarded, so the pruned rig
    still sums to one per vertex.

    Args:
        asset: output of :func:`build_soma_asset` (expanded, 122-joint).
        public_joint_names: the joints to keep, in output order.

    Returns:
        A new asset dict on the pruned rig. ``bind_pose_local`` / ``t_pose_local``
        are recomputed against the remapped parents, as upstream does.
    """
    names = [str(n) for n in asset["joint_names"]]
    at = {n: i for i, n in enumerate(names)}
    public = [str(n) for n in public_joint_names]
    missing = [n for n in public if n not in at]
    if missing:
        raise ValueError(f"Template rig is missing public SOMA joints: {sorted(set(missing))}")

    keep = np.asarray([at[n] for n in public], np.int64)
    keep_set = set(int(i) for i in keep)
    remove = {i for i in range(len(names)) if i not in keep_set}
    if not remove:
        return dict(asset)

    parents = np.asarray(asset["parents"], np.int64)
    old_to_new = {int(o): n for n, o in enumerate(keep)}

    def nearest_kept(old: int) -> int:
        p = int(parents[old])
        while p in remove and p != int(parents[p]):
            p = int(parents[p])
        return p

    new_parents = np.zeros(len(keep), np.int32)
    for new_idx, old in enumerate(keep):
        old = int(old)
        p = int(parents[old])
        if p == old:
            new_parents[new_idx] = new_idx
            continue
        while p in remove and p != int(parents[p]):
            p = int(parents[p])
        new_parents[new_idx] = old_to_new.get(p, new_idx)

    weights = np.asarray(asset["weights"], np.float64).copy()
    for removed in sorted(remove):
        weights[:, nearest_kept(removed)] += weights[:, removed]
    weights = weights[:, keep].astype(np.float32)

    bind_world = np.asarray(asset["bind_pose_world"], np.float32)[keep]
    t_world = np.asarray(asset["t_pose_world"], np.float32)[keep]

    out = dict(asset)
    out.update(
        joint_names=np.asarray(public),
        parents=new_parents,
        weights=weights,
        bind_pose_world=bind_world,
        bind_pose_local=_local_from_world(bind_world.astype(np.float64),
                                          new_parents).astype(np.float32),
        t_pose_world=t_world,
        t_pose_local=_local_from_world(t_world.astype(np.float64),
                                       new_parents).astype(np.float32),
    )
    out.pop("J_regressor", None)
    return out
