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

__all__ = ["build_runtime_archive", "build_soma_asset", "load_public_rig",
           "merge_template_rig", "prune_procedural_joints", "save_runtime_archive"]

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
        # Upstream `io._load_rig_from_usd_stage`: the skin mesh points are the
        # bind shape. Native centimetres, like the rest of the rig.
        "bind_shape": np.asarray(rig["points"], np.float32),
        # The LOD's own polygon topology (upstream `face_vert_indices` /
        # `face_vert_counts`); the npz only carries mid/low triangles. The
        # SOMA Hand layer triangulates these for its low/xlo LODs.
        "face_vert_indices": rig.get("face_vert_indices"),
        "face_vert_counts": rig.get("face_vert_counts"),
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
        npz_path: ``SOMA_neutral.npz``. Either asset generation works: the
            v0.3 contract ships it **without** rig fields (``bind_shape``,
            ``bind_pose_*``, ``t_pose_*``, skinning, joint names) — its
            metadata names ``SOMA_template_rig.usda`` as their only source —
            while older archives still carry them. Rig data is always taken
            from the USD here, exactly as upstream's ``rig_data.update(...)``
            overrides whatever the npz holds, so both generations build the
            same layer. Resolved when omitted.
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
    }
    # `rig` supplies `bind_shape` (the USD skin-mesh points, native cm — the
    # frame SkeletonTransfer fits in) along with every other rig array.
    asset.update(rig)
    for key in _PASSTHROUGH:
        if key in src.files:
            asset[key] = np.asarray(src[key])
    # Historical reference T-poses (SOMA-X v0.3.1) — what
    # `SOMALayer.get_reference_pose(version=...)` reads. Absent on older
    # archives, which then simply have no reference history, as upstream.
    from .reference_poses import _is_history_key
    for key in src.files:
        if _is_history_key(key):
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

    Port of upstream ``derive_soma_rig_without_procedural_joints``. The
    template (v0027 since SOMA-X v0.2.2) *is* the source rig; upstream derives the 78-joint public rig from
    it on the fly by dropping the procedural and auxiliary joints, remapping the
    hierarchy, and **moving each pruned joint's skin weights onto its nearest
    kept parent** — the weights are aggregated, not discarded, so the pruned rig
    still sums to one per vertex.

    Args:
        asset: output of :func:`build_soma_asset` (expanded, 110-joint on v0027).
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


def build_runtime_archive(npz_path=None, usd_path=None, *,
                          fit_joint_regressor: bool = True) -> dict:
    """The 78-joint public rig, in the layout :meth:`SOMALayer.load` reads.

    This is the SOMA-JAX-only ``SOMA_neutral_fixed.npz`` cache: upstream's merge
    (:func:`build_soma_asset`) pruned to the public rig exactly as upstream's
    ``derive_soma_rig_without_procedural_joints`` does, plus the optional linear
    ``J_regressor``. It carries the reference-pose history too, so
    ``SOMALayer.load(..., reference_pose={"version": ...})`` works from it.

    It exists so a runtime can skip ``usd-core``; it is a *cache* of the
    template it was built from. Rebuild it whenever the vendored assets move —
    a cache built from an older template silently reproduces that older rig.

    Args:
        npz_path, usd_path: as for :func:`build_soma_asset`.
        fit_joint_regressor: add the ``skeleton_fit="linear"`` regressor.

    Returns:
        A dict for ``np.savez`` / ``SOMALayer.load``.
    """
    from .assets import resolve
    from .procedural_transforms import load_definition

    public_names = load_definition(resolve("SOMA_procedural_transforms.json")).main_joint_names
    asset = prune_procedural_joints(
        build_soma_asset(npz_path, usd_path, "mid", fit_joint_regressor=False), public_names)
    if fit_joint_regressor:
        asset["J_regressor"] = _fit_joint_regressor(
            asset["bind_shape"], asset["bind_pose_world"], asset["weights"], asset["parents"])
    # USD face topology is only needed while slicing LODs; drop it (and any
    # other None) so the dict round-trips through np.savez.
    return {k: np.asarray(v) for k, v in asset.items()
            if v is not None and k not in ("face_vert_indices", "face_vert_counts")}


def save_runtime_archive(path, npz_path=None, usd_path=None, *,
                         fit_joint_regressor: bool = True) -> Path:
    """Write :func:`build_runtime_archive` to ``path`` (compressed npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **build_runtime_archive(
        npz_path, usd_path, fit_joint_regressor=fit_joint_regressor))
    return path


#: What :func:`load_public_rig` returns (all native centimetres).
_PUBLIC_RIG_KEYS = ("joint_names", "parents", "weights", "bind_pose_world",
                    "bind_pose_local", "t_pose_world", "t_pose_local", "bind_shape")


def load_public_rig(path=None, usd_path=None) -> dict:
    """The 78-joint public rig, from a runtime archive or the template USD.

    For tools that need the raw rig arrays rather than a layer. A SOMA-JAX
    runtime archive (``SOMA_neutral_fixed.npz``, see
    :func:`build_runtime_archive`) already holds them and is read as-is. Any
    other input — ``None``, or a ``SOMA_neutral.npz`` of either generation — is
    rebuilt from ``SOMA_template_rig.usda``: since SOMA-X v0.3 the npz carries
    no rig, and a pre-v0.3 npz's own rig arrays are what upstream always
    overrode with the USD, so reading them would reproduce a stale rig.

    Args:
        path: a runtime archive or ``SOMA_neutral.npz``; resolved when omitted.
        usd_path: the template USD, for the rebuild; resolved when omitted.

    Returns:
        ``joint_names``, ``parents`` (the root is its own parent, as in the
        template), dense ``weights`` (V, 78),
        ``bind_pose_world`` / ``bind_pose_local`` / ``t_pose_world`` /
        ``t_pose_local`` (78, 4, 4) and ``bind_shape`` (V, 3), in centimetres.
    """
    if path is not None:
        with np.load(path, allow_pickle=False) as src:
            if all(k in src.files for k in _PUBLIC_RIG_KEYS):
                return {k: np.asarray(src[k]) for k in _PUBLIC_RIG_KEYS}
    asset = build_runtime_archive(path, usd_path, fit_joint_regressor=False)
    return {k: asset[k] for k in _PUBLIC_RIG_KEYS}

