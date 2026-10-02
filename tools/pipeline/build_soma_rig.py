"""Build SOMA-JAX's runtime archive (``SOMA_neutral_fixed.npz``) from upstream's assets.

The archive is a SOMA-JAX-only cache: the 78-joint public rig in the layout
``SOMALayer.load`` reads, so a runtime can skip ``usd-core``. It is built from
the vendored SOMA-X assets with :func:`soma_jax.rig_build.save_runtime_archive`
— the rig from ``SOMA_template_rig.usda`` (SOMA-X v0.3 asset contract: the npz
no longer carries one), pruned to the public joints as upstream's
``derive_soma_rig_without_procedural_joints`` does, plus shape PCA, LOD maps,
segments and the reference-pose history from ``SOMA_neutral.npz``, plus the
affine ``J_regressor`` below for ``skeleton_fit="linear"``.

Rebuild it whenever the submodule moves: a cache built from an older template
silently reproduces that template's rig.

Usage::

    python tools/pipeline/build_soma_rig.py                    # -> assets/SOMA_neutral_fixed.npz
    python tools/pipeline/build_soma_rig.py --out other.npz

``build_regressor`` is also what ``soma_jax.rig_build`` uses for the regressor:
a linear+affine joint regressor restricted to each joint's skinning support,
reproducing ``bind_pose_world`` from ``bind_shape``. Upstream has no SOMA joint
regressor (it fits joints with ``SkeletonTransfer``); this is a SOMA-JAX extra.
"""
from __future__ import annotations
import argparse
import numpy as np


def _fit_affine_row(verts_support, target):
    """Min-norm affine weights a s.t. a@verts==target and sum(a)==1."""
    n = verts_support.shape[0]
    A = np.vstack([verts_support.T, np.ones(n)])           # (4, n)
    b = np.concatenate([target, [1.0]])                    # (4,)
    return A.T @ np.linalg.solve(A @ A.T + 1e-9 * np.eye(4), b)


def build_regressor(bind_shape, joints_t, W, parents, children, weight_thr=1e-4):
    V, J = W.shape
    Jreg = np.full((J, V), np.nan, np.float64)

    def support(j):
        m = W[:, j] > weight_thr
        if parents[j] != j:
            m = m & (W[:, parents[j]] > weight_thr)        # bone "tube" between j and parent
        if m.sum() < 4:
            m = W[:, j] > weight_thr                        # fall back to joint-only weight
        if m.sum() < 4:                                     # union of children's weight (e.g. Root)
            for c in children[j]:
                m = m | (W[:, c] > weight_thr)
        return np.where(m)[0]

    for j in range(J):
        ids = support(j)
        if len(ids) >= 4:
            Jreg[j, ids] = _fit_affine_row(bind_shape[ids], joints_t[j])

    # Leaf / zero-weight joints (End markers, eyes, jaw on some rigs): build_regressor
    # leaves their whole row NaN (no support → no affine fit). For these we fit a
    # NEAREST-VERTEX row that places the joint at its canonical bind position via a
    # single closest mesh vertex — that way `J_reg @ rest_verts` for any identity
    # gives the eye/jaw/tip joint a sensible position on (or very near) the head/
    # hand/foot instead of collapsing to the world origin. We use `.all()` not
    # `.any()` because a normal joint's affine row only fills its support
    # vertices, leaving the rest NaN; only fully-NaN rows are the empty ones.
    empty_rows = np.array([bool(np.isnan(Jreg[j]).all()) for j in range(J)])
    for j in np.where(empty_rows)[0]:
        canonical_j = joints_t[j]
        # Closest bind vertex to the canonical joint position.
        nearest = int(np.argmin(np.linalg.norm(bind_shape - canonical_j[None], axis=1)))
        Jreg[j] = 0.0
        Jreg[j, nearest] = 1.0
    Jreg = np.nan_to_num(Jreg, nan=0.0)
    return Jreg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="assets/SOMA_neutral_fixed.npz")
    p.add_argument("--npz", default=None, help="SOMA_neutral.npz (default: resolved)")
    p.add_argument("--usd", default=None, help="SOMA_template_rig.usda (default: resolved)")
    p.add_argument("--no-regressor", action="store_true",
                   help="skip the skeleton_fit='linear' regressor")
    args = p.parse_args()

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from soma_jax.rig_build import build_runtime_archive

    asset = build_runtime_archive(args.npz, args.usd,
                                  fit_joint_regressor=not args.no_regressor)
    if "J_regressor" in asset:
        joints = asset["bind_pose_world"][:, :3, 3].astype(np.float64)
        err = np.linalg.norm(asset["J_regressor"] @ asset["bind_shape"].astype(np.float64)
                             - joints, axis=1)
        print(f"J_regressor fit error: mean {err.mean():.3f} cm  max {err.max():.3f} cm")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **asset)
    print(f"Wrote {out}: {len(asset['joint_names'])} public joints, "
          f"{asset['v_template'].shape[0]} vertices, {len(asset)} arrays")


if __name__ == "__main__":
    main()
