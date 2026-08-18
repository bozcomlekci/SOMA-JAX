"""Rebuild a correct SOMA joint regressor from the real NVlabs/SOMA-X rig.

The repackaged `SOMA_neutral.npz` shipped a synthesized `J_regressor` that places
joints up to ~1.5 m away from the true skeleton (e.g. Root near the head). This
corrupts the skeleton drawing AND the LBS bind (FK uses the rest joints), so every
SOMA-family identity (SOMA, MHR, Anny, Garment) poses incorrectly.

The HF `SOMA_neutral.npz` ships the real rig: `bind_pose_world` (J,4,4) joint
transforms, `bind_shape` (V,3), and sparse `skinning_weights` (V,J). SOMA-X derives
joints via skinning-masked RBF regressors (`SkeletonTransfer`). Here we fit a
standard linear+affine joint regressor — restricted to each joint's skinning
support — that reproduces `bind_pose_world` from `bind_shape`, then write a fixed
asset that swaps in this regressor (everything else preserved).

Usage::

    python tools/build_soma_rig.py --hf assets/third_party/SOMA_neutral.npz \
        --local SOMA_neutral.npz --out assets/SOMA_neutral_fixed.npz
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy.sparse import csc_matrix


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf", default=None,
                   help="full-schema SOMA_neutral.npz (default: resolved via soma_jax.assets)")
    p.add_argument("--local", default="SOMA_neutral.npz")
    p.add_argument("--out", default="assets/SOMA_neutral_fixed.npz")
    args = p.parse_args()

    hf = dict(np.load(args.hf, allow_pickle=True))
    local = dict(np.load(args.local, allow_pickle=True))

    bind_shape = hf["bind_shape"].astype(np.float64)                    # (V,3) cm
    joints_t = hf["bind_pose_world"][:, :3, 3].astype(np.float64)       # (J,3) cm
    W = csc_matrix(
        (hf["skinning_weights_data"], hf["skinning_weights_indices"], hf["skinning_weights_indptr"]),
        shape=tuple(hf["skinning_weights_shape"]),
    ).toarray()                                                         # (V,J)
    parents = hf["joint_parent_ids"].astype(int).copy(); parents[0] = 0
    J = W.shape[1]
    children = {j: [k for k in range(J) if parents[k] == j and k != j] for j in range(J)}

    Jreg = build_regressor(bind_shape, joints_t, W, parents, children)

    # Validate on the canonical bind shape (regressor is unit-independent & affine).
    err = np.linalg.norm(Jreg @ bind_shape - joints_t, axis=1)
    print(f"J_regressor fit error: mean {err.mean():.3f} cm  max {err.max():.3f} cm")

    # Write a fixed asset: keep everything from the local file, swap J_regressor.
    out = dict(local)
    out["J_regressor"] = Jreg.astype(np.float32)
    np.savez(args.out, **out)
    print(f"Wrote {args.out}")

    # Report joints on our actual v_template (= mean) for sanity.
    lv = local["v_template"].astype(np.float64)
    j_local = Jreg @ lv
    print(f"Root joint on v_template: {j_local[0].round(3)} (expect near pelvis/origin)")
    print(f"Head joint on v_template: {j_local[7].round(3)}")


if __name__ == "__main__":
    main()
