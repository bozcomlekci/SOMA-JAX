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


# The fit itself lives in the package so a non-editable install produces the
# same regressor as a checkout; re-exported here for the INSTALL.md §4.2 recipe
# and for existing callers of `build_soma_rig.build_regressor`.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from soma_jax.rig_build import _fit_affine_row, build_regressor  # noqa: E402,F401


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf", default=None,
                   help="full-schema SOMA_neutral.npz (default: resolved via soma_jax.assets)")
    p.add_argument("--local", default="SOMA_neutral.npz")
    p.add_argument("--out", default="assets/SOMA_neutral_fixed.npz")
    args = p.parse_args()
    if args.hf is None:
        from soma_jax.assets import resolve
        args.hf = str(resolve("SOMA_neutral.npz"))

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
