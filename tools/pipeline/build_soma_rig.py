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


# The fit itself lives in the package so a non-editable install produces the
# same regressor as a checkout; re-exported here for existing callers of
# `build_soma_rig.build_regressor`.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from soma_jax.rig_build import _fit_affine_row, build_regressor  # noqa: E402,F401


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="assets/SOMA_neutral_fixed.npz")
    p.add_argument("--npz", default=None, help="SOMA_neutral.npz (default: resolved)")
    p.add_argument("--usd", default=None, help="SOMA_template_rig.usda (default: resolved)")
    p.add_argument("--no-regressor", action="store_true",
                   help="skip the skeleton_fit='linear' regressor")
    args = p.parse_args()
    if args.hf is None:
        from soma_jax.assets import resolve
        args.hf = str(resolve("SOMA_neutral.npz"))

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
