"""Convert a SOMA-X ``correctives_model.pt`` to a soma_jax-loadable .npz.

The upstream checkpoint (third_party/SOMA-X/soma/correctives_model.py)
stores its weights as a PyTorch dict of sparse COO tensors plus metadata:

* ``W1``      — sparse (D=J*6, K=J*C) first-layer linear weights.
* ``W2``      — sparse (K, 3V) second-layer linear weights (native unit cm).
* ``M1_mask`` — sparse (J, J) joint-to-joint anatomical mask.
* ``M2_mask`` — sparse (J, V) joint-to-vertex anatomical mask.
* ``bindpose``— (J, 3, 3) bind-pose rotations baked into the input feature.
* ``C_max``, ``use_tanh``, ``joint_indices``, ``source_num_joints``, ``meta``.

soma_jax loads the densified equivalents directly without depending on torch
at inference time. The two transformations this converter applies are:

1. **Sparse → dense.** PyTorch ``.to_dense()`` for the four weight / mask
   tensors. The downstream :class:`soma_jax.CorrectivesMLP` masks W1 / W2
   inline via :func:`jnp.where`-style multiplication, so we never need the
   sparse representation at runtime.
2. **cm → m on W2.** SOMA-X's ``CorrectivesMLP.load_checkpoint`` multiplies
   W2 by ``native_unit.meters_per_unit / output_unit.meters_per_unit`` so the
   per-vertex displacement output ends up in meters. We bake that scaling
   into the .npz once at conversion time. W1 is dimensionless, M1/M2 are
   binary masks, and ``bindpose`` is a pure rotation — none of those need
   scaling.

Usage::

  python tools/convert/convert_correctives_pt_to_npz.py \\
      assets/third_party/hf/correctives_model_v021.pt \\
      assets/correctives_model_v021.npz
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    """CLI entry point: parse args, load the PT, write the NPZ."""
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path, help="input .pt checkpoint")
    p.add_argument("dst", type=Path, help="output .npz")
    args = p.parse_args()

    # weights_only=False is required because SOMA-X stores a meta dict alongside
    # the tensors (epoch, mask paths, etc.); torch 2.4+ blocks those by default.
    ck = torch.load(args.src, map_location="cpu", weights_only=False)

    def dense(t):
        """Coerce a tensor (sparse or dense) into a float32 numpy array."""
        if torch.is_tensor(t) and t.is_sparse:
            t = t.to_dense()
        return np.asarray(t, dtype=np.float32) if torch.is_tensor(t) else np.asarray(t)

    # SOMA-X correctives ship in centimeters; convert to meters by scaling W2.
    # See third_party/SOMA-X/soma/correctives_model.py: load_checkpoint multiplies
    # W2 by `native_unit.meters_per_unit / output_unit.meters_per_unit`.
    native_unit_name = str(ck.get("unit", "centimeters")).lower()
    cm_to_m = {"centimeters": 0.01, "millimeters": 0.001, "meters": 1.0}
    scale = cm_to_m.get(native_unit_name, 0.01)

    W2_dense = dense(ck["W2"]) * scale
    out = {
        "C_max":    np.int32(int(ck["C_max"])),
        "use_tanh": np.bool_(bool(ck["use_tanh"])),
        "bindpose": dense(ck["bindpose"]),
        "W1":       dense(ck["W1"]),
        "W2":       W2_dense,
    }
    print(f"  applied native_unit={native_unit_name} -> W2 *= {scale}")
    if "M1_mask" in ck:
        out["M1_mask"] = dense(ck["M1_mask"])
    if "M2_mask" in ck:
        out["M2_mask"] = dense(ck["M2_mask"])
    if "joint_indices" in ck:
        out["joint_indices"] = np.asarray(ck["joint_indices"], dtype=np.int64)
    if "source_num_joints" in ck:
        out["source_num_joints"] = np.int32(int(ck["source_num_joints"]))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dst, **out)
    print(f"Wrote {args.dst}")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s} shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k:20s} = {v}")


if __name__ == "__main__":
    sys.exit(main() or 0)
