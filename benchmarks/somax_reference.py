"""SOMA-X's posed meshes for ``verify_fairness.py``'s agreement check.

Runs upstream ``SOMALayer`` exactly as ``bench_forward_pass.py`` times it
(``mode="warp"``, SOMA identity backend, no correctives, legacy 78-joint rig)
on seeded random identities, poses and translations, and saves inputs and
vertices. Torch and JAX keep separate CUDA stacks, so this runs in its own
process; ``verify_fairness.py`` then feeds the same inputs to the timed JAX
pipelines and compares meshes.

Usage::

    python benchmarks/somax_reference.py          # -> benchmarks/results/_somax_reference.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "SOMA-X"))

BATCH = 16
SEED = 0


def reference_inputs(num_coeffs: int) -> dict:
    """The seeded inputs both sides pose (root rotation identity, as upstream pads it)."""
    rng = np.random.default_rng(SEED)
    poses = (rng.standard_normal((BATCH, 77, 3)) * 0.2).astype(np.float32)
    return {
        "coeffs": (rng.standard_normal((BATCH, num_coeffs)) * 0.5).astype(np.float32),
        "poses": poses,                                                     # (B, 77, 3) axis-angle
        "transl": (rng.standard_normal((BATCH, 3)) * 0.1).astype(np.float32),  # metres
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset-dir", default=None, help="upstream-layout assets (default: data_root)")
    p.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "_somax_reference.npz"))
    args = p.parse_args()

    import torch
    from soma import SOMALayer

    from soma_jax.assets import data_root
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = SOMALayer(data_root=args.asset_dir or str(data_root()), device="cuda:0",
                      identity_model_type="soma", mode="warp", correctives_model_path=None,
                      enable_procedural_transforms=False).to("cuda:0")
    layer.eval()
    inputs = reference_inputs(layer.num_shape_components)
    with torch.no_grad():
        layer.prepare_identity(torch.from_numpy(inputs["coeffs"]).cuda(),
                               repose_to_bind_pose=False)
        out = layer.pose(torch.from_numpy(inputs["poses"]).cuda(),
                         transl=torch.from_numpy(inputs["transl"]).cuda(),
                         apply_correctives=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, vertices=out["vertices"].cpu().numpy(), **inputs)
    print(f"wrote {args.out}: {out['vertices'].shape[0]} meshes")


if __name__ == "__main__":
    main()
