"""Convert an SMPL-X motion (.npz) to SOMA format (pure JAX).

Mirrors NVlabs/SOMA-X's motion→SOMA conversion: it retargets a source motion
onto the SOMA 78-joint skeleton via mesh-based inverse-LBS (PoseInversion) and
writes a SOMA-format `.npz`.

Output `.npz` fields (matching SOMA-X):
    poses            (N, J, 3)   parent-relative axis-angle per SOMA joint
    root_translation (N, 3)      root position (meters)
    joint_names      (J,)        SOMA joint names
    per_vertex_error (N, V)      reconstruction error vs the source mesh (meters)
    identity_coeffs  (C,)        identity parameters used (zeros = neutral)

Examples::

    # Convert + report error
    python -m tools.motion2soma --input path/to/motion.npz \\
        --output-npz out/motion_soma.npz

    # Analytical + autograd FK refinement (best accuracy)
    python -m tools.motion2soma --input seq.npz --output-npz out/soma.npz --autograd-iters 60
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Source SMPL-X motion (.npz)")
    p.add_argument("--output-npz", required=True, help="Output SOMA-format .npz")
    p.add_argument("--soma-model", default="SOMA_neutral.npz")
    p.add_argument("--smplx-model", default="data/smplx/SMPLX_NEUTRAL.npz")
    p.add_argument("--hf-dir", default=None,
                   help="asset root (default: resolved via soma_jax.assets)")
    p.add_argument("--num-frames", type=int, default=0,
                   help="Subsample to this many frames (0 = all frames)")
    p.add_argument("--autograd-iters", type=int, default=0,
                   help="Autograd FK refinement iterations (0 = analytical only)")
    args = p.parse_args()

    import motion_pipeline as mp
    from soma_jax import SOMALayer

    # Use all frames unless subsampling requested.
    if args.num_frames > 0:
        n_frames = args.num_frames
    else:
        n_frames = int(np.load(args.input)["poses"].shape[0])
    motion = mp.load_smplx_motion(args.input, num_frames=n_frames)
    print(f"Loaded {args.input}: {motion['n_frames']} frames")

    smplx = mp.load_smplx_jax(args.smplx_model)
    soma = SOMALayer.load(args.soma_model, identity_model_type="soma")

    mode = f"analytical+{args.autograd_iters} autograd FK" if args.autograd_iters else "analytical"
    print(f"Retargeting → SOMA skeleton (inverse-LBS, {mode})...")
    poses, extras = mp.smplx_motion_to_soma_poses(
        smplx, soma, motion, hf_dir=args.hf_dir,
        refine_iters=args.autograd_iters, return_extras=True)
    err = extras["per_vertex_error"]
    print(f"  per-vertex error: mean {err.mean()*100:.2f} cm  "
          f"median {np.median(err)*100:.2f} cm  max {err.max()*100:.2f} cm")

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        poses=poses.astype(np.float32),
        root_translation=extras["root_translation"],
        joint_names=extras["joint_names"],
        per_vertex_error=err,
        identity_coeffs=np.zeros((10,), np.float32),
    )
    print(f"Wrote {args.output_npz}")


if __name__ == "__main__":
    main()
