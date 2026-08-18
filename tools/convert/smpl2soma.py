"""Convert SMPL/SMPL-X/SMPL-H animations to SOMA NPZ format.

Usage::

    python tools/convert/smpl2soma.py \\
        --input path/to/smpl_animation.npz \\
        --soma-model path/to/SOMA_neutral.npz \\
        --output path/to/output.soma.npz \\
        --model-type smpl
"""
from __future__ import annotations
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soma_jax.io import save_soma_npz, add_npz_args


def parse_args():
    p = argparse.ArgumentParser(description="Convert SMPL to SOMA NPZ format")
    p.add_argument("--input", required=True, help="Input SMPL NPZ file")
    p.add_argument("--soma-model", required=True, help="SOMA_neutral.npz model file")
    p.add_argument("--model-type", default="smpl", choices=["smpl", "smplx", "smplh"])
    p = add_npz_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)

    poses = np.array(data.get("poses", data.get("body_pose")), dtype=np.float32)
    transl = np.array(data.get("trans", data.get("transl", np.zeros((poses.shape[0], 3)))), dtype=np.float32)
    betas = np.array(data.get("betas", data.get("shape", np.zeros((1, 10)))), dtype=np.float32)

    if betas.ndim == 1:
        betas = betas[None]

    N = poses.shape[0]
    J = poses.shape[1] if poses.ndim == 3 else poses.shape[1]

    joint_names = [f"smpl_joint_{i}" for i in range(J)]

    output_path = args.output_npz or args.input.replace(".npz", ".soma.npz")

    save_soma_npz(
        output_path,
        poses=poses,
        transl=transl,
        joint_names=joint_names,
        identity_model_type=args.model_type,
        identity_coeffs=betas,
        unit=args.unit,
        absolute_pose=args.absolute_pose,
        keep_root=args.keep_root,
    )
    print(f"Saved SOMA NPZ to: {output_path}")
    print(f"  Frames: {N}, Joints: {J}, Betas: {betas.shape[1]}")


if __name__ == "__main__":
    main()
