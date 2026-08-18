"""Convert MHR animations to SOMA NPZ format.

MHR (Meta Human Rig) uses centimeters and has body-part scale parameters.
This tool packages MHR pose + identity data into the SOMA NPZ format.

Usage::

    python tools/convert/mhr2soma.py \\
        --input path/to/mhr_animation.npz \\
        --output path/to/output.soma.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soma_jax.io import save_soma_npz, add_npz_args


def parse_args():
    p = argparse.ArgumentParser(description="Convert MHR to SOMA NPZ format")
    p.add_argument("--input", required=True, help="Input MHR NPZ file")
    p.add_argument(
        "--soma-model", default=None,
        help="Optional SOMA_neutral.npz for joint name reference",
    )
    p = add_npz_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)

    # MHR fields
    poses = np.array(data.get("poses", data.get("pose")), dtype=np.float32)
    transl = np.array(
        data.get("trans", data.get("transl", np.zeros((poses.shape[0], 3)))),
        dtype=np.float32,
    )
    identity_coeffs = np.array(
        data.get("identity_coeffs", data.get("betas", np.zeros((1, 10)))),
        dtype=np.float32,
    )
    scale_params = data.get("scale_params", None)
    if scale_params is not None:
        scale_params = np.array(scale_params, dtype=np.float32)

    if identity_coeffs.ndim == 1:
        identity_coeffs = identity_coeffs[None]

    if poses.ndim == 2:
        N = poses.shape[0]
        J = poses.shape[1] // 3
        poses = poses.reshape(N, J, 3)

    N, J = poses.shape[:2]

    # Try to load joint names from SOMA model if provided
    joint_names: list[str]
    if args.soma_model is not None:
        soma_data = np.load(args.soma_model, allow_pickle=True)
        soma_names = soma_data.get("joint_names")
        if soma_names is not None:
            names_list = list(soma_names) if hasattr(soma_names, "__iter__") else [soma_names]
            joint_names = [str(n) for n in names_list[:J]]
            while len(joint_names) < J:
                joint_names.append(f"mhr_joint_{len(joint_names)}")
        else:
            joint_names = [f"mhr_joint_{i}" for i in range(J)]
    else:
        joint_names = [f"mhr_joint_{i}" for i in range(J)]

    output_path = args.output_npz or args.input.replace(".npz", ".soma.npz")
    save_soma_npz(
        output_path,
        poses=poses,
        transl=transl,
        joint_names=joint_names,
        identity_model_type="mhr",
        identity_coeffs=identity_coeffs,
        scale_params=scale_params,
        unit=args.unit,
        absolute_pose=args.absolute_pose,
        keep_root=args.keep_root,
    )
    print(f"Saved SOMA NPZ to: {output_path}")
    print(f"  Frames: {N}, Joints: {J}, Identity coeffs: {identity_coeffs.shape[1]}")
    if scale_params is not None:
        print(f"  Scale params: {scale_params.shape}")


if __name__ == "__main__":
    main()
