"""I/O utilities for SOMA-JAX NPZ animation format.

The SOMA NPZ format stores a complete animation sequence with all information
needed to replay it via SOMALayer:

Required fields:
    poses: (N, J, 3) axis-angle or (N, J, 3, 3) rotation matrices
    transl: (N, 3) root translations
    joint_names: list of J joint name strings
    identity_model_type: string identifier ('smpl', 'mhr', etc.)
    identity_coeffs: (N, C) or (1, C) shape coefficients

Optional fields:
    scale_params: (N, S) or (1, S) body-part scale parameters
    joint_orient: (J, 3, 3) T-pose joint orientation
    extra_arrays: additional custom data

Metadata:
    rotation_repr: 'rotvec' or 'matrix' (inferred from pose shape if absent)
    absolute_pose: bool — whether poses are in absolute world frame
    unit: translation unit string ('meters', 'centimeters', 'millimeters')
    keep_root: bool — whether virtual root joint (index 0) is included

Upstream: ``soma/io.py (NPZ half)``
    Partial port of that code. Shares save_soma_npz's field names; root/absolute-pose defaults differ - see docs/FAITHFULNESS.md.
"""
from __future__ import annotations
import argparse
from typing import Optional, Any
import numpy as np

from .units import Unit


def save_soma_npz(
    path: str,
    poses: np.ndarray,
    transl: np.ndarray,
    joint_names: list[str],
    identity_model_type: str,
    identity_coeffs: np.ndarray,
    scale_params: Optional[np.ndarray] = None,
    joint_orient: Optional[np.ndarray] = None,
    extra_arrays: Optional[dict[str, np.ndarray]] = None,
    rotation_repr: Optional[str] = None,
    absolute_pose: Optional[bool] = None,
    unit: str = "meters",
    keep_root: bool = False,
    global_scale: Optional[float] = None,
    hand_type: Optional[str] = None,
) -> None:
    """Save a SOMA animation sequence to a compressed NPZ file.

    Args:
        path: output file path (will add .npz if absent).
        poses: (N, J, 3) axis-angle or (N, J, 3, 3) rotation matrices.
        transl: (N, 3) root translations.
        joint_names: list of J joint name strings.
        identity_model_type: model type identifier string.
        identity_coeffs: (N, C) or (1, C) shape coefficients.
        scale_params: optional (N, S) or (1, S) scale parameters.
        joint_orient: optional (J, 3, 3) T-pose orientations.
        extra_arrays: optional dict of additional numpy arrays.
        rotation_repr: 'rotvec' or 'matrix'; inferred from poses.ndim if None.
        absolute_pose: whether poses are in absolute world frame.
        unit: translation unit string.
        keep_root: include the virtual Root joint (index 0). Upstream's default
            is ``False``, i.e. Root is **stripped** from ``poses`` and
            ``joint_names`` before writing (J=78 -> J=77).
        global_scale: optional uniform scale, stored when given.
        hand_type: optional hand-model identifier, stored when given.
    """
    poses = np.asarray(poses, dtype=np.float32)

    # Infer the representation from the pose shape exactly as upstream's
    # ``save_soma_npz`` does, including its rejection of anything else — an
    # unrecognised shape written silently would be unreadable by either side.
    if rotation_repr is None:
        if poses.ndim == 3 and poses.shape[-1] == 3:
            rotation_repr = "rotvec"
        elif poses.ndim == 4 and poses.shape[-2:] == (3, 3):
            rotation_repr = "matrix"
        else:
            raise ValueError(
                f"Cannot infer rotation representation from poses shape {poses.shape}. "
                "Expected (N, J, 3) for rotvec or (N, J, 3, 3) for matrix."
            )

    # Upstream infers this rather than taking it on faith: a clip carrying a
    # joint orient is by construction relative to it.
    _absolute_pose = (joint_orient is None) if absolute_pose is None else bool(absolute_pose)

    # Strip the Root joint unless asked to keep it — upstream does this to the
    # arrays, not just to the flag. Recording ``keep_root=False`` while leaving
    # Root in the array mislabels the file and shifts every joint by one when
    # SOMA-X reads it back.
    joint_names = list(joint_names)
    if not keep_root:
        poses = poses[:, 1:]
        joint_names = joint_names[1:]

    arrays: dict[str, Any] = {
        "poses": poses,
        "transl": np.asarray(transl, dtype=np.float32),
        "joint_names": np.array(joint_names),
        "identity_model_type": np.array(identity_model_type),
        "identity_coeffs": np.asarray(identity_coeffs, dtype=np.float32),
        "rotation_repr": np.array(rotation_repr),
        "absolute_pose": np.array(_absolute_pose),
        "unit": np.array(unit),
        "keep_root": np.array(keep_root),
    }

    if scale_params is not None:
        arrays["scale_params"] = np.asarray(scale_params, dtype=np.float32)
    if joint_orient is not None:
        arrays["joint_orient"] = np.asarray(joint_orient, dtype=np.float32)
    if global_scale is not None:
        arrays["global_scale"] = np.float32(global_scale)
    if hand_type is not None:
        arrays["hand_type"] = np.array(hand_type)

    if extra_arrays is not None:
        for k, v in extra_arrays.items():
            if k in arrays:
                raise ValueError(f"extra_arrays key {k!r} conflicts with a reserved field.")
            arrays[k] = np.asarray(v)

    np.savez_compressed(path, **arrays)


def load_soma_npz(path: str) -> dict:
    """Load a SOMA animation NPZ file into a dict of numpy arrays.

    Args:
        path: path to .npz file.

    Returns:
        Dict with all stored fields, including decoded metadata scalars.
    """
    raw = np.load(path, allow_pickle=True)
    data = {}

    for k in raw.files:
        v = raw[k]
        # Unwrap 0-d object arrays (scalars stored as np.array)
        if v.ndim == 0 and v.dtype == object:
            data[k] = v.item()
        elif v.ndim == 0:
            data[k] = v.item()
        elif v.dtype == object:
            data[k] = list(v)
        else:
            data[k] = v

    return data


def add_npz_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add SOMA NPZ output arguments to an argparse parser.

    Args:
        parser: existing ArgumentParser to extend.

    Returns:
        The same parser with SOMA NPZ arguments added.
    """
    grp = parser.add_argument_group("SOMA NPZ output")
    grp.add_argument(
        "--output-npz", type=str, default=None,
        help="Path to save output SOMA NPZ animation file.",
    )
    grp.add_argument(
        "--unit", type=str, default="meters",
        choices=["meters", "centimeters", "millimeters"],
        help="Translation unit for output NPZ (default: meters).",
    )
    # Both defaults mirror upstream ``soma.io.save_soma_npz``: Root is stripped
    # unless asked for, and absolute-vs-relative is *inferred* from whether a
    # joint orient is present rather than forced to a fixed value.
    grp.add_argument(
        "--absolute-pose", action=argparse.BooleanOptionalAction, default=None,
        help="Store poses in absolute world frame "
             "(default: inferred — absolute iff no joint_orient is written).",
    )
    grp.add_argument(
        "--keep-root", action=argparse.BooleanOptionalAction, default=False,
        help="Include the virtual Root joint in pose output "
             "(default: stripped, matching SOMA-X).",
    )
    return parser
