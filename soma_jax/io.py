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
    Partial port of that code: ``save_soma_npz`` / ``load_soma_npz`` /
    ``add_npz_args`` (shared field names; root/absolute-pose defaults differ —
    see docs/FAITHFULNESS.md) and the template-rig asset contract constants
    (``SOMA_NEUTRAL_RIG_KEYS``, ``missing_soma_neutral_rig_keys``,
    ``SOMA_TEMPLATE_RIG_FILENAME``). The USD half is :mod:`soma_jax.usd_io`.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .units import Unit

logger = logging.getLogger(__name__)


class SOMANPZData(dict):
    """Dictionary returned by :func:`load_soma_npz` (upstream's type).

    A ``dict`` (``data["poses"]``) with attribute access (``data.poses``).
    Optional fields (``scale_params``, ``joint_orient``, ``global_scale``,
    ``hand_type``) are present only if they were saved.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class RigUSDData(dict):
    """Dictionary returned by :func:`soma_jax.usd_io.load_rig_from_usd` (upstream's type).

    A ``dict`` (``rig["joint_names"]``) with attribute access
    (``rig.joint_names``). Mesh fields (``face_vert_indices``,
    ``face_vert_counts``, ``uv_data``) are present only when the body skin mesh
    carries polygon / UV data.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ---------------------------------------------------------------------------
# SOMA template-rig asset contract (upstream ``soma/io.py``)
# ---------------------------------------------------------------------------
#: Filename of the SOMA body template rig asset (in ``data_root``).
SOMA_TEMPLATE_RIG_FILENAME = "SOMA_template_rig.usda"
#: The xlo LOD is read from the same template since SOMA-X v0.2.2.
SOMA_XLO_TEMPLATE_RIG_FILENAME = SOMA_TEMPLATE_RIG_FILENAME
#: Rig keys a pre-v0.3 ``SOMA_neutral.npz`` carried. Since SOMA-X v0.3 the npz
#: ships none of them — its ``metadata`` lists them under
#: ``asset_contract.removed_npz_rig_fields`` — and ``SOMA_template_rig.usda``
#: supplies them (``soma_jax.usd_io.load_rig_from_usd``).
SOMA_NEUTRAL_RIG_KEYS = (
    "joint_names",
    "joint_parent_ids",
    "bind_pose_world",
    "bind_pose_local",
    "t_pose_world",
    "t_pose_local",
    "bind_shape",
    "skinning_weights_data",
    "skinning_weights_indices",
    "skinning_weights_indptr",
    "skinning_weights_shape",
)


def missing_soma_neutral_rig_keys(data) -> tuple[str, ...]:
    """Rig keys absent from a loaded ``SOMA_neutral.npz`` mapping (upstream ``soma.io``)."""
    return tuple(key for key in SOMA_NEUTRAL_RIG_KEYS if key not in data)


def save_soma_npz(
    out_path: str,
    poses: np.ndarray,
    transl: np.ndarray,
    *,
    joint_names: list[str],
    identity_model_type: str,
    identity_coeffs: np.ndarray,
    scale_params: Optional[np.ndarray] = None,
    joint_orient: Optional[np.ndarray] = None,
    global_scale: Optional[float] = None,
    hand_type: Optional[str] = None,
    unit: str = "meters",
    keep_root: bool = False,
    extra_arrays: Optional[dict[str, np.ndarray]] = None,
    rotation_repr: Optional[str] = None,
    absolute_pose: Optional[bool] = None,
) -> None:
    """Save a SOMA animation sequence to a compressed NPZ file.

    Upstream's signature, keyword-only after ``transl`` as upstream's is;
    ``rotation_repr`` and ``absolute_pose`` are SOMA-JAX extras that override
    what upstream infers from the pose shape and from ``joint_orient``.

    Args:
        out_path: output file path (will add .npz if absent).
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

    if extra_arrays:
        # Upstream updates the dict, so an extra array replaces a field.
        arrays.update({k: np.asarray(v) for k, v in extra_arrays.items()})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **arrays)

    pose_label = "absolute" if _absolute_pose else "relative"
    root_label = "with Root (J=78)" if keep_root else "no Root (J=77)"
    summary_lines = [f"Saved: {out_path}"]
    if hand_type is not None:
        summary_lines.append(f"  hand_type: {hand_type}")
    summary_lines.append(f"  identity_model_type: {identity_model_type}")
    summary_lines.append(f"  identity_coeffs: {arrays['identity_coeffs'].shape}")
    if scale_params is not None:
        summary_lines.append(f"  scale_params: {np.shape(scale_params)}")
    if global_scale is not None:
        summary_lines.append(f"  global_scale: {float(global_scale):.4f}")
    summary_lines.append(f"  poses: {poses.shape} ({rotation_repr}, {pose_label}, {root_label})")
    summary_lines.append(f"  transl: {np.shape(transl)} ({unit})")
    summary_lines.append(f"  joint_names: {len(joint_names)} joints")
    logger.info("\n".join(summary_lines))


def load_soma_npz(path) -> SOMANPZData:
    """Load a SOMA animation ``.npz`` saved by :func:`save_soma_npz`.

    Port of upstream ``soma.io.load_soma_npz``. Keys: ``poses`` ((N, J, 3)
    rotvec or (N, J, 3, 3) matrices), ``transl`` (N, 3), ``joint_names``,
    ``identity_model_type``, ``identity_coeffs``, ``rotation_repr``,
    ``absolute_pose``, ``unit`` and ``keep_root``; optional ``scale_params``,
    ``joint_orient``, ``global_scale`` and ``hand_type``; any extra arrays
    stored via ``extra_arrays`` as-is.

    Args:
        path: path to the ``.npz`` file.

    Returns:
        :class:`SOMANPZData` of numpy arrays and Python scalars.
    """
    data = np.load(str(path), allow_pickle=True)
    result = SOMANPZData(
        poses=data["poses"],
        transl=data["transl"],
        joint_names=list(data["joint_names"]),
        identity_model_type=str(data["identity_model_type"]),
        identity_coeffs=data["identity_coeffs"],
        rotation_repr=str(data["rotation_repr"]),
        absolute_pose=bool(data["absolute_pose"]),
        unit=str(data["unit"]),
        keep_root=bool(data.get("keep_root", False)),
    )
    if "scale_params" in data:
        result["scale_params"] = data["scale_params"]
    if "joint_orient" in data:
        result["joint_orient"] = data["joint_orient"]
    if "global_scale" in data:
        result["global_scale"] = float(data["global_scale"])
    if "hand_type" in data:
        result["hand_type"] = str(data["hand_type"])

    known = {"poses", "transl", "joint_names", "identity_model_type", "identity_coeffs",
             "rotation_repr", "absolute_pose", "unit", "keep_root", "scale_params",
             "joint_orient", "global_scale", "hand_type"}
    for key in data.files:
        if key not in known:
            result[key] = data[key]
    return result


def add_npz_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the common NPZ output arguments to an argparse parser.

    Upstream ``soma.io.add_npz_args``: ``--output-npz``, ``--keep-root``
    (Root stripped by default, J=77, matching ``SOMALayer.pose()`` input) and
    ``--output-unit``. Returns the parser for chaining (upstream returns None).
    """
    parser.add_argument("--output-npz", default=None,
                        help="Output .npz file with SOMA pose parameters.")
    parser.add_argument("--keep-root", action="store_true",
                        help="Include the virtual Root joint (J=78). Off by default (J=77) "
                             "to match SOMALayer.pose() input convention.")
    parser.add_argument("--output-unit", choices=[u.unit_name for u in Unit],
                        default=Unit.METERS.unit_name,
                        help="Unit for translational quantities in the output .npz. "
                             "Default: meters.")
    return parser
