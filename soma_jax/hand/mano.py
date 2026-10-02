"""MANO rig layer and joint conventions for SOMA-JAX hand workflows.

Upstream: ``soma/hand/mano.py`` (SOMA-X v0.3.0).

MANO model files are licensed separately and not shipped; :class:`MANOLayer`
reads ``data_root/MANO/MANO_{LEFT,RIGHT}.pkl``.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..smpl.layers import _SMPLFamilyLBSLayer
from ..units import Unit
from ._smpl_family_loader import load_mano_pkl, mano_parent_ids

__all__ = [
    "MANO_FINGERTIP_VERTEX_IDS",
    "MANO_JOINT_NAMES",
    "MANO_JOINT_NAMES_21",
    "MANO_JOINT_NAMES_WITH_FINGERTIPS",
    "MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS",
    "MANO_PARENT_IDS_21",
    "MANO_TIP_VERTEX_IDS",
    "MANOLayer",
    "build_mano_joints_with_fingertips",
]

MANO_JOINT_NAMES = [
    "Wrist",
    "Index1", "Index2", "Index3",
    "Middle1", "Middle2", "Middle3",
    "Pinky1", "Pinky2", "Pinky3",
    "Ring1", "Ring2", "Ring3",
    "Thumb1", "Thumb2", "Thumb3",
]

MANO_FINGERTIP_VERTEX_IDS = {
    "IndexTip": 320,
    "MiddleTip": 443,
    "PinkyTip": 671,
    "RingTip": 554,
    "ThumbTip": 744,
}

MANO_JOINT_NAMES_WITH_FINGERTIPS = [
    "Wrist",
    "Index1", "Index2", "Index3", "IndexTip",
    "Middle1", "Middle2", "Middle3", "MiddleTip",
    "Pinky1", "Pinky2", "Pinky3", "PinkyTip",
    "Ring1", "Ring2", "Ring3", "RingTip",
    "Thumb1", "Thumb2", "Thumb3", "ThumbTip",
]

MANO_JOINT_PARENT_NAMES_WITH_FINGERTIPS = [
    "Wrist",
    "Wrist", "Index1", "Index2", "Index3",
    "Wrist", "Middle1", "Middle2", "Middle3",
    "Wrist", "Pinky1", "Pinky2", "Pinky3",
    "Wrist", "Ring1", "Ring2", "Ring3",
    "Wrist", "Thumb1", "Thumb2", "Thumb3",
]


def _joint_parent_ids_from_names(joint_names: list[str], parent_names: list[str]) -> list[int]:
    if len(joint_names) != len(parent_names):
        raise ValueError(
            f"Expected one parent per joint, got {len(joint_names)} joints and "
            f"{len(parent_names)} parents.")
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    return [name_to_idx[parent_name] for parent_name in parent_names]


MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS = _joint_parent_ids_from_names(
    MANO_JOINT_NAMES_WITH_FINGERTIPS, MANO_JOINT_PARENT_NAMES_WITH_FINGERTIPS)

MANO_JOINT_NAMES_21 = MANO_JOINT_NAMES_WITH_FINGERTIPS
MANO_PARENT_IDS_21 = MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS
MANO_TIP_VERTEX_IDS = {
    "index": MANO_FINGERTIP_VERTEX_IDS["IndexTip"],
    "middle": MANO_FINGERTIP_VERTEX_IDS["MiddleTip"],
    "pinky": MANO_FINGERTIP_VERTEX_IDS["PinkyTip"],
    "ring": MANO_FINGERTIP_VERTEX_IDS["RingTip"],
    "thumb": MANO_FINGERTIP_VERTEX_IDS["ThumbTip"],
}


def build_mano_joints_with_fingertips(vertices: jnp.ndarray, joints16: jnp.ndarray) -> jnp.ndarray:
    """Return MANO joints in the 21-landmark convention.

    Args:
        vertices: MANO vertices, shape (..., V, 3).
        joints16: native MANO joints, shape (..., 16, 3).

    Returns:
        (..., 21, 3) ordered as :data:`MANO_JOINT_NAMES_WITH_FINGERTIPS`.
    """
    if joints16.shape[-2] != len(MANO_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(MANO_JOINT_NAMES)} native MANO joints, got {joints16.shape[-2]}.")
    max_tip_id = max(MANO_FINGERTIP_VERTEX_IDS.values())
    if vertices.shape[-2] <= max_tip_id:
        raise ValueError(f"Expected MANO vertices to include vertex {max_tip_id}.")
    tip = MANO_FINGERTIP_VERTEX_IDS
    return jnp.concatenate([
        joints16[..., 0:1, :],
        joints16[..., 1:4, :], vertices[..., tip["IndexTip"]:tip["IndexTip"] + 1, :],
        joints16[..., 4:7, :], vertices[..., tip["MiddleTip"]:tip["MiddleTip"] + 1, :],
        joints16[..., 7:10, :], vertices[..., tip["PinkyTip"]:tip["PinkyTip"] + 1, :],
        joints16[..., 10:13, :], vertices[..., tip["RingTip"]:tip["RingTip"] + 1, :],
        joints16[..., 13:16, :], vertices[..., tip["ThumbTip"]:tip["ThumbTip"] + 1, :],
    ], axis=-2)


class MANOLayer(_SMPLFamilyLBSLayer):
    """MANO LBS rig adapter implementing the PoseInversion layer contract."""

    def __init__(self, data_root, hand_type: str, device=None, mode: str = "warp",
                 output_unit=Unit.METERS) -> None:
        # ``device`` keeps upstream's positional order; accepted and ignored.
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got {hand_type!r}.")
        super().__init__(data_root, device=device, mode=mode, output_unit=output_unit)
        self.hand_type = hand_type
        self.model_type = "mano"
        self.model_spec = f"mano-{hand_type}"
        self.topology_family = "hand"
        self.identity_model_type = "mano_native"
        self.identity_model_kwargs = {"hand_type": hand_type}
        self.rig_data = {"joint_names": MANO_JOINT_NAMES}
        self.default_skin_mesh_name = f"{hand_type}_mano"
        self.base_mesh_path = self.data_root / "MANO" / f"base_hand_{hand_type}.obj"
        self.wrap_mesh_path = self.data_root / "MANO" / f"SOMA_wrap_{hand_type}.obj"

        mano = load_mano_pkl(self.data_root, hand_type)
        parent_ids = mano_parent_ids(mano["kintree_table"])
        if len(parent_ids) != len(MANO_JOINT_NAMES):
            raise ValueError(f"Expected 16 MANO joints, got {len(parent_ids)}.")
        self.num_identity_coeffs = int(mano["shapedirs"].shape[2])
        self._v_template = jnp.asarray(mano["v_template"])
        self._shapedirs = jnp.asarray(mano["shapedirs"])
        self._J_regressor = jnp.asarray(mano["J_regressor"])
        self.skinning_weights = jnp.asarray(mano["weights"])
        self.joint_parent_ids = np.asarray(parent_ids, np.int64)
        self.faces = jnp.asarray(mano["faces"])
        self.posedirs = jnp.asarray(mano["posedirs"])
        self.hands_mean = jnp.asarray(mano["hands_mean"])
        self.hands_components = jnp.asarray(mano["hands_components"])
        self._finish_init()

    def _shape_native(self, identity_coeffs):
        blend = jnp.einsum("bk,vdk->bvd", identity_coeffs, self._shapedirs)
        verts = self._v_template[None] + blend
        joints = jnp.einsum("jv,bvd->bjd", self._J_regressor, verts)
        wrist = joints[:, 0:1]
        return verts - wrist, joints - wrist
