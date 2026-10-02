"""Hand-only SOMA-JAX layers and MANO interoperability.

Upstream: ``soma/hand/`` (SOMA-X v0.3.0, "SOMA Hand").
"""
from .identity_model import (
    BaseHandIdentityModel,
    MANOHandIdentityModel,
    MHRHandIdentityModel,
    SOMAHandIdentityModel,
)
from .mano import (
    MANO_FINGERTIP_VERTEX_IDS,
    MANO_JOINT_NAMES,
    MANO_JOINT_NAMES_21,
    MANO_JOINT_NAMES_WITH_FINGERTIPS,
    MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS,
    MANO_PARENT_IDS_21,
    MANO_TIP_VERTEX_IDS,
    MANOLayer,
    build_mano_joints_with_fingertips,
)
from .soma import SOMAHandIdentity, SOMAHandLayer, SOMAHandPoseOutput

__all__ = [
    "SOMAHandLayer",
    "SOMAHandPoseOutput",
    "SOMAHandIdentity",
    "BaseHandIdentityModel",
    "SOMAHandIdentityModel",
    "MANOHandIdentityModel",
    "MHRHandIdentityModel",
    "MANOLayer",
    "MANO_FINGERTIP_VERTEX_IDS",
    "MANO_JOINT_NAMES",
    "MANO_JOINT_NAMES_WITH_FINGERTIPS",
    "MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS",
    "MANO_JOINT_NAMES_21",
    "MANO_PARENT_IDS_21",
    "MANO_TIP_VERTEX_IDS",
    "build_mano_joints_with_fingertips",
]
