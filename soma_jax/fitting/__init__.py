"""Fitting algorithms: pose inversion from posed vertices and post-fit smoothing.

Upstream: ``soma/fitting/`` (SOMA-X v0.3.3). These invert the SOMA body and hand
layers (recover skeleton rotations from vertices) and regularize the resulting
pose trajectories. They are shared by the body and hand packages and by the
conversion tools.

As upstream, the implementations live in :mod:`.pose_inversion`,
:mod:`.pose_inversion_mhr` and :mod:`.rts_smoothing`, and the pre-0.3 paths
``soma_jax.pose_inversion``, ``soma_jax.pose_inversion_mhr`` and
``soma_jax.rts_smoothing`` resolve to these same modules. The top-level
``soma_jax.PoseInversion`` is *not* this solver: it is the SOMA-JAX-only
lightweight inverter (:mod:`soma_jax.pose_inversion_lite`); upstream has no
top-level ``PoseInversion``.

``MHRPoseInversion`` needs ``MHR/MHR_base_rig.npz`` and
``MHR/parameter_transform.npz``, which are not part of upstream's public asset
set; without them it raises ``FileNotFoundError``, as upstream's does.
"""

from .pose_inversion import PoseInversion, PoseInversionResult
from .pose_inversion_mhr import MHRPoseInversion, MHRPoseInversionResult
from .rts_smoothing import (
    DEFAULT_RTS_SMOOTHING_CONFIG,
    RTS_SMOOTHING_PRESETS,
    STRONG_RTS_SMOOTHING_CONFIG,
    RTSSmoothingConfig,
    RTSSmoothingGains,
    RTSSmoothingGroups,
    smooth_pose,
)

__all__ = [
    "PoseInversion",
    "PoseInversionResult",
    "MHRPoseInversion",
    "MHRPoseInversionResult",
    "DEFAULT_RTS_SMOOTHING_CONFIG",
    "RTS_SMOOTHING_PRESETS",
    "STRONG_RTS_SMOOTHING_CONFIG",
    "RTSSmoothingConfig",
    "RTSSmoothingGains",
    "RTSSmoothingGroups",
    "smooth_pose",
]
