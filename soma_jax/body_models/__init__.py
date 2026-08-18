"""Parametric body model implementations in JAX.

This package provides standalone JAX implementations of:
    - SMPL  (24 joints, 6890 vertices) — Loper et al. 2015
    - SMPL-H (52 joints with hands)    — Romero et al. 2017
    - SMPL-X (55 joints with hands, face, eyes) — Pavlakos et al. 2019
    - MHR   (Meta Human Rig, per-body-part scales)
    - Anny  (children's body, Z-up coordinate system)

Each model can be used standalone (full forward pass with pose blend shapes
and LBS) or as an identity source for SOMA-JAX's pivot system.

Example::

    from soma_jax.body_models import SMPLModel, SMPLParams
    import jax.numpy as jnp

    model = SMPLModel.load("SMPL_NEUTRAL.pkl")
    params = SMPLModel.make_params(batch_size=1)
    output = model(params)
    # output.vertices: (1, 6890, 3)
    # output.joints:   (1, 24, 3)

Upstream: none — SOMA-JAX-only.
    Standalone parametric body models. Upstream loads SMPL-family assets (``soma/_smpl_family_loader.py``) but does not implement their forward pass.
"""
from ._base import BaseBodyModel, BodyModelOutput, shape_blend_shapes, pose_blend_shapes
from .smpl import SMPLModel, SMPLParams, SMPL_JOINT_NAMES
from .smplx import SMPLXModel, SMPLXParams, SMPLX_JOINT_NAMES
from .smplh import SMPLHModel, SMPLHParams, SMPLH_JOINT_NAMES
from .mhr import MHRModel, MHRParams
from .anny import AnnyModel, AnnyParams, to_soma_coords, from_soma_coords
from .model_io import load_smpl_data, save_smpl_data

__all__ = [
    # Base
    "BaseBodyModel",
    "BodyModelOutput",
    "shape_blend_shapes",
    "pose_blend_shapes",
    # SMPL
    "SMPLModel",
    "SMPLParams",
    "SMPL_JOINT_NAMES",
    # SMPL-X
    "SMPLXModel",
    "SMPLXParams",
    "SMPLX_JOINT_NAMES",
    # SMPL-H
    "SMPLHModel",
    "SMPLHParams",
    "SMPLH_JOINT_NAMES",
    # MHR
    "MHRModel",
    "MHRParams",
    # Anny
    "AnnyModel",
    "AnnyParams",
    "to_soma_coords",
    "from_soma_coords",
    # IO
    "load_smpl_data",
    "save_smpl_data",
]
