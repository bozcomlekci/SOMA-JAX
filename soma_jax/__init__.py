"""SOMA-JAX: JAX implementation of the SOMA-X universal human body pivot.

SOMA (Skeleton-Oriented Mean Avatar) provides a universal pivot system for
parametric human body models, enabling mix-and-match of identity sources
and pose data at inference time.

Supported identity models:
    - SMPL / SMPL-X / SMPL-H
    - MHR (high-fidelity, centimeters, body-part scales)
    - Anny (children's model)
    - SOMA (proprietary 128-coeff PCA)
    - GarmentMeasurement (CAESARS dataset)

Example::

    import numpy as np
    import jax.numpy as jnp
    from soma_jax import SOMALayer, SOMAParams

    layer = SOMALayer.load("assets/SOMA_neutral_fixed.npz")
    params = SOMAParams(
        poses=jnp.zeros((1, 78, 3)),          # axis-angle; Root (index 0) must be zero
        transl=jnp.zeros((1, 3)),
        identity_coeffs=jnp.zeros((1, 10)),
    )
    output = layer(params)
    # output.vertices: (1, V, 3)
    # output.joints:   (1, J, 3)

Upstream: ``soma/__init__.py``
    Public re-export surface. The correspondence map for every module is in ``docs/FAITHFULNESS.md``.
"""

__version__ = "0.1.0"

from .soma import SOMALayer, SomaLayer, get_assets_dir, remove_joint_orient_local
from .units import Unit
from .identity_model import BaseIdentityModel, create_identity_model
from .types import SOMAParams, SOMAOutput
from .io import save_soma_npz, load_soma_npz, add_npz_args
from .usd_io import (
    save_soma_usd,
    save_vertex_animation_usd,
    export_soma_usd,
    load_usd_mesh,
    load_usd_skeleton,
    load_usd_animation,
    load_usd_skinning,
    list_usd_meshes,
    write_usd_mesh,
    fan_triangulate,
)
from .pose_inversion import PoseInversion, apply_dof_constraints
from .pose_inversion_soma import SOMAPoseInversion, PoseInversionResult
from .correctives_model import CorrectivesMLP
from .procedural_transforms import (
    ProceduralTransforms,
    SOMAProceduralTransformDefinition,
    SOMATwistSegmentSpec,
    load_definition as load_procedural_transform_definition,
    SOMA_ALIGNED_X_SWING_TWIST_MODE,
    SOMA_LOCAL_X_EULER_TWIST_MODE,
    SOMA_LOCAL_X_SWING_TWIST_MODE,
    SOMA_PROCEDURAL_TRANSFORM_MODES,
)
from .geometry import (
    BatchedSkinning,
    PoseMirror,
    chamfer_distance,
    apply_joint_orient_local,
    precompute_joint_orient,
    infer_joint_orient_from_rest,
    PoseMirrorSOMA,
    PoseMirrorMHR,
)
from .body_models import (
    SMPLModel,
    SMPLParams,
    SMPLXModel,
    SMPLXParams,
    SMPLHModel,
    SMPLHParams,
    MHRModel,
    MHRParams,
    AnnyModel,
    AnnyParams,
    BodyModelOutput,
    load_smpl_data,
)

__all__ = [
    # Main model
    "SOMALayer",
    "SomaLayer",
    # Parameters & outputs
    "SOMAParams",
    "SOMAOutput",
    # Identity models
    "BaseIdentityModel",
    "create_identity_model",
    # Utilities
    "Unit",
    "get_assets_dir",
    "remove_joint_orient_local",
    "apply_joint_orient_local",
    "precompute_joint_orient",
    "infer_joint_orient_from_rest",
    "PoseMirrorSOMA",
    "PoseMirrorMHR",
    # IO
    "save_soma_npz",
    "load_soma_npz",
    "add_npz_args",
    # USD IO (requires the optional usd-core package)
    "save_soma_usd",
    "save_vertex_animation_usd",
    "export_soma_usd",
    "load_usd_mesh",
    "load_usd_skeleton",
    "load_usd_animation",
    "load_usd_skinning",
    "list_usd_meshes",
    "write_usd_mesh",
    "fan_triangulate",
    # Pose inversion
    "SOMAPoseInversion",
    "PoseInversionResult",
    "PoseInversion",
    "apply_dof_constraints",
    # Correctives
    "CorrectivesMLP",
    # Geometry
    "BatchedSkinning",
    "PoseMirror",
    "chamfer_distance",
    # Body models
    "SMPLModel",
    "SMPLParams",
    "SMPLXModel",
    "SMPLXParams",
    "SMPLHModel",
    "SMPLHParams",
    "MHRModel",
    "MHRParams",
    "AnnyModel",
    "AnnyParams",
    "BodyModelOutput",
    "load_smpl_data",
]
