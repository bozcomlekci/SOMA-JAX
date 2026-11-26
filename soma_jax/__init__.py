"""SOMA-JAX: JAX implementation of the SOMA-X universal human body pivot.

SOMA (Skeleton-Oriented Mean Avatar) provides a universal pivot system for
parametric human body models, enabling mix-and-match of identity sources
and pose data at inference time.

Supported identity models (upstream's six backends):
    - SOMA's own 128-coefficient shape PCA (the default here; upstream defaults to MHR)
    - MHR (centimetres, with body-part scales)
    - Anny (anthropometric phenotypes, infants to elders)
    - SMPL / SMPL-X / SMPL-H (licensed model files supplied by the user)
    - GarmentMeasurement (the GarmentMeasurements shape PCA)

Example::

    import numpy as np
    import jax.numpy as jnp
    from soma_jax import SOMALayer, SOMAParams

    layer = SOMALayer.from_upstream_assets()   # rig from the template USD
    params = SOMAParams(
        poses=jnp.zeros((1, 78, 3)),          # axis-angle; Root (index 0) must be zero
        transl=jnp.zeros((1, 3)),
        identity_coeffs=jnp.zeros((1, 128)),  # layer.identity_model.num_identity_coeffs
    )
    output = layer(params)
    # output.vertices: (1, 18056, 3)
    # output.joints:   (1, 78, 3)             # Root included; forward() returns upstream's 77

Upstream: ``soma/__init__.py``
    Public re-export surface, tracking SOMA-X v0.3.3: everything upstream's
    ``soma`` exports except ``setup_warp_for_ddp`` (Warp/torch-distributed
    plumbing), plus the ``soma_jax.body`` / ``soma_jax.fitting`` /
    ``soma_jax.hand`` namespaces. ``SOMAOutput`` (what ``SOMALayer.__call__``
    returns) and the rest of this list beyond upstream's are SOMA-JAX extras. The correspondence map for every module is in
    ``docs/FAITHFULNESS.md``.
"""

__version__ = "0.1.0"

import sys as _sys

from .body.soma import (
    SOMALayer,
    SOMAPoseOutput,
    SOMAPublicRigView,
    SomaLayer,
    remove_joint_orient_local,
)
from .assets import get_assets_dir
from .units import Unit
from .identity_model import BaseIdentityModel
from .body import create_identity_model
from .types import SOMAParams, SOMAOutput
from .io import (
    SOMA_TEMPLATE_RIG_FILENAME,
    SOMA_XLO_TEMPLATE_RIG_FILENAME,
    save_soma_npz,
    load_soma_npz,
    add_npz_args,
)
from .usd_io import (
    find_lod_skin_mesh_name,
    load_lod_rig_from_usd,
    load_lod_rigs_from_usd,
    load_rig_from_usd,
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
from .pose_inversion_lite import PoseInversion, apply_dof_constraints
from .fitting.pose_inversion import SOMAPoseInversion, PoseInversionResult
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
from .hand import MANOLayer, SOMAHandLayer, SOMAHandPoseOutput
from .smpl import (
    SMPLFamilyPoseTransferResult,
    SMPLFamilyTopologyBridge,
    SMPLLayer,
    SMPLXLayer,
    create_smpl_family_layer,
)
from .smpl import transfer_smpl_family_pose_parameters
from . import body, fitting, hand  # noqa: E402  (upstream's v0.3 package layout)

# Pre-0.3 module paths, registered as upstream's ``soma/__init__.py`` registers
# its own: ``import soma_jax.soma`` / ``from soma_jax.pose_inversion import X``
# and pickled class references resolve to the implementation modules, with no
# shim files. (The SOMA-JAX-only lightweight inverter that used to live at
# ``soma_jax.pose_inversion`` is ``soma_jax.pose_inversion_lite``.)
_LEGACY_MODULES = {
    "soma_jax.soma": body.soma,
    "soma_jax.pose_inversion": fitting.pose_inversion,
    "soma_jax.pose_inversion_mhr": fitting.pose_inversion_mhr,
    "soma_jax.rts_smoothing": fitting.rts_smoothing,
}
for _name, _module in _LEGACY_MODULES.items():
    _sys.modules.setdefault(_name, _module)
    setattr(_sys.modules[__name__], _name.rsplit(".", 1)[1], _module)

__all__ = [
    # Main model
    "SOMALayer",
    "SomaLayer",
    # Parameters & outputs
    "SOMAParams",
    "SOMAOutput",
    "SOMAPoseOutput",
    "SOMAPublicRigView",
    # SOMA Hand (SOMA-X v0.3.0)
    "SOMAHandLayer",
    "SOMAHandPoseOutput",
    "MANOLayer",
    # SMPL-family rig layers and transfer
    "SMPLLayer",
    "SMPLXLayer",
    "create_smpl_family_layer",
    "SMPLFamilyPoseTransferResult",
    "SMPLFamilyTopologyBridge",
    "transfer_smpl_family_pose_parameters",
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
    "SOMA_TEMPLATE_RIG_FILENAME",
    "SOMA_XLO_TEMPLATE_RIG_FILENAME",
    "find_lod_skin_mesh_name",
    "load_lod_rig_from_usd",
    "load_lod_rigs_from_usd",
    "load_rig_from_usd",
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
