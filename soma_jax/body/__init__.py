"""Full-body SOMA-JAX layer exports.

Upstream: ``soma/body/`` (SOMA-X v0.3.0 reorganized the full-body layer and its
identity backends under ``soma.body``). The layer lives in
:mod:`soma_jax.body.soma` — also importable from the pre-0.3 path
``soma_jax.soma``, as upstream's is from ``soma.soma`` — and the identity
backends in :mod:`soma_jax.body.identity_model`, built from
the asset directory exactly as upstream's are, with upstream's
``create_identity_model(identity_model_type, data_root, low_lod, ...)``
signature. (SOMA-JAX's older pack-based backends —
``soma_jax.identity_packs.create_identity_model(type, soma_data, model_data)`` —
remain available under their own module.)

``SOMAPoseOutput`` (what :meth:`SOMALayer.forward` returns) and
``SOMAPublicRigView`` (what :meth:`SOMALayer.public_rig_view` returns) are
upstream's types; :meth:`SOMALayer.__call__` returns SOMA-JAX's
:class:`~soma_jax.types.SOMAOutput` NamedTuple.
"""
from .identity_model import (
    AnnyIdentityModel,
    GarmentMeasurementIdentityModel,
    MHRIdentityModel,
    SMPLIdentityModel,
    SOMAIdentityModel,
    create_identity_model,
)
from .soma import SOMALayer, SOMAPoseOutput, SOMAPublicRigView

__all__ = [
    "SOMALayer",
    "SOMAPoseOutput",
    "SOMAPublicRigView",
    "AnnyIdentityModel",
    "GarmentMeasurementIdentityModel",
    "MHRIdentityModel",
    "SMPLIdentityModel",
    "SOMAIdentityModel",
    "create_identity_model",
]
