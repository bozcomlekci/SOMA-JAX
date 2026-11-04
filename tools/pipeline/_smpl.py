"""SMPL / SMPL-X body models for the pipeline tools.

These tools are written against SMPL-JAX (the ``third_party/SMPL-JAX``
submodule: ``pip install -e third_party/SMPL-JAX``) and use it when it is
installed. Without it they fall back to :mod:`soma_jax.body_models`, which gives
bit-identical vertices and joints for every model these tools build, so they
also work on a plain ``pip install -e .``.
"""
try:
    from smpl_jax import SMPLModel, SMPLParams, SMPLXModel, SMPLXParams
except ModuleNotFoundError:
    from soma_jax.body_models import SMPLModel, SMPLParams, SMPLXModel, SMPLXParams

__all__ = ["SMPLModel", "SMPLParams", "SMPLXModel", "SMPLXParams"]
