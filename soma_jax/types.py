"""Type definitions for SOMA-JAX.

Upstream: none — SOMA-JAX-only.
    SOMAParams / SOMAOutput dataclasses; upstream passes tensors positionally.
"""
from __future__ import annotations
from typing import NamedTuple, Optional
import jax.numpy as jnp


class SOMAParams(NamedTuple):
    """Inputs for a SOMA forward pass."""
    poses: jnp.ndarray               # (B, J, 3, 3) rotation matrices or (B, J, 3) axis-angle
    transl: jnp.ndarray              # (B, 3) root translation
    identity_coeffs: jnp.ndarray     # (B, C) identity/shape coefficients
    scale_params: Optional[jnp.ndarray] = None   # (B, S) optional body-part scales
    joint_orient: Optional[jnp.ndarray] = None   # (J, 3, 3) optional T-pose joint orientation


class SOMAOutput(NamedTuple):
    """Outputs from a SOMA forward pass.

    ``transforms`` mirrors SOMA-X's ``SOMAPoseOutput["transforms"]``: the full
    world transform of every joint, not just its position. ``vertices`` is
    ``None`` when ``pose(fk_only=True)`` skipped skinning.
    """
    vertices: jnp.ndarray                          # (B, V, 3) — None if fk_only
    joints: jnp.ndarray                            # (B, J, 3)
    transforms: Optional[jnp.ndarray] = None       # (B, J, 4, 4) world transforms
