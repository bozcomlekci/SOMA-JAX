"""Reusable RTS smoothing for SOMA pose sequences.

Upstream: ``soma/fitting/rts_smoothing.py`` (SOMA-X v0.2.2; also importable
upstream as ``soma.rts_smoothing``, and here as ``soma_jax.rts_smoothing``).

The rotation smoother is an error-state Rauch-Tung-Striebel smoother on SO(3).
It keeps the filter state as a unit quaternion plus angular velocity and uses
shortest-arc quaternion log residuals instead of smoothing raw quaternion
components or extracting axes from matrix logs. Translations use the same
constant-velocity model per scalar channel.

Faithful port. JAX-mechanical differences:

* The forward filter and backward smoother are ``jax.lax.scan`` loops rather
  than Python frame loops, so :func:`rts_smooth_rotations` /
  :func:`rts_smooth_euclidean` are ``jit``-compatible (the joint grouping is
  static).
* Upstream always smooths in ``float64`` and casts back. JAX only has float64
  with ``jax_enable_x64``; without it the recursion runs in float32. Enable x64
  for bit-level agreement with upstream on long sequences.
* ``device=`` arguments are dropped (JAX places arrays itself).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.rig_utils import get_joint_descendents
from ..geometry.transforms import (
    matrix_to_quaternion_xyzw,
    project_rotations_to_so3,
    quaternion_conjugate_xyzw,
    quaternion_exp_xyzw,
    quaternion_log_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_xyzw_to_rotmat,
)
from ..reference_poses import _apply_orient, _orient_pair, _remove_orient

RotationConvention = Literal["absolute", "relative"]


@dataclass(frozen=True)
class RTSSmoothingGains:
    """Scalar gains for a constant-velocity RTS smoother."""

    r_obs: float = 0.06
    q_pos: float = 0.1
    q_vel: float = 4.5


@dataclass(frozen=True)
class RTSSmoothingConfig:
    """Configuration for :func:`smooth_pose`."""

    fps: float = 30.0
    rotation: RTSSmoothingGains = field(default_factory=RTSSmoothingGains)
    hand: RTSSmoothingGains | None = field(
        default_factory=lambda: RTSSmoothingGains(r_obs=0.005, q_pos=0.1, q_vel=35.0)
    )
    translation: RTSSmoothingGains | None = field(default_factory=RTSSmoothingGains)
    smooth_root_translation: bool = True


@dataclass(frozen=True)
class RTSSmoothingGroups:
    """Joint-name-derived smoothing groups."""

    hand_indices: frozenset[int] = frozenset()
    limb_indices: frozenset[int] = frozenset()

    def fast_indices(self, include_limbs: bool = False) -> frozenset[int]:
        if include_limbs:
            return self.hand_indices | self.limb_indices
        return self.hand_indices


DEFAULT_RTS_SMOOTHING_CONFIG = RTSSmoothingConfig()
STRONG_RTS_SMOOTHING_CONFIG = RTSSmoothingConfig(
    rotation=RTSSmoothingGains(r_obs=0.16, q_pos=0.05, q_vel=2.0),
    hand=RTSSmoothingGains(r_obs=0.024, q_pos=0.05, q_vel=18.0),
    translation=RTSSmoothingGains(r_obs=0.16, q_pos=0.05, q_vel=2.0),
)
RTS_SMOOTHING_PRESETS: dict[str, RTSSmoothingConfig] = {
    "default": DEFAULT_RTS_SMOOTHING_CONFIG,
    "strong": STRONG_RTS_SMOOTHING_CONFIG,
}


def _work_dtype():
    """Upstream smooths in float64; JAX can only when x64 is enabled."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _right_compose_delta(quaternions: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    return quaternion_normalize_xyzw(
        quaternion_multiply_xyzw(quaternions, quaternion_exp_xyzw(delta))
    )


def _quaternion_residual(predicted: jnp.ndarray, observed: jnp.ndarray) -> jnp.ndarray:
    relative = quaternion_multiply_xyzw(quaternion_conjugate_xyzw(predicted), observed)
    return quaternion_log_xyzw(relative)


def _validate_fps(fps: float) -> float:
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    return fps


def _gain_index_groups(
    num_joints: int,
    default_gains: RTSSmoothingGains,
    joint_gains: Mapping[int, RTSSmoothingGains] | None,
) -> list[tuple[np.ndarray, RTSSmoothingGains]]:
    grouped: dict[RTSSmoothingGains, list[int]] = {}
    joint_gains = joint_gains or {}
    for joint_idx in range(num_joints):
        gains = joint_gains.get(joint_idx, default_gains)
        grouped.setdefault(gains, []).append(joint_idx)
    return [(np.asarray(indices, dtype=np.int64), gains)
            for gains, indices in grouped.items() if indices]


def _rts_smooth_channels(measurements: jnp.ndarray, *, fps: float,
                         gains: RTSSmoothingGains) -> jnp.ndarray:
    """Smooth ``(T, C)`` independent scalar channels with one RTS recursion."""
    num_frames, num_channels = measurements.shape
    if num_frames < 2 or num_channels == 0:
        return measurements
    dt = 1.0 / _validate_fps(fps)
    dtype = measurements.dtype
    f_mat = jnp.asarray([[1.0, dt], [0.0, 1.0]], dtype=dtype)
    q_mat = jnp.diag(jnp.asarray([gains.q_pos * dt, gains.q_vel * dt], dtype=dtype))
    r_obs = jnp.asarray(float(gains.r_obs), dtype=dtype)

    state0 = jnp.zeros((num_channels, 2), dtype).at[:, 0].set(measurements[0])
    cov0 = jnp.zeros((num_channels, 2, 2), dtype)
    cov0 = cov0.at[:, 0, 0].set(r_obs).at[:, 1, 1].set(max(float(gains.q_vel) * dt, 1e-2))

    def forward(carry, measurement):
        state, cov = carry
        state_pred = state @ f_mat.T
        cov_pred = f_mat @ cov @ f_mat.T + q_mat
        innovation_cov = cov_pred[:, 0, 0] + r_obs
        kalman_gain = cov_pred[:, :, 0] / innovation_cov[:, None]
        residual = measurement - state_pred[:, 0]
        state_new = state_pred + kalman_gain * residual[:, None]
        cov_new = cov_pred - kalman_gain[:, :, None] * cov_pred[:, 0:1, :]
        cov_new = 0.5 * (cov_new + jnp.swapaxes(cov_new, -2, -1))
        return (state_new, cov_new), (state_new, cov_new)

    _, (states, covs) = jax.lax.scan(forward, (state0, cov0), measurements[1:])
    state_fwd = jnp.concatenate([state0[None], states], axis=0)
    cov_fwd = jnp.concatenate([cov0[None], covs], axis=0)

    def backward(state_next_smooth, xs):
        state_f, cov_f = xs
        next_pred_cov = f_mat @ cov_f @ f_mat.T + q_mat
        gain = cov_f @ f_mat.T @ jnp.linalg.inv(next_pred_cov)
        next_pred_state = state_f @ f_mat.T
        residual = state_next_smooth - next_pred_state
        smoothed = state_f + (gain @ residual[..., None])[..., 0]
        return smoothed, smoothed

    _, smoothed = jax.lax.scan(backward, state_fwd[-1],
                               (state_fwd[:-1], cov_fwd[:-1]), reverse=True)
    state_smooth = jnp.concatenate([smoothed, state_fwd[-1][None]], axis=0)
    return state_smooth[:, :, 0]


def rts_smooth_euclidean(
    values: jnp.ndarray,
    *,
    fps: float,
    gains: RTSSmoothingGains,
    joint_gains: Mapping[int, RTSSmoothingGains] | None = None,
) -> jnp.ndarray:
    """Smooth ``(T, J, D)`` Euclidean values with a constant-velocity RTS model."""
    values = jnp.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"Expected values with shape (T, J, D), got {values.shape}.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("values must be a floating-point array.")
    fps = _validate_fps(fps)
    num_frames, num_joints, dims = values.shape
    if num_frames < 2:
        return values
    values_w = values.astype(_work_dtype())
    result = jnp.empty_like(values_w)
    for indices, group_gains in _gain_index_groups(num_joints, gains, joint_gains):
        grouped = values_w[:, indices].reshape(num_frames, -1)
        smoothed = _rts_smooth_channels(grouped, fps=fps, gains=group_gains)
        result = result.at[:, indices].set(smoothed.reshape(num_frames, len(indices), dims))
    return result.astype(values.dtype)


def _smooth_rotation_group(quaternions: jnp.ndarray, *, fps: float,
                           gains: RTSSmoothingGains) -> jnp.ndarray:
    num_frames, num_channels = quaternions.shape[:2]
    if num_frames < 2 or num_channels == 0:
        return quaternions
    dt = 1.0 / _validate_fps(fps)
    dtype = quaternions.dtype
    i3 = jnp.eye(3, dtype=dtype)
    i6 = jnp.eye(6, dtype=dtype)
    f_mat = jnp.eye(6, dtype=dtype).at[:3, 3:].set(dt * i3)
    h_mat = jnp.zeros((3, 6), dtype).at[:, :3].set(i3)
    q_mat = jnp.diag(jnp.asarray([gains.q_pos * dt] * 3 + [gains.q_vel * dt] * 3, dtype=dtype))
    r_mat = i3 * gains.r_obs

    q0 = quaternions[0]
    v0 = jnp.zeros((num_channels, 3), dtype)
    p0 = jnp.zeros((num_channels, 6, 6), dtype)
    p0 = p0.at[:, :3, :3].set(i3 * gains.r_obs)
    p0 = p0.at[:, 3:, 3:].set(i3 * max(float(gains.q_vel) * dt, 1e-2))

    def forward(carry, observed):
        q_prev, v_prev, p_prev = carry
        q_pred = _right_compose_delta(q_prev, v_prev * dt)
        v_pred = v_prev
        p_pred = f_mat @ p_prev @ f_mat.T + q_mat
        residual = _quaternion_residual(q_pred, observed)
        innovation_cov = h_mat @ p_pred @ h_mat.T + r_mat
        kalman_gain = p_pred @ h_mat.T @ jnp.linalg.inv(innovation_cov)
        delta = (kalman_gain @ residual[..., None])[..., 0]
        q_new = _right_compose_delta(q_pred, delta[:, :3])
        v_new = v_pred + delta[:, 3:]
        p_new = (i6 - kalman_gain @ h_mat) @ p_pred
        p_new = 0.5 * (p_new + jnp.swapaxes(p_new, -2, -1))
        return (q_new, v_new, p_new), (q_new, v_new, p_new, q_pred, v_pred)

    _, (qf, vf, pf, qp, vp) = jax.lax.scan(forward, (q0, v0, p0), quaternions[1:])
    quat_fwd = jnp.concatenate([q0[None], qf], axis=0)
    vel_fwd = jnp.concatenate([v0[None], vf], axis=0)
    cov_fwd = jnp.concatenate([p0[None], pf], axis=0)
    # quat_pred[0] = observation[0], vel_pred[0] = 0 — only [1:] is read below.
    quat_pred = jnp.concatenate([q0[None], qp], axis=0)
    vel_pred = jnp.concatenate([v0[None], vp], axis=0)

    def backward(carry, xs):
        q_next_s, v_next_s = carry
        q_f, v_f, p_f, q_next_pred, v_next_pred = xs
        next_pred_cov = f_mat @ p_f @ f_mat.T + q_mat
        gain = p_f @ f_mat.T @ jnp.linalg.inv(next_pred_cov)
        rot_residual = _quaternion_residual(q_next_pred, q_next_s)
        vel_residual = v_next_s - v_next_pred
        residual = jnp.concatenate([rot_residual, vel_residual], axis=-1)
        delta = (gain @ residual[..., None])[..., 0]
        q_s = _right_compose_delta(q_f, delta[:, :3])
        v_s = v_f + delta[:, 3:]
        return (q_s, v_s), q_s

    _, q_smooth = jax.lax.scan(
        backward, (quat_fwd[-1], vel_fwd[-1]),
        (quat_fwd[:-1], vel_fwd[:-1], cov_fwd[:-1], quat_pred[1:], vel_pred[1:]),
        reverse=True)
    quat_smooth = jnp.concatenate([q_smooth, quat_fwd[-1][None]], axis=0)
    return quaternion_normalize_xyzw(quat_smooth)


def rts_smooth_rotations(
    rotations: jnp.ndarray,
    *,
    fps: float,
    gains: RTSSmoothingGains,
    joint_gains: Mapping[int, RTSSmoothingGains] | None = None,
) -> jnp.ndarray:
    """Smooth ``(T, J, 3, 3)`` rotations with an SO(3) error-state RTS model."""
    rotations = jnp.asarray(rotations)
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape (T, J, 3, 3), got {rotations.shape}.")
    if not jnp.issubdtype(rotations.dtype, jnp.floating):
        raise TypeError("rotations must be a floating-point array.")
    fps = _validate_fps(fps)
    num_frames, num_joints = rotations.shape[:2]
    if num_frames < 2:
        return rotations
    rotations_w = project_rotations_to_so3(rotations.astype(_work_dtype()))
    quaternions = matrix_to_quaternion_xyzw(rotations_w)
    result = jnp.empty_like(quaternions)
    for indices, group_gains in _gain_index_groups(num_joints, gains, joint_gains):
        result = result.at[:, indices].set(
            _smooth_rotation_group(quaternions[:, indices], fps=fps, gains=group_gains))
    return quaternion_xyzw_to_rotmat(result).astype(rotations.dtype)


def _joint_names_from_layer(soma_layer) -> list[str] | None:
    if soma_layer is None:
        return None
    if hasattr(soma_layer, "public_joint_names"):
        return [str(n) for n in soma_layer.public_joint_names]
    return None


def _parent_ids_from_layer(soma_layer):
    if soma_layer is None:
        return None
    if hasattr(soma_layer, "output_joint_parent_ids"):
        return soma_layer.output_joint_parent_ids
    return None


def _matching_names_and_parents(num_joints: int, joint_names, joint_parent_ids):
    if joint_names is None:
        return None, None
    names = [str(name) for name in joint_names]
    if len(names) == num_joints:
        parents = (joint_parent_ids
                   if joint_parent_ids is not None and len(joint_parent_ids) == num_joints
                   else None)
        return names, parents
    if names and names[0] == "Root" and len(names) - 1 == num_joints:
        return names[1:], None
    raise ValueError(f"Got {num_joints} rotations but {len(names)} joint names.")


def _name_matches_any(name: str, tokens: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(token.lower() in lowered for token in tokens)


def derive_smoothing_groups(
    joint_names: Sequence[str],
    joint_parent_ids: Sequence[int] | np.ndarray | None = None,
) -> RTSSmoothingGroups:
    """Derive hand and limb smoothing groups from joint names and topology."""
    names = [str(name) for name in joint_names]
    parents = None
    if joint_parent_ids is not None:
        parents = [int(p) for p in np.asarray(joint_parent_ids).tolist()]

    hand_indices: set[int] = set()
    if parents is not None:
        for idx, name in enumerate(names):
            lowered = name.lower()
            if lowered.endswith("hand") or lowered.endswith("wrist"):
                hand_indices.add(idx)
                hand_indices.update(get_joint_descendents(np.asarray(parents), idx))

    hand_tokens = ("hand", "thumb", "index", "middle", "ring", "pinky")
    for idx, name in enumerate(names):
        if _name_matches_any(name, hand_tokens):
            hand_indices.add(idx)

    limb_tokens = ("shoulder", "arm", "forearm", "leg", "shin", "foot", "toe")
    limb_indices = {
        idx for idx, name in enumerate(names)
        if idx not in hand_indices and _name_matches_any(name, limb_tokens)
    }
    return RTSSmoothingGroups(hand_indices=frozenset(hand_indices),
                              limb_indices=frozenset(limb_indices))


def _joint_gain_map(groups: RTSSmoothingGroups, config: RTSSmoothingConfig, *,
                    use_hand_gains: bool, include_limb_gains: bool
                    ) -> dict[int, RTSSmoothingGains]:
    if not use_hand_gains or config.hand is None:
        return {}
    return {idx: config.hand for idx in groups.fast_indices(include_limb_gains)}


def _public_orient_from_layer(soma_layer, dtype):
    """Upstream ``_public_orient_from_layer``: the skinning rig's T-pose at the
    public joints, with the public hierarchy — or ``None`` for a layer without one."""
    if not all(hasattr(soma_layer, attr) for attr in
               ("public_transform_joint_indices", "public_joint_parent_ids", "t_pose_world")):
        return None
    if soma_layer.t_pose_world is None:
        return None
    public_indices = np.asarray(soma_layer.public_transform_joint_indices)
    t_pose_world = jnp.asarray(soma_layer.t_pose_world, dtype)[public_indices]
    return _orient_pair(t_pose_world, np.asarray(soma_layer.public_joint_parent_ids))


def _resolve_joint_orient(*, expected_joints: int, dtype, soma_layer=None,
                          t_pose_orient=None, t_pose_orient_parent_T=None):
    if t_pose_orient is not None or t_pose_orient_parent_T is not None:
        if t_pose_orient is None or t_pose_orient_parent_T is None:
            raise ValueError("Pass both t_pose_orient and t_pose_orient_parent_T, or neither.")
        orient = jnp.asarray(t_pose_orient, dtype)
        orient_parent_t = jnp.asarray(t_pose_orient_parent_T, dtype)
    elif soma_layer is not None:
        public_orient = _public_orient_from_layer(soma_layer, dtype)
        if public_orient is not None:
            orient, orient_parent_t = public_orient
        else:
            orient = getattr(soma_layer, "_t_pose_orient", None)
            orient_parent_t = getattr(soma_layer, "_t_pose_orient_parent_T", None)
            if orient is None or orient_parent_t is None:
                raise ValueError("SOMA convention conversion requires joint orient tensors.")
            orient = jnp.asarray(orient, dtype)
            orient_parent_t = jnp.asarray(orient_parent_t, dtype)
    else:
        raise ValueError(
            "rotation_convention='absolute' requires soma_layer or explicit joint orient tensors.")
    if orient.shape[0] != expected_joints or orient_parent_t.shape[0] != expected_joints:
        raise ValueError(
            "Joint orient tensors must match the rotation joint count. "
            f"Got orient={orient.shape[0]}, parent={orient_parent_t.shape[0]}, "
            f"rotations={expected_joints}.")
    return orient, orient_parent_t


def _config_from_preset(preset: str, config: RTSSmoothingConfig | None,
                        fps: float | None) -> RTSSmoothingConfig:
    if config is None:
        if preset not in RTS_SMOOTHING_PRESETS:
            raise ValueError(f"Unknown RTS smoothing preset: {preset!r}.")
        config = RTS_SMOOTHING_PRESETS[preset]
    if fps is not None:
        config = replace(config, fps=float(fps))
    _validate_fps(config.fps)
    return config


def smooth_pose(
    rotations: jnp.ndarray,
    root_translation: jnp.ndarray | None = None,
    *,
    soma_layer=None,
    t_pose_orient: jnp.ndarray | None = None,
    t_pose_orient_parent_T: jnp.ndarray | None = None,
    joint_names: Sequence[str] | None = None,
    joint_parent_ids: Sequence[int] | np.ndarray | None = None,
    fps: float | None = None,
    preset: str = "default",
    config: RTSSmoothingConfig | None = None,
    rotation_convention: RotationConvention = "absolute",
    output_rotation_convention: RotationConvention | None = None,
    use_hand_gains: bool = True,
    include_limb_gains: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Smooth SOMA rotations and root translation.

    Args:
        rotations: ``(T, J, 3, 3)`` local rotation matrices.
        root_translation: Optional ``(T, 3)`` root translation.
        soma_layer: Optional :class:`~soma_jax.SOMALayer` supplying public joint
            names, parents, and the T-pose joint orient for convention
            conversion.
        rotation_convention: ``"absolute"`` for PoseInversion-style rotations
            with joint orient baked in, or ``"relative"`` for T-pose-relative
            rotations. The default output convention matches the input.

    Returns:
        ``(rotations, root_translation)``, smoothed.
    """
    if rotation_convention not in ("absolute", "relative"):
        raise ValueError(f"Unsupported rotation_convention: {rotation_convention!r}.")
    if output_rotation_convention is None:
        output_rotation_convention = rotation_convention
    if output_rotation_convention not in ("absolute", "relative"):
        raise ValueError(f"Unsupported output_rotation_convention: {output_rotation_convention!r}.")
    rotations = jnp.asarray(rotations)
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape (T, J, 3, 3), got {rotations.shape}.")
    if root_translation is not None:
        root_translation = jnp.asarray(root_translation)
        if root_translation.ndim != 2 or root_translation.shape != (rotations.shape[0], 3):
            raise ValueError(
                "root_translation must have shape (T, 3) matching rotations; "
                f"got {root_translation.shape} for rotations {rotations.shape}.")

    config = _config_from_preset(preset, config, fps)
    joint_names = joint_names or _joint_names_from_layer(soma_layer)
    joint_parent_ids = (joint_parent_ids if joint_parent_ids is not None
                        else _parent_ids_from_layer(soma_layer))

    if rotation_convention == "absolute":
        orient, orient_parent_t = _resolve_joint_orient(
            expected_joints=rotations.shape[1], dtype=rotations.dtype,
            soma_layer=soma_layer, t_pose_orient=t_pose_orient,
            t_pose_orient_parent_T=t_pose_orient_parent_T)
        working_rotations = _remove_orient(rotations, orient, orient_parent_t)
    else:
        orient = orient_parent_t = None
        working_rotations = rotations

    group_names, group_parents = _matching_names_and_parents(
        working_rotations.shape[1], joint_names, joint_parent_ids)
    groups = (derive_smoothing_groups(group_names, group_parents)
              if group_names is not None else RTSSmoothingGroups())

    smoothed_relative = rts_smooth_rotations(
        working_rotations, fps=config.fps, gains=config.rotation,
        joint_gains=_joint_gain_map(groups, config, use_hand_gains=use_hand_gains,
                                    include_limb_gains=include_limb_gains))

    smoothed_root = root_translation
    if (root_translation is not None and config.smooth_root_translation
            and config.translation is not None):
        smoothed_root = rts_smooth_euclidean(
            root_translation[:, None, :], fps=config.fps, gains=config.translation)[:, 0]

    if output_rotation_convention == "relative":
        return smoothed_relative, smoothed_root
    if orient is None or orient_parent_t is None:
        orient, orient_parent_t = _resolve_joint_orient(
            expected_joints=smoothed_relative.shape[1], dtype=smoothed_relative.dtype,
            soma_layer=soma_layer, t_pose_orient=t_pose_orient,
            t_pose_orient_parent_T=t_pose_orient_parent_T)
    return _apply_orient(smoothed_relative, orient, orient_parent_t), smoothed_root


def so3_angular_velocity(rotations: jnp.ndarray) -> jnp.ndarray:
    """Return per-frame SO(3) step vectors with shape ``(T - 1, J, 3)``."""
    rotations = jnp.asarray(rotations)
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape (T, J, 3, 3), got {rotations.shape}.")
    if rotations.shape[0] < 2:
        return jnp.zeros((0, rotations.shape[1], 3), rotations.dtype)
    quaternions = matrix_to_quaternion_xyzw(project_rotations_to_so3(rotations))
    relative = quaternion_multiply_xyzw(quaternion_conjugate_xyzw(quaternions[:-1]),
                                        quaternions[1:])
    return quaternion_log_xyzw(relative.reshape(-1, 4)).reshape(rotations.shape[0] - 1, -1, 3)


def so3_angular_acceleration(rotations: jnp.ndarray) -> jnp.ndarray:
    """Return finite-difference angular acceleration vectors."""
    velocity = so3_angular_velocity(rotations)
    if velocity.shape[0] < 2:
        return jnp.zeros((0, velocity.shape[1], 3), velocity.dtype)
    return velocity[1:] - velocity[:-1]


def euclidean_acceleration(values: jnp.ndarray) -> jnp.ndarray:
    """Return second differences for ``(T, D)`` or ``(T, J, D)`` values."""
    values = jnp.asarray(values)
    if values.shape[0] < 3:
        return jnp.zeros((0, *values.shape[1:]), values.dtype)
    return values[2:] - 2.0 * values[1:-1] + values[:-2]


__all__ = [
    "DEFAULT_RTS_SMOOTHING_CONFIG",
    "RTSSmoothingConfig",
    "RTSSmoothingGains",
    "RTSSmoothingGroups",
    "RTS_SMOOTHING_PRESETS",
    "STRONG_RTS_SMOOTHING_CONFIG",
    "derive_smoothing_groups",
    "euclidean_acceleration",
    "rts_smooth_euclidean",
    "rts_smooth_rotations",
    "smooth_pose",
    "so3_angular_acceleration",
    "so3_angular_velocity",
]
