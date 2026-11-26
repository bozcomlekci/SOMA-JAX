"""Shared optimization and SOMA NPZ helpers for identity backend conversion.

Upstream: ``tools/identity_conversion.py`` (SOMA-X v0.3.0). Same objective and
schedule: evaluate the source identity in the SOMA bind pose, then fit the
target backend's parameters by Adam on the mean squared distance between the
centred bind-pose vertices, plus a small L2 pull toward the neutral parameters,
keeping the best iterate.

JAX port. Differences are mechanical: gradients come from ``jax.value_and_grad``
over one jitted step (optax's Adam, whose defaults are torch's), layers are
immutable so the SOMA bone-scale re-evaluation passes the neutral scales
explicitly instead of swapping ``layer._cached_scale_params``, and a layer is
any :class:`soma_jax.SOMALayer` or :class:`soma_jax.hand.SOMAHandLayer`.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from soma_jax.io import load_soma_npz, save_soma_npz

logger = logging.getLogger(__name__)

_STANDARD_NPZ_KEYS = {
    "poses", "transl", "joint_names", "identity_model_type", "identity_coeffs",
    "rotation_repr", "absolute_pose", "unit", "keep_root", "scale_params",
    "joint_orient", "global_scale", "hand_type",
    # NumPy versions before 2.5 serialize save_soma_npz's allow_pickle=False
    # argument as an array instead of consuming it as a writer option.
    "allow_pickle",
}


@dataclass
class IdentityConversionResult:
    """Optimized target parameters and bind-pose reconstruction diagnostics."""

    identity_coeffs: np.ndarray
    scale_params: np.ndarray | None
    global_scale: float
    vertex_error: np.ndarray
    loss_history: list[float]


def _is_hand(layer: Any) -> bool:
    return hasattr(layer, "hand_type")


def _scale_param_count(layer: Any) -> int | None:
    if layer.identity_model_type == "soma" and _is_hand(layer):
        return len(layer.joint_parent_ids) - 1
    value = getattr(layer, "num_scale_params", None)
    if value is not None:
        return int(value)
    value = getattr(layer.identity_model, "num_scale_params", None)
    return None if value is None else int(value)


def _identity_param_count(layer: Any) -> int:
    return int(layer.identity_model.num_identity_coeffs)


def neutral_scale_params(layer: Any, batch_size: int) -> jnp.ndarray | None:
    """Backend-neutral scale parameters: 1 for SOMA bone scales, 0 otherwise."""
    count = _scale_param_count(layer)
    if count is None:
        return None
    fill_value = 1.0 if layer.identity_model_type == "soma" else 0.0
    return jnp.full((batch_size, count), fill_value, jnp.float32)


def _as_parameter_rows(values, *, rows: int, width: int | None, name: str, default=None):
    if width is None:
        if values is not None:
            raise ValueError(f"{name} were provided, but this backend does not use them")
        return None
    if values is None:
        if default is None:
            raise ValueError(f"Missing required {name}")
        return default
    array = jnp.asarray(values, jnp.float32)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Expected {name} with shape (N, {width}); got {tuple(array.shape)}")
    if array.shape[0] == 1 and rows > 1:
        array = jnp.broadcast_to(array, (rows, width))
    elif array.shape[0] != rows:
        raise ValueError(f"Expected {name} to have 1 or {rows} rows; got {array.shape[0]}")
    return array


def _bind_pose_rotations(layer: Any, batch_size: int) -> jnp.ndarray:
    """Local bind-pose rotations; the body's virtual Root is identity, as upstream pads it."""
    if _is_hand(layer):
        local = jnp.asarray(layer.bind_pose_local)[..., :3, :3]
    else:
        local = jnp.asarray(np.asarray(layer._bind_pose_local_np, np.float32))[..., :3, :3]
        local = local.at[0].set(jnp.eye(3, dtype=local.dtype))
    return jnp.broadcast_to(local[None], (batch_size,) + local.shape)


def bind_pose_vertices(layer: Any, identity_coeffs, scale_params, global_scale) -> jnp.ndarray:
    """Evaluate an identity as SOMA-topology vertices in the SOMA bind pose.

    Upstream ``bind_pose_vertices``: ``prepare_identity(repose_to_bind_pose=True)``;
    SOMA bone scales act at pose time, so their effect is added as the
    difference between the bind pose posed with and without them.
    """
    B = identity_coeffs.shape[0]
    if _is_hand(layer):
        identity = layer.prepare_identity(identity_coeffs, scale_params=scale_params,
                                          repose_to_bind_pose=True, global_scale=global_scale)
        rest = identity.rest_shape
        if layer.identity_model_type != "soma" or scale_params is None:
            return rest
        R = _bind_pose_rotations(layer, B)
        kw = dict(pose2rot=False, apply_correctives=False, absolute_pose=True)
        scaled = layer.pose(R, identity, **kw)["vertices"]
        neutral = layer.pose(R, identity._replace(scale_params=neutral_scale_params(layer, B)),
                             **kw)["vertices"]
        return rest + scaled - neutral

    rest, joints, binds = layer.prepare_identity(
        identity_coeffs, scale_params, repose_to_bind_pose=True, global_scale=global_scale,
        return_bind_transforms=True)
    if layer.identity_model_type != "soma" or scale_params is None:
        return rest
    R = _bind_pose_rotations(layer, B)
    zeros = jnp.zeros((B, 3), rest.dtype)
    kw = dict(bind_transforms=binds, absolute_pose=True, apply_correctives=False)
    scaled = layer.pose(R, zeros, rest, joints, bone_scales=scale_params, **kw).vertices
    neutral = layer.pose(R, zeros, rest, joints,
                         bone_scales=neutral_scale_params(layer, B), **kw).vertices
    return rest + scaled - neutral


def _center_vertices(vertices: jnp.ndarray) -> jnp.ndarray:
    return vertices - vertices.mean(axis=1, keepdims=True)


def _unit_scale(layer: Any, unit: str | None) -> float:
    """Factor taking the layer's vertices into ``unit``.

    Upstream builds both layers with ``output_unit=unit``, so the loss — and
    with it the balance against the unitless regularizer — is measured in the
    NPZ's unit. A layer with its own ``output_unit`` (the hand) is built that
    way by the factory; the body layer works in metres and is rescaled here.
    """
    if unit is None or hasattr(layer, "output_unit"):
        return 1.0
    from soma_jax.units import Unit
    return 1.0 / Unit.from_name(unit).meters_per_unit


def convert_identity_parameters(
    source_layer: Any,
    target_layer: Any,
    source_identity_coeffs,
    *,
    source_scale_params=None,
    global_scale: float = 1.0,
    optimize_scale_params: bool = True,
    optimize_global_scale: bool = False,
    iterations: int = 200,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    unit: str | None = None,
) -> IdentityConversionResult:
    """Fit target-backend parameters to source geometry in the SOMA bind pose.

    ``unit`` (SOMA-JAX extra) names the unit the loss is measured in when a
    layer does not carry its own ``output_unit``; see :func:`_unit_scale`.
    """
    import optax

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if global_scale <= 0:
        raise ValueError("global_scale must be greater than zero")
    source_coeffs = jnp.asarray(source_identity_coeffs, jnp.float32)
    if source_coeffs.ndim == 1:
        source_coeffs = source_coeffs[None]
    if source_coeffs.ndim != 2:
        raise ValueError(
            f"source identity_coeffs must have shape (N, C); got {tuple(source_coeffs.shape)}")
    expected_source_width = _identity_param_count(source_layer)
    if source_coeffs.shape[1] != expected_source_width:
        raise ValueError(
            f"Source backend '{source_layer.identity_model_type}' expects "
            f"{expected_source_width} identity coefficients; got {source_coeffs.shape[1]}")

    rows = source_coeffs.shape[0]
    source_scales = _as_parameter_rows(
        source_scale_params, rows=rows, width=_scale_param_count(source_layer),
        name="source scale_params", default=neutral_scale_params(source_layer, rows))
    source_vertices = _center_vertices(
        bind_pose_vertices(source_layer, source_coeffs, source_scales, global_scale)
        * _unit_scale(source_layer, unit))
    target_unit_scale = _unit_scale(target_layer, unit)

    target_neutral_scale = neutral_scale_params(target_layer, rows)
    params = {"coeffs": jnp.zeros((rows, _identity_param_count(target_layer)), jnp.float32)}
    fixed_scales = target_neutral_scale
    if target_neutral_scale is not None and optimize_scale_params:
        params["scales"] = target_neutral_scale
        fixed_scales = None
    log_global_scale0 = float(np.log(global_scale))
    if optimize_global_scale:
        params["log_global_scale"] = jnp.asarray(log_global_scale0, jnp.float32)

    def _unpack(p):
        scales = p.get("scales", fixed_scales)
        gs = jnp.exp(p["log_global_scale"]) if optimize_global_scale else global_scale
        return p["coeffs"], scales, gs

    def loss_fn(p):
        coeffs, scales, gs = _unpack(p)
        target_vertices = _center_vertices(
            bind_pose_vertices(target_layer, coeffs, scales, gs) * target_unit_scale)
        if target_vertices.shape != source_vertices.shape:
            raise ValueError(
                "Source and target layers produced different SOMA topology shapes: "
                f"{tuple(source_vertices.shape)} vs {tuple(target_vertices.shape)}")
        data_loss = jnp.mean((target_vertices - source_vertices) ** 2)
        reg_loss = jnp.mean(coeffs ** 2)
        if "scales" in p:
            reg_loss = reg_loss + jnp.mean((p["scales"] - target_neutral_scale) ** 2)
        if optimize_global_scale:
            reg_loss = reg_loss + (p["log_global_scale"] - log_global_scale0) ** 2
        return data_loss + regularization * reg_loss

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        return loss, optax.apply_updates(p, updates), state

    loss_history: list[float] = []
    best_loss = float("inf")
    best = params
    for _ in range(iterations):
        loss, new_params, opt_state = step(params, opt_state)
        loss_value = float(loss)
        loss_history.append(loss_value)
        if loss_value < best_loss:        # the loss was evaluated at `params`
            best_loss, best = loss_value, params
        params = new_params

    coeffs, scales, gs = _unpack(best)
    final_vertices = _center_vertices(
        bind_pose_vertices(target_layer, coeffs, scales, gs) * target_unit_scale)
    vertex_error = jnp.linalg.norm(final_vertices - source_vertices, axis=-1).mean(axis=-1)
    best_scales = best.get("scales", fixed_scales)
    return IdentityConversionResult(
        identity_coeffs=np.asarray(coeffs),
        scale_params=None if best_scales is None else np.asarray(best_scales),
        global_scale=float(gs),
        vertex_error=np.asarray(vertex_error),
        loss_history=loss_history,
    )


def _poses_for_resave(data: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    poses = np.asarray(data["poses"])
    joint_names = [str(n) for n in data["joint_names"]]
    if bool(data["keep_root"]):
        return poses, joint_names
    root_shape = list(poses.shape)
    root_shape[1] = 1
    if str(data["rotation_repr"]) == "matrix":
        root_pose = np.broadcast_to(np.eye(3, dtype=poses.dtype), root_shape).copy()
    else:
        root_pose = np.zeros(root_shape, dtype=poses.dtype)
    return np.concatenate([root_pose, poses], axis=1), ["Root", *joint_names]


def convert_soma_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_backend: str,
    layer_factory: Callable[[str, str, str], Any],
    optimize_scale_params: bool = True,
    optimize_global_scale: bool = False,
    iterations: int = 200,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    expected_hand_type: str | None = None,
) -> IdentityConversionResult:
    """Load, convert, and resave a canonical SOMA NPZ animation."""
    data = load_soma_npz(str(input_path))
    source_backend = str(data["identity_model_type"]).lower()
    file_hand_type = data.get("hand_type")
    if file_hand_type is not None:
        file_hand_type = str(file_hand_type)
    if expected_hand_type is None and file_hand_type is not None:
        raise ValueError("Full-body conversion does not accept a hand SOMA NPZ")
    if expected_hand_type is not None and file_hand_type is not None \
            and file_hand_type != expected_hand_type:
        raise ValueError(
            f"Input hand_type is '{file_hand_type}', but '{expected_hand_type}' was requested")

    unit = str(data["unit"])
    source_layer = layer_factory(source_backend, unit, "source")
    target_layer = layer_factory(target_backend, unit, "target")
    input_global_scale = float(data.get("global_scale", 1.0))
    result = convert_identity_parameters(
        source_layer, target_layer, data["identity_coeffs"],
        source_scale_params=data.get("scale_params"),
        global_scale=input_global_scale,
        optimize_scale_params=optimize_scale_params,
        optimize_global_scale=optimize_global_scale,
        iterations=iterations, learning_rate=learning_rate, regularization=regularization,
        unit=unit,
    )

    extra_arrays: dict[str, Any] = {
        key: value for key, value in data.items() if key not in _STANDARD_NPZ_KEYS}
    extra_arrays.update({
        "conversion_source_identity_model_type": np.array(source_backend),
        "conversion_source_identity_coeffs": np.asarray(data["identity_coeffs"]),
        "conversion_vertex_error": result.vertex_error.astype(np.float32),
        "conversion_iterations": np.int32(iterations),
        "conversion_loss_history": np.asarray(result.loss_history, dtype=np.float32),
    })
    if "scale_params" in data:
        extra_arrays["conversion_source_scale_params"] = np.asarray(data["scale_params"])
    if "global_scale" in data:
        extra_arrays["conversion_source_global_scale"] = np.float32(input_global_scale)

    poses, joint_names = _poses_for_resave(data)
    output_global_scale = (
        result.global_scale if optimize_global_scale or "global_scale" in data else None)
    save_soma_npz(
        str(output_path), poses, np.asarray(data["transl"]),
        joint_names=joint_names,
        identity_model_type=target_backend,
        identity_coeffs=result.identity_coeffs,
        scale_params=result.scale_params,
        joint_orient=data.get("joint_orient"),
        global_scale=output_global_scale,
        hand_type=expected_hand_type,
        unit=unit,
        keep_root=bool(data["keep_root"]),
        extra_arrays=extra_arrays,
    )
    logger.info("Converted %s -> %s with mean bind-pose vertex error %.6f %s",
                source_backend, target_backend, float(result.vertex_error.mean()), unit)
    return result


def model_kwargs(model_path: str | None) -> Mapping[str, Any] | None:
    """Identity-model constructor kwargs for optional licensed model files."""
    return None if model_path is None else {"model_path": model_path}
