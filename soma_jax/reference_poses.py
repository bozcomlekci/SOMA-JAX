"""Validation and offline lookup of reference orientations in the core NPZ asset.

Upstream: ``soma/reference_poses.py`` (SOMA-X v0.3.1).

A *reference pose* is the set of world joint orientations that pose rotations are
expressed relative to — by default the layer's current T-pose. SOMA-X v0.3.1 lets
callers choose another one (``SOMALayer(reference_pose=...)``,
``pose(..., reference_pose=...)``) and re-express rotations between two
(``convert_reference``). Historical T-poses ship inside ``SOMA_neutral.npz``
under ``reference_pose_history_metadata`` so they can be looked up without
downloading older releases.

Faithful port. Differences are JAX-mechanical, not semantic:

* Arrays are ``jax.numpy`` / ``numpy`` instead of ``torch.Tensor``; the
  ``TypeError`` for non-tensor input becomes one for non-array input.
* Value checks (finite, orthonormal, identity Root) need concrete numbers. Under
  ``jax.jit`` a reference is a tracer, so :func:`validate_reference_pose` keeps
  the shape/dtype checks and skips the value checks — validate eagerly, before
  tracing, if that matters. Upstream has no traced mode to worry about.
* The orient algebra uses upstream's explicit ``(orient, orient_parent_T)``
  pair (``R_out = orient_parent_T @ R @ orient``) with the *raw* parent index,
  not :func:`soma_jax.geometry.rig_utils.apply_joint_orient_local`'s
  ``parents`` form, which pins the root's parent orient to identity. The two
  agree for body references (whose virtual Root is identity) but not for hand
  references, whose joint 0 is a real wrist.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "REFERENCE_POSE_ATOL",
    "HISTORY_METADATA_KEY",
    "ReferencePoseHistory",
    "convert_reference_rotations",
    "normalize_soma_version",
    "reference_pose_array_keys",
    "reference_pose_key",
    "semantic_version_key",
    "validate_reference_pose",
]

REFERENCE_POSE_ATOL = 1e-4
HISTORY_METADATA_KEY = "reference_pose_history_metadata"
_SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def normalize_soma_version(version: str) -> str:
    """Validate SemVer and remove an optional leading v from a caller's version."""
    normalized = version.removeprefix("v") if isinstance(version, str) else None
    match = _SEMVER.fullmatch(normalized) if normalized is not None else None
    if match is None or (
        match[4] is not None
        and any(
            part.isdigit() and len(part) > 1 and part.startswith("0")
            for part in match[4].split(".")
        )
    ):
        raise ValueError("version must be a semantic version, optionally prefixed with 'v'.")
    return normalized


def semantic_version_key(version: str) -> tuple:
    """SemVer precedence: numeric components, prerelease identifiers, no build metadata."""
    match = _SEMVER.fullmatch(normalize_soma_version(version))
    prerelease = match[4]
    identifiers = (
        tuple((0, int(part)) if part.isdigit() else (1, part) for part in prerelease.split("."))
        if prerelease is not None
        else ()
    )
    return (int(match[1]), int(match[2]), int(match[3]), prerelease is None, identifiers)


def _validate_data_key(data_key: str) -> None:
    if (
        not isinstance(data_key, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", data_key) is None
        or "__" in data_key
        or data_key.startswith("reference_pose_history_")
    ):
        raise ValueError("data_key must be an identifier without the reserved '__' delimiter.")


def _validate_alias(alias: str) -> None:
    if not isinstance(alias, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", alias) is None:
        raise ValueError("alias must be a nonempty name using letters, digits, '.', '_' or '-'.")


def reference_pose_key(
    *,
    soma_version: str | None = None,
    data_key: str = "t_pose_world",
    asset_revision: str | None = None,
) -> str:
    """Return the canonical NPZ array key for an exact release or asset revision."""
    if (soma_version is None) == (asset_revision is None):
        raise ValueError("Specify exactly one of soma_version or asset_revision.")
    _validate_data_key(data_key)
    if soma_version is not None:
        return f"{data_key}__v{normalize_soma_version(soma_version)}"
    if (
        not isinstance(asset_revision, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", asset_revision) is None
    ):
        raise ValueError("asset_revision must be 'sha256:' followed by 64 lowercase hex digits.")
    return f"{data_key}__sha256_{asset_revision[7:]}"


def reference_pose_array_keys(reference_id: str) -> tuple[str, str, str]:
    """Return the rotation array and its joint metadata companion keys."""
    return reference_id, f"{reference_id}__joint_names", f"{reference_id}__parent_ids"


def _is_history_key(key: str) -> bool:
    return (
        key.startswith(("reference_pose_history_", "t_pose_world__"))
        or re.match(r"[A-Za-z][A-Za-z0-9_]*__(?:v[0-9]|sha256_)", key) is not None
    )


def _is_array(x) -> bool:
    return isinstance(x, (np.ndarray, jax.Array))


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def validate_reference_pose(
    reference_pose,
    joint_count: int,
    *,
    require_identity_root: bool = True,
):
    """Return the rotation block of a reference pose after validating it.

    Accept one joint reference. Body references require identity virtual Root;
    hand references include a real wrist and set ``require_identity_root=False``.
    SO(3) and Root checks use absolute tolerance ``1e-4`` (zero relative
    tolerance). Translation and the bottom row of 4x4 transforms are unused.

    Args:
        reference_pose: ``(J, 3, 3)`` rotations or ``(J, 4, 4)`` transforms.
        joint_count: expected ``J``, including the root.
        require_identity_root: body references (virtual Root) must have an
            identity joint 0.

    Returns:
        The ``(J, 3, 3)`` rotation block, gradients intact.

    Raises:
        TypeError: not a floating-point array.
        ValueError: wrong shape, or (when concrete) non-finite, non-SO(3), or a
            non-identity Root. Under ``jax.jit`` the value checks are skipped.
    """
    if not _is_array(reference_pose):
        raise TypeError("reference_pose must be a jax or numpy array.")
    if tuple(reference_pose.shape) not in ((joint_count, 3, 3), (joint_count, 4, 4)):
        raise ValueError(
            f"reference_pose must have shape ({joint_count}, 3, 3) or "
            f"({joint_count}, 4, 4), in joint order including the root; "
            f"got {tuple(reference_pose.shape)}. Batched/internal-joint references are unsupported."
        )
    if not jnp.issubdtype(reference_pose.dtype, jnp.floating):
        raise TypeError("reference_pose must contain floating-point rotation matrices.")
    rotations = reference_pose[..., :3, :3]
    if _is_tracer(rotations):
        return rotations
    # Concrete: run upstream's checks in float64/float32 on host.
    checked = np.asarray(rotations)
    checked = checked.astype(np.float64 if checked.dtype == np.float64 else np.float32)
    if not np.isfinite(checked).all():
        raise ValueError("reference_pose rotation blocks must be finite.")
    eye = np.eye(3, dtype=checked.dtype)
    if require_identity_root and not np.allclose(checked[0], eye, atol=REFERENCE_POSE_ATOL, rtol=0):
        raise ValueError("reference_pose virtual Root orientation must be identity.")
    # Elementwise products rather than a matmul, as upstream does, so a caller's
    # reduced-precision matmul policy cannot push valid rotations out of tolerance.
    gram = (checked[..., :, :, None] * checked[..., :, None, :]).sum(axis=-3)
    determinant = (np.cross(checked[..., :, 0], checked[..., :, 1]) * checked[..., :, 2]).sum(axis=-1)
    if not np.allclose(gram, np.broadcast_to(eye, checked.shape), atol=REFERENCE_POSE_ATOL, rtol=0) \
            or not np.allclose(determinant, np.ones(joint_count, checked.dtype),
                               atol=REFERENCE_POSE_ATOL, rtol=0):
        raise ValueError(
            "reference_pose rotation blocks must be orthonormal with determinant +1 "
            f"(absolute tolerance {REFERENCE_POSE_ATOL:g})."
        )
    return rotations


class ReferencePoseHistory:
    """Private, immutable-by-API snapshots loaded from the current core asset."""

    def __init__(self, data: Mapping[str, np.ndarray]):
        self._references: list[dict[str, Any]] = []
        self._world: list[np.ndarray] = []
        self._joint_names: list[np.ndarray] = []
        self._parent_ids: list[np.ndarray] = []
        self.default_reference_id: str | None = None
        present = {key for key in data if _is_history_key(key)}
        if not present:
            return
        if HISTORY_METADATA_KEY not in data:
            raise ValueError(
                "Incomplete reference pose history in SOMA_neutral.npz: missing metadata."
            )
        metadata_array = np.asarray(data[HISTORY_METADATA_KEY])
        if metadata_array.shape != () or metadata_array.dtype.kind != "U":
            raise ValueError("Reference pose history metadata must be a Unicode JSON scalar.")
        try:
            metadata = json.loads(metadata_array.item())
        except json.JSONDecodeError as error:
            raise ValueError("Invalid reference pose history metadata JSON.") from error
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
            raise ValueError("Unsupported reference pose history schema_version; expected 2.")
        if "lookups" in metadata:
            raise ValueError("Reference pose history schema 2 stores named arrays without lookups.")
        references = metadata.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError("Reference pose history must contain a nonempty references list.")
        if not isinstance(metadata.get("conventions"), dict) or not metadata["conventions"]:
            raise ValueError("Reference pose history must document its conventions.")
        reference_ids = set()
        version_keys = set()
        alias_keys = set()
        expected_keys = {HISTORY_METADATA_KEY}
        for reference in references:
            if not isinstance(reference, dict) or not isinstance(reference.get("id"), str):
                raise ValueError("Each reference pose history entry must have a string id.")
            reference_id = reference["id"]
            selectors = set(reference) & {"soma_version", "asset_revision"}
            if len(selectors) != 1 or "data_key" not in reference:
                raise ValueError(
                    "Each reference needs data_key and one soma_version or asset_revision."
                )
            canonical = reference_pose_key(
                data_key=reference["data_key"],
                **{selector: reference[selector] for selector in selectors},
            )
            if "soma_version" in reference:
                version = reference["soma_version"]
                if version != normalize_soma_version(version):
                    raise ValueError("Stored soma_version must not have a leading 'v'.")
                version_key = (reference["data_key"], semantic_version_key(version))
                if version_key in version_keys:
                    raise ValueError(
                        "Reference pose history has ambiguous semantic version precedence."
                    )
                version_keys.add(version_key)
            if reference_id != canonical or reference_id in reference_ids:
                raise ValueError("Reference pose history IDs must be unique canonical NPZ keys.")
            reference_ids.add(reference_id)
            aliases = reference.get("aliases", [])
            if not isinstance(aliases, list):
                raise ValueError("Reference pose aliases must be a list of names.")
            for alias in aliases:
                _validate_alias(alias)
                alias_key = (reference["data_key"], alias)
                if alias_key in alias_keys:
                    raise ValueError("Reference pose aliases must be unique within each data_key.")
                alias_keys.add(alias_key)
            keys = reference_pose_array_keys(reference_id)
            expected_keys.update(keys)
            if any(key not in data for key in keys):
                raise ValueError(f"Incomplete reference pose history arrays for {reference_id!r}.")
            world, names, parents = (np.asarray(data[key]) for key in keys)
            if world.ndim != 3 or world.shape[-2:] != (3, 3):
                raise ValueError("Reference pose history world must have shape (J, 3, 3).")
            joint_count = world.shape[0]
            if joint_count == 0 or world.dtype.kind != "f":
                raise ValueError(
                    "Reference pose history world must contain floating-point rotations."
                )
            if names.shape != (joint_count,) or names.dtype.kind != "U":
                raise ValueError("Reference pose history joint_names must be a Unicode (J,) array.")
            if parents.shape != (joint_count,) or parents.dtype.kind not in "iu":
                raise ValueError("Reference pose history parent_ids must be an integer (J,) array.")
            if names[0] != "Root" or len(set(names)) != joint_count:
                raise ValueError("Reference pose history joints must be unique with Root first.")
            if np.any(names == ""):
                raise ValueError("Reference pose history joint names must be nonempty.")
            if parents[0] != 0 or np.any(parents < 0) or np.any(parents >= joint_count):
                raise ValueError(
                    "Reference pose history parents must be valid indices with Root parent 0."
                )
            for joint in range(1, joint_count):
                visited = set()
                ancestor = joint
                while ancestor != 0:
                    if ancestor in visited:
                        raise ValueError("Reference pose history hierarchy contains a cycle.")
                    visited.add(ancestor)
                    ancestor = int(parents[ancestor])
            validate_reference_pose(world.copy(), joint_count)
            self._world.append(world.copy())
            self._joint_names.append(names.copy())
            self._parent_ids.append(parents.copy())
        if present != expected_keys:
            raise ValueError(
                f"Unindexed reference pose history arrays: {sorted(present - expected_keys)}"
            )
        if (
            not isinstance(metadata.get("default_reference_id"), str)
            or metadata["default_reference_id"] not in reference_ids
        ):
            raise ValueError("Reference pose history default_reference_id must identify an entry.")
        self._references = deepcopy(references)
        self.default_reference_id = metadata["default_reference_id"]

    def list_reference_poses(self) -> list[dict[str, Any]]:
        """Return independent records with public selector names in asset order."""
        references = deepcopy(self._references)
        for reference in references:
            if "soma_version" in reference:
                reference["version"] = reference.pop("soma_version")
        return references

    def resolve_reference_id(
        self,
        reference_id: str | None = None,
        *,
        soma_version: str | None = None,
        data_key: str = "t_pose_world",
        asset_revision: str | None = None,
        alias: str | None = None,
    ) -> str:
        """Resolve the newest same-key revision at or before a requested version.

        Full NPZ keys, asset revisions and aliases remain exact selectors. Build
        metadata does not affect semantic version precedence.
        """
        selectors = (reference_id, soma_version, asset_revision, alias)
        if sum(value is not None for value in selectors) != 1:
            raise ValueError(
                "Specify exactly one of reference_id, version, asset_revision or alias."
            )
        if alias is not None:
            _validate_alias(alias)
            _validate_data_key(data_key)
            for reference in self._references:
                if reference["data_key"] == data_key and alias in reference.get("aliases", []):
                    return reference["id"]
            raise KeyError(
                f"Unknown reference pose alias {alias!r} for data_key={data_key!r} "
                "in the current SOMA_neutral.npz. Use list_reference_poses() to inspect aliases."
            )
        if reference_id is not None:
            if not isinstance(reference_id, str) or not reference_id:
                raise ValueError("Reference selectors must be nonempty strings.")
            if data_key != "t_pose_world":
                raise ValueError(
                    "data_key selects versioned data and cannot modify a reference_id."
                )
        else:
            reference_id = reference_pose_key(
                soma_version=soma_version, data_key=data_key, asset_revision=asset_revision
            )
            if soma_version is not None:
                requested = semantic_version_key(soma_version)
                candidates = [
                    (semantic_version_key(reference["soma_version"]), reference["id"])
                    for reference in self._references
                    if reference["data_key"] == data_key
                    and "soma_version" in reference
                    and semantic_version_key(reference["soma_version"]) <= requested
                ]
                if not candidates:
                    raise KeyError(
                        f"No reference pose for data_key={data_key!r} at or before "
                        f"version={soma_version!r} in the current SOMA_neutral.npz."
                    )
                reference_id = max(candidates)[1]
        if reference_id not in {reference["id"] for reference in self._references}:
            raise KeyError(
                f"Unknown reference pose {reference_id!r} in the current SOMA_neutral.npz. "
                "Use list_reference_poses() to inspect stored references. "
                "Full keys and asset revisions require an exact match; no download is performed."
            )
        return reference_id

    def get_reference_pose(
        self,
        reference_id: str,
        joint_names: Sequence[str],
        parent_ids,
        *,
        device=None,
        dtype=jnp.float32,
    ) -> jnp.ndarray:
        """Return a fresh array after matching names and parent-name hierarchy.

        ``device`` is upstream's torch device, accepted for call compatibility
        and ignored: JAX places arrays on the default device.
        """
        reference_ids = [reference["id"] for reference in self._references]
        if reference_id not in reference_ids:
            raise KeyError(
                f"Unknown reference pose {reference_id!r}; available IDs: {reference_ids}. "
                "References are loaded only from the current SOMA_neutral.npz. "
                "For an unlisted reference, pass its saved array to reference_pose."
            )
        index = reference_ids.index(reference_id)
        source_names = self._joint_names[index].tolist()
        joint_names = [str(n) for n in joint_names]
        if len(joint_names) != len(source_names) or set(joint_names) != set(source_names):
            raise ValueError(
                f"Reference pose {reference_id!r} has incompatible public joint names."
            )
        source_by_name = {name: offset for offset, name in enumerate(source_names)}
        order = [source_by_name[name] for name in joint_names]
        current_parents = [int(p) for p in np.asarray(parent_ids).tolist()]
        for joint, source_joint in enumerate(order):
            source_parent_name = source_names[self._parent_ids[index][source_joint]]
            if source_parent_name != joint_names[current_parents[joint]]:
                raise ValueError(
                    f"Reference pose {reference_id!r} has an incompatible public hierarchy."
                )
        return jnp.asarray(self._world[index][order], dtype=dtype)


def _orient_pair(world_rotations, parent_ids):
    """Upstream ``precompute_joint_orient``: ``(orient, orient[parents].T)``.

    Indexes with the raw parent array, exactly as upstream does — a root whose
    parent is itself therefore conjugates by its own orientation.
    """
    orient = world_rotations[..., :3, :3]
    parents = np.asarray(parent_ids, dtype=np.int64)
    return orient, jnp.swapaxes(orient[parents], -2, -1)


def _apply_orient(local, orient, orient_parent_T):
    """Upstream ``apply_joint_orient_local``: ``orient_parent_T @ R @ orient``."""
    return orient_parent_T[None] @ local @ orient[None]


def _remove_orient(local, orient, orient_parent_T):
    """Upstream ``remove_joint_orient_local``: ``orient_parent_T.T @ R @ orient.T``."""
    return (jnp.swapaxes(orient_parent_T, -2, -1)[None] @ local
            @ jnp.swapaxes(orient, -2, -1)[None])


def convert_reference_rotations(
    rotations,
    from_ref,
    to_ref,
    parent_ids,
    *,
    virtual_root: bool,
) -> jnp.ndarray:
    """Re-express local rotations from one reference pose in another.

    Shared body/hand conversion with explicit references and no layer state.
    Absolute local rotations are preserved: ``rotations`` posed against
    ``from_ref`` and the result posed against ``to_ref`` give the same world
    pose. Differentiable and ``jit``-compatible.

    Args:
        rotations: ``(B, P, 3, 3)`` local rotations, where ``P = J - 1`` when
            ``virtual_root`` (the body's identity Root is omitted) else ``J``.
        from_ref, to_ref: ``(J, 3, 3)`` or ``(J, 4, 4)`` reference world
            orientations in ``parent_ids`` order.
        parent_ids: ``(J,)`` parent indices.
        virtual_root: body references (identity Root, omitted from
            ``rotations``) vs hand references (real wrist at joint 0).

    Returns:
        ``(B, P, 3, 3)`` rotations relative to ``to_ref``.
    """
    joint_count = len(parent_ids)
    pose_count = joint_count - int(virtual_root)
    if not _is_array(rotations):
        raise TypeError("rotations must be a jax or numpy array.")
    if rotations.ndim != 4 or tuple(rotations.shape[1:]) != (pose_count, 3, 3):
        raise ValueError(f"rotations must have shape (B, {pose_count}, 3, 3).")
    if not jnp.issubdtype(rotations.dtype, jnp.floating):
        raise TypeError("rotations must contain floating-point rotation matrices.")
    if not _is_tracer(rotations) and not bool(np.isfinite(np.asarray(rotations)).all()):
        raise ValueError("rotations must be finite.")
    rotations = jnp.asarray(rotations)
    source = jnp.asarray(validate_reference_pose(
        from_ref, joint_count, require_identity_root=virtual_root), rotations.dtype)
    target = jnp.asarray(validate_reference_pose(
        to_ref, joint_count, require_identity_root=virtual_root), rotations.dtype)
    source_orient, source_parent_t = _orient_pair(source, parent_ids)
    target_orient, target_parent_t = _orient_pair(target, parent_ids)
    if virtual_root:
        root = jnp.broadcast_to(jnp.eye(3, dtype=rotations.dtype),
                                (rotations.shape[0], 1, 3, 3))
        rotations = jnp.concatenate([root, rotations], axis=1)
    absolute = _apply_orient(rotations, source_orient, source_parent_t)
    converted = _remove_orient(absolute, target_orient, target_parent_t)
    return converted[:, 1:] if virtual_root else converted
