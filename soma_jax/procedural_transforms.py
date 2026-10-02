"""Procedural twist-joint transforms for SOMA template rigs (JAX port).

Faithful JAX port of the core of
``third_party/SOMA-X/soma/procedural_transforms.py``: parse
``SOMA_procedural_transforms.json``, extract per-segment twist channels from
the 78 public-rig rotations, then distribute them through the trained sparse
parameter matrix to emit additional twist-joint local rotations that extend the
public rig. :meth:`ProceduralTransforms.extend_public_rotations` emits the 110
joints the JSON describes (78 public + 32 twist);
:meth:`ProceduralTransforms.extend_to_template_rig` emits every joint of the
template in its authored order. Since SOMA-X v0.2.2's simplified v0027 template
those are exactly the same 110; any USD-only helper bone (v0026 carried 12, for
122 in total) would take identity.

All three channel-extraction modes are implemented:

* ``aligned_x_swing_twist`` — operates in the bind-aligned absolute frame
  (rotations after :func:`apply_joint_orient_local`). This is the mode the
  v0.2.1 trained corrective checkpoint was distilled against.
* ``local_x_swing_twist`` — swing-twist decomposition in the local frame
  around each segment's configured axis (does not require joint orient).
* ``local_x_euler`` — Euler-XYZ decomposition; extracts the rotation around
  the segment's configured Euler axis. Numerically less stable than the
  swing-twist variants near gimbal-lock; included for parity with the JSON
  schema.

Public API
==========
* :func:`load_definition` — parse the JSON into a
  :class:`SOMAProceduralTransformDefinition`.
* :class:`ProceduralTransforms` — wraps the definition + joint-name lookup
  tables and exposes :py:meth:`extend_public_rotations`, which takes a public
  ``(B, 78, 3, 3)`` rotmat tensor and returns the full
  ``(B, 110, 3, 3)``; :meth:`extend_to_template_rig` gives the template's
  joints in template order (``(B, 110, 3, 3)`` on the v0027 template).

Translation parameter matrix
----------------------------
The 64-entry translation matrix maps each twist joint's world position to a
convex combination of two public joints' positions (e.g.
``LeftArmTwist1 = 0.95·LeftArm + 0.05·LeftForeArm``). This is implemented in
:py:meth:`ProceduralTransforms.emit_twist_world_positions`, which produces
the twist joints' bind-pose world positions directly from the public bind
positions — no USD parsing required.

SOMALayer integration
---------------------
:py:meth:`SOMALayer.extend_rig_with_procedural_transforms` wires the
procedural module into the SOMA pipeline and returns a tuple of
``(full_rotations, full_bind_positions, full_joint_names, full_parents)``
ready to feed an extended :class:`BatchedSkinning`. The expanded-rig integration
is "additive" in the sense that the original 78-joint rig output stays
identical; the 32 twist joints are leaves under their segment's start joint
and contribute only to LBS via their skinning weights.

What's NOT included
===================
* USD parsing for the upstream ``SOMA_template_rig.usda`` (28 MB since v0.2.2;
  345 MB before). Not required for runtime — the procedural module derives the twist-joint bind
  positions analytically from the public bind positions via the translation
  matrix above. The USD is the authored DCC representation, redundant with
  the JSON for inference.

Upstream: ``soma/procedural_transforms.py``
    **Complete port.** ``extend_to_template_rig`` emits all template joints in
    authored order — 110 on the v0027 template, the same set the JSON describes;
    USD-only helper bones (12 on v0026) would take identity.
    ``expand_world_transforms_from_source_fk`` reproduces upstream's expansion of
    public FK world transforms onto the procedural rig — each twist joint a
    single local step off its public parent — and ``twist_rotations_from_source``
    matches upstream to 2.4e-4 given the same posed world transforms. Per-joint
    rotation-extraction modes are dispatched as upstream does, by routing each
    twist joint's parameter-matrix row into its own mode and summing.

    ``SOMALayer.from_upstream_assets()`` drives the expanded rig end to end at
    **0.34–1.16 mm** against upstream's default constructor
    (``tests/test_procedural_parity.py``), including bone scales.

"""
from __future__ import annotations
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp

from .geometry.lbs import batch_rodrigues, compute_skeleton_levels
from .geometry.rig_utils import (
    apply_joint_orient_local,
    compute_skeleton_levels as rig_compute_skeleton_levels,
    joint_local_to_world_levelorder,
    joint_world_to_local,
    precompute_joint_orient,
)
from .geometry.transforms import (
    matrix_to_euler_xyz,
    matrix_to_quaternion_xyzw,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_twist_angle_xyzw,
    se3_from_rt,
    single_axis_rotation_matrices,
)


# Upstream constants (``soma.procedural_transforms``).
SOMA_LOCAL_X_EULER_TWIST_MODE = "local_x_euler"
SOMA_LOCAL_X_SWING_TWIST_MODE = "local_x_swing_twist"
SOMA_ALIGNED_X_SWING_TWIST_MODE = "aligned_x_swing_twist"
SOMA_PROCEDURAL_TRANSFORM_MODES = (
    SOMA_LOCAL_X_EULER_TWIST_MODE,
    SOMA_LOCAL_X_SWING_TWIST_MODE,
    SOMA_ALIGNED_X_SWING_TWIST_MODE,
)
SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME = "SOMA_procedural_transforms.json"
SOMA_PROCEDURAL_AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SOMATwistSegmentSpec:
    """Generated twist helpers for one public SOMA limb segment (upstream's)."""

    start_joint: str
    end_joint: str
    twist_joints: tuple[str, ...]
    reverse: bool = False
    parent_joint: str | None = None
    source_axis: int = 0
    source_sign: float = 1.0


@dataclass(frozen=True)
class SOMANamedMatrixEntry:
    """One named sparse procedural matrix entry from the JSON sidecar."""

    row: str
    column: str
    value: float


@dataclass(frozen=True)
class SOMAProceduralParameterMatrices:
    """Compiled SOMA procedural parameter matrices (float32 host arrays)."""

    rotation: np.ndarray
    translation: np.ndarray
    segment_fractions: np.ndarray


@dataclass(frozen=True)
class SOMAProceduralTransformOutput:
    """Optional outputs from one procedural parameter transform call."""

    rotations: jnp.ndarray | None = None
    transforms: jnp.ndarray | None = None


@dataclass(frozen=True)
class SOMAProceduralTransformDefinition:
    """Portable SOMA procedural-control rig definition loaded from JSON.

    Upstream's fields, plus ``template_joint_count`` — the JSON's
    ``template_asset.joint_count``, which upstream does not read (SOMA-JAX
    extra, ``None`` when absent).
    """

    schema_version: str
    modes: tuple[str, ...]
    rotation_extraction_modes: tuple[str, ...]
    public_joint_names: tuple[str, ...]
    segments: tuple[SOMATwistSegmentSpec, ...]
    rotation_entries: tuple[SOMANamedMatrixEntry, ...]
    translation_entries: tuple[SOMANamedMatrixEntry, ...]
    path: Path | None = None
    template_joint_count: int | None = None

    @property
    def main_joint_names(self) -> tuple[str, ...]:
        """Main non-procedural joint names from the portable JSON schema."""
        return self.public_joint_names

    @property
    def twist_joint_names(self) -> tuple[str, ...]:
        """Procedural joints in segment order (SOMA-JAX convenience)."""
        return tuple(_twist_joint_names(self.segments))

    def full_joint_names(self) -> tuple[str, ...]:
        """Public joints followed by every twist joint (SOMA-JAX convenience)."""
        return tuple(self.public_joint_names) + self.twist_joint_names

    @property
    def rotation_extraction_mode(self) -> str:
        """The extraction mode a single-mode caller gets (SOMA-JAX convenience).

        Upstream stores one mode per procedural joint; every published asset
        uses one mode for all of them. Mixed definitions are dispatched per
        joint by :class:`ProceduralTransforms` and
        :class:`SOMAProceduralParameterTransform`; this reports the most
        common mode.
        """
        modes = self.rotation_extraction_modes
        if not modes:
            return SOMA_ALIGNED_X_SWING_TWIST_MODE
        return max(set(modes), key=modes.count)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_mode(value: Any, field: str, modes: Sequence[str]) -> str:
    mode = _require_string(value, field)
    if mode not in modes:
        raise ValueError(f"{field} must be one of {tuple(modes)}, got {mode!r}")
    return mode


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    strings = tuple(_require_string(item, f"{field}[]") for item in _require_sequence(value, field))
    duplicates = sorted({item for item in strings if strings.count(item) > 1})
    if duplicates:
        raise ValueError(f"{field} contains duplicate names: {duplicates}")
    return strings


def _axis_id(value: Any, field: str) -> int:
    if isinstance(value, str):
        axis = SOMA_PROCEDURAL_AXIS_TO_ID.get(value.lower())
        if axis is not None:
            return axis
    elif isinstance(value, int) and value in (0, 1, 2):
        return value
    raise ValueError(f"{field} must be one of 'x', 'y', 'z', 0, 1, or 2")


def _sign(value: Any, field: str) -> float:
    if value in (-1, -1.0):
        return -1.0
    if value in (1, 1.0):
        return 1.0
    raise ValueError(f"{field} must be -1.0 or 1.0")


def _require_segments(segments, context: str) -> tuple[SOMATwistSegmentSpec, ...]:
    if segments is None:
        raise ValueError(
            f"{context} requires explicit procedural twist segments. Load "
            f"{SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME} with "
            "load_soma_procedural_transform_definition() and pass definition.segments.")
    return tuple(segments)


def _require_rotation_extraction_modes(rotation_extraction_modes, twist_joint_names,
                                       context: str) -> tuple[str, ...]:
    if rotation_extraction_modes is None:
        raise ValueError(
            f"{context} requires explicit rotation extraction modes. Load "
            f"{SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME} with "
            "load_soma_procedural_transform_definition() and pass "
            "definition.rotation_extraction_modes.")
    modes = tuple(rotation_extraction_modes)
    if len(modes) != len(twist_joint_names):
        raise ValueError(
            f"{context} expected {len(twist_joint_names)} rotation extraction modes, "
            f"got {len(modes)}")
    for mode in modes:
        _require_mode(mode, f"{context} rotation_extraction_modes[]",
                      SOMA_PROCEDURAL_TRANSFORM_MODES)
    return modes


def _parse_rotation_extraction_modes(root: Mapping[str, Any], modes: Sequence[str],
                                     twist_joint_names: Sequence[str]) -> tuple[str, ...]:
    """Which twist extractor drives each procedural joint.

    The JSON's ``rotation_extraction`` is a bare mode applied to every
    procedural joint, or a mapping with ``default`` plus ``per_procedural_joint``
    overrides; returns one mode per twist joint.
    """
    raw = root.get("rotation_extraction")
    if raw is None:
        raise ValueError("rotation_extraction is required")
    if isinstance(raw, str):
        return (_require_mode(raw, "rotation_extraction", modes),) * len(twist_joint_names)

    config = _require_mapping(raw, "rotation_extraction")
    per_joint = _require_mapping(config.get("per_procedural_joint", {}),
                                 "rotation_extraction.per_procedural_joint")
    unknown_joints = sorted(set(per_joint) - set(twist_joint_names))
    if unknown_joints:
        raise ValueError(
            "rotation_extraction.per_procedural_joint references unknown procedural joints: "
            f"{unknown_joints}")

    default = config.get("default")
    if default is None:
        missing = [name for name in twist_joint_names if name not in per_joint]
        if missing:
            raise ValueError(
                "rotation_extraction.default is required unless every procedural joint "
                f"has an override; missing: {missing}")
        default_mode = None
    else:
        default_mode = _require_mode(default, "rotation_extraction.default", modes)

    return tuple(
        _require_mode(per_joint.get(joint_name, default_mode),
                      f"rotation_extraction.per_procedural_joint[{joint_name}]", modes)
        for joint_name in twist_joint_names)


def _parse_named_sparse_matrix(matrix_data: Any, matrix_name: str, valid_rows: set[str],
                               valid_columns: set[str],
                               require_entries: bool = True) -> tuple[SOMANamedMatrixEntry, ...]:
    matrix = _require_mapping(matrix_data, f"parameter_matrices.{matrix_name}")
    if matrix.get("format") not in (None, "sparse_coo_named"):
        raise ValueError(f"parameter_matrices.{matrix_name}.format must be 'sparse_coo_named'")
    if matrix.get("dtype") not in (None, "float32"):
        raise ValueError(f"parameter_matrices.{matrix_name}.dtype must be 'float32'")
    entries = _require_sequence(matrix.get("entries", []),
                                f"parameter_matrices.{matrix_name}.entries")
    if require_entries and not entries:
        raise ValueError(f"parameter_matrices.{matrix_name}.entries must not be empty")
    seen = set()
    parsed_entries = []
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"parameter_matrices.{matrix_name}.entries[{index}]")
        row = _require_string(entry.get("row"), f"parameter_matrices.{matrix_name}.entries[].row")
        column = _require_string(entry.get("column"),
                                 f"parameter_matrices.{matrix_name}.entries[].column")
        if row not in valid_rows:
            raise ValueError(f"unknown {matrix_name} matrix row: {row!r}")
        if column not in valid_columns:
            raise ValueError(f"unknown {matrix_name} matrix column: {column!r}")
        try:
            float(entry["value"])
        except KeyError as e:
            raise ValueError(
                f"parameter_matrices.{matrix_name}.entries[].value is required") from e
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"parameter_matrices.{matrix_name}.entries[].value must be numeric") from e
        key = (row, column)
        if key in seen:
            raise ValueError(f"duplicate {matrix_name} matrix entry for {row!r}, {column!r}")
        seen.add(key)
        parsed_entries.append(SOMANamedMatrixEntry(row=row, column=column,
                                                   value=float(entry["value"])))
    return tuple(parsed_entries)


def parse_soma_procedural_transform_definition(
    data: Mapping[str, Any],
    path: str | Path | None = None,
) -> SOMAProceduralTransformDefinition:
    """Validate and parse a portable SOMA procedural-control rig definition."""
    root = _require_mapping(data, "definition")
    schema_version = _require_string(root.get("schema_version"), "schema_version")
    modes = _string_tuple(root.get("modes"), "modes")
    unknown_modes = sorted(set(modes) - set(SOMA_PROCEDURAL_TRANSFORM_MODES))
    if unknown_modes:
        raise ValueError(f"unknown procedural transform modes: {unknown_modes}")

    channel_extractors = _require_mapping(root.get("channel_extractors"), "channel_extractors")
    unknown_extractors = sorted(set(channel_extractors) - set(SOMA_PROCEDURAL_TRANSFORM_MODES))
    if unknown_extractors:
        raise ValueError(f"unknown channel extractors: {unknown_extractors}")
    missing_extractors = [mode for mode in modes if mode not in channel_extractors]
    if missing_extractors:
        raise ValueError(f"missing channel extractors for modes: {missing_extractors}")

    public_rig = _require_mapping(root.get("public_rig_derivation"), "public_rig_derivation")
    public_joint_names = _string_tuple(public_rig.get("main_joint_names"),
                                       "public_rig_derivation.main_joint_names")
    public_joint_set = set(public_joint_names)

    segments = []
    for index, raw_segment in enumerate(_require_sequence(root.get("segments"), "segments")):
        segment_data = _require_mapping(raw_segment, f"segments[{index}]")
        start_joint = _require_string(segment_data.get("start_joint"), "segments[].start_joint")
        end_joint = _require_string(segment_data.get("end_joint"), "segments[].end_joint")
        parent_joint_raw = segment_data.get("parent_joint")
        parent_joint = (_require_string(parent_joint_raw, "segments[].parent_joint")
                        if parent_joint_raw is not None else None)
        twist_joints = _string_tuple(segment_data.get("twist_joints"), "segments[].twist_joints")
        control_names = (start_joint, end_joint)
        if parent_joint is not None:
            control_names = (*control_names, parent_joint)
        missing_controls = [name for name in control_names if name not in public_joint_set]
        if missing_controls:
            raise ValueError(
                f"segments[{index}] references joints outside the public rig: {missing_controls}")
        segments.append(SOMATwistSegmentSpec(
            start_joint=start_joint,
            end_joint=end_joint,
            twist_joints=twist_joints,
            reverse=bool(segment_data.get("reverse", False)),
            parent_joint=parent_joint,
            source_axis=_axis_id(segment_data.get("source_axis", "x"), "segments[].source_axis"),
            source_sign=_sign(segment_data.get("source_sign", 1.0), "segments[].source_sign"),
        ))

    parsed_segments = tuple(segments)
    twist_joint_names = tuple(_twist_joint_names(parsed_segments))
    duplicate_outputs = sorted(
        {name for name in twist_joint_names if twist_joint_names.count(name) > 1})
    if duplicate_outputs:
        raise ValueError(f"duplicate procedural outputs: {duplicate_outputs}")
    rotation_extraction_modes = _parse_rotation_extraction_modes(root, modes, twist_joint_names)

    parameter_matrices = _require_mapping(root.get("parameter_matrices"), "parameter_matrices")
    rotation_entries = _parse_named_sparse_matrix(
        parameter_matrices.get("rotation"), "rotation", set(twist_joint_names), public_joint_set)
    translation_entries = _parse_named_sparse_matrix(
        parameter_matrices.get("translation"), "translation", set(twist_joint_names),
        public_joint_set | set(twist_joint_names))

    template_asset = root.get("template_asset")
    template_joint_count = (template_asset.get("joint_count")
                            if isinstance(template_asset, Mapping) else None)
    return SOMAProceduralTransformDefinition(
        schema_version=schema_version,
        modes=modes,
        rotation_extraction_modes=rotation_extraction_modes,
        public_joint_names=public_joint_names,
        segments=parsed_segments,
        rotation_entries=rotation_entries,
        translation_entries=translation_entries,
        path=Path(path) if path is not None else None,
        template_joint_count=None if template_joint_count is None else int(template_joint_count),
    )


def load_soma_procedural_transform_definition(path: str | Path) -> SOMAProceduralTransformDefinition:
    """Load a portable SOMA procedural-control rig definition JSON file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid SOMA procedural transform definition JSON at {path}: {e}") from e
    try:
        return parse_soma_procedural_transform_definition(data, path=path)
    except ValueError as e:
        raise ValueError(f"Invalid SOMA procedural transform definition at {path}: {e}") from e


#: SOMA-JAX's name for :func:`load_soma_procedural_transform_definition`.
load_definition = load_soma_procedural_transform_definition


def _name_list(joint_names: Sequence[str]) -> list[str]:
    return [str(name) for name in joint_names]


def _twist_joint_names(segments: Sequence[SOMATwistSegmentSpec]) -> list[str]:
    return [twist_joint for segment in segments for twist_joint in segment.twist_joints]

# ----------------------------------------------------------------------------
# Channel extractors (delegated to soma_jax.geometry.transforms public API)
# ----------------------------------------------------------------------------
def _euler_xyz_angle(R: jnp.ndarray, axis_idx: int) -> jnp.ndarray:
    """Extract one twist-source channel from a rotation matrix.

    Port of upstream ``_local_euler_xyz_from_matrix`` (the source extraction the
    ``local_x_euler`` twist mode gathers from, ``procedural_transforms.py:1368``)
    — **not** of ``matrix_to_euler_xyz``. Upstream stacks three channels:

    ==== ==================================== ===========================
    axis upstream source                      formula
    ==== ==================================== ===========================
    X    ``local_x_euler_from_matrix``        ``atan2(R[2,1], R[1,1])``
    Y    ``matrix_to_euler_xyz(...)[..., 1]`` ``asin(-R[2,0])``
    Z    ``matrix_to_euler_xyz(...)[..., 2]`` ``atan2(R[1,0], R[0,0])``
    ==== ==================================== ===========================

    The X channel is deliberately **not** the standard Euler X
    (``atan2(R[2,1], R[2,2])``). Upstream uses the ``R[1,1]`` denominator so the
    angle stays meaningful when the matrix carries swing as well as twist, which
    is exactly the case for the arm/leg segments that drive the twist helpers.
    Substituting the standard Euler X diverges by >1e-3 on generic rotations and
    silently corrupts the channel the twist joints are driven by — see
    ``tests/test_soma_x_parity_modules.py::TestProceduralTwistExtraction``.

    The Y channel uses the ``atan2`` form rather than upstream's ``asin``: for a
    valid rotation matrix ``sqrt(R[2,1]^2 + R[2,2]^2) == cos(y) >= 0`` makes the
    two identical, and ``atan2`` is better conditioned near gimbal lock. Parity
    is pinned to 1e-5 by the test above.
    """
    if axis_idx == 0:
        return jnp.arctan2(R[..., 2, 1], R[..., 1, 1])
    if axis_idx == 1:
        return jnp.arctan2(
            -R[..., 2, 0],
            jnp.sqrt(R[..., 2, 1] ** 2 + R[..., 2, 2] ** 2),
        )
    return jnp.arctan2(R[..., 1, 0], R[..., 0, 0])


#: SOMA-JAX alias of :data:`SOMA_PROCEDURAL_AXIS_TO_ID`.
_AXIS_NAME_TO_IDX = SOMA_PROCEDURAL_AXIS_TO_ID


class ProceduralTransforms:
    """Runtime helper that extends the 78-joint public rig with the procedural
    twist joints defined by ``SOMA_procedural_transforms.json``.

    The shipped public JSON ships 8 segments × 4 twist joints = 32 derived
    joints (e.g. ``LeftArmTwist1..4``, ``LeftForeArmTwist1..4`` etc. for both
    sides, both upper / lower arms and upper / lower legs). The internal rig
    in the upstream USD additionally has finger / spine micro-twists which
    are not exposed through this JSON.

    All three channel-extraction modes are implemented:

    * ``aligned_x_swing_twist`` (default, used by the trained correctives) —
      swing-twist decomposition in the bind-aligned absolute frame, twist
      axis fixed at +X.
    * ``local_x_swing_twist`` — same math but caller supplies inputs in the
      local parent-relative frame; per-segment configured axis.
    * ``local_x_euler`` — Euler-XYZ decomposition; selects the segment's
      configured Euler axis. Numerically less stable near gimbal lock.

    The :py:meth:`emit_twist_world_positions` helper applies the translation
    parameter matrix (the 5/35/65/95% segment distribution) to derive twist
    joint world positions from the public joint positions.
    """

    def __init__(
        self,
        definition: SOMAProceduralTransformDefinition,
        mode: str | None = None,
    ):
        # Upstream takes the mode from the definition's ``rotation_extraction``
        # block rather than a library constant, so an asset that asks for a
        # different extractor is honoured. ``mode`` remains an explicit override.
        if mode is None:
            mode = definition.rotation_extraction_mode
        if mode not in SOMA_PROCEDURAL_TRANSFORM_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Use one of {SOMA_PROCEDURAL_TRANSFORM_MODES}."
            )
        self.definition = definition
        self.mode = mode

        # Source-joint index for each segment (the joint we extract twist from).
        name_to_idx = {n: i for i, n in enumerate(definition.main_joint_names)}
        self._segment_source_idx = np.asarray(
            [name_to_idx[s.start_joint] for s in definition.segments],
            dtype=np.int32,
        )
        # Per-segment +/- sign and axis idx of the extracted scalar channel.
        self._segment_signs = np.asarray(
            [s.source_sign for s in definition.segments], dtype=np.float32,
        )
        self._segment_axes = np.asarray(
            [int(s.source_axis) for s in definition.segments], dtype=np.int32,
        )

        # Densify the sparse rotation parameter matrix.
        # Rows are extracted channels — one per segment in the current mode.
        # Columns are (twist_joint, axis) pairs — we group by output joint
        # below so the per-twist-joint rotation is a single angle.
        # Build the dense (n_segments, n_twist_axes) matrix indexed by the
        # twist joint's serial index in the full twist list.
        twist_names = definition.twist_joint_names
        twist_name_to_idx = {n: i for i, n in enumerate(twist_names)}

        # Per-twist output joint we store ONE angle and an axis. The JSON
        # entries reference an extracted-channel "row" (e.g. "LeftArmTwist1"
        # is the row label — matches the output joint here for aligned_x
        # mode) and a "column" that is one of the source main joints. The
        # weight is a scalar multiplier on the extracted twist angle.
        # The output twist joint axis (we rotate around) follows the segment's
        # source_axis convention.
        n_twist = len(twist_names)
        weights = np.zeros((n_twist, len(definition.main_joint_names)),
                            dtype=np.float32)
        for entry in definition.rotation_entries:
            tj = entry.row
            src_joint = entry.column
            if tj not in twist_name_to_idx:
                continue
            if src_joint not in name_to_idx:
                continue
            weights[twist_name_to_idx[tj], name_to_idx[src_joint]] = float(entry.value)
        self._twist_source_weights = weights        # (n_twist, n_public)

        # Per-twist joint, store the output axis and sign — defaulting to
        # the segment the joint belongs to.
        twist_to_segment: dict[str, int] = {}
        for si, seg in enumerate(definition.segments):
            for n in seg.twist_joints:
                twist_to_segment[n] = si
        self._twist_axis = np.asarray(
            [int(definition.segments[twist_to_segment[n]].source_axis) for n in twist_names],
            dtype=np.int32,
        )
        self._twist_sign = np.asarray(
            [definition.segments[twist_to_segment[n]].source_sign
             for n in twist_names], dtype=np.float32,
        )
        self.n_public = len(definition.main_joint_names)

        # Optional bind data enabling upstream's `aligned` extraction. Set via
        # `set_bind_data`; without it `extract_twist_angles` falls back to the
        # start-joint scalar, which is not upstream-equivalent.
        self._bind_world = None
        name_to_idx = {n: i for i, n in enumerate(definition.main_joint_names)}
        self._segment_start_ids = np.asarray(
            [name_to_idx[s.start_joint] for s in definition.segments], dtype=np.int64)
        self._segment_end_ids = np.asarray(
            [name_to_idx[s.end_joint] for s in definition.segments], dtype=np.int64)
        self._segment_parent_ids = np.asarray(
            [name_to_idx.get(s.parent_joint, name_to_idx[s.start_joint])
             for s in definition.segments], dtype=np.int64)
        self._segment_reverse_mask = np.asarray(
            [bool(getattr(s, "reverse", False)) for s in definition.segments], dtype=bool)
        self.n_twist = n_twist

        # Translation matrix: for each twist joint, store the two source-joint
        # public indices and their weights. The standard distribution has at
        # most 2 sources per twist (start_joint + end_joint of the segment).
        # Pre-densify to a (n_twist, n_public) matrix for one matmul.
        tr_weights = np.zeros((n_twist, self.n_public), dtype=np.float32)
        for entry in definition.translation_entries:
            tj = entry.row
            src_joint = entry.column
            if tj not in twist_name_to_idx or src_joint not in name_to_idx:
                continue
            tr_weights[twist_name_to_idx[tj], name_to_idx[src_joint]] = float(entry.value)
        self._twist_position_weights = tr_weights

    # --------------------------- evaluation ---------------------------------
    def set_bind_data(self, bind_world: jnp.ndarray) -> None:
        """Supply the skin bind pose to enable upstream's `aligned` mode.

        Args:
            bind_world: (n_public, 4, 4) source-joint world transforms of the
                **USD skin bind pose** — upstream's ``target_bind_pose_world``
                restricted to the public joints. Both ``bind_quaternions`` and
                the segment alignment quaternions come from it. SOMA-X v0.2.2
                moved both off ``target_t_pose_world``; passing the T-pose here
                reproduces the pre-v0.2.2 behaviour, whose twist is zero at rest
                and so misses upstream's rest mesh by ~2 cm on the limb twist
                segments. Without any bind data, ``aligned`` falls back to a
                start-joint scalar that is not upstream-equivalent.
        """
        bw = jnp.asarray(bind_world)
        if bw.shape[0] != self.n_public or bw.shape[-2:] != (4, 4):
            raise ValueError(
                f"Expected ({self.n_public}, 4, 4) bind transforms, got {bw.shape}")
        self._bind_world = bw

    def extract_twist_angles(self, public_rotmats: jnp.ndarray,
                             source_world_transforms: jnp.ndarray | None = None,
                             mode: str | None = None,
                             ) -> jnp.ndarray:
        """Extract the per-segment scalar twist channels from public local
        rotations using the configured mode.

        Args:
            public_rotmats: (B, 78, 3, 3) public-rig local rotmats. The frame
                interpretation depends on ``self.mode``:

                * ``aligned_x_swing_twist`` — bind-aligned absolute frame
                  (post-``apply_joint_orient_local``); twist axis fixed at +X.
                * ``local_x_swing_twist`` — local parent-relative frame;
                  twist axis from each segment's ``source_axis``.
                * ``local_x_euler`` — local parent-relative frame; Euler-XYZ
                  decomposition with the segment's configured axis.

        Returns:
            (B, n_public) scalar twist angles per public joint. Most entries
            are zero — only joints referenced as segment sources are non-zero.
        """
        mode = self.mode if mode is None else mode
        B = public_rotmats.shape[0]
        J = self.n_public

        # `aligned` is the mode the trained checkpoint was distilled against.
        # When the caller has supplied bind data, use upstream's real
        # formulation: bind-aligned virtual quaternions, local twist written to
        # the segment END joint, inherited twist to the start of reverse
        # segments. Sampling a scalar at the start joint (the fallback below)
        # leaves the forearm/shin helpers identically zero, because those
        # segments have no nonzero start column in the parameter matrix.
        if (mode == SOMA_ALIGNED_X_SWING_TWIST_MODE
                and getattr(self, "_bind_world", None) is not None):
            # Upstream's `aligned_x_swing_twist` reads the **posed world**
            # rotations (`_twist_angles_from_source(source_rotations,
            # source_world_transforms)`), not the local ones. Passing local
            # rotmats here is what made the emitted twist rotations differ from
            # upstream by 1.87 on identical inputs.
            world_rot = (public_rotmats if source_world_transforms is None
                         else source_world_transforms[..., :3, :3])
            return aligned_twist_channels(
                world_rot, self._bind_world,
                self._segment_start_ids, self._segment_end_ids,
                self._segment_parent_ids, self._segment_reverse_mask, J,
            )

        # Swing-twist modes share the same quaternion math; ``aligned`` and
        # ``local`` differ only in which frame the caller passes the rotmats
        # in. Convert to quats once for both swing-twist modes so the
        # per-segment loop is just a per-axis projection.
        if mode != SOMA_LOCAL_X_EULER_TWIST_MODE:
            quats = matrix_to_quaternion_xyzw(public_rotmats)  # (B, J, 4)

        # Build the (B, n_public) sparse angle vector. Only entries
        # corresponding to a segment source joint are non-zero.
        angles_public = jnp.zeros((B, J), dtype=public_rotmats.dtype)
        for si in range(len(self.definition.segments)):
            src_i = int(self._segment_source_idx[si])
            ax = int(self._segment_axes[si])
            sgn = float(self._segment_signs[si])
            if mode == SOMA_LOCAL_X_EULER_TWIST_MODE:
                ang = _euler_xyz_angle(public_rotmats[:, src_i], ax) * sgn
            else:
                ang = quaternion_twist_angle_xyzw(quats[:, src_i], ax) * sgn
            angles_public = angles_public.at[:, src_i].set(ang)
        return angles_public

    def emit_twist_rotmats(self, public_rotmats: jnp.ndarray,
                           source_world_transforms: jnp.ndarray | None = None,
                           ) -> jnp.ndarray:
        """Compute (B, n_twist, 3, 3) twist-joint local rotmats from public
        rotations via the sparse parameter matrix."""
        modes = tuple(self.definition.rotation_extraction_modes)
        W = jnp.asarray(self._twist_source_weights)                  # (n_twist, J_pub)
        if len(set(modes)) > 1:
            # Upstream allows a different extractor per procedural joint. It
            # splits the parameter matrix into one matrix per mode — each row
            # routed to the mode that joint asks for, all other rows zero
            # (`_build_rotation_parameter_matrices_by_mode`) — then sums the
            # per-mode contributions
            # (`_twist_angles_from_source`, procedural_transforms.py:1366).
            twist_angles = 0.0
            for mode in sorted(set(modes)):
                rows = jnp.asarray(
                    np.asarray([m == mode for m in modes], np.float32))[:, None]
                angles = self.extract_twist_angles(
                    public_rotmats, source_world_transforms, mode=mode)
                twist_angles = twist_angles + angles @ (W * rows).T
        else:
            angles_public = self.extract_twist_angles(
                public_rotmats, source_world_transforms)              # (B, J_pub)
            twist_angles = angles_public @ W.T                        # (B, n_twist)
        # Per-twist sign was rolled into W via the segment sign already.
        # Build rotation matrices per (twist joint, configured axis).
        # axis indices vary per twist joint -> do per-joint branchless build.
        B, n_t = twist_angles.shape
        # Allocate output
        out = jnp.broadcast_to(jnp.eye(3), (B, n_t, 3, 3))
        # Build per-axis rotmats once then gather by axis index
        rx = single_axis_rotation_matrices(twist_angles, 0)                  # (B, n_t, 3, 3)
        ry = single_axis_rotation_matrices(twist_angles, 1)
        rz = single_axis_rotation_matrices(twist_angles, 2)
        ax = jnp.asarray(self._twist_axis)                           # (n_t,)
        # Stack into (3, B, n_t, 3, 3) and gather per joint
        stacked = jnp.stack([rx, ry, rz], axis=0)                    # (3, B, n_t, 3, 3)
        out = jnp.take_along_axis(
            stacked, ax[None, None, :, None, None], axis=0,
        )[0]                                                          # (B, n_t, 3, 3)
        return out

    def extend_public_rotations(self, public_rotmats: jnp.ndarray) -> jnp.ndarray:
        """Concatenate public local rotmats with the derived twist-joint
        rotmats into a full-rig tensor.

        Args:
            public_rotmats: (B, 78, 3, 3) local rotations. Frame depends on
                the configured mode (see :py:meth:`extract_twist_angles`).

        Returns:
            (B, 78 + n_twist, 3, 3) full-rig local rotations. The first 78
            entries are the public rotations untouched; the remainder are the
            derived twist-joint rotations.
        """
        twist = self.emit_twist_rotmats(public_rotmats)
        return jnp.concatenate([public_rotmats, twist], axis=1)

    def extend_to_template_rig(
        self,
        public_rotmats: jnp.ndarray,
        joint_names: Optional[list[str]] = None,
    ) -> tuple[jnp.ndarray, list[str]]:
        """Emit the **full** template rig (110 joints on v0027), in authored order.

        :meth:`extend_public_rotations` returns the 110 joints the JSON
        describes (78 public + 32 twist), concatenated. The template rig has 12
        more — ``Nose``, ``ChestCenter`` and the ``ChestTo*`` chains — which are
        authored only in the USD. They carry no twist channel, so they take
        identity rotations; what they *do* need is to appear at their authored
        index, because downstream FK and skinning index by position.

        Args:
            public_rotmats: (B, 78, 3, 3) public local rotations.
            joint_names: template joint order; read from the USD when omitted
                (needs ``usd-core``).

        Returns:
            ``(rotmats, names)`` with rotmats ``(B, J_template, 3, 3)`` ordered to
            match ``names``.
        """
        if joint_names is None:
            joint_names = template_joint_names()

        derived = self.extend_public_rotations(public_rotmats)      # (B, 110, 3, 3)
        derived_names = list(self.definition.full_joint_names())
        index = {n: i for i, n in enumerate(derived_names)}

        B = public_rotmats.shape[0]
        eye = jnp.broadcast_to(jnp.eye(3, dtype=public_rotmats.dtype), (B, 3, 3))
        cols = [derived[:, index[n]] if n in index else eye for n in joint_names]
        return jnp.stack(cols, axis=1), list(joint_names)

    # ------------------------- translation matrix --------------------------
    def expand_world_transforms_from_source_fk(
        self,
        source_rotations: jnp.ndarray,
        source_world_transforms: jnp.ndarray,
        target_base_rotations: jnp.ndarray,
        target_local_translations: jnp.ndarray,
        control_target_ids: np.ndarray,
        twist_target_ids: np.ndarray,
        twist_parent_target_ids: np.ndarray,
        target_parents: np.ndarray,
    ) -> jnp.ndarray:
        """Expand public FK world transforms onto the full procedural rig.

        Port of upstream ``expand_world_transforms_from_source_fk``
        (``procedural_transforms.py:1405``), which upstream reaches from
        ``soma.py:1580`` via ``transform_expander=``.

        This is **not** "expand the rotations then run FK". Upstream runs FK on
        the public joints only, copies those world transforms into the expanded
        rig, and gives each twist joint a *single local step* off its public
        parent::

            target_world[control_target_ids] = source_world[:]
            twist_local  = SE3(base_rot[twist] @ twist_rot, local_t[twist])
            target_world[twist] = target_world[twist_parent] @ twist_local
            # remaining joints: level-order fill from parents

        Treating a twist joint as a link in a general FK chain instead lets the
        bind absorb its rotation exactly, which is what made the posed output
        reproduce the non-procedural rig.

        Args:
            source_rotations: (B, n_public, 3, 3) absolute public rotations.
            source_world_transforms: (B, n_public, 4, 4) public FK result.
            target_base_rotations: (n_target, 3, 3) target T-pose local rotations
                (upstream's ``target_t_pose_local_rotations``).
            target_local_translations: (n_target, 3) target local translations.
            control_target_ids: (n_public,) where each public joint lands.
            twist_target_ids, twist_parent_target_ids: (n_twist,) twist joint
                slots and their parents' slots.
            target_parents: (n_target,) parent index per target joint.

        Returns:
            (B, n_target, 4, 4) world transforms for the expanded rig.
        """
        from .geometry.transforms import se3_from_rt

        B = source_world_transforms.shape[0]
        J = int(np.asarray(target_parents).shape[0])
        base = jnp.broadcast_to(jnp.asarray(target_base_rotations), (B, J, 3, 3))
        loc_t = jnp.broadcast_to(jnp.asarray(target_local_translations), (B, J, 3))

        out = jnp.broadcast_to(
            jnp.eye(4, dtype=source_world_transforms.dtype), (B, J, 4, 4))
        ctrl = np.asarray(control_target_ids)
        out = out.at[:, ctrl].set(source_world_transforms)
        assigned = np.zeros(J, dtype=bool)
        assigned[ctrl] = True

        tw = np.asarray(twist_target_ids)
        if tw.size:
            twist_rot = self.emit_twist_rotmats(
                source_rotations, source_world_transforms)
            twist_local = se3_from_rt(
                jnp.einsum("bjmn,bjnp->bjmp", base[:, tw], twist_rot), loc_t[:, tw])
            out = out.at[:, tw].set(
                jnp.einsum("bjmn,bjnp->bjmp", out[:, np.asarray(twist_parent_target_ids)],
                           twist_local))
            assigned[tw] = True

        # Remaining joints (USD-only helpers; none on v0027) inherit their parent's world
        # transform composed with their own bind-local step, in level order.
        if not assigned.all():
            local = se3_from_rt(base, loc_t)
            parents = np.asarray(target_parents)
            for level in compute_skeleton_levels(parents)[1:]:
                ids = np.asarray([j for j in np.asarray(level) if not assigned[j]],
                                 dtype=np.int64)
                if ids.size == 0:
                    continue
                out = out.at[:, ids].set(jnp.einsum(
                    "bjmn,bjnp->bjmp", out[:, parents[ids]], local[:, ids]))
                assigned[ids] = True
        return out

    def emit_twist_world_positions(
        self,
        public_positions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the twist joints' world positions from the public-joint
        world positions, using the sparse translation parameter matrix.

        The standard 4-helper distribution places twist joints at 5%, 35%,
        65%, and 95% along each segment — e.g. for the LeftArm segment,
        ``LeftArmTwist1 = 0.95·LeftArm + 0.05·LeftForeArm``. The translation
        matrix's rows always sum to 1 (convex combination), so the twist
        joints inherit the public joints' parent transforms cleanly under FK.

        Args:
            public_positions: (..., n_public, 3) world positions of the 78
                public joints.

        Returns:
            (..., n_twist, 3) world positions of the derived twist joints.
        """
        W = jnp.asarray(self._twist_position_weights)              # (n_twist, n_public)
        return jnp.einsum("nj,...jd->...nd", W, public_positions)

    def full_rig_bind_world(
        self,
        public_bind_world: np.ndarray | jnp.ndarray,
    ) -> np.ndarray:
        """Build the full-rig (78 + n_twist) bind world transforms from the
        public ones.

        The twist joints' bind rotation is identity (twist helpers are added
        at rest with no offset rotation — they activate only when the source
        joint twists). Bind translation comes from the translation parameter
        matrix.

        Args:
            public_bind_world: (n_public, 4, 4) public bind world transforms.

        Returns:
            (n_public + n_twist, 4, 4) full-rig bind world transforms.
        """
        pbw = np.asarray(public_bind_world, dtype=np.float32)
        pub_pos = pbw[:, :3, 3]                                    # (J, 3)
        twist_pos = np.einsum(
            "nj,jd->nd", self._twist_position_weights, pub_pos,
        )
        twist_T = np.broadcast_to(np.eye(4, dtype=np.float32),
                                   (self.n_twist, 4, 4)).copy()
        twist_T[:, :3, 3] = twist_pos
        return np.concatenate([pbw, twist_T], axis=0)

    # ----------------------- parent / name expansion -----------------------
    def full_rig_parents(self, public_parents: np.ndarray) -> np.ndarray:
        """Extend the (n_public,) parent-id array to (n_public + n_twist,).

        Twist joints attach as leaves to their segment's ``start_joint`` (the
        rotation source), matching SOMA-X's authored hierarchy where
        ``LeftArmTwist1..4`` are children of ``LeftArm``.
        """
        public_parents = np.asarray(public_parents, dtype=np.int32)
        name_to_idx = {n: i for i, n in enumerate(self.definition.main_joint_names)}
        twist_parents = np.empty(self.n_twist, dtype=np.int32)
        cursor = 0
        for seg in self.definition.segments:
            parent_id = name_to_idx[seg.start_joint]
            for _ in seg.twist_joints:
                twist_parents[cursor] = parent_id
                cursor += 1
        return np.concatenate([public_parents, twist_parents], axis=0)

    def full_rig_joint_names(self) -> tuple[str, ...]:
        """``(public_names..., twist_names...)`` in the same order the
        rotation / translation tensors use."""
        return self.definition.full_joint_names()


# ---------------------------------------------------------------------------
# Full template rig (110 joints on v0027, 122 on v0026)
# ---------------------------------------------------------------------------


def template_joint_names(usd_path=None) -> list[str]:
    """Authored joint names of the template rig, in template order.

    The procedural-transform JSON only describes 110 joints — the 78 public
    joints plus 32 twist helpers. The remaining 12 are authored directly in
    ``SOMA_template_rig.usda`` (``Nose``, ``ChestCenter`` and the ``ChestTo*``
    chains): pure geometric helpers with no twist channel, which take identity
    rotations and are placed by the template's bind transforms. Reading the USD
    is the only way to recover them and, importantly, their authored *order*.

    Requires ``usd-core``.

    Args:
        usd_path: template rig; defaults to the resolved asset.

    Returns:
        The template's joint names (110 on v0027), template order.
    """
    from pxr import Usd, UsdSkel
    if usd_path is None:
        from .assets import resolve
        usd_path = resolve("SOMA_template_rig.usda")
    stage = Usd.Stage.Open(str(usd_path))
    skeletons = [pr for pr in stage.Traverse() if pr.IsA(UsdSkel.Skeleton)]
    if not skeletons:
        raise ValueError(f"No UsdSkel.Skeleton in {usd_path}")
    paths = UsdSkel.Skeleton(skeletons[0]).GetJointsAttr().Get()
    return [str(pth).split("/")[-1] for pth in paths]


# ---------------------------------------------------------------------------
# Bind-aligned twist extraction (upstream's `aligned_x_swing_twist` machinery)
# ---------------------------------------------------------------------------


def _normalize_vectors(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), eps)


def _project_to_plane(v: jnp.ndarray, n: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return _normalize_vectors(v - jnp.sum(v * n, axis=-1, keepdims=True) * n, eps=eps)


def bind_alignment_quaternions(
    bind_world: jnp.ndarray, start_ids: np.ndarray, end_ids: np.ndarray
) -> jnp.ndarray:
    """Per-segment frame that puts local +X along the bone at bind.

    Port of upstream ``_bind_alignment_quaternions``. The segment's X axis is
    the normalised start->end span, signed so it agrees with the start joint's
    own X; Y is the start joint's Y projected off that axis, with a documented
    fallback chain (start Z, then world Y, then world Z) for the degenerate
    cases where the projection vanishes.

    Args:
        bind_world: (J, 4, 4) bind world transforms.
        start_ids, end_ids: (S,) segment endpoint indices.

    Returns:
        (S, 4) xyzw quaternions.
    """
    start = bind_world[np.asarray(start_ids)]
    end = bind_world[np.asarray(end_ids)]
    start_rot = start[..., :3, :3]
    x_axis = _normalize_vectors(end[..., :3, 3] - start[..., :3, 3])

    ex = jnp.asarray([1.0, 0.0, 0.0]); ey = jnp.asarray([0.0, 1.0, 0.0]); ez = jnp.asarray([0.0, 0.0, 1.0])
    up_x = jnp.einsum("sij,j->si", start_rot, ex)
    sign = jnp.where(jnp.sum(up_x * x_axis, axis=-1, keepdims=True) >= 0.0, 1.0, -1.0)
    x_axis = x_axis * sign

    y_cand = jnp.einsum("sij,j->si", start_rot, ey)
    z_cand = jnp.einsum("sij,j->si", start_rot, ez)
    world_y = jnp.broadcast_to(ey, x_axis.shape)
    world_z = jnp.broadcast_to(ez, x_axis.shape)

    def _resid(v):
        return jnp.linalg.norm(v - jnp.sum(v * x_axis, -1, keepdims=True) * x_axis, axis=-1)

    y_axis = _project_to_plane(y_cand, x_axis)
    y_n, z_n, wy_n = _resid(y_cand), _resid(z_cand), _resid(world_y)
    y_axis = jnp.where((y_n > 1e-8)[:, None], y_axis, _project_to_plane(z_cand, x_axis))
    y_axis = jnp.where(((y_n > 1e-8) | (z_n > 1e-8))[:, None], y_axis,
                       _project_to_plane(world_y, x_axis))
    y_axis = jnp.where(((y_n > 1e-8) | (z_n > 1e-8) | (wy_n > 1e-8))[:, None], y_axis,
                       _project_to_plane(world_z, x_axis))

    z_axis = _normalize_vectors(jnp.cross(x_axis, y_axis))
    y_axis = _normalize_vectors(jnp.cross(z_axis, x_axis))
    align_rot = jnp.stack((x_axis, y_axis, z_axis), axis=-1)
    return matrix_to_quaternion_xyzw(align_rot)


def aligned_virtual_quaternions(
    world_rotations: jnp.ndarray,
    bind_quaternions: jnp.ndarray,
    align_quaternions: jnp.ndarray,
    segment_ids: np.ndarray,
    joint_ids: np.ndarray,
) -> jnp.ndarray:
    """``q_current * conj(q_bind) * q_align`` per gathered joint.

    Port of upstream ``_aligned_virtual_quaternions``: takes each joint's world
    rotation into the segment's bind-aligned frame, so the residual twist is
    measured about the bone axis rather than an arbitrary local axis.
    """
    q_cur = matrix_to_quaternion_xyzw(world_rotations[:, np.asarray(joint_ids)])
    q_bind_inv = quaternion_conjugate_xyzw(bind_quaternions[np.asarray(joint_ids)])
    q_align = align_quaternions[np.asarray(segment_ids)][None]
    q = quaternion_multiply_xyzw(
        quaternion_multiply_xyzw(q_cur, q_bind_inv[None]), q_align)
    return quaternion_normalize_xyzw(q)


def aligned_twist_channels(
    world_rotations: jnp.ndarray,
    bind_world: jnp.ndarray,
    start_ids: np.ndarray,
    end_ids: np.ndarray,
    parent_ids: np.ndarray,
    reverse_mask: np.ndarray,
    n_source_joints: int,
) -> jnp.ndarray:
    """Per-joint twist angles in the bind-aligned frame — upstream's `aligned` mode.

    Port of ``SOMAProceduralParameterTransform._aligned_twist_channels_from_world``.
    Two channels come out of each segment:

    * **local twist** ``twist(conj(q_start) * q_end)`` — how much the bone has
      twisted between its own start and end, written to the **end** joint. This
      is the part the previous implementation missed entirely: it sampled a
      scalar at the *start* joint, which for the forearm/shin segments has no
      nonzero column in the parameter matrix, so those helpers came out zero.
    * **inherited twist** ``twist(conj(q_parent) * q_start)`` — written to the
      start joint, and only for segments flagged ``reverse``.

    Args:
        world_rotations: (B, J, 3, 3) source-joint world rotations.
        bind_world: (J, 4, 4) source bind world transforms (T-pose).
        start_ids, end_ids, parent_ids: (S,) segment joint indices.
        reverse_mask: (S,) bool; reverse segments also emit inherited twist.
        n_source_joints: J, width of the returned channel vector.

    Returns:
        (B, J) twist angle per source joint, zero where no segment writes.
    """
    start_ids = np.asarray(start_ids); end_ids = np.asarray(end_ids)
    parent_ids = np.asarray(parent_ids); reverse_mask = np.asarray(reverse_mask, dtype=bool)
    S = len(start_ids)

    align_q = bind_alignment_quaternions(bind_world, start_ids, end_ids)
    bind_q = matrix_to_quaternion_xyzw(bind_world[..., :3, :3])

    seg_ids = np.concatenate([np.arange(S)] * 3)
    joint_ids = np.concatenate([end_ids, start_ids, parent_ids])
    q = aligned_virtual_quaternions(
        world_rotations[..., :3, :3], bind_q, align_q, seg_ids, joint_ids)

    q_end, q_start, q_parent = q[:, :S], q[:, S:2 * S], q[:, 2 * S:]
    local_twist = quaternion_twist_angle_xyzw(
        quaternion_normalize_xyzw(
            quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_start), q_end)), 0)
    inherited_twist = quaternion_twist_angle_xyzw(
        quaternion_normalize_xyzw(
            quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_parent), q_start)), 0)

    B = world_rotations.shape[0]
    out = jnp.zeros((B, n_source_joints), dtype=local_twist.dtype)
    out = out.at[:, end_ids].set(local_twist)
    if reverse_mask.any():
        rev = np.where(reverse_mask)[0]
        out = out.at[:, start_ids[rev]].set(inherited_twist[:, rev])
    return out


# ---------------------------------------------------------------------------
# Upstream compiled transform (``SOMAProceduralParameterTransform``) and helpers
# ---------------------------------------------------------------------------


def _dense_named_matrix(entries, row_names, column_names, matrix_name: str, *,
                        base_identity: bool = False) -> np.ndarray:
    if base_identity:
        if len(row_names) != len(column_names) or tuple(row_names) != tuple(column_names):
            raise ValueError(f"{matrix_name} identity base requires matching row/column names")
        matrix = np.eye(len(row_names), dtype=np.float32)
    else:
        matrix = np.zeros((len(row_names), len(column_names)), dtype=np.float32)
    rows = {name: index for index, name in enumerate(row_names)}
    columns = {name: index for index, name in enumerate(column_names)}
    cleared_rows = set()
    for entry in entries:
        try:
            row = rows[entry.row]
            column = columns[entry.column]
        except KeyError as e:
            raise ValueError(
                f"{matrix_name} matrix references unknown row/column: "
                f"{entry.row!r}, {entry.column!r}") from e
        if base_identity and row not in cleared_rows:
            matrix[row] = 0.0
            cleared_rows.add(row)
        matrix[row, column] = float(entry.value)
    return matrix


def _build_parameter_matrices(source_by_name, target_by_name, segments, rotation_entries,
                              translation_entries) -> SOMAProceduralParameterMatrices:
    """Compile SOMA-owned procedural rotation and translation matrices."""
    twist_names = tuple(_twist_joint_names(segments))
    rotation_matrix = _dense_named_matrix(rotation_entries, twist_names, tuple(source_by_name),
                                          "rotation")
    translation_matrix = _dense_named_matrix(translation_entries, tuple(target_by_name),
                                             tuple(target_by_name), "translation",
                                             base_identity=True)
    fractions = [translation_matrix[target_by_name[twist_joint], target_by_name[segment.end_joint]]
                 for segment in segments for twist_joint in segment.twist_joints]
    return SOMAProceduralParameterMatrices(
        rotation=rotation_matrix,
        translation=translation_matrix,
        segment_fractions=(np.asarray(fractions, np.float32) if fractions
                           else np.empty(0, np.float32)),
    )


def _build_rotation_parameter_matrices_by_mode(rotation_parameter_matrix, mode_names,
                                               rotation_extraction_modes) -> np.ndarray:
    mode_to_idx = {mode: index for index, mode in enumerate(mode_names)}
    matrices = np.zeros((len(mode_names), *rotation_parameter_matrix.shape),
                        dtype=rotation_parameter_matrix.dtype)
    for row, mode in enumerate(rotation_extraction_modes):
        matrices[mode_to_idx[mode], row] = rotation_parameter_matrix[row]
    return matrices


def has_soma_twist_joints(joint_names, segments=None) -> bool:
    """Return whether all SOMA procedural twist joints are present."""
    segments = _require_segments(segments, "has_soma_twist_joints()")
    names = set(_name_list(joint_names))
    return all(name in names for name in _twist_joint_names(segments))


def derive_soma_rig_without_procedural_joints(rig_data, public_joint_names=None,
                                              segments=None) -> dict:
    """Derive the public SOMA rig and aggregate removed joint weights to parents.

    Upstream ``derive_soma_rig_without_procedural_joints``, on upstream's rig
    dict (as :func:`soma_jax.usd_io.load_lod_rig_from_usd` returns it): prune
    the generated procedural and auxiliary joints, remap the hierarchy, and
    move each pruned joint's skin weights onto its nearest kept parent.
    (:func:`soma_jax.rig_build.prune_procedural_joints` does the same on
    SOMA-JAX's asset dict.)
    """
    from scipy.sparse import csc_matrix

    from .geometry.rig_utils import joint_world_to_local

    joint_names = _name_list(rig_data["joint_names"])
    if public_joint_names is None:
        segments = _require_segments(
            segments, "derive_soma_rig_without_procedural_joints() without public_joint_names")
        twist_names = set(_twist_joint_names(segments))
        keep_ids = np.array([idx for idx, name in enumerate(joint_names)
                             if name not in twist_names], dtype=np.int64)
    else:
        public_names = _name_list(public_joint_names)
        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        missing_public = [name for name in public_names if name not in name_to_idx]
        if missing_public:
            raise ValueError(
                f"Template rig is missing public SOMA joints: {sorted(set(missing_public))}")
        keep_ids = np.asarray([name_to_idx[name] for name in public_names], dtype=np.int64)
    keep_id_set = {int(idx) for idx in keep_ids}
    remove_ids = {idx for idx in range(len(joint_names)) if idx not in keep_id_set}
    if not remove_ids:
        return dict(rig_data)

    parent_ids = np.asarray(rig_data["joint_parent_ids"], dtype=np.int64)
    old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(keep_ids)}

    def nearest_kept_parent(old_idx: int) -> int:
        parent = int(parent_ids[old_idx])
        while parent in remove_ids and parent != int(parent_ids[parent]):
            parent = int(parent_ids[parent])
        return parent

    new_parent_ids = np.zeros((len(keep_ids),), dtype=np.int32)
    for new_idx, old_idx_np in enumerate(keep_ids):
        old_idx = int(old_idx_np)
        if int(parent_ids[old_idx]) == old_idx:
            new_parent_ids[new_idx] = new_idx
            continue
        new_parent_ids[new_idx] = old_to_new[nearest_kept_parent(old_idx)]

    weights = np.asarray(csc_matrix(
        (rig_data["skinning_weights_data"], rig_data["skinning_weights_indices"],
         rig_data["skinning_weights_indptr"]),
        shape=rig_data["skinning_weights_shape"]).todense(), dtype=np.float32)
    for removed_idx in sorted(remove_ids):
        weights[:, nearest_kept_parent(removed_idx)] += weights[:, removed_idx]
    weights = weights[:, keep_ids]
    weights_sparse = csc_matrix(weights)

    bind_pose_world = np.asarray(rig_data["bind_pose_world"], dtype=np.float32)[keep_ids]
    t_pose_world = np.asarray(rig_data["t_pose_world"], dtype=np.float32)[keep_ids]
    bind_pose_local = np.asarray(joint_world_to_local(jnp.asarray(bind_pose_world), new_parent_ids))
    t_pose_local = np.asarray(joint_world_to_local(jnp.asarray(t_pose_world), new_parent_ids))

    out = dict(rig_data)
    out.update(
        joint_names=np.asarray([joint_names[int(idx)] for idx in keep_ids]),
        joint_parent_ids=new_parent_ids,
        bind_pose_world=bind_pose_world.astype(np.float32),
        bind_pose_local=bind_pose_local.astype(np.float32),
        t_pose_world=t_pose_world.astype(np.float32),
        t_pose_local=t_pose_local.astype(np.float32),
        skinning_weights_data=weights_sparse.data.astype(np.float32),
        skinning_weights_indices=weights_sparse.indices.astype(np.int32),
        skinning_weights_indptr=weights_sparse.indptr.astype(np.int32),
        skinning_weights_shape=np.array(weights_sparse.shape, dtype=np.int32),
    )
    return out


def _build_source_twist_channels(source_by_name, segments) -> tuple[np.ndarray, np.ndarray]:
    axis_ids = np.zeros((len(source_by_name),), dtype=np.int64)
    signs = np.ones((len(source_by_name),), dtype=np.float32)
    assigned = {}
    for segment in segments:
        if segment.source_axis not in (0, 1, 2):
            raise ValueError(f"source_axis must be 0, 1, or 2, got {segment.source_axis}")
        for name in (segment.start_joint, segment.end_joint):
            spec = (segment.source_axis, float(segment.source_sign))
            if name in assigned and assigned[name] != spec:
                raise ValueError(f"Conflicting SOMA twist source channel for joint {name!r}")
            assigned[name] = spec
    for name, (axis, sign) in assigned.items():
        idx = source_by_name[name]
        axis_ids[idx] = axis
        signs[idx] = sign
    return axis_ids, signs


def _build_twist_output_channels(segments) -> tuple[np.ndarray, np.ndarray]:
    axis_ids, signs = [], []
    for segment in segments:
        axis_ids.extend([segment.source_axis] * len(segment.twist_joints))
        signs.extend([float(segment.source_sign)] * len(segment.twist_joints))
    return np.asarray(axis_ids, dtype=np.int64), np.asarray(signs, dtype=np.float32)


def local_x_euler_from_matrix(rotations: jnp.ndarray) -> jnp.ndarray:
    """Return the local X Euler angle for matrices that may include swing.

    This is the source extraction used by the ``local_x_euler`` twist mode.
    ``local_x_swing_twist`` uses quaternion projection instead.
    """
    return jnp.arctan2(rotations[..., 2, 1], rotations[..., 1, 1])


def _local_euler_xyz_from_matrix(rotations: jnp.ndarray) -> jnp.ndarray:
    euler = matrix_to_euler_xyz(rotations)
    return jnp.stack((local_x_euler_from_matrix(rotations), euler[..., 1], euler[..., 2]),
                     axis=-1)


def _axis_rotations(angles: jnp.ndarray, axis_ids, axis_signs) -> jnp.ndarray:
    signed_angles = angles * jnp.asarray(axis_signs, angles.dtype)[None]
    one_hot = jax.nn.one_hot(jnp.asarray(axis_ids), 3, dtype=angles.dtype)
    rotvecs = signed_angles[..., None] * one_hot[None]
    return batch_rodrigues(rotvecs.reshape(-1, 3), dtype=angles.dtype).reshape(
        *angles.shape, 3, 3)


def _swing_twist_channels_from_matrix(rotations: jnp.ndarray, axis_ids, axis_signs) -> jnp.ndarray:
    quaternions = matrix_to_quaternion_xyzw(rotations)
    twist_angles = quaternion_twist_angle_xyzw(quaternions, jnp.asarray(axis_ids))
    return twist_angles * jnp.asarray(axis_signs, rotations.dtype)[None]


def _bind_alignment_quaternions(bind_world_transforms, start_ids, end_ids) -> jnp.ndarray:
    return bind_alignment_quaternions(jnp.asarray(bind_world_transforms, jnp.float32),
                                      np.asarray(start_ids), np.asarray(end_ids))


_aligned_virtual_quaternions = aligned_virtual_quaternions


class SOMAProceduralParameterTransform:
    """Expand the public SOMA pose with procedural parameter matrices.

    Port of upstream's ``SOMAProceduralParameterTransform`` (an ``nn.Module``)
    as a plain object: the same constructor, attributes (upstream's buffers, as
    NumPy index arrays and JAX float arrays) and methods; :meth:`forward` /
    calling it returns :class:`SOMAProceduralTransformOutput`. Methods are
    traceable. SOMA-JAX's layers drive the same math through
    :class:`ProceduralTransforms`.
    """

    def __init__(
        self,
        source_joint_names,
        target_joint_names,
        rotation_extraction_modes=None,
        segments=None,
        rotation_entries=None,
        translation_entries=None,
        target_t_pose_world=None,
        target_joint_parent_ids=None,
        target_bind_pose_world=None,
    ) -> None:
        source_names = _name_list(source_joint_names)
        target_names = _name_list(target_joint_names)
        segments = _require_segments(segments, "SOMAProceduralParameterTransform")
        source_by_name = {name: idx for idx, name in enumerate(source_names)}
        target_by_name = {name: idx for idx, name in enumerate(target_names)}
        twist_names = _twist_joint_names(segments)
        rotation_extraction_modes = _require_rotation_extraction_modes(
            rotation_extraction_modes, twist_names, "SOMAProceduralParameterTransform")
        if rotation_entries is None or translation_entries is None:
            raise ValueError(
                "SOMAProceduralParameterTransform requires JSON sidecar matrix entries. "
                f"Load {SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME} with "
                "load_soma_procedural_transform_definition() and pass "
                "definition.rotation_entries and definition.translation_entries.")
        mode_names = tuple(mode_name for mode_name in SOMA_PROCEDURAL_TRANSFORM_MODES
                           if mode_name in rotation_extraction_modes)

        missing_source = [name for segment in segments
                          for name in (segment.start_joint, segment.end_joint)
                          if name not in source_by_name]
        missing_target = [name for name in twist_names if name not in target_by_name]
        missing_control_targets = [name for name in source_names if name not in target_by_name]
        duplicate_targets = sorted({name for name in twist_names if twist_names.count(name) > 1})
        if missing_source or missing_target or missing_control_targets or duplicate_targets:
            parts = []
            if missing_source:
                parts.append(f"missing source joints: {sorted(set(missing_source))}")
            if missing_target:
                parts.append(f"missing twist joints: {sorted(set(missing_target))}")
            if missing_control_targets:
                parts.append(
                    f"missing public SOMA joints in twist rig: {sorted(missing_control_targets)}")
            if duplicate_targets:
                parts.append(f"duplicate twist joints: {duplicate_targets}")
            raise ValueError("Invalid SOMA procedural twist rig mapping; " + "; ".join(parts))

        source_axis_ids, source_axis_signs = _build_source_twist_channels(source_by_name, segments)
        twist_axis_ids, twist_axis_signs = _build_twist_output_channels(segments)

        self.mode = (rotation_extraction_modes[0]
                     if len(set(rotation_extraction_modes)) == 1 else None)
        self.rotation_extraction_modes = rotation_extraction_modes
        self.rotation_extraction_mode_names = mode_names
        self.source_joint_names = tuple(source_names)
        self.target_joint_names = tuple(target_names)
        self.segments = segments
        self.twist_joint_names = tuple(twist_names)
        self.control_source_ids = np.asarray([source_by_name[n] for n in source_names], np.int64)
        self.control_target_ids = np.asarray([target_by_name[n] for n in source_names], np.int64)
        self.twist_target_ids = np.asarray([target_by_name[n] for n in twist_names], np.int64)
        self.source_twist_axis_ids = source_axis_ids
        self.source_twist_axis_signs = jnp.asarray(source_axis_signs)
        self.twist_axis_ids = twist_axis_ids
        self.twist_axis_signs = jnp.asarray(twist_axis_signs)

        if target_t_pose_world is not None:
            target_t_pose_world = np.asarray(target_t_pose_world)
            if target_t_pose_world.shape[-2:] != (4, 4):
                raise ValueError("target_t_pose_world must have shape (J, 4, 4), "
                                 f"got {target_t_pose_world.shape}")
            if target_t_pose_world.shape[0] != len(target_names):
                raise ValueError(
                    "target_t_pose_world must have the same joint count as target_joint_names")
        if target_bind_pose_world is not None:
            target_bind_pose_world = np.asarray(target_bind_pose_world)
            if target_bind_pose_world.shape[-2:] != (4, 4):
                raise ValueError("target_bind_pose_world must have shape (J, 4, 4), "
                                 f"got {target_bind_pose_world.shape}")
            if target_bind_pose_world.shape[0] != len(target_names):
                raise ValueError(
                    "target_bind_pose_world must have the same joint count as target_joint_names")
        if target_joint_parent_ids is not None:
            target_joint_parent_ids = np.asarray(target_joint_parent_ids, np.int64)
            if target_joint_parent_ids.shape != (len(target_names),):
                raise ValueError("target_joint_parent_ids must have shape "
                                 f"({len(target_names)},), got {target_joint_parent_ids.shape}")
        target_t_pose_local = None
        if target_t_pose_world is not None and target_joint_parent_ids is not None:
            target_t_pose_local = joint_world_to_local(jnp.asarray(target_t_pose_world),
                                                       target_joint_parent_ids)
        self.target_joint_parent_ids = (target_joint_parent_ids if target_joint_parent_ids
                                        is not None else np.empty(0, np.int64))
        self.target_t_pose_local_rotations = (
            target_t_pose_local[..., :3, :3] if target_t_pose_local is not None
            else jnp.empty((0, 3, 3), jnp.float32))
        parameter_matrices = _build_parameter_matrices(
            source_by_name, target_by_name, segments, rotation_entries, translation_entries)

        source_parent_ids = None
        source_t_pose_local = None
        source_joint_orient = None
        source_joint_orient_parent_t = None
        bind_quaternions = None
        bind_align_quaternions = None
        segment_start_ids, segment_end_ids, segment_parent_ids = [], [], []
        segment_reverse_mask = []
        aligned_virtual_segment_ids, aligned_virtual_joint_ids = [], []
        twist_parent_target_ids = []
        if target_joint_parent_ids is not None:
            twist_parent_target_ids = [int(target_joint_parent_ids[target_by_name[name]])
                                       for name in twist_names]
            source_target_ids = np.asarray([target_by_name[name] for name in source_names],
                                           np.int64)
            target_to_source = {int(t): s for s, t in enumerate(source_target_ids)}
            source_parent_list = []
            for target_idx in source_target_ids:
                parent_idx = int(target_joint_parent_ids[int(target_idx)])
                while parent_idx not in target_to_source:
                    # A negative parent is a root here; upstream's tensor
                    # indexing would wrap it to the last joint.
                    if parent_idx < 0:
                        break
                    next_parent_idx = int(target_joint_parent_ids[parent_idx])
                    if next_parent_idx == parent_idx:
                        break
                    parent_idx = next_parent_idx
                source_parent_list.append(target_to_source.get(parent_idx, 0))
            source_parent_ids = np.asarray(source_parent_list, np.int64)
            source_t_pose_world = (jnp.asarray(target_t_pose_world[source_target_ids], jnp.float32)
                                   if target_t_pose_world is not None else None)
            source_bind_pose_world = (
                jnp.asarray(target_bind_pose_world[source_target_ids], jnp.float32)
                if target_bind_pose_world is not None else None)
            if source_t_pose_world is not None:
                source_t_pose_local = joint_world_to_local(source_t_pose_world, source_parent_ids)
                source_joint_orient, source_joint_orient_parent_t = precompute_joint_orient(
                    source_t_pose_world, source_parent_ids)
            if source_bind_pose_world is not None:
                bind_quaternions = matrix_to_quaternion_xyzw(source_bind_pose_world[..., :3, :3])
            for segment in segments:
                start_idx = source_by_name[segment.start_joint]
                segment_start_ids.append(start_idx)
                segment_end_ids.append(source_by_name[segment.end_joint])
                segment_parent_ids.append(
                    source_by_name[segment.parent_joint] if segment.parent_joint is not None
                    else int(source_parent_ids[start_idx]))
                segment_reverse_mask.append(bool(segment.reverse))
            if source_bind_pose_world is not None:
                bind_align_quaternions = _bind_alignment_quaternions(
                    source_bind_pose_world, segment_start_ids, segment_end_ids)
            segment_ids = list(range(len(segment_start_ids)))
            aligned_virtual_segment_ids = segment_ids + segment_ids + segment_ids
            aligned_virtual_joint_ids = segment_end_ids + segment_start_ids + segment_parent_ids
        single_twist_axis = None
        if twist_axis_ids.size > 0 and bool(np.all(twist_axis_ids == twist_axis_ids[0])):
            single_twist_axis = int(twist_axis_ids[0])

        def _index(values):
            return np.asarray(values, np.int64) if len(values) else np.empty(0, np.int64)

        self.segment_fractions = jnp.asarray(parameter_matrices.segment_fractions)
        self.rotation_parameter_matrix = jnp.asarray(parameter_matrices.rotation)
        self.rotation_parameter_matrices_by_mode = jnp.asarray(
            _build_rotation_parameter_matrices_by_mode(
                parameter_matrices.rotation, mode_names, rotation_extraction_modes))
        self.translation_parameter_matrix = jnp.asarray(parameter_matrices.translation)
        self.source_parent_ids = (source_parent_ids if source_parent_ids is not None
                                  else np.empty(0, np.int64))
        self.source_t_pose_local = (source_t_pose_local if source_t_pose_local is not None
                                    else jnp.empty((0, 4, 4), jnp.float32))
        self.source_joint_orient = (source_joint_orient if source_joint_orient is not None
                                    else jnp.empty((0, 3, 3), jnp.float32))
        self.source_joint_orient_parent_t = (
            source_joint_orient_parent_t if source_joint_orient_parent_t is not None
            else jnp.empty((0, 3, 3), jnp.float32))
        self.source_bind_quaternions = (bind_quaternions if bind_quaternions is not None
                                        else jnp.empty((0, 4), jnp.float32))
        self.segment_bind_align_quaternions = (
            bind_align_quaternions if bind_align_quaternions is not None
            else jnp.empty((0, 4), jnp.float32))
        self.segment_start_source_ids = _index(segment_start_ids)
        self.segment_end_source_ids = _index(segment_end_ids)
        self.segment_parent_source_ids = _index(segment_parent_ids)
        self.segment_source_ids = np.arange(len(segment_start_ids), dtype=np.int64)
        self.segment_reverse_mask = np.asarray(segment_reverse_mask, dtype=bool)
        self.aligned_virtual_segment_ids = _index(aligned_virtual_segment_ids)
        self.aligned_virtual_joint_ids = _index(aligned_virtual_joint_ids)
        self.twist_parent_target_ids = _index(twist_parent_target_ids)
        self._single_twist_axis = single_twist_axis

    @property
    def twist_joint_indices(self) -> tuple[int, ...]:
        return tuple(int(idx) for idx in self.twist_target_ids)

    def apply_source_joint_orient(self, source_rotations: jnp.ndarray) -> jnp.ndarray:
        """Convert source rotations from T-pose-relative to absolute local rotations."""
        if self.source_joint_orient.size == 0:
            return source_rotations
        return apply_joint_orient_local(
            source_rotations, self.source_joint_orient.astype(source_rotations.dtype),
            self.source_joint_orient_parent_t.astype(source_rotations.dtype))

    def _target_local_rotations(self, batch_size: int, target_joint_count: int, dtype,
                                target_local_rotations) -> jnp.ndarray:
        if target_local_rotations is None:
            if self.target_t_pose_local_rotations.shape[:1] == (target_joint_count,):
                target_local_rotations = self.target_t_pose_local_rotations
            else:
                return jnp.broadcast_to(jnp.eye(3, dtype=dtype),
                                        (batch_size, target_joint_count, 3, 3))
        target_local_rotations = jnp.asarray(target_local_rotations)
        if target_local_rotations.ndim == 3:
            target_local_rotations = target_local_rotations[None]
        if tuple(target_local_rotations.shape[-3:]) != (target_joint_count, 3, 3):
            raise ValueError(
                "target_local_rotations must have shape (B, target_joint_count, 3, 3), "
                f"got {target_local_rotations.shape}")
        target_local_rotations = target_local_rotations.astype(dtype)
        if target_local_rotations.shape[0] == 1 and batch_size > 1:
            return jnp.broadcast_to(target_local_rotations,
                                    (batch_size,) + target_local_rotations.shape[1:])
        if target_local_rotations.shape[0] != batch_size:
            raise ValueError(
                "target_local_rotations batch must match source rotations; "
                f"got {target_local_rotations.shape[0]} and {batch_size}")
        return target_local_rotations

    def _check_source_rotations(self, source_rotations) -> jnp.ndarray:
        source_rotations = jnp.asarray(source_rotations)
        if source_rotations.ndim != 4 or source_rotations.shape[-2:] != (3, 3):
            raise ValueError(
                f"source_rotations must have shape (B, J, 3, 3), got {source_rotations.shape}")
        expected = len(self.source_joint_names)
        if source_rotations.shape[1] != expected:
            raise ValueError(f"Expected {expected} source joints, got {source_rotations.shape[1]}")
        return source_rotations

    def expand_source_rotations_with_identity_twists(self, source_rotations,
                                                     target_local_rotations=None) -> jnp.ndarray:
        """Copy public source rotations into target order and keep target bind rotations elsewhere."""
        source_rotations = self._check_source_rotations(source_rotations)
        target_rotations = self._target_local_rotations(
            source_rotations.shape[0], len(self.target_joint_names), source_rotations.dtype,
            target_local_rotations)
        return target_rotations.at[:, self.control_target_ids].set(
            source_rotations[:, self.control_source_ids])

    def _source_world_transforms_from_rotations(self, source_rotations) -> jnp.ndarray:
        if self.source_parent_ids.size == 0 or self.source_t_pose_local.size == 0:
            raise RuntimeError(
                "aligned_x_swing_twist requires source_world_transforms, or construction "
                "with target_t_pose_world and target_joint_parent_ids")
        if source_rotations.ndim == 3:
            source_rotations = source_rotations[None]
        local_t = self.source_t_pose_local.astype(source_rotations.dtype)[..., :3, 3]
        local_t = jnp.broadcast_to(local_t[None], (source_rotations.shape[0],) + local_t.shape)
        local_transforms = se3_from_rt(source_rotations, local_t)
        return joint_local_to_world_levelorder(
            local_transforms, rig_compute_skeleton_levels(self.source_parent_ids))

    def _aligned_twist_channels_from_world(self, source_world_transforms) -> jnp.ndarray:
        if self.source_bind_quaternions.size == 0:
            raise RuntimeError(
                "aligned_x_swing_twist requires bind data from target_bind_pose_world "
                "and target_joint_parent_ids")
        dtype = source_world_transforms.dtype
        segment_count = self.segment_source_ids.size
        virtual_quaternions = _aligned_virtual_quaternions(
            source_world_transforms[..., :3, :3],
            self.source_bind_quaternions.astype(dtype),
            self.segment_bind_align_quaternions.astype(dtype),
            self.aligned_virtual_segment_ids, self.aligned_virtual_joint_ids)
        q_end = virtual_quaternions[:, :segment_count]
        q_start = virtual_quaternions[:, segment_count:2 * segment_count]
        q_parent = virtual_quaternions[:, 2 * segment_count:]
        segment_local_twist = quaternion_twist_angle_xyzw(quaternion_normalize_xyzw(
            quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_start), q_end)), 0)
        segment_inherited_twist = quaternion_twist_angle_xyzw(quaternion_normalize_xyzw(
            quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_parent), q_start)), 0)
        twist_values = jnp.zeros((source_world_transforms.shape[0], len(self.source_joint_names)),
                                 dtype=dtype)
        twist_values = twist_values.at[:, self.segment_end_source_ids].set(segment_local_twist)
        if self.segment_reverse_mask.any():
            reverse = self.segment_reverse_mask
            twist_values = twist_values.at[:, self.segment_start_source_ids[reverse]].set(
                segment_inherited_twist[:, reverse])
        return twist_values

    def _apply_translation_parameters(self, target_world_transforms) -> jnp.ndarray:
        """Apply the compiled translation parameter matrix to fitted transforms.

        Rotation generation follows the segment driver topology, but the template
        helper translations must be owned by SOMA. The translation matrix keeps
        non-procedural joints unchanged and places twist helpers on the fitted
        public segment so identity or body-part stretch remains coherent.
        """
        target_world_transforms = jnp.asarray(target_world_transforms)
        added_batch = False
        if target_world_transforms.ndim == 3:
            target_world_transforms = target_world_transforms[None]
            added_batch = True
        elif target_world_transforms.ndim != 4:
            raise ValueError(
                "target_world_transforms must have shape (J, 4, 4) or (B, J, 4, 4), "
                f"got {target_world_transforms.shape}")
        if target_world_transforms.shape[-2:] != (4, 4):
            raise ValueError("target_world_transforms must have shape (..., J, 4, 4), "
                             f"got {target_world_transforms.shape}")
        expected = len(self.target_joint_names)
        if target_world_transforms.shape[-3] != expected:
            raise ValueError(
                f"Expected {expected} target joints, got {target_world_transforms.shape[-3]}")
        matrix = self.translation_parameter_matrix.astype(target_world_transforms.dtype)
        out = target_world_transforms.at[..., :3, 3].set(
            jnp.matmul(matrix[None], target_world_transforms[..., :3, 3]))
        return out[0] if added_batch else out

    def _apply_rotation_parameters(self, source_rotations, source_world_transforms=None,
                                   target_local_rotations=None) -> jnp.ndarray:
        target_rotations = self.expand_source_rotations_with_identity_twists(
            source_rotations, target_local_rotations=target_local_rotations)
        twist_rotations = self.twist_rotations_from_source(source_rotations,
                                                           source_world_transforms)
        ids = self.twist_target_ids
        return target_rotations.at[:, ids].set(target_rotations[:, ids] @ twist_rotations)

    def _twist_angles_from_source(self, source_rotations,
                                  source_world_transforms=None) -> jnp.ndarray:
        source_rotations = self._check_source_rotations(source_rotations)
        batch_size = source_rotations.shape[0]
        if (source_world_transforms is None
                and SOMA_ALIGNED_X_SWING_TWIST_MODE in self.rotation_extraction_mode_names):
            source_world_transforms = self._source_world_transforms_from_rotations(
                source_rotations)
        elif source_world_transforms is not None:
            source_world_transforms = jnp.asarray(source_world_transforms)
            if source_world_transforms.ndim == 3:
                source_world_transforms = source_world_transforms[None]
            if source_world_transforms.shape[:2] != source_rotations.shape[:2]:
                raise ValueError(
                    "source_world_transforms must match source_rotations batch/joint shape; "
                    f"got {source_world_transforms.shape[:2]} and {source_rotations.shape[:2]}")
            if source_world_transforms.shape[-2:] != (4, 4):
                raise ValueError("source_world_transforms must have shape (B, J, 4, 4), "
                                 f"got {source_world_transforms.shape}")

        dtype = source_rotations.dtype
        source_axis_signs = self.source_twist_axis_signs.astype(dtype)
        matrices_by_mode = self.rotation_parameter_matrices_by_mode.astype(dtype)
        twist_angles = jnp.zeros((batch_size, len(self.twist_joint_names)), dtype=dtype)
        for mode_index, mode in enumerate(self.rotation_extraction_mode_names):
            if mode == SOMA_LOCAL_X_EULER_TWIST_MODE:
                euler_channels = _local_euler_xyz_from_matrix(source_rotations)
                gather_ids = jnp.broadcast_to(
                    jnp.asarray(self.source_twist_axis_ids)[None, :, None],
                    (batch_size, len(self.source_joint_names), 1))
                twist_values = (jnp.take_along_axis(euler_channels, gather_ids, axis=-1)[..., 0]
                                * source_axis_signs[None])
            elif mode == SOMA_LOCAL_X_SWING_TWIST_MODE:
                twist_values = _swing_twist_channels_from_matrix(
                    source_rotations, self.source_twist_axis_ids, source_axis_signs)
            elif mode == SOMA_ALIGNED_X_SWING_TWIST_MODE:
                if source_world_transforms is None:
                    raise RuntimeError("source_world_transforms is required for aligned twist")
                twist_values = self._aligned_twist_channels_from_world(source_world_transforms)
            else:
                raise RuntimeError(f"Unsupported SOMA procedural twist mode: {mode!r}")
            twist_angles = twist_angles + twist_values @ matrices_by_mode[mode_index].T
        return twist_angles

    def twist_rotations_from_source(self, source_rotations,
                                    source_world_transforms=None) -> jnp.ndarray:
        """Evaluate only procedural twist helper rotations from public source data."""
        twist_angles = self._twist_angles_from_source(source_rotations, source_world_transforms)
        if self._single_twist_axis is not None:
            return single_axis_rotation_matrices(twist_angles, self._single_twist_axis,
                                                 self.twist_axis_signs)
        return _axis_rotations(twist_angles, self.twist_axis_ids, self.twist_axis_signs)

    def expand_world_transforms_from_source_fk(
        self,
        source_rotations,
        source_world_transforms,
        target_local_rotations,
        target_local_translations,
        target_joint_count: int,
    ) -> jnp.ndarray:
        """Expand public FK world transforms to the full procedural skinning rig."""
        source_rotations = self._check_source_rotations(source_rotations)
        if target_joint_count != len(self.target_joint_names):
            raise ValueError(f"Expected target_joint_count={len(self.target_joint_names)}, "
                             f"got {target_joint_count}")
        source_world_transforms = jnp.asarray(source_world_transforms)
        if source_world_transforms.ndim == 3:
            source_world_transforms = source_world_transforms[None]
        if source_world_transforms.shape[:2] != source_rotations.shape[:2]:
            raise ValueError(
                "source_world_transforms must match source_rotations batch/joint shape; "
                f"got {source_world_transforms.shape[:2]} and {source_rotations.shape[:2]}")
        if source_world_transforms.shape[-2:] != (4, 4):
            raise ValueError("source_world_transforms must have shape (B, J, 4, 4), "
                             f"got {source_world_transforms.shape}")

        batch_size = source_world_transforms.shape[0]
        dtype = source_world_transforms.dtype
        target_local_translations = jnp.asarray(target_local_translations)
        if target_local_translations.ndim == 2:
            target_local_translations = target_local_translations[None]
        if tuple(target_local_translations.shape[-2:]) != (target_joint_count, 3):
            raise ValueError(
                "target_local_translations must have shape (B, target_joint_count, 3), "
                f"got {target_local_translations.shape}")
        if target_local_translations.shape[0] == 1 and batch_size > 1:
            target_local_translations = jnp.broadcast_to(
                target_local_translations, (batch_size,) + target_local_translations.shape[1:])
        elif target_local_translations.shape[0] != batch_size:
            raise ValueError(
                "target_local_translations batch must match source_world_transforms; "
                f"got {target_local_translations.shape[0]} and {batch_size}")
        target_local_translations = target_local_translations.astype(dtype)
        target_base_rotations = self._target_local_rotations(
            batch_size, target_joint_count, dtype, target_local_rotations)

        target_world_transforms = jnp.broadcast_to(jnp.eye(4, dtype=dtype),
                                                   (batch_size, target_joint_count, 4, 4))
        target_world_transforms = target_world_transforms.at[:, self.control_target_ids].set(
            source_world_transforms[:, self.control_source_ids])
        assigned = np.zeros(target_joint_count, dtype=bool)
        assigned[self.control_target_ids] = True

        twist_target_ids = self.twist_target_ids
        if twist_target_ids.size > 0 and self.twist_parent_target_ids.size != twist_target_ids.size:
            raise RuntimeError(
                "expand_world_transforms_from_source_fk requires target_joint_parent_ids "
                "at construction time.")
        if twist_target_ids.size > 0:
            twist_rotations = self.twist_rotations_from_source(
                source_rotations=source_rotations,
                source_world_transforms=source_world_transforms).astype(dtype)
            twist_local_transforms = se3_from_rt(
                target_base_rotations[:, twist_target_ids] @ twist_rotations,
                target_local_translations[:, twist_target_ids])
            target_world_transforms = target_world_transforms.at[:, twist_target_ids].set(
                target_world_transforms[:, self.twist_parent_target_ids] @ twist_local_transforms)
            assigned[twist_target_ids] = True

        if (~assigned).any():
            if self.target_joint_parent_ids.size != target_joint_count:
                raise RuntimeError(
                    "expand_world_transforms_from_source_fk requires target_joint_parent_ids "
                    "to fill non-public, non-procedural target joints.")
            target_parent_ids = self.target_joint_parent_ids
            target_local_transforms = se3_from_rt(target_base_rotations,
                                                  target_local_translations)
            for joint_ids, _parent_ids in rig_compute_skeleton_levels(target_parent_ids):
                fill_ids = joint_ids[~assigned[joint_ids]]
                if fill_ids.size == 0:
                    continue
                target_world_transforms = target_world_transforms.at[:, fill_ids].set(
                    target_world_transforms[:, target_parent_ids[fill_ids]]
                    @ target_local_transforms[:, fill_ids])
                assigned[fill_ids] = True
        return target_world_transforms

    def forward(self, source_rotations=None, source_world_transforms=None,
                target_world_transforms=None,
                target_local_rotations=None) -> SOMAProceduralTransformOutput:
        """Apply the compiled SOMA procedural parameter transform."""
        if source_rotations is None and target_world_transforms is None:
            raise ValueError("Provide source_rotations, target_world_transforms, or both")
        rotations = (self._apply_rotation_parameters(
            source_rotations, source_world_transforms,
            target_local_rotations=target_local_rotations)
            if source_rotations is not None else None)
        transforms = (self._apply_translation_parameters(target_world_transforms)
                      if target_world_transforms is not None else None)
        return SOMAProceduralTransformOutput(rotations=rotations, transforms=transforms)

    __call__ = forward
