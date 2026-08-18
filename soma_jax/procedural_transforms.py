"""Procedural twist-joint transforms for SOMA template rigs (JAX port).

Faithful JAX port of the core of
``third_party/SOMA-X/soma/procedural_transforms.py``: parse
``SOMA_procedural_transforms.json``, extract per-segment twist channels from
the 78 public-rig rotations, then distribute them through the trained sparse
parameter matrix to emit additional twist-joint local rotations that extend the
public rig. :meth:`ProceduralTransforms.extend_public_rotations` emits the 110
joints the JSON describes (78 public + 32 twist);
:meth:`ProceduralTransforms.extend_to_template_rig` emits the full **122** in
authored template order, filling the 12 USD-only helper bones with identity.

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
  ``(B, 110, 3, 3)``; :meth:`extend_to_template_rig` gives the full
  ``(B, 122, 3, 3)`` in template order.

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
ready to feed an extended :class:`BatchedSkinning`. The 122-joint integration
is "additive" in the sense that the original 78-joint rig output stays
identical; the 32 twist joints are leaves under their segment's start joint
and contribute only to LBS via their skinning weights.

What's NOT included
===================
* USD parsing for the upstream ``SOMA_template_rig.usda`` (345 MB). Not
  required for runtime — the procedural module derives the twist-joint bind
  positions analytically from the public bind positions via the translation
  matrix above. The USD is the authored DCC representation, redundant with
  the JSON for inference.

Upstream: ``soma/procedural_transforms.py``
    **Complete port.** ``extend_to_template_rig`` emits all **122** joints in the
    authored template order (the 12 USD-only helper bones come from
    ``SOMA_template_rig.usda`` and take identity).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

from .geometry.lbs import compute_skeleton_levels
from .geometry.transforms import (
    matrix_to_quaternion_xyzw,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_twist_angle_xyzw,
    single_axis_rotation_matrices,
)


# Modes from soma.procedural_transforms.SOMA_PROCEDURAL_TRANSFORM_MODES.
SOMA_LOCAL_X_EULER_TWIST_MODE = "local_x_euler"
SOMA_LOCAL_X_SWING_TWIST_MODE = "local_x_swing_twist"
SOMA_ALIGNED_X_SWING_TWIST_MODE = "aligned_x_swing_twist"
SOMA_PROCEDURAL_TRANSFORM_MODES = (
    SOMA_LOCAL_X_EULER_TWIST_MODE,
    SOMA_LOCAL_X_SWING_TWIST_MODE,
    SOMA_ALIGNED_X_SWING_TWIST_MODE,
)


@dataclass
class SOMATwistSegmentSpec:
    """One procedural twist chain (e.g. LeftArm with 4 twist helpers).

    Matches ``soma.procedural_transforms.SOMATwistSegmentSpec``.
    """
    start_joint: str           # rotation source (e.g. "LeftArm")
    end_joint: str             # bone end (e.g. "LeftForeArm")
    parent_joint: str | None   # parent context (e.g. "LeftShoulder"); optional in JSON
    twist_joints: tuple[str, ...]   # output twist joint names
    source_axis: str = "x"  # axis to extract twist around
    source_sign: float = 1.0
    reverse: bool = False   # twist accumulates from end -> start when True


@dataclass
class SOMAProceduralTransformDefinition:
    """Parsed ``SOMA_procedural_transforms.json``.

    Holds the public 78-joint name list, the 122-joint full-rig name list,
    the 8 twist segments, and the sparse rotation parameter matrix.

    Matches ``soma.procedural_transforms.SOMAProceduralTransformDefinition``.
    """
    schema_version: str
    template_joint_count: int
    main_joint_names: tuple[str, ...]        # 78 public joints
    segments: tuple[SOMATwistSegmentSpec, ...]
    modes: tuple[str, ...]
    # Sparse parameter matrix in {row, column, value} COO-named form. Rows
    # are extracted twist channels (one per (segment, mode)), columns are
    # output (twist_joint, axis) labels — keep them as the raw upstream
    # strings so we can defer index mapping until evaluation time.
    rotation_matrix_rows: tuple[str, ...]
    rotation_matrix_cols: tuple[str, ...]
    rotation_matrix_entries: tuple[dict, ...]
    # Translation matrix: maps each twist joint's world position to a convex
    # combination of two public joints' positions (5/35/65/95% along the
    # segment for the standard 4-helper distribution).
    translation_matrix_rows: tuple[str, ...]
    translation_matrix_cols: tuple[str, ...]
    translation_matrix_entries: tuple[dict, ...]

    #: One extraction mode per procedural (twist) joint, in
    #: :py:attr:`twist_joint_names` order — upstream's
    #: ``SOMAProceduralTransformDefinition.rotation_extraction_modes``.
    rotation_extraction_modes: tuple[str, ...] = ()

    @property
    def twist_joint_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for s in self.segments:
            names.extend(s.twist_joints)
        return tuple(names)

    def full_joint_names(self) -> tuple[str, ...]:
        """Public joints followed by every twist joint, in order."""
        return tuple(self.main_joint_names) + self.twist_joint_names

    @property
    def rotation_extraction_mode(self) -> str:
        """The single extraction mode this definition asks for.

        Upstream stores one mode *per procedural joint*. Every published SOMA-X
        asset assigns the same mode to all of them, and
        :py:class:`ProceduralTransforms` evaluates one mode per layer, so this
        collapses the per-joint tuple and refuses rather than silently picking
        one when an asset really is mixed.
        """
        if not self.rotation_extraction_modes:
            return SOMA_ALIGNED_X_SWING_TWIST_MODE
        distinct = set(self.rotation_extraction_modes)
        if len(distinct) > 1:
            # Mixed definitions are handled per joint inside
            # `ProceduralTransforms.emit_twist_rotmats`; this reports the mode a
            # single-mode caller would get, which is the most common one.
            return max(distinct, key=self.rotation_extraction_modes.count)
        return next(iter(distinct))


def _parse_rotation_extraction_modes(
    raw: dict, modes: tuple[str, ...], twist_joint_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Which twist extractor drives each procedural joint.

    Port of upstream ``_parse_rotation_extraction_modes``. The JSON's
    ``rotation_extraction`` is either a bare mode string applied to every
    procedural joint, or a mapping with ``default`` plus
    ``per_procedural_joint`` overrides. Returns one mode **per twist joint**,
    matching upstream's ``SOMAProceduralTransformDefinition.rotation_extraction_modes``.

    The asset shipped with SOMA-X v0.2.1 is the bare-string form
    (``"aligned_x_swing_twist"``), so all 32 procedural joints share one mode.
    """
    spec = raw.get("rotation_extraction")
    if spec is None:
        raise ValueError("rotation_extraction is required")

    def _check(mode, where):
        if mode not in modes:
            raise ValueError(f"{where}: unknown mode {mode!r}; expected one of {modes}")
        return str(mode)

    if isinstance(spec, str):
        return (_check(spec, "rotation_extraction"),) * len(twist_joint_names)

    per_joint = dict(spec.get("per_procedural_joint", {}))
    unknown = sorted(set(per_joint) - set(twist_joint_names))
    if unknown:
        raise ValueError(
            "rotation_extraction.per_procedural_joint references unknown procedural "
            f"joints: {unknown}"
        )
    default = spec.get("default")
    if default is None:
        missing = [n for n in twist_joint_names if n not in per_joint]
        if missing:
            raise ValueError(
                "rotation_extraction.default is required unless every procedural joint "
                f"has an override; missing: {missing}"
            )
    else:
        default = _check(default, "rotation_extraction.default")
    return tuple(
        _check(per_joint.get(n, default), f"rotation_extraction.per_procedural_joint[{n}]")
        for n in twist_joint_names
    )


def load_definition(path: str | Path) -> SOMAProceduralTransformDefinition:
    """Parse ``SOMA_procedural_transforms.json`` into a definition object."""
    with open(path) as f:
        raw = json.load(f)

    segments = tuple(
        SOMATwistSegmentSpec(
            start_joint=s["start_joint"],
            end_joint=s["end_joint"],
            parent_joint=s.get("parent_joint"),
            twist_joints=tuple(s["twist_joints"]),
            source_axis=str(s.get("source_axis", "x")).lower(),
            source_sign=float(s.get("source_sign", 1.0)),
            reverse=bool(s.get("reverse", False)),
        )
        for s in raw["segments"]
    )

    twist_joint_names = tuple(n for s in segments for n in s.twist_joints)
    modes = tuple(raw.get("modes", SOMA_PROCEDURAL_TRANSFORM_MODES))

    rot = raw["parameter_matrices"]["rotation"]
    tr = raw["parameter_matrices"]["translation"]
    return SOMAProceduralTransformDefinition(
        schema_version=str(raw.get("schema_version", "")),
        template_joint_count=int(raw["template_asset"]["joint_count"]),
        main_joint_names=tuple(raw["public_rig_derivation"]["main_joint_names"]),
        segments=segments,
        modes=modes,
        rotation_extraction_modes=_parse_rotation_extraction_modes(
            raw, modes, twist_joint_names),
        rotation_matrix_rows=tuple(rot["rows"]),
        rotation_matrix_cols=tuple(rot["columns"]),
        rotation_matrix_entries=tuple(rot["entries"]),
        translation_matrix_rows=tuple(tr["rows"]),
        translation_matrix_cols=tuple(tr["columns"]),
        translation_matrix_entries=tuple(tr["entries"]),
    )


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


_AXIS_NAME_TO_IDX = {"x": 0, "y": 1, "z": 2}


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
            [_AXIS_NAME_TO_IDX[s.source_axis] for s in definition.segments],
            dtype=np.int32,
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
        for entry in definition.rotation_matrix_entries:
            tj = entry["row"]
            src_joint = entry["column"]
            if tj not in twist_name_to_idx:
                continue
            if src_joint not in name_to_idx:
                continue
            weights[twist_name_to_idx[tj], name_to_idx[src_joint]] = float(
                entry["value"]
            )
        self._twist_source_weights = weights        # (n_twist, n_public)

        # Per-twist joint, store the output axis and sign — defaulting to
        # the segment the joint belongs to.
        twist_to_segment: dict[str, int] = {}
        for si, seg in enumerate(definition.segments):
            for n in seg.twist_joints:
                twist_to_segment[n] = si
        self._twist_axis = np.asarray(
            [_AXIS_NAME_TO_IDX[definition.segments[twist_to_segment[n]].source_axis]
             for n in twist_names],
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
        for entry in definition.translation_matrix_entries:
            tj = entry["row"]
            src_joint = entry["column"]
            if tj not in twist_name_to_idx or src_joint not in name_to_idx:
                continue
            tr_weights[twist_name_to_idx[tj], name_to_idx[src_joint]] = float(
                entry["value"]
            )
        self._twist_position_weights = tr_weights

    # --------------------------- evaluation ---------------------------------
    def set_bind_data(self, bind_world: jnp.ndarray) -> None:
        """Supply T-pose bind transforms to enable upstream's `aligned` mode.

        Args:
            bind_world: (n_public, 4, 4) source-joint world transforms at bind
                (upstream's ``target_t_pose_world`` restricted to the public
                joints). Without this, ``aligned`` falls back to a start-joint
                scalar that is not upstream-equivalent.
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
        """Emit the **full 122-joint** template rig, in authored order.

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
            ``(rotmats, names)`` with rotmats ``(B, 122, 3, 3)`` ordered to
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

        # Remaining joints (the USD-only helpers) inherit their parent's world
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
# Full 122-joint template rig
# ---------------------------------------------------------------------------


def template_joint_names(usd_path=None) -> list[str]:
    """Authored joint names of the 122-joint template rig, in template order.

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
        122 joint names, template order.
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
