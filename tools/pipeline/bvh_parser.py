"""Minimal BVH parser for SOMA-skeleton motion clips.

Expects BVH files whose hierarchy is the canonical SOMA 78-joint skeleton (same
names, same DFS order). This parser returns SOMA-ready arrays: per-frame
axis-angle rotations + root translation + source FPS.

Output of `load_soma_bvh(path)`:
    poses              (N, J, 3)   per-joint axis-angle (from `rotmats`)
    rotmats            (N, J, 3, 3) per-joint rotation, composed in the order
                                   each joint's CHANNELS line declares
    root_translation   (N, 3)      root position (meters; BVH offsets are cm)
    joint_names        list[str]
    parents            (J,)        int parent indices (root: -1)
    channel_orders     list[str]   per-joint Euler order as declared, e.g. "ZXY"
    source_fps         float
    source_total_frames int
    source_duration_s  float

Convention note: this returns each joint's channel rotation verbatim. The SOMA
clips these tools consume bake the bind orientation in, so `gen_motion.py` and
`demo_soma_vis.py` feed them as `absolute_pose=True` (no joint-orient step).
One consequence: a joint whose rotation channels are all zero comes back as
IDENTITY, which under that convention means world-axis-aligned, not "at rest".
For joints with skinning weight (e.g. Jaw) that is a real, if small,
deformation -- check before trusting a clip that leaves joints unanimated.
"""
from __future__ import annotations
import re
import numpy as np


def _parse_hierarchy(text: str):
    """Walk the HIERARCHY section. Returns (names, parents, channels_per_joint)."""
    names: list[str] = []
    parents: list[int] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    # Tokenize the HIERARCHY block by line, ignoring End Site (we only consume
    # JOINT / ROOT — some exporters encode "End" markers as full JOINTs).
    hier = text.split("MOTION", 1)[0]
    i_joint = -1
    for line in hier.splitlines():
        s = line.strip()
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            name = s.split(None, 1)[1]
            i_joint += 1
            names.append(name)
            parents.append(stack[-1] if stack else -1)
            channels.append([])
            stack.append(i_joint)
        elif s.startswith("End Site"):
            # Skip End Site blocks entirely (no channels, no joint added).
            stack.append(None)
        elif s.startswith("CHANNELS "):
            parts = s.split()
            n = int(parts[1])
            if stack and stack[-1] is not None:
                channels[stack[-1]] = parts[2 : 2 + n]
        elif s == "}":
            if stack:
                stack.pop()
    return names, np.array(parents, dtype=np.int64), channels


def _parse_motion(text: str):
    motion_section = text.split("MOTION", 1)[1]
    m = re.search(r"Frames:\s*(\d+)", motion_section)
    n_frames = int(m.group(1))
    m = re.search(r"Frame Time:\s*([0-9.eE+-]+)", motion_section)
    frame_time = float(m.group(1))
    # data starts after "Frame Time: X" line
    data_start = motion_section.find(str(frame_time)) + len(str(frame_time))
    rows = motion_section[data_start:].split()
    arr = np.asarray(rows, dtype=np.float32).reshape(n_frames, -1)
    return arr, n_frames, frame_time


def _axis_rotmat(axis: str, deg: np.ndarray) -> np.ndarray:
    """Elementary right-handed rotation about X/Y/Z for column vectors. (...,) -> (...,3,3)."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    R = np.zeros(deg.shape + (3, 3), dtype=np.float32)
    if axis == "X":
        R[..., 0, 0] = 1.0
        R[..., 1, 1] = c; R[..., 1, 2] = -s
        R[..., 2, 1] = s; R[..., 2, 2] = c
    elif axis == "Y":
        R[..., 1, 1] = 1.0
        R[..., 0, 0] = c; R[..., 0, 2] = s
        R[..., 2, 0] = -s; R[..., 2, 2] = c
    elif axis == "Z":
        R[..., 2, 2] = 1.0
        R[..., 0, 0] = c; R[..., 0, 1] = -s
        R[..., 1, 0] = s; R[..., 1, 1] = c
    else:
        raise ValueError(f"unknown rotation axis {axis!r}")
    return R


def _euler_deg_to_rotmat(order: str, vals: np.ndarray) -> np.ndarray:
    """Compose Euler angles in the order the BVH declares them.

    ``order`` is the joint's rotation-channel order as written in its CHANNELS
    line (e.g. "ZXY" for "Zrotation Xrotation Yrotation"); ``vals`` is (N, k)
    degrees in that same order. BVH channel order is the composition order for
    column vectors, so "ZXY" means R = Rz @ Rx @ Ry.

    The order is per joint and per file -- BVH does not mandate one. Assuming a
    fixed order silently yields a wrong rotation whenever a file disagrees
    (tens of degrees on joints that rotate about more than one axis).
    """
    R = None
    for k, axis in enumerate(order):
        Rk = _axis_rotmat(axis, vals[..., k])
        R = Rk if R is None else R @ Rk
    if R is None:
        R = np.broadcast_to(np.eye(3, dtype=np.float32), vals.shape[:-1] + (3, 3)).copy()
    return R.astype(np.float32)


def _rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) axis-angle (trace formula; unstable near pi)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    th = np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    axis = np.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], axis=-1)
    sin_th = np.sin(th)
    denom = np.where(sin_th > 1e-6, 2.0 * sin_th, 1.0)
    return (axis / denom[..., None] * th[..., None]).astype(np.float32)


def load_soma_bvh(path: str, units_to_meters: float = 0.01) -> dict:
    """Parse a SOMA-skeleton BVH motion file."""
    text = open(path).read()
    names, parents, channels = _parse_hierarchy(text)
    motion, n_frames, frame_time = _parse_motion(text)

    J = len(names)
    # Map channel layout: for each joint, slice the motion frame and pick
    # translation (if any) + its rotation channels IN THE ORDER IT DECLARES.
    rotmats = np.zeros((n_frames, J, 3, 3), dtype=np.float32)
    rotmats[:, :] = np.eye(3, dtype=np.float32)
    root_trans = np.zeros((n_frames, 3), dtype=np.float32)
    hips_trans = np.zeros((n_frames, 3), dtype=np.float32)
    orders: list[str] = []
    col = 0
    for j, chans in enumerate(channels):
        nc = len(chans)
        block = motion[:, col : col + nc]
        col += nc
        # Translation channels: SOMA-format BVHs put position channels on BOTH
        # Root (joint 0) and Hips (joint 1). Root usually carries the static
        # "rig origin" offset (often all-zero), while Hips carries the actual
        # per-frame world translation (the character walking forward).
        if "Xposition" in chans:
            xi, yi, zi = chans.index("Xposition"), chans.index("Yposition"), chans.index("Zposition")
            xyz = block[:, [xi, yi, zi]].astype(np.float32) * units_to_meters
            if j == 0:
                root_trans = xyz
            elif names[j] == "Hips":
                hips_trans = xyz
        # Rotation channels: BVH declares both WHICH axes and in WHAT ORDER they
        # compose. Read them positionally so a file using e.g. "Zrotation
        # Xrotation Yrotation" is composed Rz @ Rx @ Ry, not a fixed ZYX.
        rot_cols = [(c[0].upper(), k) for k, c in enumerate(chans)
                    if c.lower().endswith("rotation")]
        order = "".join(a for a, _ in rot_cols)
        orders.append(order)
        if rot_cols:
            vals = block[:, [k for _, k in rot_cols]].astype(np.float32)
            rotmats[:, j] = _euler_deg_to_rotmat(order, vals)
    assert col == motion.shape[1], f"channel count mismatch: parsed {col}, motion has {motion.shape[1]}"

    poses = _rotmat_to_axis_angle(rotmats)
    fps = 1.0 / frame_time
    # Prefer Hips position when it carries the per-frame walk; many SOMA BVHs
    # leave Root all-zero and animate only Hips. If both are populated, sum
    # them (Root = static rig offset, Hips = per-frame motion in Root space).
    if np.abs(hips_trans).sum() > 0 and np.abs(root_trans).sum() == 0:
        effective_trans = hips_trans
    else:
        effective_trans = root_trans + hips_trans
    return dict(
        poses=poses,                                # (N, J, 3) axis-angle
        rotmats=rotmats.astype(np.float32),         # (N, J, 3, 3) preferred (exact)
        channel_orders=orders,                      # per-joint Euler order as declared
        root_translation=effective_trans,           # (N, 3) meters
        joint_names=names,
        parents=parents,
        source_fps=float(fps),
        source_total_frames=int(n_frames),
        source_duration_s=n_frames / fps,
        n_frames=int(n_frames),
    )
