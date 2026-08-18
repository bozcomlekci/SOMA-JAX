"""Minimal BVH parser for SOMA-skeleton motion clips.

Expects BVH files whose hierarchy is the canonical SOMA 78-joint skeleton (same
names, same DFS order). This parser returns SOMA-ready arrays: per-frame
axis-angle rotations + root translation + source FPS.

Output of `load_soma_bvh(path)`:
    poses              (N, J, 3)   parent-relative axis-angle per joint
    root_translation   (N, 3)      root position (meters; BVH offsets are cm)
    joint_names        list[str]
    parents            (J,)        int parent indices (root: -1)
    source_fps         float
    source_total_frames int
    source_duration_s  float
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


def _euler_zyx_deg_to_rotmat(rots_zyx_deg: np.ndarray) -> np.ndarray:
    """(N, J, 3) ZYX euler in degrees -> (N, J, 3, 3) rotation matrices.

    BVH "Zrotation Yrotation Xrotation" channel order means
        R = Rz(z) @ Ry(y) @ Rx(x)   applied to column vectors.
    """
    r = np.deg2rad(rots_zyx_deg)
    cz, sz = np.cos(r[..., 0]), np.sin(r[..., 0])
    cy, sy = np.cos(r[..., 1]), np.sin(r[..., 1])
    cx, sx = np.cos(r[..., 2]), np.sin(r[..., 2])
    R = np.zeros(rots_zyx_deg.shape + (3,), dtype=np.float32)
    R[..., 0, 0] = cz * cy
    R[..., 0, 1] = cz * sy * sx - sz * cx
    R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy
    R[..., 1, 1] = sz * sy * sx + cz * cx
    R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cy * cx
    return R


def _euler_zyx_deg_to_axis_angle(rots_zyx_deg: np.ndarray) -> np.ndarray:
    """ZYX Euler degrees -> axis-angle (trace formula; unstable near π — prefer the rotmat path)."""
    R = _euler_zyx_deg_to_rotmat(rots_zyx_deg)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(cos_th)
    sin_th = np.sin(th)
    axis = np.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], axis=-1)
    denom = np.where(sin_th > 1e-6, 2.0 * sin_th, 1.0)
    return (axis / denom[..., None] * th[..., None]).astype(np.float32)


def load_soma_bvh(path: str, units_to_meters: float = 0.01) -> dict:
    """Parse a SOMA-skeleton BVH motion file."""
    text = open(path).read()
    names, parents, channels = _parse_hierarchy(text)
    motion, n_frames, frame_time = _parse_motion(text)

    J = len(names)
    # Map channel layout: for each joint, slice the motion frame and pick
    # translation (if any) + ZYX rotation channels.
    rots = np.zeros((n_frames, J, 3), dtype=np.float32)   # [Zrot, Yrot, Xrot] deg
    root_trans = np.zeros((n_frames, 3), dtype=np.float32)
    hips_trans = np.zeros((n_frames, 3), dtype=np.float32)
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
        # Rotation channels — ZYX order
        zi = chans.index("Zrotation") if "Zrotation" in chans else None
        yi = chans.index("Yrotation") if "Yrotation" in chans else None
        xi = chans.index("Xrotation") if "Xrotation" in chans else None
        if zi is not None:
            rots[:, j, 0] = block[:, zi]
        if yi is not None:
            rots[:, j, 1] = block[:, yi]
        if xi is not None:
            rots[:, j, 2] = block[:, xi]
    assert col == motion.shape[1], f"channel count mismatch: parsed {col}, motion has {motion.shape[1]}"

    poses = _euler_zyx_deg_to_axis_angle(rots)
    rotmats = _euler_zyx_deg_to_rotmat(rots)        # (N, J, 3, 3) — exact, no axis-angle round-trip
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
        root_translation=effective_trans,           # (N, 3) meters
        joint_names=names,
        parents=parents,
        source_fps=float(fps),
        source_total_frames=int(n_frames),
        source_duration_s=n_frames / fps,
        n_frames=int(n_frames),
    )
