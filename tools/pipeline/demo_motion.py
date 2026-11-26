"""Real motion capture on the SOMA rig: CMU mocap -> SOMA-JAX -> render.

Fetches a trial from the CMU Graphics Lab Motion Capture Database
(mocap.cs.cmu.edu: an ASF skeleton and an AMC motion), retargets it onto the
SOMA skeleton, poses the SOMA body with SOMA-JAX and renders it headless::

    python tools/pipeline/demo_motion.py                     # a walk (CMU 35_08)
    python tools/pipeline/demo_motion.py --motion cartwheel  # an example; --list shows them
    python tools/pipeline/demo_motion.py --gallery           # every example, and a gallery
    python tools/pipeline/demo_motion.py --trial 14_01 --start 5 --seconds 4   # any trial
    python tools/pipeline/demo_motion.py --skeleton          # SOMA skeleton in a see-through body
    python tools/pipeline/demo_motion.py --identity-model garment_measurement \\
        --identity-path assets/identity/identity_garment.npz --identity-coeffs 1.5 -1 0.5

A motion writes to ``demo_renders/<name>`` (or ``--output-dir``): ``<name>.gif``,
``<name>.mp4`` (when imageio-ffmpeg is installed), ``<name>_strip.png`` (a pose
every ``--strip-every`` seconds in one picture) and ``<name>.soma.npz``, the
retargeted motion in SOMA's NPZ format, which the other tools replay
(``tools/vis/vis_pyrender.py --animation demo_renders/walk/walk.soma.npz``).
``--gallery`` renders every example that way and then ``demo_renders/gallery``:
``gallery.gif`` (and ``.mp4``) plays them side by side, ``gallery.png`` holds
their strips.

Staging. A motion that travels (the hips end over a metre from where they
start) is turned to travel along X, front rather than back to the camera
(a cartwheel faces it), and the camera follows; a motion in place is turned
three-quarters to a camera that holds still. Grounding puts the feet on the
floor in all but the lowest 5% of frames.

Retargeting. The ASF skeleton rests in a T-pose facing +Z with its left side
on +X, as SOMA's does, and every ASF rest frame is world-aligned, so a bone's
world rotation is its rotation away from rest. SOMA's pose input is relative
to its own T-pose too: for world-aligned deltas ``D`` (a joint's world rotation
times its T-pose orientation transposed), the local input is
``R_j = D_parent^T @ D_j``. The two rest poses are the same posture (upright,
arms out, head level, feet flat) except for the legs, which CMU rests 20
degrees apart, so each SOMA joint takes the world rotation of the matching CMU
bone as it is, and only the thighs and shins first turn from SOMA's rest
direction onto CMU's; the posed legs then follow the captured ones exactly.
Matching the rest directions of the other bones too would bend SOMA into CMU's
proportions, as the skeletons place the neck, clavicle and foot joints
differently: CMU's neck rises nearly upright where SOMA's leans forward like a
real one, and tipping SOMA's back to match pressed the chin into the throat.
Joints CMU does not track (fingers, eyes, jaw) follow their parent, and the
root path is scaled by the ratio of leg lengths so the stride fits SOMA's legs.

Data: "The data used in this project was obtained from mocap.cs.cmu.edu. The
database was created with funding from NSF EIA-0196217." The clips may be used
freely, in commercial products too, but not resold; they are downloaded on
first use into ``assets/third_party/cmu_mocap`` and are not part of this
repository. Most were captured at 120 fps; the subjects in ``CMU_60FPS`` at 60.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

CMU_URL = "http://mocap.cs.cmu.edu/subjects/{subject}/{name}"
CMU_FPS = 120.0
#: Subjects (and the one trial) the CMU motion index lists at 60 fps, not 120.
CMU_60FPS = {33, 34, 42, 60, 61, 62, 74, 75, 77, 79, 80, 87, 88, 89, "136_19"}

#: Example motions: name -> (CMU trial, start (s), length (s) or None for the
#: rest of the trial, caption).
EXAMPLES = {
    "walk": ("35_08", 0.0, None, "walk"),
    "run": ("16_55", 0.0, None, "run"),
    "jump": ("13_11", 0.8, 2.2, "forward jump"),
    "cartwheel": ("90_02", 2.6, 2.6, "cartwheel"),
    "backflip": ("87_03", 1.6, 2.4, "backflip"),
    "charleston": ("93_03", 0.0, None, "Charleston"),
    "ballet": ("05_17", 3.2, 4.2, "ballet: grand jete"),
    "boxing": ("13_17", 7.3, 4.0, "boxing"),
    "kick": ("135_07", 3.0, 4.0, "karate: roundhouse kick"),
    "basketball": ("124_05", 2.0, 3.0, "basketball: jump shot"),
    "jumping-jacks": ("13_29", 0.0, 4.0, "jumping jacks"),
    "pitch": ("124_01", 1.6, 3.6, "baseball pitch"),
}


def cmu_fps(trial: str) -> float:
    """Capture rate of a CMU trial such as ``"35_08"`` (see :data:`CMU_60FPS`)."""
    subject = trial.split("_")[0]
    slow = trial in CMU_60FPS or (subject.isdigit() and int(subject) in CMU_60FPS)
    return 60.0 if slow else CMU_FPS

#: SOMA joint -> (CMU bone whose world rotation drives it, SOMA child joint that
#: ends the joint's bone). ``None``: the pelvis, which has no single bone
#: direction. Unlisted joints follow their parent.
SOMA_FROM_CMU = {
    "Hips": ("root", None),
    "Spine1": ("lowerback", "Spine2"),
    "Spine2": ("upperback", "Chest"),
    "Chest": ("thorax", "Neck1"),
    "Neck1": ("lowerneck", "Neck2"),
    "Neck2": ("upperneck", "Head"),
    "Head": ("head", "HeadEnd"),
}
for _side, _s in (("Left", "l"), ("Right", "r")):
    for _joint, _bone, _child in (
        ("Shoulder", "clavicle", "Arm"), ("Arm", "humerus", "ForeArm"),
        ("ForeArm", "radius", "Hand"), ("Hand", "hand", "HandMiddle1"),
        ("Leg", "femur", "Shin"), ("Shin", "tibia", "Foot"),
        ("Foot", "foot", "ToeBase"), ("ToeBase", "toes", "ToeEnd"),
    ):
        SOMA_FROM_CMU[_side + _joint] = (_s + _bone, _side + _child)

#: Joints whose rest pose differs between the two skeletons: every CMU skeleton
#: rests its thighs and shins 20 degrees apart, along (+-0.342, -0.940, 0), while
#: SOMA's hang straight down. These first turn SOMA's rest bone onto CMU's.
REST_ALIGNED = ("LeftLeg", "LeftShin", "RightLeg", "RightShin")


# --------------------------------------------------------------------------- ASF / AMC
def fetch_cmu_trial(trial: str, cache_dir: Path) -> tuple[Path, Path]:
    """Download (once) the ASF + AMC of a CMU trial such as ``"35_08"``."""
    subject = trial.split("_")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in (f"{subject}.asf", f"{trial}.amc"):
        path = cache_dir / name
        if not path.exists():
            url = CMU_URL.format(subject=subject, name=name)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        data = r.read()
                    break
                except OSError as e:
                    if attempt == 3:
                        sys.exit(f"Could not download {url}: {e}\n"
                                 "Pass --asf / --amc to use local files instead.")
                    time.sleep(2 ** attempt)
            path.with_suffix(path.suffix + ".part").write_bytes(data)
            path.with_suffix(path.suffix + ".part").replace(path)
            print(f"  downloaded {url}")
        paths.append(path)
    return paths[0], paths[1]


def parse_asf(text: str) -> dict:
    """Skeleton of an Acclaim ASF file: bones, hierarchy and the length unit."""
    bones, hierarchy, root = {}, {}, {}
    section, bone, length_unit = None, None, 1.0
    for raw in text.replace("\r", "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(":"):
            section = line.split()[0][1:]
            continue
        tok = line.split()
        if section == "units" and tok[0] == "length":
            length_unit = float(tok[1])
        elif section == "root":
            root[tok[0]] = tok[1:]
        elif section == "bonedata":
            if tok[0] == "begin":
                bone = {"dof": []}
            elif tok[0] == "end":
                bones[bone["name"]] = bone
            elif tok[0] == "name":
                bone["name"] = tok[1]
            elif tok[0] == "direction":
                bone["direction"] = np.asarray(tok[1:4], np.float64)
            elif tok[0] == "length":
                bone["length"] = float(tok[1])
            elif tok[0] == "axis":
                bone["axis"] = np.asarray(tok[1:4], np.float64)
                bone["axis_order"] = tok[4] if len(tok) > 4 else "XYZ"
            elif tok[0] == "dof":
                bone["dof"] = [d.lower() for d in tok[1:]]
        elif section == "hierarchy" and tok[0] not in ("begin", "end"):
            hierarchy[tok[0]] = tok[1:]
    parent = {c: p for p, cs in hierarchy.items() for c in cs}
    return {"bones": bones, "parent": parent, "hierarchy": hierarchy, "root": root,
            "meters_per_unit": 0.0254 / length_unit}    # ASF lengths are inches / unit


def parse_amc(text: str) -> list[dict]:
    """Frames of an AMC file: ``[{bone: values}, ...]`` in the ASF's dof order."""
    frames = []
    for raw in text.replace("\r", "").splitlines():
        line = raw.strip()
        if not line or line[0] in "#:":
            continue
        tok = line.split()
        if len(tok) == 1 and tok[0].isdigit():
            frames.append({})
        elif frames:
            frames[-1][tok[0]] = np.asarray(tok[1:], np.float64)
    return frames


def _axis_rot(axis: str, deg: np.ndarray) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    m = {"x": [[o, z, z], [z, c, -s], [z, s, c]],
         "y": [[c, z, s], [z, o, z], [-s, z, c]],
         "z": [[c, -s, z], [s, c, z], [z, z, o]]}[axis]
    return np.moveaxis(np.asarray(m), (0, 1), (-2, -1))


def euler(deg_xyz: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """(..., 3) angles (x, y, z, degrees) -> rotation applying the axes in
    ``order``, first letter first — the ASF/AMC convention (XYZ: Rz @ Ry @ Rx)."""
    deg_xyz = np.asarray(deg_xyz, np.float64)
    R = np.broadcast_to(np.eye(3), deg_xyz.shape[:-1] + (3, 3))
    for ax in order.lower():
        R = _axis_rot(ax, deg_xyz[..., "xyz".index(ax)]) @ R
    return R


def cmu_forward_kinematics(asf: dict, frames: list[dict]):
    """World rotations and bone end positions (metres) of every ASF bone.

    Standard Acclaim FK: ``W_b = W_parent @ C_b @ M_b @ C_b^T`` with ``C_b``
    the bone's ``axis`` frame and ``M_b`` its AMC rotation, and the bone's end
    at its start plus ``length * W_b @ direction``.

    Returns ``(W, end)``: dicts of ``(T, 3, 3)`` and ``(T, 3)`` arrays.
    """
    T, scale = len(frames), asf["meters_per_unit"]
    order = [d.lower() for d in asf["root"].get("order", "TX TY TZ RX RY RZ".split())]
    root_vals = np.stack([f["root"] for f in frames])
    root_t = np.stack([root_vals[:, order.index(f"t{a}")] for a in "xyz"], -1)
    root_r = np.stack([root_vals[:, order.index(f"r{a}")] for a in "xyz"], -1)
    W = {"root": euler(root_r, asf["root"].get("axis", ["XYZ"])[0])}
    end = {"root": root_t * scale}
    queue = list(asf["hierarchy"].get("root", []))
    while queue:
        name = queue.pop(0)
        queue.extend(asf["hierarchy"].get(name, []))
        bone, parent = asf["bones"][name], asf["parent"][name]
        C = euler(bone["axis"], bone["axis_order"])
        angles = np.zeros((T, 3))
        for i, d in enumerate(bone["dof"]):
            angles[:, "xyz".index(d[1])] = [f[name][i] if name in f else 0.0 for f in frames]
        W[name] = W[parent] @ C @ euler(angles) @ C.T
        end[name] = end[parent] + bone["length"] * scale * (W[name] @ bone["direction"])
    return W, end


# --------------------------------------------------------------------------- retarget
def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-9:                       # opposite: half-turn about any normal
        n = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        n /= np.linalg.norm(n)
        return 2 * np.outer(n, n) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def retarget_to_soma(asf: dict, W: dict, end: dict, joint_names, parents, tpose_joints):
    """CMU world rotations -> SOMA T-pose-relative local rotations + hips path.

    Args:
        asf, W, end: from :func:`parse_asf` / :func:`cmu_forward_kinematics`.
        joint_names, parents: the SOMA rig's joints (parents before children).
        tpose_joints: (J, 3) SOMA joint positions in its T-pose (zero pose).

    Returns:
        ``(local (T, J, 3, 3), hips (T, 3) metres, stride_scale)``.
    """
    names = list(joint_names)
    index = {n: i for i, n in enumerate(names)}
    T, J = W["root"].shape[0], len(names)
    D = np.broadcast_to(np.eye(3), (T, J, 3, 3)).copy()      # world-aligned deltas
    for j, name in enumerate(names):
        p = int(parents[j])
        if name in SOMA_FROM_CMU:
            bone, child = SOMA_FROM_CMU[name]
            D[:, j] = W[bone]
            if name in REST_ALIGNED:
                D[:, j] = D[:, j] @ _rotation_between(
                    tpose_joints[index[child]] - tpose_joints[j], asf["bones"][bone]["direction"])
        elif p != j:
            D[:, j] = D[:, p]
    parent_idx = np.where(np.asarray(parents) == np.arange(J), 0, np.asarray(parents))
    local = np.swapaxes(D[:, parent_idx], -1, -2) @ D
    local[:, 0] = np.eye(3)                                    # virtual Root
    local[:, index["Hips"]] = D[:, index["Hips"]]

    soma_leg = sum(np.linalg.norm(tpose_joints[index[b]] - tpose_joints[index[a]])
                   for a, b in (("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot")))
    cmu_leg = (asf["bones"]["lfemur"]["length"] + asf["bones"]["ltibia"]["length"]) \
        * asf["meters_per_unit"]
    scale = soma_leg / cmu_leg
    return local, end["root"] * scale, scale


def yaw_matrix(angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about +Y (positive turns +Z towards +X)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def heading_yaw(path: np.ndarray) -> np.ndarray:
    """Rotation about +Y that turns the start-to-end direction of travel to +X."""
    d = path[-1] - path[0]
    return yaw_matrix(np.arctan2(d[2], d[0]))


def stage_yaw(path, root_rot, azimuth_deg: float = 15.0, travel: float = 1.0):
    """Rotation about +Y that stages a motion for a camera ``azimuth_deg`` round
    from +Z towards +X.

    A motion whose ``path`` (T, 3) ends more than ``travel`` metres from where it
    starts turns to travel along +X, or -X where +X would show the camera the
    performer's back; one in place turns to face 35 degrees past the camera.
    ``root_rot`` (T, 3, 3) are the pelvis's world rotations away from rest,
    which faces +Z. Returns ``(G, travels)``.
    """
    face = root_rot @ np.array([0.0, 0.0, 1.0])
    upright = (root_rot @ np.array([0.0, 1.0, 0.0]))[:, 1] > 0.5   # not mid-flip
    face = (face[upright] if upright.any() else face).mean(0) * [1.0, 0.0, 1.0]
    d = path[-1] - path[0]
    if np.hypot(d[0], d[2]) > travel:
        G = heading_yaw(path)
        if (G @ face)[2] < -0.3 * np.linalg.norm(face):
            G = yaw_matrix(np.pi) @ G
        return G, True
    return yaw_matrix(np.radians(azimuth_deg + 35.0) - np.arctan2(face[0], face[2])), False


def bone_direction_error(joints_soma, asf, W, G, joint_names, tpose_joints, frames=slice(None)):
    """Largest angle (degrees) between a posed SOMA bone and where the capture
    turns it: along the captured bone for the legs (:data:`REST_ALIGNED`), and
    elsewhere SOMA's T-pose bone turned as its CMU bone turned from rest."""
    index = {n: i for i, n in enumerate(joint_names)}
    worst = 0.0
    for name, (bone, child) in SOMA_FROM_CMU.items():
        if child is None:
            continue
        rest = (asf["bones"][bone]["direction"] if name in REST_ALIGNED
                else tpose_joints[index[child]] - tpose_joints[index[name]])
        a = joints_soma[:, index[child]] - joints_soma[:, index[name]]
        b = (W[bone][frames] @ rest) @ G.T
        cos = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        worst = max(worst, float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max()))
    return worst


# --------------------------------------------------------------------------- render
def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """(4, 4) camera pose at ``eye`` looking at ``target`` (OpenGL: looks along -Z)."""
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([r, np.cross(r, f), -f], axis=1)
    pose[:3, 3] = eye
    return pose


def fit_camera(points, width: int, height: int, azimuth_deg: float = 15.0,
               elevation_deg: float = 8.0, yfov: float = np.pi / 3.0,
               fill: float = 0.85) -> np.ndarray:
    """Camera on the +Z side of ``points`` (..., 3), turned ``azimuth_deg``
    towards +X and raised ``elevation_deg``, aimed at the middle of their
    bounding box and pulled back until every point projects inside ``fill`` of
    the frame (pass a subsample of the vertices: that frames the body itself,
    which a box round it would not)."""
    corners = np.asarray(points, np.float64).reshape(-1, 3)
    target = (corners.min(0) + corners.max(0)) / 2
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    view = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    tan_y = np.tan(yfov / 2)
    tan_x = tan_y * width / height

    def fits(dist):
        pose = look_at(target + dist * view, target)
        pc = (corners - pose[:3, 3]) @ pose[:3, :3]          # camera coordinates
        depth = -pc[:, 2]
        return bool((depth > 0.05).all() and (np.abs(pc[:, 0]) <= fill * tan_x * depth).all()
                    and (np.abs(pc[:, 1]) <= fill * tan_y * depth).all())

    lo, hi = 0.1, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return look_at(target + hi * view, target)


def checker_floor(x_range, z_range, tile: float = 0.5):
    """Two-tone tiled floor at y = 0 (``pyrender.Mesh`` list) to show the motion."""
    import pyrender
    import trimesh
    quads = {0: [], 1: []}
    xs = np.arange(np.floor(x_range[0] / tile), np.ceil(x_range[1] / tile)) * tile
    zs = np.arange(np.floor(z_range[0] / tile), np.ceil(z_range[1] / tile)) * tile
    for x in xs:
        for z in zs:
            quads[int(round(x / tile + z / tile)) % 2].append(
                [[x, 0, z], [x + tile, 0, z], [x + tile, 0, z + tile], [x, 0, z + tile]])
    meshes = []
    for shade, color in ((0, 0.86), (1, 0.74)):
        v = np.asarray(quads[shade], np.float32).reshape(-1, 3)
        base = np.arange(len(quads[shade]))[:, None] * 4
        f = np.concatenate([base + [0, 2, 1], base + [0, 3, 2]])   # +Y normals
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[color, color, color + 0.03, 1.0], metallicFactor=0.0,
            roughnessFactor=0.95)
        meshes.append(pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices=v, faces=f, process=False), material=mat, smooth=False))
    return meshes


BODY_COLOR = (0.55, 0.7, 0.9)


def save_gif(images, path, fps: float, colors: int | None = None):
    """Frames (arrays or PIL images) -> looping GIF that plays in real time:
    GIF delays are whole centiseconds, so 33.3 ms frames go out as 30/40/30 ms
    rather than all 30 ms, which would run 10% fast.

    ``colors``: map every frame onto one shared palette of that many colours,
    undithered, rather than a palette per frame: the file comes out about half
    the size and nothing flickers between frames.
    """
    from PIL import Image
    frames = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images]
    if colors:
        sample = [f.convert("RGB") for f in frames[::max(1, len(frames) // 16)]]
        montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for k, f in enumerate(sample):
            montage.paste(f, (0, k * f.height))
        palette = montage.quantize(colors, method=getattr(Image, "Quantize", Image).MEDIANCUT)
        frames = [f.convert("RGB").quantize(palette=palette,
                                            dither=getattr(Image, "Dither", Image).NONE)
                  for f in frames]
    ticks = np.round(np.arange(len(frames) + 1) * 100.0 / fps).astype(int)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[int(d) * 10 for d in np.diff(ticks)], loop=0, optimize=True)


def save_mp4(images, path, fps: float) -> bool:
    """Frames -> MP4 when imageio and imageio-ffmpeg are installed (else False)."""
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return False
    # imageio-ffmpeg runs os.setpgrp in the forked child before it starts ffmpeg
    # unless told not to; Python running between fork and exec can deadlock a
    # process that has JAX's threads (JAX warns as much), a plain fork-exec not.
    os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
    imageio.mimsave(path, [np.asarray(im) for im in images], fps=fps, quality=8,
                    macro_block_size=2)
    return True


def load_clip(asf_path, amc_path, source_fps: float, fps: float, start: float = 0.0,
              seconds: float | None = None):
    """Parse an ASF/AMC pair, keep ``seconds`` from ``start`` (the rest of the
    trial when None) and resample to about ``fps``.

    Returns ``(asf, W, end, fps)``: :func:`cmu_forward_kinematics` of the clip
    and the frame rate it ends up at.
    """
    asf = parse_asf(Path(asf_path).read_text())
    frames = parse_amc(Path(amc_path).read_text())
    first = int(round(start * source_fps))
    last = len(frames) if seconds is None else first + int(round(seconds * source_fps))
    clip = frames[first:last]
    if len(clip) < 2:
        sys.exit(f"{amc_path}: {len(frames) / source_fps:.1f} s long, nothing left "
                 f"from {start:g} s")
    stride = max(1, int(round(source_fps / fps)))
    W, end = cmu_forward_kinematics(asf, clip[::stride])
    return asf, W, end, source_fps / stride


def pose_clip(layer, coeffs, asf, W, end, tpose, azimuth_deg: float) -> dict:
    """Retarget a clip onto SOMA, stage it for the camera (:func:`stage_yaw`),
    pose the body in one jitted batch and stand it on y = 0.

    Returns a dict: ``local`` (T, J, 3, 3) pose input, ``hips`` (T, 3) path,
    posed ``verts`` and ``joints``, whether it ``travels``, the stride
    ``scale`` and ``err``, the largest :func:`bone_direction_error` (degrees).
    """
    import equinox as eqx
    import jax.numpy as jnp

    from soma_jax import SOMAParams
    names, parents = list(layer.public_joint_names), layer.public_joint_parent_ids
    local, hips, scale = retarget_to_soma(asf, W, end, names, parents, tpose)
    G, travels = stage_yaw(hips, W["root"], azimuth_deg)
    hip = names.index("Hips")
    local[:, hip] = G @ local[:, hip]
    hips = hips @ G.T
    hips[:, [0, 2]] -= hips[:, [0, 2]].mean(0)
    T = local.shape[0]
    out = eqx.filter_jit(layer)(SOMAParams(
        poses=jnp.asarray(local, jnp.float32), transl=jnp.asarray(hips, jnp.float32),
        identity_coeffs=jnp.broadcast_to(jnp.asarray(coeffs), (T, len(coeffs)))))
    verts, joints = np.array(out.vertices), np.array(out.joints)      # writable copies
    ground = float(np.percentile(verts[..., 1].min(axis=1), 5))      # feet on y = 0
    verts[..., 1] -= ground
    joints[..., 1] -= ground
    hips[:, 1] -= ground
    err = bone_direction_error(joints.astype(np.float64), asf, W, G, names, tpose)
    return {"local": local, "hips": hips, "verts": verts, "joints": joints,
            "travels": travels, "scale": scale, "err": err}


def _floor_for(lo_x: float, hi_x: float, cameras):
    """Tiled floor wide enough for the motion, ending behind the nearest camera."""
    near = max(float(c[2, 3]) for c in cameras) + 2.0
    return checker_floor((lo_x - 12.0, hi_x + 12.0), (-40.0, near))


def render_clip(verts, faces, hips, fps: float, camera: str, width: int, height: int,
                azimuth_deg: float, joints=None, parents=None):
    """Render every frame. ``camera``: "follow" frames the performer at one
    distance and slides along X with the hips' path, averaged over ~1 s (odd
    reflection at the ends keeps a steady motion centred there too), so the
    shot neither zooms nor jerks; "fixed" holds the whole motion."""
    from demo_soma_vis import _make_ground_and_shadow, render_mesh_png
    T = len(verts)
    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    every = max(1, T // 300)                  # every frame of a clip, every 20th vertex:
    sample = verts[::every, ::20].copy()      # enough to frame the body at its extremes
    cams = [fit_camera(sample, width, height, azimuth_deg)] * T
    if camera == "follow":
        k = max(1, min(int(round(fps / 2)), T - 1))
        padded = np.pad(hips[:, 0], k, mode="reflect", reflect_type="odd")
        track = np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        sample[..., 0] -= track[::every, None]              # the body relative to the track
        window = fit_camera(sample, width, height, azimuth_deg)
        cams = []
        for t in range(T):
            cams.append(window.copy())
            cams[-1][0, 3] += track[t]
    floor = _floor_for(lo[0], hi[0], cams)
    images = []
    for t in range(T):
        skel = ({} if joints is None else
                {"joints": joints[t], "parents": np.asarray(parents), "body_alpha": 0.45})
        shadow = _make_ground_and_shadow(verts[t], faces)[1]
        images.append(render_mesh_png(verts[t], faces, None, width, height, color=BODY_COLOR,
                                      camera_pose=cams[t], extra_meshes=floor + [shadow],
                                      **skel))
    return images


def strip_offsets(x, widths, factor: float = 0.8) -> np.ndarray:
    """Shifts along X that keep poses at path positions ``x`` in order, each at
    least ``factor`` times the median body width past the one before (so a
    motion in place lines up evenly). Returns one shift per pose."""
    x = np.asarray(x, np.float64)
    way = 1.0 if x[-1] >= x[0] else -1.0
    step = factor * float(np.median(widths))
    shift = np.zeros(len(x))
    for k in range(1, len(x)):
        shift[k] = shift[k - 1] + max(0.0, step - way * (x[k] - x[k - 1]))
    return way * shift


def render_strip(verts, faces, hips, picks, width: int, height: int, azimuth_deg: float,
                 path):
    """The poses ``picks`` in one picture, spaced by :func:`strip_offsets`."""
    from demo_soma_vis import _make_ground_and_shadow, render_mesh_png
    poses = [verts[t].copy() for t in picks]
    for v, dx in zip(poses, strip_offsets(hips[picks, 0], [np.ptp(v[:, 0]) for v in poses]),
                     strict=True):
        v[:, 0] += dx
    strip_v = np.concatenate(poses)
    strip_f = np.concatenate([faces + k * verts.shape[1] for k in range(len(poses))])
    cam = fit_camera(strip_v[::10], width, height, azimuth_deg)
    floor = _floor_for(strip_v[:, 0].min(), strip_v[:, 0].max(), [cam])
    return render_mesh_png(strip_v, strip_f, str(path), width, height, color=BODY_COLOR,
                           camera_pose=cam,
                           extra_meshes=floor + [_make_ground_and_shadow(strip_v, strip_f)[1]])


def run_motion(layer, coeffs, tpose, name: str, caption: str, asf_path, amc_path,
               source_fps: float, args, start: float = 0.0, seconds: float | None = None,
               out_dir=None):
    """Load, retarget, pose and render one clip into ``out_dir`` (default
    ``demo_renders/<name>``). Returns ``(frames, fps, strip)``."""
    import jax.numpy as jnp
    from _common import ensure_headless_gl

    from soma_jax import save_soma_npz
    from soma_jax.geometry.transforms import rotmat_to_axis_angle
    out_dir = Path(out_dir or REPO / "demo_renders" / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    asf, W, end, fps = load_clip(asf_path, amc_path, source_fps, args.fps, start, seconds)
    T, source = W["root"].shape[0], f"CMU {Path(amc_path).stem}"
    print(f"{caption} ({source}, {start:g} s in): {T} frames at {fps:g} fps ({T / fps:.1f} s)")
    m = pose_clip(layer, coeffs, asf, W, end, tpose, args.azimuth)
    hips = m["hips"]
    moved = float(np.linalg.norm(hips[-1, [0, 2]] - hips[0, [0, 2]]))
    print(f"  retargeted: stride scale {m['scale']:.3f}, " + (
        f"travels {moved:.2f} m ({moved / (T / fps):.2f} m/s)" if m["travels"] else "in place"))
    print(f"  posed {m['verts'].shape[1]} vertices x {T} frames; posed bones turn with the "
          f"captured ones to {m['err']:.2g} deg")
    save_soma_npz(str(out_dir / f"{name}.soma.npz"), np.asarray(rotmat_to_axis_angle(
        jnp.asarray(m["local"], jnp.float32))), hips, list(layer.public_joint_names),
        args.identity_model, np.asarray(coeffs)[None], absolute_pose=False,
        extra_arrays={"fps": np.asarray(fps), "source": np.asarray(source)})

    ensure_headless_gl()                 # before the first pyrender import
    camera = args.camera if args.camera != "auto" else ("follow" if m["travels"] else "fixed")
    faces = np.asarray(layer.faces)
    t0 = time.time()
    images = render_clip(m["verts"], faces, hips, fps, camera, args.width, args.height,
                         args.azimuth, *((m["joints"], layer.public_joint_parent_ids) if args.skeleton else ()))
    print(f"  rendered {T} frames ({camera} camera) in {time.time() - t0:.1f} s")
    save_gif(images, out_dir / f"{name}.gif", fps)
    print(f"  wrote {out_dir / f'{name}.gif'}")
    if save_mp4(images, out_dir / f"{name}.mp4", fps):
        print(f"  wrote {out_dir / f'{name}.mp4'}")
    else:
        print(f"  ({name}.mp4 needs imageio and imageio-ffmpeg: "
              "pip install imageio imageio-ffmpeg)")

    # One pose every --strip-every seconds; at most 10 along a path, 8 in place.
    every = max(1, int(round(args.strip_every * fps)), -(-T // (10 if m["travels"] else 8)))
    picks = list(range(0, T, every))
    strip = render_strip(m["verts"], faces, hips, picks, args.width, args.height,
                         args.azimuth, out_dir / f"{name}_strip.png")
    print(f"  wrote {out_dir / f'{name}_strip.png'} ({len(picks)} poses)")
    return images, fps, strip


def compose_gallery(clips, out_dir, cols: int = 4, tile=(384, 216), fps: float = 15.0,
                    colors: int | None = 128):
    """Lay out ``clips`` [(caption, frames, fps, strip), ...]: ``gallery.gif``
    (and ``.mp4``) plays them side by side, each looping, at ``fps``, with
    ``colors`` shared colours (see :func:`save_gif`), and ``gallery.png`` holds
    their strips. Returns the paths written."""
    from PIL import Image, ImageDraw, ImageFont
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=max(10, tile[1] // 14))
    except TypeError:                                # Pillow < 10.1: fixed size
        font = ImageFont.load_default()
    rows = -(-len(clips) // cols)
    tw, th = tile

    def sheet(images, captions, size):
        out = Image.new("RGB", (cols * size[0], rows * size[1]), "white")
        draw = ImageDraw.Draw(out)
        for i, (im, text) in enumerate(zip(images, captions, strict=True)):
            x, y = (i % cols) * size[0], (i // cols) * size[1]
            if not isinstance(im, Image.Image):
                im = Image.fromarray(np.asarray(im))
            out.paste(im if im.size == size else im.resize(size, Image.LANCZOS), (x, y))
            draw.text((x + 8, y + 6), text, fill=(25, 25, 25), font=font)
        return out

    captions = [c[0] for c in clips]
    shrunk = [[f if isinstance(f, Image.Image) and f.size == tile
               else Image.fromarray(np.asarray(f)).resize(tile, Image.LANCZOS) for f in c[1]]
              for c in clips]
    seconds = max(len(c[1]) / c[2] for c in clips)
    frames = [sheet([s[int(k * c[2] / fps) % len(s)] for s, c in zip(shrunk, clips, strict=True)],
                    captions, tile) for k in range(max(1, int(round(seconds * fps))))]
    written = [out_dir / "gallery.gif"]
    save_gif(frames, written[0], fps, colors)
    if save_mp4(frames, out_dir / "gallery.mp4", fps):
        written.append(out_dir / "gallery.mp4")
    sheet([c[3] for c in clips], captions, (2 * tw, 2 * th)).save(out_dir / "gallery.png")
    return written + [out_dir / "gallery.png"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", default="walk", choices=list(EXAMPLES),
                   help="an example motion (default walk; --list describes them)")
    p.add_argument("--list", action="store_true", help="list the example motions and exit")
    p.add_argument("--gallery", action="store_true",
                   help="render every example (or --motions) and a gallery of them")
    p.add_argument("--motions", nargs="+", choices=list(EXAMPLES), default=None,
                   help="the examples --gallery renders (default: all)")
    p.add_argument("--gallery-width", type=int, default=1536,
                   help="width of gallery.gif in pixels, four clips across (default 1536; "
                        "the README's assets/media/mocap_gallery.gif is 1024)")
    p.add_argument("--trial", default=None,
                   help="any CMU trial instead of an example: subject_trial, e.g. 49_06")
    p.add_argument("--asf", default=None, help="local ASF file instead of downloading")
    p.add_argument("--amc", default=None, help="local AMC file instead of downloading")
    p.add_argument("--start", type=float, default=None,
                   help="seconds into the trial to start (default: the example's, else 0)")
    p.add_argument("--seconds", type=float, default=None,
                   help="length of the clip (default: the example's, else the whole trial)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="capture rate of the AMC (default: CMU's for the trial, 120 or 60)")
    p.add_argument("--soma-model", default=None,
                   help="SOMA asset: NVIDIA's SOMA_neutral.npz or a SOMA-JAX archive "
                        "(default: build from the resolved upstream assets)")
    p.add_argument("--identity-model", default="soma", help="identity backend")
    p.add_argument("--identity-path", default=None, help="identity pack for that backend")
    p.add_argument("--identity-coeffs", type=float, nargs="*", default=[],
                   help="leading identity coefficients (default: the mean body)")
    p.add_argument("--fps", type=float, default=30.0, help="output frame rate")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--camera", choices=["auto", "follow", "fixed"], default="auto",
                   help="follow the performer, or fixed: one shot holding the whole motion "
                        "(default auto: follow a motion that travels)")
    p.add_argument("--azimuth", type=float, default=15.0,
                   help="degrees the camera is turned from a pure side view")
    p.add_argument("--skeleton", action="store_true",
                   help="draw the SOMA skeleton inside a see-through body")
    p.add_argument("--strip-every", type=float, default=0.4,
                   help="seconds between the poses of <name>_strip.png")
    p.add_argument("--output-dir", default=None,
                   help="output folder (default demo_renders/<name>; for --gallery the "
                        "gallery's, default demo_renders/gallery)")
    args = p.parse_args()

    if args.list:
        for name, (trial, start, seconds, caption) in EXAMPLES.items():
            span = f"{start:g}-{start + seconds:g} s" if seconds else f"{start:g} s-end"
            print(f"{name:14s} CMU {trial:7s} {span:11s} {caption}")
        return

    import jax.numpy as jnp
    from _common import load_layer

    from soma_jax import SOMAParams
    # The SOMA body, and its joints in the T-pose its pose input is relative to.
    layer = load_layer(args.soma_model, identity_model_type=args.identity_model,
                       identity_model_path=args.identity_path)
    n_coeffs = int(getattr(layer.identity_model, "n_betas", 128))
    coeffs = np.zeros(n_coeffs, np.float32)
    coeffs[:len(args.identity_coeffs)] = args.identity_coeffs[:n_coeffs]
    J = len(layer.public_joint_names)
    zero = layer(SOMAParams(poses=jnp.zeros((1, J, 3)), transl=jnp.zeros((1, 3)),
                            identity_coeffs=jnp.asarray(coeffs)[None]), fk_only=True)
    tpose = np.asarray(zero.joints[0], np.float64)
    cache = REPO / "assets" / "third_party" / "cmu_mocap"

    if args.gallery:
        from PIL import Image
        tw = args.gallery_width // 4
        tile = (tw, int(round(tw * args.height / args.width)))
        clips = []
        for name in args.motions or EXAMPLES:
            trial, start, seconds, caption = EXAMPLES[name]
            images, fps, strip = run_motion(layer, coeffs, tpose, name, caption,
                                            *fetch_cmu_trial(trial, cache), cmu_fps(trial),
                                            args, start, seconds)
            clips.append((caption, [Image.fromarray(im).resize(tile, Image.LANCZOS)
                                    for im in images], fps, strip))
        for path in compose_gallery(clips, args.output_dir or REPO / "demo_renders" / "gallery",
                                    tile=tile):
            print(f"wrote {path}")
        return

    if args.asf and args.amc:
        asf_path, amc_path = Path(args.asf), Path(args.amc)
        name = caption = trial = amc_path.stem
        start, seconds = args.start or 0.0, args.seconds
    elif args.trial:
        name = caption = trial = args.trial
        asf_path, amc_path = fetch_cmu_trial(trial, cache)
        start, seconds = args.start or 0.0, args.seconds
    else:
        name = args.motion
        trial, start, seconds, caption = EXAMPLES[name]
        asf_path, amc_path = fetch_cmu_trial(trial, cache)
        start = start if args.start is None else args.start
        seconds = seconds if args.seconds is None else args.seconds
    run_motion(layer, coeffs, tpose, name, caption, asf_path, amc_path,
               args.source_fps or cmu_fps(trial), args, start, seconds, args.output_dir)


if __name__ == "__main__":
    main()
