"""A real walk on the SOMA rig: CMU motion capture -> SOMA-JAX -> render.

Fetches a walking trial from the CMU Graphics Lab Motion Capture Database
(mocap.cs.cmu.edu: an ASF skeleton and an AMC motion, captured at 120 fps),
retargets it onto the SOMA skeleton, poses the SOMA body with SOMA-JAX and
renders it headless::

    python tools/pipeline/demo_walk.py                    # trial 35_08, "walk"
    python tools/pipeline/demo_walk.py --trial 07_01 --width 1280 --height 720
    python tools/pipeline/demo_walk.py --skeleton         # SOMA skeleton in a see-through body
    python tools/pipeline/demo_walk.py --identity-model garment_measurement \\
        --identity-path assets/identity/identity_garment.npz --identity-coeffs 1.5 -1 0.5

Writes to ``--output-dir`` (default ``demo_renders/walk``): ``walk.gif``,
``walk.mp4`` (when imageio-ffmpeg is installed), ``walk_strip.png`` (a pose
every ``--strip-every`` seconds in one picture) and ``walk.soma.npz``, the
retargeted motion in SOMA's NPZ format, which the other tools replay
(``tools/vis/vis_pyrender.py --animation demo_renders/walk/walk.soma.npz``).

Retargeting. The ASF skeleton rests in a T-pose facing +Z with its left side
on +X, as SOMA's does, and every ASF rest frame is world-aligned, so a bone's
world rotation is its rotation away from rest. SOMA's pose input is relative
to its own T-pose too: for world-aligned deltas ``D`` (a joint's world rotation
times its T-pose orientation transposed), the local input is
``R_j = D_parent^T @ D_j``. Each SOMA joint therefore takes the world rotation
of the matching CMU bone, preceded by the fixed rotation that turns SOMA's rest
bone direction into CMU's (CMU's thighs, for one, rest 20 degrees apart), so the
posed bone directions follow the capture. Joints CMU does not track (fingers,
eyes, jaw) follow their parent, and the root path is scaled by the ratio of leg
lengths so the stride fits SOMA's legs.

Data: "The data used in this project was obtained from mocap.cs.cmu.edu. The
database was created with funding from NSF EIA-0196217." The clips may be used
freely, in commercial products too, but not resold; they are downloaded on
first use into ``assets/third_party/cmu_mocap`` and are not part of this
repository.
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
DEFAULT_TRIAL = "35_08"            # subject 35, trial 8: "walk"

#: SOMA joint -> (CMU bone whose world rotation drives it, SOMA child joint that
#: gives the joint's bone direction). ``None`` takes the full orientation (the
#: pelvis); unlisted joints follow their parent.
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
            if child is None:
                D[:, j] = W[bone]
            else:
                rest = _rotation_between(tpose_joints[index[child]] - tpose_joints[j],
                                         asf["bones"][bone]["direction"])
                D[:, j] = W[bone] @ rest
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


def heading_yaw(path: np.ndarray) -> np.ndarray:
    """Rotation about +Y that turns the start-to-end walking direction to +X."""
    d = path[-1] - path[0]
    a = np.arctan2(d[2], d[0])
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def bone_direction_error(joints_soma, asf, end, G, joint_names, frames=slice(None)):
    """Largest angle (degrees) between posed SOMA bones and the captured bones."""
    index = {n: i for i, n in enumerate(joint_names)}
    worst = 0.0
    for name, (bone, child) in SOMA_FROM_CMU.items():
        if child is None:
            continue
        a = joints_soma[:, index[child]] - joints_soma[:, index[name]]
        b = (end[bone] - end[asf["parent"][bone]])[frames] @ G.T
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


def fit_camera(box_min, box_max, width: int, height: int, azimuth_deg: float = 15.0,
               elevation_deg: float = 8.0, yfov: float = np.pi / 3.0,
               fill: float = 0.9) -> np.ndarray:
    """Camera in front of a box (a walk along X), turned ``azimuth_deg`` towards
    +X and raised ``elevation_deg``, pulled back until every corner of the box
    projects inside ``fill`` of the frame."""
    box_min, box_max = np.asarray(box_min, np.float64), np.asarray(box_max, np.float64)
    corners = np.stack(np.meshgrid(*zip(box_min, box_max, strict=True), indexing="ij"),
                        -1).reshape(-1, 3)
    target = (box_min + box_max) / 2
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial", default=DEFAULT_TRIAL,
                   help="CMU trial id, subject_trial (default 35_08; 07_01-07_12, 08_01-08_10 "
                        "and 35_01-35_16 are walks)")
    p.add_argument("--asf", default=None, help="local ASF file instead of downloading")
    p.add_argument("--amc", default=None, help="local AMC file instead of downloading")
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
    p.add_argument("--camera", choices=["follow", "fixed"], default="follow",
                   help="follow: track the walker (default); fixed: one shot holding the "
                        "whole walk, as walk_strip.png uses")
    p.add_argument("--azimuth", type=float, default=15.0,
                   help="degrees the camera is turned from a pure side view")
    p.add_argument("--skeleton", action="store_true",
                   help="draw the SOMA skeleton inside a see-through body")
    p.add_argument("--strip-every", type=float, default=0.4,
                   help="seconds between the poses of walk_strip.png")
    p.add_argument("--output-dir", default=str(REPO / "demo_renders" / "walk"))
    args = p.parse_args()

    import jax.numpy as jnp
    import equinox as eqx
    from _common import ensure_headless_gl, load_layer
    from soma_jax import SOMAParams, save_soma_npz
    from soma_jax.geometry.transforms import rotmat_to_axis_angle

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. The capture.
    if args.asf and args.amc:
        asf_path, amc_path = Path(args.asf), Path(args.amc)
    else:
        asf_path, amc_path = fetch_cmu_trial(args.trial, REPO / "assets" / "third_party"
                                             / "cmu_mocap")
    asf = parse_asf(asf_path.read_text())
    frames = parse_amc(amc_path.read_text())
    stride = max(1, int(round(CMU_FPS / args.fps)))
    frames = frames[::stride]
    fps = CMU_FPS / stride
    W, end = cmu_forward_kinematics(asf, frames)
    print(f"CMU {amc_path.stem}: {len(frames)} frames at {fps:g} fps "
          f"({len(frames) / fps:.1f} s)")

    # 2. The SOMA body, and its joints in the T-pose its pose input is relative to.
    layer = load_layer(args.soma_model, identity_model_type=args.identity_model,
                       identity_model_path=args.identity_path)
    n_coeffs = int(getattr(layer.identity_model, "n_betas", 128))
    coeffs = np.zeros(n_coeffs, np.float32)
    coeffs[:len(args.identity_coeffs)] = args.identity_coeffs[:n_coeffs]
    names, parents = list(layer.public_joint_names), layer.parents
    J = len(names)
    zero = layer(SOMAParams(poses=jnp.zeros((1, J, 3)), transl=jnp.zeros((1, 3)),
                            identity_coeffs=jnp.asarray(coeffs)[None]), fk_only=True)
    tpose = np.asarray(zero.joints[0], np.float64)

    # 3. Retarget, walk along +X, centred on the origin.
    local, hips, scale = retarget_to_soma(asf, W, end, names, parents, tpose)
    G = heading_yaw(hips)
    local[:, names.index("Hips")] = G @ local[:, names.index("Hips")]
    hips = hips @ G.T
    hips[:, [0, 2]] -= hips[:, [0, 2]].mean(0)
    walked = float(np.linalg.norm(hips[-1, [0, 2]] - hips[0, [0, 2]]))
    print(f"  retargeted: stride scale {scale:.3f}, {walked:.2f} m walked "
          f"({walked / (len(frames) / fps):.2f} m/s)")

    # 4. Pose the body with SOMA-JAX (one batched, jitted forward).
    T = local.shape[0]
    out = eqx.filter_jit(layer)(SOMAParams(
        poses=jnp.asarray(local, jnp.float32), transl=jnp.asarray(hips, jnp.float32),
        identity_coeffs=jnp.broadcast_to(jnp.asarray(coeffs), (T, n_coeffs))))
    verts, joints = np.array(out.vertices), np.array(out.joints)      # writable copies
    ground = float(np.percentile(verts[..., 1].min(axis=1), 5))   # feet on y = 0
    verts[..., 1] -= ground
    joints[..., 1] -= ground
    hips[:, 1] -= ground
    err = bone_direction_error(joints.astype(np.float64), asf, end, G, names)
    print(f"  posed {verts.shape[1]} vertices x {T} frames; posed bones follow the "
          f"capture to {err:.2g} deg")

    save_soma_npz(str(out_dir / "walk.soma.npz"), np.asarray(rotmat_to_axis_angle(
        jnp.asarray(local, jnp.float32))), hips, names, args.identity_model,
        coeffs[None], absolute_pose=False,
        extra_arrays={"fps": np.asarray(fps), "source": np.asarray(f"CMU {amc_path.stem}")})

    # 5. Render.
    ensure_headless_gl()
    from demo_soma_vis import _make_ground_and_shadow, render_mesh_png
    faces = np.asarray(layer.faces)
    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    floor = checker_floor((lo[0] - 12.0, hi[0] + 12.0), (-40.0, 6.0))
    fixed = fit_camera(lo, hi, args.width, args.height, args.azimuth)
    # The follow shot frames a 2 m window round the walker at one distance and
    # slides it along the hips' path, averaged over ~1 s (odd reflection at the
    # ends keeps a steady walk centred there too), so it neither zooms nor jerks.
    window = fit_camera([-1.0, 0.0, lo[2]], [1.0, hi[1], hi[2]], args.width, args.height,
                        args.azimuth)
    k = max(1, min(int(round(fps / 2)), T - 1))
    padded = np.pad(hips[:, 0], k, mode="reflect", reflect_type="odd")
    track = np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    t0 = time.time()
    images = []
    for t in range(T):
        cam = fixed
        if args.camera == "follow":
            cam = window.copy()
            cam[0, 3] += track[t]
        skel = (dict(joints=joints[t], parents=np.asarray(parents), body_alpha=0.45)
                if args.skeleton else {})
        shadow = _make_ground_and_shadow(verts[t], faces)[1]
        images.append(render_mesh_png(verts[t], faces, None, args.width, args.height,
                                      color=(0.55, 0.7, 0.9), camera_pose=cam,
                                      extra_meshes=floor + [shadow], **skel))
    print(f"  rendered {T} frames ({args.camera} camera) in {time.time() - t0:.1f} s")

    from PIL import Image
    # GIF delays are whole centiseconds: spread them so the clip plays in real
    # time (33.3 ms frames as 30/40/30 ms, not all 30 ms, which runs 10% fast).
    ticks = np.round(np.arange(T + 1) * 100.0 / fps).astype(int)
    gif = out_dir / "walk.gif"
    Image.fromarray(images[0]).save(
        gif, save_all=True, append_images=[Image.fromarray(im) for im in images[1:]],
        duration=[int(d) * 10 for d in np.diff(ticks)], loop=0, optimize=True)
    print(f"  wrote {gif}")
    try:                                         # optional: imageio + imageio-ffmpeg
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        print("  (walk.mp4 needs imageio and imageio-ffmpeg: pip install imageio imageio-ffmpeg)")
    else:
        imageio.mimsave(out_dir / "walk.mp4", images, fps=fps, quality=8)
        print(f"  wrote {out_dir / 'walk.mp4'}")

    every = max(1, int(round(args.strip_every * fps)))
    picks = list(range(0, T, every))
    strip_v = np.concatenate([verts[t] for t in picks])
    strip_f = np.concatenate([faces + k * verts.shape[1] for k in range(len(picks))])
    strip = render_mesh_png(strip_v, strip_f, str(out_dir / "walk_strip.png"),
                            args.width, args.height, color=(0.55, 0.7, 0.9),
                            camera_pose=fixed, extra_meshes=floor + [
                                _make_ground_and_shadow(strip_v, strip_f)[1]])
    print(f"  wrote {out_dir / 'walk_strip.png'} ({len(picks)} poses, "
          f"{strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
