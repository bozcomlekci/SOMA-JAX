"""Build the shared pose sequence for the SOMA-X vs SOMA-JAX comparison.

Two sources:

* ``--bvh PATH``  — a SOMA-skeleton BVH motion clip (preferred).
  Parsed via ``tools/pipeline/bvh_parser.load_soma_bvh``. These clips store rotations as
  ABSOLUTE skinning-frame rotmats (bind orientation baked in), NOT
  T-pose-relative — so we save the exact rotmats and mark ``absolute=True`` so
  the posers skip the T-pose joint-orient step (mirrors demo_soma_vis.py, which
  calls ``sk.pose(R, absolute_pose=True)``). Per-frame root translation drives
  the walk; the clip is evenly subsampled to ``--frames`` spanning the whole
  motion. The SOMA rig is already Y-up, so no view rotation is applied.

* ``--usd PATH`` — a USD SkelAnimation of the SOMA joints, e.g. upstream's
  ``assets/example_animation.usd`` (since v0.3.4), the motion SOMA-X's own
  ``tools/demo_soma_vis.py`` plays, read with :func:`soma_jax.load_soma_animation`
  at the file's own frame rate. Its rotations are absolute local rotations like
  the BVH clips', and the Hips translation is the root translation.
* ``--npy PATH`` — a clip of SOMA local transforms ``(T, 78|94, 4, 4)``, e.g.
  the ``example_animation.npy`` upstream shipped before v0.3.4 (played at
  30 fps; 94-joint clips are reduced to the 78 SOMA joints with the table that
  release's demo used). Its rotations are absolute local rotations, and joint
  1's translation is the root translation in metres.
* neither — a synthetic in-place march + arm swing (T-pose-relative,
  ``absolute=False``).

Both pipelines consume the identical arrays: SOMA-JAX uses all 78 joints,
SOMA-X uses ``rotmats[:, 1:]`` (its 77 public joints; Root is identity and
padded internally).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _synthetic(names, T):
    idx = {n: i for i, n in enumerate(names)}
    poses = np.zeros((T, len(names), 3), dtype=np.float32)
    ph = np.linspace(0, 2 * np.pi, T, endpoint=False)

    def set_aa(joint, axis, ang):
        if joint not in idx:
            return
        poses[:, idx[joint], {"x": 0, "y": 1, "z": 2}[axis]] = ang

    set_aa("LeftUpLeg", "x", 0.6 * np.sin(ph))
    set_aa("RightUpLeg", "x", 0.6 * np.sin(ph + np.pi))
    set_aa("LeftLeg", "x", 0.5 * (1 - np.cos(ph)) * 0.5)
    set_aa("RightLeg", "x", 0.5 * (1 - np.cos(ph + np.pi)) * 0.5)
    set_aa("LeftArm", "x", 0.7 * np.sin(ph + np.pi))
    set_aa("RightArm", "x", 0.7 * np.sin(ph))
    set_aa("LeftArm", "z", 0.25)
    set_aa("RightArm", "z", -0.25)
    set_aa("LeftForeArm", "z", 0.5 + 0.3 * np.sin(ph + np.pi))
    set_aa("RightForeArm", "z", -0.5 - 0.3 * np.sin(ph))
    set_aa("Spine1", "y", 0.12 * np.sin(ph))
    set_aa("Spine2", "y", 0.10 * np.sin(ph))
    set_aa("Hips", "y", 0.10 * np.sin(ph))
    set_aa("Neck1", "y", -0.10 * np.sin(ph))
    from scipy.spatial.transform import Rotation
    rotmats = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    rotmats = rotmats.reshape(T, len(names), 3, 3).astype(np.float32)
    trans = np.zeros((T, 3), dtype=np.float32)
    return rotmats, trans, 30.0


#: SOMA-X's pre-v0.3.4 demo clip (Root + its 93-joint NV skeleton): the
#: indices of the 77 SOMA joints in it, from that release's tools/demo_soma_vis.py.
_NVSKEL93 = (
    "Hips Spine1 Spine2 Chest Neck1 Neck2 Head HeadEnd Jaw LeftEye RightEye LeftShoulder "
    "LeftArm LeftForeArm LeftHand LeftHandThumb1 LeftHandThumb2 LeftHandThumb3 "
    "LeftHandThumbEnd LeftHandIndex1 LeftHandIndex2 LeftHandIndex3 LeftHandIndex4 "
    "LeftHandIndexEnd LeftHandMiddle1 LeftHandMiddle2 LeftHandMiddle3 LeftHandMiddle4 "
    "LeftHandMiddleEnd LeftHandRing1 LeftHandRing2 LeftHandRing3 LeftHandRing4 "
    "LeftHandRingEnd LeftHandPinky1 LeftHandPinky2 LeftHandPinky3 LeftHandPinky4 "
    "LeftHandPinkyEnd LeftForeArmTwist1 LeftForeArmTwist2 LeftArmTwist1 LeftArmTwist2 "
    "RightShoulder RightArm RightForeArm RightHand RightHandThumb1 RightHandThumb2 "
    "RightHandThumb3 RightHandThumbEnd RightHandIndex1 RightHandIndex2 RightHandIndex3 "
    "RightHandIndex4 RightHandIndexEnd RightHandMiddle1 RightHandMiddle2 RightHandMiddle3 "
    "RightHandMiddle4 RightHandMiddleEnd RightHandRing1 RightHandRing2 RightHandRing3 "
    "RightHandRing4 RightHandRingEnd RightHandPinky1 RightHandPinky2 RightHandPinky3 "
    "RightHandPinky4 RightHandPinkyEnd RightForeArmTwist1 RightForeArmTwist2 "
    "RightArmTwist1 RightArmTwist2 LeftLeg LeftShin LeftFoot LeftToeBase LeftToeEnd "
    "LeftShinTwist1 LeftShinTwist2 LeftLegTwist1 LeftLegTwist2 RightLeg RightShin "
    "RightFoot RightToeBase RightToeEnd RightShinTwist1 RightShinTwist2 RightLegTwist1 "
    "RightLegTwist2").split()
_NVSKEL77 = [n for n in _NVSKEL93 if "Twist" not in n]
_NVSKEL93_TO_77 = [_NVSKEL93.index(n) for n in _NVSKEL77]


def _public_joint_names(soma_npz=None):
    """The 78 public SOMA joint names in rig order (Root first)."""
    if soma_npz is not None:
        with np.load(soma_npz, allow_pickle=True) as d:
            if "joint_names" in d.files:
                return [str(n) for n in d["joint_names"]]
    sys.path.insert(0, str(REPO))
    from soma_jax.rig_build import load_public_rig
    return [str(n) for n in load_public_rig(soma_npz)["joint_names"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soma-npz", default=None,
                   help="archive to read joint names from for the synthetic motion "
                        "(default: SOMA_neutral_fixed.npz if built, else the downloaded "
                        "SOMA_neutral.npz — both carry joint_names)")
    p.add_argument("--bvh", default=None, help="SOMA-skeleton BVH motion clip")
    p.add_argument("--usd", default=None,
                   help="USD SkelAnimation of the SOMA joints, e.g. upstream's "
                        "assets/example_animation.usd")
    p.add_argument("--npy", default=None,
                   help="SOMA local-transform clip (T, 78|94, 4, 4), e.g. upstream's "
                        "assets/example_animation.npy")
    p.add_argument("--npy-fps", type=float, default=30.0,
                   help="source frame rate of --npy (SOMA-X's demo plays its clip at 30)")
    p.add_argument("--seconds", type=float, default=6.0,
                   help="length of the clip window to extract (realtime playback duration)")
    p.add_argument("--start-frac", type=float, default=0.15,
                   help="where to start the window in the clip (skip the calibration T-pose intro)")
    p.add_argument("--play-fps", type=float, default=30.0,
                   help="realtime playback fps -> T = seconds*play_fps frames span the window")
    p.add_argument("--keep-translation", action="store_true",
                   help="keep full root translation (default: freeze horizontal drift so the "
                        "body stays centered/constant-size — an in-place 'treadmill' — while "
                        "keeping vertical bob; better for a side-by-side speed comparison)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    play_fps = float(args.play_fps)
    if args.bvh:
        sys.path.insert(0, str(REPO / "tools" / "pipeline"))
        from bvh_parser import load_soma_bvh
        bvh = load_soma_bvh(args.bvh)
        names = [str(n) for n in bvh["joint_names"]]
        N = bvh["n_frames"]; src_fps = float(bvh["source_fps"])
        # Realtime window: extract `seconds` of motion, sampled at `play_fps` so
        # GIF playback at play_fps runs at true wall-clock speed (item: realtime).
        win = min(args.seconds, bvh["source_duration_s"])
        T = max(2, int(round(win * play_fps)))
        f0 = int(args.start_frac * N)
        f1 = min(N - 1, f0 + int(round(win * src_fps)))
        idx = np.linspace(f0, f1, T).astype(int)
        rotmats = bvh["rotmats"][idx].astype(np.float32)          # (T,78,3,3) exact, ABSOLUTE
        trans = bvh["root_translation"][idx].astype(np.float32)   # (T,3) meters (Y-up world)
        if not args.keep_translation:
            # Freeze horizontal (X,Z ground-plane) drift to frame 0, keep vertical
            # (Y) bob → body walks in place at a constant, well-framed size.
            trans[:, 0] = trans[0, 0]; trans[:, 2] = trans[0, 2]
        absolute = True
        duration_s = win
        print(f"loaded BVH {Path(args.bvh).name}: {N}f @ {src_fps:.0f} FPS "
              f"({bvh['source_duration_s']:.1f}s) -> {win:.1f}s window @ {play_fps:.0f} FPS = {T} frames")
    elif args.usd or args.npy:
        if args.usd:
            sys.path.insert(0, str(REPO))
            from soma_jax import load_soma_animation
            names = _public_joint_names(args.soma_npz)
            clip, src_fps = load_soma_animation(args.usd, names)        # (T, 78, 4, 4), metres
            src_fps = float(src_fps)
        else:
            clip = np.load(args.npy).astype(np.float32)
            if clip.shape[1] == 94:
                # SOMA-X's pre-v0.3.4 demo layout: Root + the 93-joint NV skeleton.
                clip = clip[:, [0] + [i + 1 for i in _NVSKEL93_TO_77]]
            names = _public_joint_names(args.soma_npz)
            src_fps = float(args.npy_fps)
        assert clip.shape[1] == len(names), (clip.shape, len(names))
        N = clip.shape[0]
        win = min(args.seconds, N / src_fps)
        T = max(2, int(round(win * play_fps)))
        f0 = int(args.start_frac * N)
        f1 = min(N - 1, f0 + int(round(win * src_fps)))
        idx = np.linspace(f0, f1, T).astype(int)
        rotmats = clip[idx, :, :3, :3]                            # (T,78,3,3) ABSOLUTE
        trans = clip[idx, 1, :3, 3].copy()                        # (T,3) metres, Y-up
        if not args.keep_translation:
            trans[:, 0] = trans[0, 0]; trans[:, 2] = trans[0, 2]
        absolute = True
        duration_s = win
        print(f"loaded {Path(args.usd or args.npy).name}: {N}f @ {src_fps:.0f} FPS "
              f"({N / src_fps:.1f}s) -> {win:.1f}s window @ {play_fps:.0f} FPS = {T} frames")
    else:
        names = _public_joint_names(args.soma_npz)
        T = max(2, int(round(args.seconds * play_fps)))
        rotmats, trans, _ = _synthetic(names, T)
        absolute = False
        duration_s = args.seconds

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, rotmats=rotmats, trans=trans, absolute=np.bool_(absolute),
             joint_names=np.asarray(names), duration_s=np.float32(duration_s),
             play_fps=np.float32(play_fps))
    print(f"wrote {args.out}: rotmats {rotmats.shape}  trans {trans.shape}  "
          f"absolute={absolute}  duration={duration_s:.1f}s @ {play_fps:.0f}fps  (T={T})")


if __name__ == "__main__":
    main()
