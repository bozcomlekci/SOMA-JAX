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

* no ``--bvh``   — a synthetic in-place march + arm swing (T-pose-relative,
  ``absolute=False``); fallback for when the dataset isn't mounted.

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soma-npz", default=str(REPO / "assets" / "SOMA_neutral_fixed.npz"))
    p.add_argument("--bvh", default=None, help="SOMA-skeleton BVH motion clip")
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
    else:
        d = dict(np.load(args.soma_npz, allow_pickle=True))
        names = [str(n) for n in d["joint_names"]]
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
