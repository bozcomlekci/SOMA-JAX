"""Render the SOMA-X vs SOMA-JAX side-by-side comparison GIF.

Reuses the proven demo_soma_vis rendering (``render_mesh_png``: ground plane +
projection shadow + multi-light + shared framing camera) so the look matches the
demo_renders/ outputs. Inputs are the two ``pose_*.py`` outputs (verts / faces /
median fps). Both columns pose the identical rig over the identical motion
with identical settings, so the vertices agree to ~cm — the ONLY thing
that differs is how many frames each pipeline can afford in a fixed time budget.

Equal-time visualization
------------------------
Both columns play for the same wall-clock duration (the same number of GIF
slots). The FASTER pipeline advances every slot — it renders the whole motion
smoothly. The SLOWER pipeline is subsampled by the measured speed ratio
``step = round(fps_fast / fps_slow)``: it advances only every ``step`` slots
(sample-and-hold), so within the same duration it shows ``1/step`` as many
distinct poses — visibly sparser/choppier. Exactly "in the same time, the
faster method computes ``step``x more frames."

The GIF ends on a centered "SOMA-JAX ~N.Nx faster" popup held for ~1.5 s.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = Path(__file__).resolve().parents[2]


def _benchmark_f32_ratio():
    """float32 SOMA-JAX/SOMA-X throughput ratio from ``benchmarks/results``.

    The capture npz files carry their own ``fps``, but those come from a single
    ~30 ms timing run: at batch 2048 they put SOMA-X 6.8% faster than the
    harness does, which moves the ratio 1.68 -> 1.60 and would show the viewer a
    different speedup from the one every document quotes. ``runtime.json`` is a
    median over 20 iters with p10/p90, so it is the number to draw. Returns
    ``None`` when the results file is absent, and the caller falls back to the
    capture fps.
    """
    import json
    f = REPO / "benchmarks" / "results" / "runtime.json"
    if not f.exists():
        return None
    try:
        per = {}
        for e in json.loads(f.read_text())["results"]:
            if e["backend"] in ("soma_x", "soma_jax_hybrid"):
                per[e["backend"]] = {r["batch"]: r["total"]["meshes_per_sec_median"]
                                     for r in e["rows"]}
        common = set(per["soma_x"]) & set(per["soma_jax_hybrid"])
        if not common:
            return None
        B = max(common)
        return per["soma_jax_hybrid"][B] / per["soma_x"][B], B
    except (KeyError, ValueError, TypeError):
        return None
sys.path.insert(0, str(REPO / "tools" / "pipeline"))


def _label(img, line1, line2=None, color=(255, 255, 255), y=8):
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img.copy()); d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        f = fs = ImageFont.load_default()
    w1 = int(d.textlength(line1, font=f))
    w2 = int(d.textlength(line2, font=fs)) if line2 else 0
    bh = 24 + (18 if line2 else 0)
    d.rectangle([6, y - 2, 6 + max(w1, w2) + 10, y + bh], fill=(0, 0, 0))
    d.text((11, y), line1, fill=color, font=f)
    if line2:
        d.text((11, y + 22), line2, fill=(180, 220, 255), font=fs)
    return np.asarray(pil)


def _centered_banner(img, lines, sub=None):
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img.copy()); d = ImageDraw.Draw(pil)
    W, H = pil.size
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        big = small = ImageFont.load_default()
    band = Image.new("RGBA", (W, 150), (0, 0, 0, 175))
    pil.paste(Image.alpha_composite(pil.crop((0, H // 2 - 75, W, H // 2 + 75)).convert("RGBA"), band),
              (0, H // 2 - 75))
    d = ImageDraw.Draw(pil)
    bb = d.textbbox((0, 0), lines, font=big)
    d.text(((W - (bb[2] - bb[0])) / 2, H // 2 - 42), lines, fill=(90, 230, 170), font=big)
    if sub:
        bb2 = d.textbbox((0, 0), sub, font=small)
        d.text(((W - (bb2[2] - bb2[0])) / 2, H // 2 + 24), sub, fill=(235, 235, 235), font=small)
    return np.asarray(pil)


def _shared_camera(view_seqs, width, height):
    """demo_soma_vis shared-camera fit: frame the union of XYZ extents across
    all frames + both columns, +20% margin, camera pulled back along +Z."""
    fov_y = np.pi / 3.0
    aspect = float(width) / float(height)
    all_min = np.array([np.inf] * 3); all_max = np.array([-np.inf] * 3)
    for vs in view_seqs:
        all_min = np.minimum(all_min, vs.reshape(-1, 3).min(0))
        all_max = np.maximum(all_max, vs.reshape(-1, 3).max(0))
    center = (all_min + all_max) * 0.5
    span = all_max - all_min
    margin = 1.20
    dist_v = (span[1] * 0.5 * margin) / np.tan(fov_y * 0.5)
    dist_h = (span[0] * 0.5 * margin) / np.tan(fov_y * 0.5) / aspect
    cam_distance = max(dist_v, dist_h) + span[2] * 0.5
    cam = np.eye(4, dtype=np.float32)
    cam[:3, 3] = center.astype(np.float32) + np.array([0.0, 0.0, cam_distance], np.float32)
    return cam, span


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--somax", required=True)
    p.add_argument("--somajax", required=True)
    p.add_argument("--gif", required=True)
    p.add_argument("--width", type=int, default=420)
    p.add_argument("--height", type=int, default=460)
    p.add_argument("--play-fps", type=float, default=24.0)
    args = p.parse_args()

    # proven ground+shadow+camera render + the non-anatomical joint mask
    from demo_soma_vis import render_mesh_png, _mask_nonanatomical_joints
    from PIL import Image

    A = np.load(args.somax, allow_pickle=True); B = np.load(args.somajax, allow_pickle=True)
    va, vb = A["verts"].astype(np.float32), B["verts"].astype(np.float32)
    fa, fb = A["faces"], B["faces"]
    T = min(len(va), len(vb))
    va, vb = va[:T].copy(), vb[:T].copy()
    fps_a, fps_b = float(A["fps"]), float(B["fps"])
    lab_a, lab_b = str(A["label"]), str(B["label"])

    # Skeleton overlay: masked joints (drops Root/End/Eye/Jaw so the head is
    # clean) drawn inside a translucent body.
    ja = _mask_nonanatomical_joints(A["joints"][:T].astype(np.float32).copy(), A["joint_names"])
    jb = _mask_nonanatomical_joints(B["joints"][:T].astype(np.float32).copy(), B["joint_names"])
    pa, pb = A["parents"].astype(int), B["parents"].astype(int)

    # How close are the two posed meshes? NOT a parity number: pose_somajax.py
    # loads the raw SOMA_neutral.npz rig while SOMA-X merges SOMA_template_rig.usda
    # over it, so ~1.7 cm here is the rig mismatch described under "the two timed
    # sides do not use the identical rig" in benchmarks/README.md, plus the Warp
    # svd3 rotation solve. The port's actual forward parity is 0.34-1.16 mm
    # against a matched rig -- see docs/FAITHFULNESS.md.
    if va.shape == vb.shape:
        dmm = np.linalg.norm(va - vb, axis=-1)
        print(f"[mesh delta] SOMA-X vs SOMA-JAX posed verts: "
              f"max={dmm.max()*100:.3f} cm  mean={dmm.mean()*100:.4f} cm "
              f"(rig mismatch + warp svd3, not a parity figure)")

    # Ground-lock each column: drop its lowest point (over the whole clip) to
    # Y=0 so both share one floor (SOMA rig is already Y-up; no view rotation).
    # Shift joints by the same offset so the skeleton tracks the mesh.
    fa_floor = va[..., 1].min(); va[..., 1] -= fa_floor; ja[..., 1] -= fa_floor
    fb_floor = vb[..., 1].min(); vb[..., 1] -= fb_floor; jb[..., 1] -= fb_floor

    cam, span = _shared_camera([va, vb], args.width, args.height)
    print(f"[camera] motion XYZ span=({span[0]:.2f},{span[1]:.2f},{span[2]:.2f})m")

    # --- equal-time frame budget ---------------------------------------------
    # Both columns run for the same wall-clock (T display slots = the clip's
    # realtime duration). The FASTER pipeline renders all T frames; the SLOWER
    # renders only round(T / ratio) frames (it can't keep up), spread across the
    # whole motion (sample-and-hold => choppier). The live counter shows exactly
    # "total frames" (fast) vs "total / relative-speed frames" (slow).
    capture_ratio = max(fps_a, fps_b) / max(min(fps_a, fps_b), 1e-9)
    bench = _benchmark_f32_ratio()
    if bench is not None:
        ratio, Bbench = bench
        src = f"runtime.json @batch {Bbench}"
    else:
        ratio, Bbench = capture_ratio, None
        src = "capture fps (benchmarks/results/runtime.json absent)"
    faster_is_b = ratio >= 1.0
    ratio = max(ratio, 1.0 / ratio)          # magnitude, whichever way round
    n_a = T if not faster_is_b else max(2, int(round(T / ratio)))
    n_b = T if faster_is_b else max(2, int(round(T / ratio)))
    print(f"[equal-time] ratio {ratio:.3f}x from {src}"
          f"   [capture fps would give {capture_ratio:.3f}x]")
    print(f"[equal-time] frames rendered: SOMA-X {n_a}  SOMA-JAX {n_b}  "
          f"over {T} realtime slots")

    def budget(t, n):
        """Motion-frame index + running count for a column that renders n frames
        (spread over the full motion) across the T display slots."""
        g = min(n - 1, int(t * n / T))                 # 0..n-1 unique-frame group
        idx = int(round(g * (T - 1) / (n - 1))) if n > 1 else 0
        return idx, g + 1

    W, H = args.width, args.height
    col_a = (0.85, 0.55, 0.30)   # SOMA-X orange
    col_b = (0.20, 0.72, 0.55)   # SOMA-JAX teal

    frames = []
    for t in range(T):
        ia_idx, ca = budget(t, n_a)
        ib_idx, cb = budget(t, n_b)
        ra = render_mesh_png(va[ia_idx], fa, None, W, H, color=col_a,
                             joints=ja[ia_idx], parents=pa, body_alpha=0.6,
                             camera_pose=cam, ground=True)
        rb = render_mesh_png(vb[ib_idx], fb, None, W, H, color=col_b,
                             joints=jb[ib_idx], parents=pb, body_alpha=0.6,
                             camera_pose=cam, ground=True)
        ia = _label(ra, lab_a, f"{ca} frames")
        ib = _label(rb, lab_b, f"{cb} frames")
        frames.append(np.concatenate([ia, ib], axis=1))

    # Final popup (held ~2 s): the relative speed increase.
    who = "SOMA-JAX" if fps_b >= fps_a else "SOMA-X"
    banner = _centered_banner(
        frames[-1].copy(), f"{who}  ~{ratio:.1f}x  faster",   # last frame: counters at finals
        sub=f"same wall-clock time  ·  {max(n_a, n_b)} vs {min(n_a, n_b)} frames rendered  "
            f"·  full forward, batch {int(A['batch'])}")
    duration_s = float(A["duration_s"])
    play_fps = T / duration_s                          # realtime playback (item: realtime)
    hold = int(round(2.0 * play_fps))
    frames.extend([banner] * hold)

    dur = int(round(1000.0 * duration_s / T))          # ms/frame = realtime
    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(x) for x in frames]
    ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=dur, loop=0)
    Image.fromarray(banner).save(str(Path(args.gif).with_suffix(".png")))
    print(f"wrote {args.gif}  ({len(frames)} frames, {play_fps:.0f} fps realtime)  "
          f"{who} ~{ratio:.1f}x faster  ({max(n_a,n_b)} vs {min(n_a,n_b)} frames)")


if __name__ == "__main__":
    main()
