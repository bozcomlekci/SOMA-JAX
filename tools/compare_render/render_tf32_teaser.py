"""Render the SOMA-JAX float32 -> TF32 speedup teaser GIF (assets/media/).

Same side-by-side compare as ``render_compare.py`` (SOMA-X left, SOMA-JAX
right, identical rig + motion), but built to show the **TF32 speedup** with the
top-left panel itself acting as the progress bar:

  * Each column's top-left panel IS a horizontal progress bar: the method name
    at the left, the precision at the right end, growing in width as frames are
    processed (the bar moves "from the name to the precision"). Bars share one
    px/frame scale, so their final widths read the speedup directly.
  * SOMA-X's speed is constant, so its baseline bar reaches the same length
    every run; SOMA-JAX renders ratio x more frames in the same wall-clock, so
    its bar reaches ~1.6x the SOMA-X bar at float32 and ~2.7x at TF32.
  * The runtime multiplier sits big in the CENTRE of the composite (in the gap
    between the two figures).
  * The body animates the first half at float32 (bar -> 1.6x), then PAUSES on
    the completed float32 state so it can be read, then switches to TF32 and
    resumes: SOMA-JAX renders every-frame-smooth while SOMA-X stays choppy, and
    its bar extends 1.6x -> 2.7x as the multiplier counts up and the precision
    flips. The final 2.7x frame holds for a couple seconds before the clip loops.

Precision honesty (the whole point of this repo's benchmark work):
  * float32 is the ONLY like-for-like comparison; SOMA-X cannot use TF32 (its
    Warp scalar kernels + sparse RBF are not tensor-core-eligible).
  * float32 and TF32 pose the SAME mesh to ~0.02 mm, so ONE SOMA-JAX vertex
    sequence is reused for both phases.
  * The TF32 SOMA-JAX throughput is not re-measured here; it is the measured
    float32 teaser throughput scaled by the benchmark's TF32/float32 factor at
    the same batch (see benchmarks/results/runtime_tf32.json). The fair float32
    ratio is the teaser's own SOMA-JAX/SOMA-X measurement.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "pipeline"))

# palette (matches the figures): SOMA-X orange ref, SOMA-JAX teal, TF32 accent.
COL_A = (0.85, 0.55, 0.30)     # SOMA-X mesh
COL_B = (0.20, 0.72, 0.55)     # SOMA-JAX mesh
SLATE = (138, 151, 171)        # SOMA-X baseline bar — neutral (distinct from teal/amber)
TEAL = (45, 190, 150)          # SOMA-JAX float32 accent
AMBER = (233, 148, 40)         # SOMA-JAX TF32 accent (Okabe-Ito vermillion-ish)

# Liberation Sans (Helvetica-like) reads cleaner than DejaVu and — unlike Noto
# Sans in this build — hints without glyph-spacing gaps at the small 13-16 px
# label sizes. Only Regular/Bold ship, so medium/semibold map to those.
_LIB = "/usr/share/fonts/truetype/liberation"
_FONT_FILES = {"regular": "LiberationSans-Regular.ttf", "medium": "LiberationSans-Regular.ttf",
               "semibold": "LiberationSans-Bold.ttf", "bold": "LiberationSans-Bold.ttf"}


def _font(size, weight="bold"):
    from PIL import ImageFont
    for path in (f"{_LIB}/{_FONT_FILES.get(weight, 'LiberationSans-Bold.ttf')}",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


_BAR_H = 30   # height of the top-left panel/progress bar


def _panel_bar(img, name, precision, bar_len, accent):
    """The top-left panel IS the progress bar: a rounded horizontal bar with the
    method name at the left and the precision at the right end, growing in width
    as frames are processed. Bars share one px/frame scale, so their final widths
    read the speedup directly (SOMA-JAX ~1.6x float32, ~2.7x TF32 of SOMA-X).
    No separate bar, no frame count."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img.copy()).convert("RGBA")
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    fN, fP = _font(14, "semibold"), _font(12, "medium")
    x0, y0 = 12, 12
    nW = int(d.textlength(name, font=fN)); pW = int(d.textlength(precision, font=fP))
    L = max(int(round(bar_len)), nW + 22)              # never clip the name
    d.rounded_rectangle([x0, y0, x0 + L, y0 + _BAR_H], radius=9, fill=accent + (240,))
    d.text((x0 + 11, y0 + 8), name, fill=(18, 20, 26, 255), font=fN)
    if L >= nW + pW + 34:                              # precision at the right end, once there's room
        d.text((x0 + L - pW - 11, y0 + 9), precision, fill=(18, 20, 26, 235), font=fP)
    return np.asarray(Image.alpha_composite(pil, ov).convert("RGB"))


def _center_mult(comp, mult, accent):
    """Big runtime multiplier centered on the FULL composite (in the gap between
    the two figures), on a subtle dark pill for legibility."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(comp.copy()).convert("RGBA")
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    W, H = pil.size
    f = _font(56, "bold")
    txt = f"{mult:.1f}×"
    bb = d.textbbox((0, 0), txt, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cx, cy = W / 2, H / 2
    d.rounded_rectangle([cx - tw / 2 - 20, cy - th / 2 - 14,
                         cx + tw / 2 + 20, cy + th / 2 + 14], radius=16, fill=(15, 17, 22, 170))
    d.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), txt, fill=accent + (255,), font=f)
    return np.asarray(Image.alpha_composite(pil, ov).convert("RGB"))


def _shared_camera(view_seqs, width, height):
    fov_y = np.pi / 3.0
    aspect = float(width) / float(height)
    lo = np.array([np.inf] * 3); hi = np.array([-np.inf] * 3)
    for vs in view_seqs:
        lo = np.minimum(lo, vs.reshape(-1, 3).min(0))
        hi = np.maximum(hi, vs.reshape(-1, 3).max(0))
    center = (lo + hi) * 0.5; span = hi - lo
    dist_v = (span[1] * 0.5 * 1.2) / np.tan(fov_y * 0.5)
    dist_h = (span[0] * 0.5 * 1.2) / np.tan(fov_y * 0.5) / aspect
    cam = np.eye(4, dtype=np.float32)
    cam[:3, 3] = center.astype(np.float32) + np.array(
        [0.0, 0.0, max(dist_v, dist_h) + span[2] * 0.5], np.float32)
    return cam, span


def _f32_ratio():
    """Benchmark-measured float32 SOMA-JAX/SOMA-X throughput ratio.

    Read from ``runtime.json`` rather than from the capture's own ``fps``
    fields. Both measure the same thing at the same batch, but the capture is a
    single ~30 ms timing run while ``runtime.json`` is a median over 20 iters
    with p10/p90 -- and they disagree: the capture put SOMA-X 6.8% faster,
    pulling the ratio to 1.60 where the harness gives 1.68. Sourcing the badge
    from the harness keeps the GIF, the READMEs and the benchmark table on one
    number instead of three.
    """
    def tot(p):
        for e in json.load(open(REPO / "benchmarks" / "results" / p))["results"]:
            if e["backend"] in ("soma_x", "soma_jax_hybrid"):
                yield e["backend"], {r["batch"]: r["total"]["meshes_per_sec_median"]
                                     for r in e["rows"]}
    d = dict(tot("runtime.json"))
    B = max(set(d["soma_x"]) & set(d["soma_jax_hybrid"]))
    return d["soma_jax_hybrid"][B] / d["soma_x"][B], B


def _tf32_factor():
    """Benchmark-measured TF32/float32 throughput factor for the full-fit
    (Warp-SVD) pipeline at the largest common batch."""
    def tot(p, b):
        for e in json.load(open(REPO / "benchmarks" / "results" / p))["results"]:
            if e["backend"] == "soma_jax_hybrid":
                return {r["batch"]: r["total"]["meshes_per_sec_median"] for r in e["rows"]}
    f32, tf = tot("runtime.json", 0), tot("runtime_tf32.json", 0)
    B = max(set(f32) & set(tf))
    return tf[B] / f32[B], B


def _budget(t, n, T):
    """Motion-frame index + running count for a column rendering n frames
    (sample-and-hold, spread over the whole clip) across T display slots."""
    g = min(n - 1, int(t * n / T))
    idx = int(round(g * (T - 1) / (n - 1))) if n > 1 else 0
    return idx, g + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--somax", default=str(REPO / "demo_renders" / "compare" / "somax.npz"))
    p.add_argument("--somajax", default=str(REPO / "demo_renders" / "compare" / "somajax.npz"))
    p.add_argument("--gif",
                   default=str(REPO / "assets" / "media" / "soma_jax_tf32_teaser.gif"))
    p.add_argument("--width", type=int, default=400)
    p.add_argument("--height", type=int, default=440)
    args = p.parse_args()

    from demo_soma_vis import render_mesh_png, _mask_nonanatomical_joints
    from PIL import Image

    A = np.load(args.somax, allow_pickle=True); B = np.load(args.somajax, allow_pickle=True)
    va, vb = A["verts"].astype(np.float32), B["verts"].astype(np.float32)
    fa, fb = A["faces"], B["faces"]
    T = min(len(va), len(vb)); va, vb = va[:T].copy(), vb[:T].copy()
    fps_sx, fps_jx = float(A["fps"]), float(B["fps"])
    ja = _mask_nonanatomical_joints(A["joints"][:T].astype(np.float32).copy(), A["joint_names"])
    jb = _mask_nonanatomical_joints(B["joints"][:T].astype(np.float32).copy(), B["joint_names"])
    pa, pb = A["parents"].astype(int), B["parents"].astype(int)

    # Speedups: both from the benchmark harness, so the badge matches the table.
    ratio_f32, Bf32 = _f32_ratio()
    tf32_factor, Bbench = _tf32_factor()
    ratio_tf32 = ratio_f32 * tf32_factor
    print(f"[speedups] float32 {ratio_f32:.3f}x @batch {Bf32} (runtime.json)"
          f"   [capture fps would give {fps_jx / fps_sx:.3f}x]")
    print(f"[speedups] TF32 factor {tf32_factor:.3f} @batch {Bbench}  ->  TF32 {ratio_tf32:.2f}x")

    # Ground-lock both columns to a shared floor (SOMA rig is Y-up).
    fl = va[..., 1].min(); va[..., 1] -= fl; ja[..., 1] -= fl
    fl = vb[..., 1].min(); vb[..., 1] -= fl; jb[..., 1] -= fl
    cam, span = _shared_camera([va, vb], args.width, args.height)
    W, H = args.width, args.height

    # ---- render each unique posed frame ONCE, reuse across both phases -------
    print(f"[render] {T} SOMA-X + {T} SOMA-JAX frames (cached, reused for both phases)")
    cache_a = [render_mesh_png(va[i], fa, None, W, H, color=COL_A, joints=ja[i],
                               parents=pa, body_alpha=0.6, camera_pose=cam, ground=True)
               for i in range(T)]
    cache_b = [render_mesh_png(vb[i], fb, None, W, H, color=COL_B, joints=jb[i],
                               parents=pb, body_alpha=0.6, camera_pose=cam, ground=True)
               for i in range(T)]

    # ONE continuous motion pass. The body switches float32 -> TF32 at the MIDDLE
    # of the motion, so the TF32 speed is shown LIVE: SOMA-X stays choppy
    # throughout, SOMA-JAX renders smoother at float32 and every-frame-smooth
    # after the TF32 switch, and its bar leaps from ~1.6x to ~2.7x the SOMA-X bar.
    max_bar = W - 28
    baseline = max_bar / ratio_tf32                  # width of a 1.0x (SOMA-X) bar
    half = T // 2                                    # switch at the middle of the motion
    ramp = 16                                        # slots to extend the bar 1.6x -> 2.7x
    # sample-and-hold quantisation = how many display slots each pose is held for;
    # bigger = choppier. SOMA-X is the choppy reference throughout; SOMA-JAX gets
    # smoother (fewer held slots) once TF32 turns on at the midpoint.
    Q_X, Q_JF32, Q_JTF32 = 4, 2, 1

    def _hold(t, q):
        return min((t // q) * q, T - 1)

    def _mult(t):                                    # ramps up just after the midpoint
        if t <= half: return ratio_f32
        if t >= half + ramp: return ratio_tf32
        return ratio_f32 + (ratio_tf32 - ratio_f32) * (t - half) / ramp

    def compose(t):
        m = _mult(t)
        tf32 = t > half
        precision, accent = ("TF32", AMBER) if tf32 else ("float32", TEAL)
        # SOMA-X bar fills 0 -> baseline over the FIRST HALF, then holds; SOMA-JAX
        # bar = mult x it, so it reads a clean 1.6x at the midpoint and then
        # extends to 2.7x when TF32 turns on. Motion plays through both halves.
        sx_bar = min(1.0, t / half) * baseline
        sj_bar = m * sx_bar
        left = _panel_bar(cache_a[_hold(t, Q_X)], "SOMA-X", "float32", sx_bar, SLATE)
        right = _panel_bar(cache_b[_hold(t, Q_JTF32 if tf32 else Q_JF32)],
                           "SOMA-JAX", precision, sj_bar, accent)
        return _center_mult(np.concatenate([left, right], axis=1), m, accent)

    duration_s = float(A["duration_s"]); play_fps = T / duration_s
    frames = [compose(t) for t in range(half + 1)]          # first half: float32, bar -> 1.6x
    # PAUSE on the completed float32 state (1.6x) so it can be read, then resume.
    frames += [frames[-1]] * max(1, int(round(1.5 * play_fps)))
    frames += [compose(t) for t in range(half + 1, T)]      # resume: switch to TF32, bar -> 2.7x
    # Hold the final 2.7x image for a couple seconds, then loop.
    frames += [frames[-1]] * max(1, int(round(2.0 * play_fps)))

    dur = int(round(1000.0 * duration_s / T))       # ms/frame = realtime playback
    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(x).convert("P", palette=Image.ADAPTIVE, colors=96)
           for x in frames]
    ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=dur,
                loop=0, optimize=True)
    # Poster still-frame goes to the git-ignored render dir (assets/media/ holds gifs).
    poster = REPO / "demo_renders" / "compare" / (Path(args.gif).stem + "_poster.png")
    poster.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[-1]).save(str(poster))
    print(f"wrote {args.gif}  ({len(frames)} frames @ {play_fps:.0f} fps realtime)")
    print(f"  motion switches float32 -> TF32 at pose {half}/{T}; "
          f"SOMA-JAX bar {ratio_f32:.1f}x -> {ratio_tf32:.1f}x the SOMA-X bar")


if __name__ == "__main__":
    main()
