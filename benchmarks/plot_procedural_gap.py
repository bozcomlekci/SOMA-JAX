"""Measure and plot what upstream's 122-joint procedural rig actually changes.

Writes ``figures/procedural_rig_gap.png`` (+ ``.pdf``) and prints the numbers
quoted in ``docs/FAITHFULNESS.md`` under "The procedural rig".

This is an **upstream-vs-upstream** measurement: the same SOMA-X layer built
with ``enable_procedural_transforms=True`` and ``False``, driven by the same
identity and the same 77-joint pose. It therefore measures the rig, not the
port -- it answers "is the twist skeleton worth reproducing at all?", which is
the question the port's motivation rests on.

Two panels:

* **left** -- maximum surface change against pose amplitude, as a median over
  ``--seeds`` random clips with the per-seed range shaded. The effect is
  strongly seed-dependent (a clip that happens to twist a forearm hard moves
  far more surface than one that does not), so a single number without its
  spread is not reproducible; the shaded band is the honest form of the claim.
* **right** -- where on the body it acts, at ``--sigma-map``.

Requires torch + the ``third_party/SOMA-X`` submodule + the HF rig.

    python benchmarks/plot_procedural_gap.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SOMA-X"))
# CPU keeps this reproducible run to run; it is a correctness measurement, not
# a throughput one, and upstream's reference path is CPU torch anyway.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FIGURES = Path(__file__).resolve().parent / "figures"
SIGMAS = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2)


def _upstream(procedural: bool, data_root: str):
    from soma.soma import SOMALayer

    return SOMALayer(
        data_root=data_root,
        identity_model_type="soma",
        device="cpu",
        mode="dense",
        enable_procedural_transforms=procedural,
        correctives_model_path=None,
    )


def _pose(rng_seed: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return (rng.standard_normal((1, 77, 3)) * sigma).astype(np.float32)


def measure(seeds: int, data_root: str):
    """Returns (sigmas, curves[seed, sigma] in mm, gap_mm at sigma_map, verts, faces)."""
    import torch

    proc, plain = _upstream(True, data_root), _upstream(False, data_root)
    zero = torch.zeros(1, 128)
    proc.prepare_identity(zero)
    plain.prepare_identity(zero)

    def both(poses):
        t = torch.tensor(poses)
        with torch.no_grad():
            a = proc.pose(t, pose2rot=True, apply_correctives=False)["vertices"].numpy()
            b = plain.pose(t, pose2rot=True, apply_correctives=False)["vertices"].numpy()
        return a[0], b[0]

    curves = np.empty((seeds, len(SIGMAS)))
    for s in range(seeds):
        for j, sigma in enumerate(SIGMAS):
            a, b = both(_pose(s, sigma))
            curves[s, j] = np.linalg.norm(a - b, axis=-1).max() * 1000.0
    return curves, proc, both


def render(verts, faces, size=460, elev=12.0, azim=28.0, values=None, cmap="viridis"):
    """Depth-sorted triangle rasteriser -- no GL context, so this runs headless."""
    import matplotlib.pyplot as plt

    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int64)
    c = 0.5 * (V.min(0) + V.max(0))
    r = np.abs(V - c).max()
    e, a = np.radians(elev), np.radians(azim)
    fwd = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    P = (V - c) @ np.stack([right, up, fwd], 1) / (r * 1.05)
    tri = P[F]

    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    n /= ln
    light = np.array([-0.4, 0.5, -1.0])
    light /= np.linalg.norm(light)
    lam = np.abs(n @ light) * 0.62 + 0.38

    if values is None:
        face_col = np.tile([0.78, 0.81, 0.87], (len(F), 1))
        vmax = None
    else:
        vals = np.asarray(values, np.float64)
        vmax = max(vals.max(), 1e-12)
        face_col = plt.get_cmap(cmap)(
            0.12 + 0.88 * np.clip(vals[F].mean(1) / vmax, 0, 1))[:, :3]

    img = np.ones((size, size, 3), np.float32)
    zbuf = np.full((size, size), np.inf)
    xy = (tri[:, :, :2] * 0.5 + 0.5) * (size - 1)
    xy[:, :, 1] = (size - 1) - xy[:, :, 1]

    for t in np.argsort(-tri[:, :, 2].mean(1)):
        p = xy[t]
        x0, x1 = int(max(0, np.floor(p[:, 0].min()))), int(min(size - 1, np.ceil(p[:, 0].max())))
        y0, y1 = int(max(0, np.floor(p[:, 1].min()))), int(min(size - 1, np.ceil(p[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0])
             + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
        if abs(d) < 1e-12:
            continue
        w0 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / d
        w1 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / d
        w2 = 1.0 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        z = w0 * tri[t, 0, 2] + w1 * tri[t, 1, 2] + w2 * tri[t, 2, 2]
        yy, xx = ys[m], xs[m]
        keep = z[m] < zbuf[yy, xx]
        yy, xx = yy[keep], xx[keep]
        zbuf[yy, xx] = z[m][keep]
        img[yy, xx] = face_col[t] * lam[t]
    return img, vmax


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5, help="random clips per amplitude")
    ap.add_argument("--sigma-map", type=float, default=0.45,
                    help="amplitude for the spatial panel")
    ap.add_argument("--data-root", default=None,
                    help="SOMA asset root (default: soma_jax.assets.data_root())")
    ap.add_argument("--out", default=str(FIGURES / "procedural_rig_gap.png"))
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import soma.soma  # noqa: F401
    except ImportError as exc:
        print(f"needs torch + the SOMA-X submodule: {exc}", file=sys.stderr)
        return 1

    data_root = args.data_root
    if data_root is None:
        from soma_jax.assets import data_root as _dr
        data_root = str(_dr(materialise=True))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves, proc, both = measure(args.seeds, data_root)
    med, lo, hi = np.median(curves, 0), curves.min(0), curves.max(0)

    print(f"upstream procedural vs non-procedural, {args.seeds} seeds")
    print(f"{'sigma':>7} {'median':>9} {'min':>9} {'max':>9}   (mm)")
    for j, s in enumerate(SIGMAS):
        print(f"{s:7.2f} {med[j]:9.1f} {lo[j]:9.1f} {hi[j]:9.1f}")

    a, b = both(_pose(0, args.sigma_map))
    gap = np.linalg.norm(a - b, axis=-1) * 1000.0
    faces = np.asarray(proc.rig_data["triangles"], np.int64)
    print(f"\nspatial map at sigma={args.sigma_map}: "
          f"max {gap.max():.1f} mm, mean {gap.mean():.2f} mm")

    fig = plt.figure(figsize=(9.6, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    ax.fill_between(SIGMAS, lo, hi, alpha=0.2, color="#3b6ea5",
                    label=f"per-seed range ({args.seeds} seeds)")
    ax.plot(SIGMAS, med, "o-", color="#3b6ea5", ms=3.5, label="median")
    ax.set_xlabel("pose amplitude σ (rad)")
    ax.set_ylabel("max surface change (mm)")
    ax.set_title("What the 122-joint twist rig changes\n"
                 "(upstream procedural vs upstream non-procedural)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    img, _ = render(a, faces)
    axb = fig.add_subplot(gs[1])
    axb.imshow(img)
    axb.axis("off")
    axb.set_title(f"posed, procedural rig\n(σ = {args.sigma_map})", fontsize=9)

    img, vmax = render(a, faces, values=gap)
    axc = fig.add_subplot(gs[2])
    axc.imshow(img)
    axc.axis("off")
    axc.set_title(f"where the twist joints act\nmax {gap.max():.1f} mm", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=axc.inset_axes([1.02, 0.12, 0.035, 0.76]))
    cb.set_label("mm", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nwrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
