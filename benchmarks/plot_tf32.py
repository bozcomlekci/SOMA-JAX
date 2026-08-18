"""Plot the TF32 vs float32 trade-off for SOMA-JAX — speed bought, precision paid.

Reads ``results/runtime.json`` (float32, fair) + ``results/runtime_tf32.json``
(TF32, JAX-only) + ``results/tf32_precision.json`` (measured TF32 vertex error)
and writes ``figures/tf32.pdf`` + ``tf32.png``.

The whole point of the figure is to show TF32 WITHOUT letting it be read as a
fair comparison against SOMA-X:

* **Left** — throughput vs batch. The two *float32* curves (SOMA-X and the
  matched SOMA-JAX full fit) are the only like-for-like pair; the SOMA-JAX
  *TF32* curve is drawn in a set-apart warm colour + dashed/open style and
  labelled "JAX-only, lower precision" so it never reads as head-to-head.
* **Right** — the trade at B=2048: the *fair* float32 speedup (both float32)
  vs the TF32 speedup, which is greyed/hatched and explicitly marked "not a
  like-for-like number", annotated with the measured TF32 vertex error.

SOMA-X CANNOT use TF32 (its work is Warp scalar-float32 kernels + a sparse RBF
matmul, none tensor-core-eligible), so TF32 is a JAX/XLA-only deployment lever.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tueplots import bundles

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "benchmarks" / "results"

# Colours: keep the family from the other figures (SOMA-X black, SOMA-JAX teal)
# and add ONE set-apart warm hue for TF32. Vermillion is Okabe-Ito (the
# colorblind-safe reference palette) and reads as "caution / different" against
# both black and teal for every CVD type.
C_SOMAX = "#000000"   # SOMA-X, float32 reference
C_F32   = "#0d9488"   # SOMA-JAX full fit, float32 (the fair match)
C_TF32  = "#d55e00"   # SOMA-JAX full fit, TF32 (JAX-only, lower precision)

# The full-fit pipeline whose TF32 gain is largest (its covariance GEMMs are the
# tensor-core-eligible work); "total" = whole forward throughput.
BACKEND = "soma_jax_hybrid"


def _totals(path: Path, backend: str) -> tuple[np.ndarray, np.ndarray]:
    """(batch, meshes/sec) for one backend's whole-forward ('total') metric."""
    results = json.load(open(path))["results"]
    for e in results:
        if e.get("backend") == backend:
            rows = e["rows"]
            b = np.array([r["batch"] for r in rows])
            mps = np.array([r["total"]["meshes_per_sec_median"] for r in rows])
            return b, mps
    raise KeyError(f"{backend} not in {path}")


def main() -> None:
    for p in ("runtime.json", "runtime_tf32.json", "tf32_precision.json"):
        if not (RES / p).exists():
            sys.exit(f"missing {RES / p}")

    b_sx, mps_sx = _totals(RES / "runtime.json", "soma_x")
    b_f32, mps_f32 = _totals(RES / "runtime.json", BACKEND)
    b_tf, mps_tf = _totals(RES / "runtime_tf32.json", BACKEND)
    err = json.load(open(RES / "tf32_precision.json"))
    e_mean = err["vertex_error_mm"]["mean"]
    e_max = err["vertex_error_mm"]["max"]
    e_rel = err["relative_mean"]

    # Speedups at the largest common batch (the reported operating point).
    B = int(min(b_sx.max(), b_f32.max(), b_tf.max()))
    at = lambda bb, mm: float(mm[np.where(bb == B)[0][0]])
    sx_B, f32_B, tf_B = at(b_sx, mps_sx), at(b_f32, mps_f32), at(b_tf, mps_tf)
    sp_fair = f32_B / sx_B      # both float32 -> the honest number
    sp_tf32 = tf_B / sx_B       # TF32 vs float32 -> NOT a fair number

    rc = bundles.cvpr2024(usetex=False, nrows=1, ncols=2)
    rc["figure.figsize"] = (7.0, 4.2)
    rc["figure.constrained_layout.use"] = True
    rc["figure.constrained_layout.h_pad"] = 0.06
    with plt.rc_context(rc):
        fig, (ax_thr, ax_bar) = plt.subplots(1, 2)
        fig.get_layout_engine().set(rect=(0.0, 0.11, 1.0, 0.80))

        # ---- left: throughput vs batch (two float32 + one TF32) ----
        ax_thr.plot(b_sx, mps_sx, color=C_SOMAX, ls="-", marker="o", markersize=3,
                    label="SOMA-X  ·  float32")
        ax_thr.plot(b_f32, mps_f32, color=C_F32, ls="-", marker="o", markersize=3,
                    label="SOMA-JAX full fit  ·  float32  (fair match)")
        ax_thr.plot(b_tf, mps_tf, color=C_TF32, ls="--", marker="s", markersize=3.5,
                    markerfacecolor="none", label="SOMA-JAX full fit  ·  TF32  (JAX-only)")
        # Shade the gap TF32 opens over its own float32 — the precision-bought speed.
        ax_thr.fill_between(b_f32, mps_f32, mps_tf, where=(mps_tf >= mps_f32),
                            color=C_TF32, alpha=0.10, linewidth=0)
        ax_thr.set_xscale("log", base=2)
        ax_thr.set_yscale("log")
        ax_thr.set_xlabel("Batch size")
        ax_thr.set_ylabel("Throughput (meshes/sec)")
        ax_thr.set_title("Throughput — float32 is the only fair pair", fontsize=7.5)
        ax_thr.grid(True, which="both", linewidth=0.3, alpha=0.4)
        ax_thr.legend(loc="upper left", frameon=False, fontsize=5.8,
                      handletextpad=0.5, borderaxespad=0.3)

        # ---- right: the trade at B=2048 (speedup vs SOMA-X + precision cost) ----
        xs = [0, 1]
        ax_bar.bar(0, sp_fair, width=0.62, color=C_F32,
                   label="float32 (fair)")
        ax_bar.bar(1, sp_tf32, width=0.62, color=C_TF32, alpha=0.55,
                   hatch="////", edgecolor=C_TF32, linewidth=0.0,
                   label="TF32 (not comparable)")
        ax_bar.axhline(1.0, color="0.35", ls="--", lw=0.9)
        ax_bar.text(1.47, 1.05, "SOMA-X baseline (1.0×)", fontsize=5.6, color="0.35",
                    va="bottom", ha="right")
        ax_bar.text(0, sp_fair + 0.06, f"{sp_fair:.2f}×", ha="center", va="bottom",
                    fontsize=7.5, color=C_F32, fontweight="bold")
        ax_bar.text(1, sp_tf32 + 0.06, f"{sp_tf32:.2f}×", ha="center", va="bottom",
                    fontsize=7.5, color=C_TF32, fontweight="bold")
        # The crossed-out marker on the TF32 bar: it is NOT a like-for-like number.
        ax_bar.text(1, sp_tf32 * 0.5, "≠ fair\ncomparison", ha="center", va="center",
                    fontsize=5.6, color="white", fontweight="bold", linespacing=1.2)
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels(["SOMA-JAX\nfloat32", "SOMA-JAX\nTF32"], fontsize=6.5)
        ax_bar.set_ylabel(f"Speedup vs SOMA-X  (batch {B})")
        ax_bar.set_ylim(0, sp_tf32 * 1.28)
        ax_bar.set_title("What TF32 buys, and what it costs", fontsize=7.5)
        ax_bar.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        # Precision-cost annotation — the "error" the user must see.
        cost = (f"TF32 precision cost (measured):\n"
                f"vertex error  mean {e_mean:.3f} mm · max {e_max:.2f} mm\n"
                f"(relative {e_rel:.1e} — sub-millimetre)\n"
                f"SOMA-X cannot use TF32.")
        ax_bar.text(0.5, sp_tf32 * 1.24, cost, transform=ax_bar.transData,
                    ha="center", va="top", fontsize=5.4, color="0.20", linespacing=1.45,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7ed",
                              edgecolor=C_TF32, linewidth=0.6))

        fig.suptitle("SOMA-JAX TF32 vs float32 on RTX 5080 — speed bought, precision paid",
                     fontsize=8)
        foot = ("float32 (highest) is the matched, like-for-like comparison; TF32 is a "
                "JAX/XLA tensor-core mode SOMA-X's Warp kernels cannot use.   ·   "
                "precision cost = TF32 (10-bit-mantissa) emulation of the identity-blend GEMM")
        fig.text(0.5, 0.015, foot, ha="center", va="bottom", fontsize=5.2,
                 color="0.30", linespacing=1.6)

        out_pdf = REPO / "benchmarks" / "figures" / "tf32.pdf"
        out_png = REPO / "benchmarks" / "figures" / "tf32.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=220)
        print(f"wrote {out_pdf}")
        print(f"wrote {out_png}")
        print(f"  B={B}: fair {sp_fair:.2f}x, TF32 {sp_tf32:.2f}x; "
              f"TF32 err mean {e_mean:.3f}mm max {e_max:.2f}mm")


if __name__ == "__main__":
    main()
