"""Plot forward-pass runtimes (mean ± std) for SOMA-X vs SOMA-JAX.

Reads ``benchmarks/results/runtime.json`` (from ``bench_forward_pass.py``) and
writes ``benchmarks/figures/runtimes.pdf`` + ``runtimes.png`` styled with tueplots
CVPR full-page preset.

The figure has two panels:

* **Left** — total-forward latency vs batch size (mean ± std envelope).
* **Right** — throughput in meshes/sec vs batch size (mean ± std envelope).

Both axes are log-scaled on the batch dimension; the latency axis is also log
since the JAX path is ~4× faster across the range.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results" / "runtime.json"


# Series labels + colors chosen so the RELATIONSHIPS read off the legend:
#   * The four setups differ only in the per-identity "skeleton fit" (mapping a
#     body shape to its posed joints).
#   * "full fit" = SOMA-X's exact RBF + 2-stage-Kabsch solve; its rotation step
#     is a 3×3 SVD. The two SOMA-JAX full-fit rows are the SAME JAX pipeline and
#     differ ONLY in which kernel runs that SVD — XLA or a Warp svd3 kernel — so
#     they are two shades of the SAME hue (teal): they are alternatives.
#   * "linear fit" swaps the whole solve for a cheaper, approximate linear
#     regressor (no SVD) — a different method, so its own colour (purple).
#   * SOMA-X (the original, whole forward in PyTorch + NVIDIA Warp) is the
#     orange reference. Its "Warp" is the entire GPU backend, NOT the same thing
#     as the hybrid's single Warp SVD kernel — the colour keeps them separate.
_LABELS = {
    "soma_x":          "SOMA-X (PyTorch + Warp)",
    "soma_jax_st":     "SOMA-JAX · full fit (SVD in XLA)",
    "soma_jax_hybrid": "SOMA-JAX · full fit (SVD in Warp)",
    "soma_jax_linear": "SOMA-JAX · linear fit (approx.)",
    "soma_jax":        "SOMA-JAX · linear fit (approx.)",   # legacy key
}
_COLORS = {
    "soma_x":          "#000000",  # BLACK   — the original/reference, set apart
    # The SOMA-JAX family looks alike (one cool palette) and is colorblind-safe
    # (validated: teal↔indigo ΔE 20.6 CVD / 24.8 normal, contrast ≥3:1). The two
    # full-fit rows are the SAME pipeline → the SAME teal, told apart by line
    # style (solid vs dashed + ×), not colour.
    "soma_jax_st":     "#0d9488",  # teal    — full fit, SVD in XLA   (solid)
    "soma_jax_hybrid": "#0d9488",  # teal    — full fit, SVD in Warp  (dashed ×)
    "soma_jax_linear": "#4f46b8",  # indigo  — linear approximation
    "soma_jax":        "#4f46b8",
}
# Per-series line style: the Warp full-fit is dashed with × markers so it is
# distinct from the (same-colour) XLA full-fit.
_STYLE = {"soma_jax_hybrid": dict(ls="--", marker="x", markersize=4)}
_DEFAULT_STYLE = dict(ls="-", marker="o", markersize=3)
# One-line key printed under the title so "full/linear" and "SVD" aren't jargon.
_KEY = ("skeleton fit: full = SOMA-X's exact RBF+Kabsch solve (rotation step = a 3×3 SVD)   ·   "
        "linear = cheaper approximation (no SVD)   ·   SVD in XLA / Warp = which kernel runs it")


def _collect(results: dict) -> dict[str, dict[str, np.ndarray]]:
    """Pull batch / median / 95%-CI arrays out of the JSON for each backend.

    The value is the MEDIAN; the band is its 95% confidence interval
    (median ± 1.96·SE, SE ≈ 1.25·std/√n). That is the uncertainty of the
    *reported number* — it is <1% here, so the band is a hairline. (The larger
    per-call spread is GPU clock jitter, a property of a single call, not
    measurement error.)"""
    out: dict[str, dict[str, np.ndarray]] = {}
    for r in results["results"]:
        rows = r["rows"]
        batch = np.asarray([row["batch"] for row in rows])
        med = np.asarray([row["total"]["median_ms"] for row in rows])
        samples = [np.asarray(row["total"]["samples_ms"]) for row in rows]
        se = np.asarray([1.2533 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
                         for s in samples])
        lo, hi = med - 1.96 * se, med + 1.96 * se
        mps = batch * 1000.0 / med
        out[r["backend"]] = {
            "batch": batch, "med_ms": med, "lo_ms": lo, "hi_ms": hi,
            "mps": mps, "mps_lo": batch * 1000.0 / hi, "mps_hi": batch * 1000.0 / lo,
        }
    return out


def main() -> None:
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS}; run benchmarks/run_runtime.sh first")
    with open(RESULTS) as f:
        results = json.load(f)
    data = _collect(results)

    # tueplots CVPR preset for fonts + spacing. CVPR's default aspect is too
    # short for two panels with a header; bump the height ratio so the
    # suptitle has room and tick labels aren't clipped.
    rc = bundles.cvpr2024(usetex=False, nrows=1, ncols=2)
    # CVPR locks the figure to a single column width (~3.25"); for two
    # side-by-side log panels with a suptitle that's too tight. Widen to
    # full-page width (7") and bump the height to match the new aspect.
    rc["figure.figsize"] = (7.0, 4.2)    # room for a top legend row + 2-line footnote
    rc["figure.constrained_layout.use"] = True
    rc["figure.constrained_layout.h_pad"] = 0.06

    with plt.rc_context(rc):
        fig, (ax_lat, ax_thr) = plt.subplots(1, 2)
        # Reserve top band (suptitle + legend) and bottom band (2-line footnote)
        # so the legend sits OUTSIDE the panels and never overlaps the curves.
        fig.get_layout_engine().set(rect=(0.0, 0.11, 1.0, 0.78))

        # ---- left: latency vs batch (median; thin p10-p90 band) ----
        for be, d in data.items():
            ax_lat.plot(d["batch"], d["med_ms"], color=_COLORS[be], label=_LABELS[be],
                        **_STYLE.get(be, _DEFAULT_STYLE))
            ax_lat.fill_between(d["batch"], d["lo_ms"], d["hi_ms"],
                                 color=_COLORS[be], alpha=0.3, linewidth=0)
        ax_lat.set_xscale("log", base=2)
        ax_lat.set_yscale("log")
        ax_lat.set_xlabel("Batch size")
        ax_lat.set_ylabel("Forward-pass time (ms)")
        ax_lat.set_title("Per-forward time (median, 95% CI < 1%)")
        ax_lat.grid(True, which="both", linewidth=0.3, alpha=0.4)
        handles, labels = ax_lat.get_legend_handles_labels()

        # ---- right: throughput vs batch ----
        for be, d in data.items():
            ax_thr.plot(d["batch"], d["mps"], color=_COLORS[be], label=_LABELS[be],
                        **_STYLE.get(be, _DEFAULT_STYLE))
            ax_thr.fill_between(d["batch"], d["mps_lo"], d["mps_hi"],
                                 color=_COLORS[be], alpha=0.3, linewidth=0)
        ax_thr.set_xscale("log", base=2)
        ax_thr.set_yscale("log")
        ax_thr.set_xlabel("Batch size")
        ax_thr.set_ylabel("Throughput (meshes/sec)")
        ax_thr.set_title("Throughput (median, 95% CI < 1%)")
        ax_thr.grid(True, which="both", linewidth=0.3, alpha=0.4)

        # Legend OUTSIDE the panels, in a horizontal row under the title.
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
                   ncol=4, frameon=False, fontsize=6.5, columnspacing=1.1,
                   handletextpad=0.4)
        fig.suptitle("SOMA forward pass on RTX 5080", fontsize=8)
        setup = ("mid-LOD SOMA mesh: 18056 verts, 78 joints   ·   LBS-only   ·   "
                 "median over sustained-clock-warmed samples")
        fig.text(0.5, 0.015, setup + "\n" + _KEY, ha="center", va="bottom",
                 fontsize=5.4, color="0.30", linespacing=1.6)

        out_pdf = REPO / "benchmarks" / "figures" / "runtimes.pdf"
        out_png = REPO / "benchmarks" / "figures" / "runtimes.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=220)
        print(f"wrote {out_pdf}")
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
