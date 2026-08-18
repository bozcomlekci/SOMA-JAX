"""Plot peak GPU memory vs batch size for SOMA-X vs SOMA-JAX.

Reads ``benchmarks/results/memory.jsonl`` (from ``run_memory.sh``) and writes
``benchmarks/figures/memory.pdf`` + ``memory.png``.

One panel, one metric: ``peak_mib`` — CUDA context plus the high-water mark of
live device bytes the process demands, with NVIDIA Warp's allocations counted
on both sides. Linear GiB axis against the 16 GB card limit, so "how much
memory / how close to the wall" reads directly.

Methodology lives in ``benchmarks/README.md``; the figure carries only what is
needed to read it.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results" / "memory.jsonl"
FIELD = "peak_mib"
CARD_GIB = 16.0

# Labels + colours match plot_runtimes.py: the two "full fit" rows are the same
# JAX pipeline differing only in the rotation kernel (teal pair); "linear fit"
# is the approximate method (indigo); SOMA-X is the black reference.
_LABELS = {
    "soma_x": "SOMA-X (PyTorch + Warp)",
    "fair":   "SOMA-JAX · full fit (SVD in XLA)",
    "hybrid": "SOMA-JAX · full fit (SVD in Warp)",
    "linear": "SOMA-JAX · linear fit (approx.)",
}
_COLORS = {
    "soma_x": "#000000",  # BLACK   — the original/reference, set apart
    "fair":   "#0d9488",  # teal    — full fit, SVD in XLA   (solid)
    "hybrid": "#0d9488",  # teal    — full fit, SVD in Warp  (dashed ×)
    "linear": "#4f46b8",  # indigo  — linear approximation
}
_ORDER = ["soma_x", "fair", "hybrid", "linear"]


def _rows():
    for line in open(RESULTS):
        line = line.strip()
        if line:
            yield json.loads(line)


def _collect():
    data = {}
    for r in _rows():
        if r.get("status") != "ok" or r.get(FIELD) is None:
            continue
        data.setdefault(r["method"], []).append(r)
    for m in data:
        data[m].sort(key=lambda r: r["batch"])
    return data


def _first_oom(method: str):
    """Smallest batch where *method* failed to allocate, or None if it never did."""
    batches = [r["batch"] for r in _rows()
               if r.get("method") == method and r.get(FIELD) is None]
    return min(batches) if batches else None


def main():
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS}; run run_memory.sh first")
    data = _collect()

    rc = bundles.cvpr2024(usetex=False, nrows=1, ncols=1)
    rc["figure.figsize"] = (5.4, 3.9)
    rc["figure.constrained_layout.use"] = False
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(left=0.135, right=0.975, top=0.855, bottom=0.135)

        for m in _ORDER:
            rows = data.get(m)
            if not rows:
                continue
            b = np.array([r["batch"] for r in rows])
            gib = np.array([r[FIELD] for r in rows]) / 1024.0
            style = dict(ls="--", marker="x", markersize=5) if m == "hybrid" \
                else dict(ls="-", marker="o", markersize=3.5)
            ax.plot(b, gib, color=_COLORS[m], label=_LABELS[m],
                    linewidth=1.5, **style)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Peak GPU memory (GiB)")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        ax.set_ylim(0, CARD_GIB + 1.0)

        ax.axhline(CARD_GIB, color="0.35", ls="--", lw=1.0)
        ax.text(0.985, CARD_GIB - 0.3, "16 GB card", fontsize=6.5,
                color="0.35", va="top", ha="right",
                transform=ax.get_yaxis_transform())

        oom = _first_oom("soma_x")
        rows = data.get("soma_x") or []
        if oom is not None and rows:
            last = rows[-1]
            ax.annotate(f"OOM at B={oom}",
                        xy=(last["batch"], last[FIELD] / 1024.0),
                        xytext=(-2, 8), textcoords="offset points",
                        fontsize=6.0, color="0.15", ha="right")

        # Anchored below the card line (at CARD_GIB/ylim_top of the axes
        # height) so the top entry cannot collide with it.
        ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.90),
                  frameon=False, fontsize=6.4, handletextpad=0.4)
        fig.suptitle("SOMA forward-pass peak GPU memory (RTX 5080)",
                     fontsize=9, y=0.985)
        ax.set_title("CUDA context + live allocator high-water, Warp included",
                     fontsize=6.8, color="0.35", pad=3)

        out = REPO / "benchmarks" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "memory.pdf")
        fig.savefig(out / "memory.png", dpi=200)
        print(f"wrote {out}/memory.{{pdf,png}}")


if __name__ == "__main__":
    main()
