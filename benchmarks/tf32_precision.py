"""Measure what TF32 costs SOMA-JAX in vertex accuracy -> results/tf32_precision.json.

The identity-blend GEMM ``mean + (coeffs * sqrt(eig)) @ shapedirs`` is the
largest matmul in the forward and sets the vertex positions. It is computed
twice on the CPU: once in full float32, and once with both operands rounded to
TF32's 10-bit mantissa (round-to-nearest-even) and accumulated in float32 —
what an Ampere+ tensor core does with ``jax_default_matmul_precision="default"``.
The difference is the TF32 vertex error. SOMA-X cannot use TF32 (its heavy
kernels are Warp scalar float32 code), which is why the TF32 throughput is
never compared head-to-head with it.

Usage::

    python benchmarks/tf32_precision.py      # -> benchmarks/results/tf32_precision.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def to_tf32(x: np.ndarray) -> np.ndarray:
    """Round float32 values to TF32's 10-bit mantissa, round-to-nearest-even."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0xFFF) + ((bits >> np.uint32(13)) & np.uint32(1))) \
        & np.uint32(0xFFFFE000)
    return rounded.view(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "tf32_precision.json"))
    args = p.parse_args()

    from soma_jax.assets import data_root
    with np.load(data_root() / "SOMA_neutral.npz", allow_pickle=False) as core:
        mean = np.asarray(core["mean"], np.float32).reshape(-1)           # (3V,) cm
        shapedirs = np.asarray(core["shapedirs"], np.float32)              # (K, 3V) cm
        eigenvalues = np.asarray(core["eigenvalues"], np.float32)          # (K,)
    coeffs = np.random.default_rng(args.seed).standard_normal(
        (args.batch, eigenvalues.shape[0])).astype(np.float32)
    weighted = coeffs * np.sqrt(eigenvalues)

    full = mean[None] + weighted @ shapedirs
    tf32 = mean[None] + to_tf32(weighted) @ to_tf32(shapedirs)
    err_mm = np.linalg.norm((tf32 - full).reshape(args.batch, -1, 3), axis=-1) * 10.0
    # Body scale: each identity's mean vertex distance from its centroid.
    bodies_mm = full.reshape(args.batch, -1, 3) * 10.0
    radius_mm = float(np.linalg.norm(
        bodies_mm - bodies_mm.mean(axis=1, keepdims=True), axis=-1).mean())

    result = {
        "method": ("TF32 numerical-error emulation: the identity-blend GEMM "
                   "'mean + (coeffs*sqrt(eig)) @ shapedirs' (the largest matmul in the "
                   "forward, which sets vertex positions) computed twice on CPU -- once in "
                   "full float32, once with both operands rounded to a 10-bit mantissa "
                   "(TF32, round-to-nearest-even) and accumulated in float32, exactly as an "
                   f"Ampere+ TF32 tensor-core GEMM does. B={args.batch} random SOMA "
                   f"identities (seed {args.seed}). Reproduce: python benchmarks/tf32_precision.py"),
        "vertex_error_mm": {"mean": round(float(err_mm.mean()), 4),
                            "median": round(float(np.median(err_mm)), 4),
                            "p99": round(float(np.percentile(err_mm, 99)), 4),
                            "max": round(float(err_mm.max()), 4)},
        "relative_mean": float(f"{err_mm.mean() / radius_mm:.3g}"),
        "body_radius_mm": round(radius_mm, 1),
        "note": ("SOMA-X runs full float32 and structurally cannot use TF32 (Warp scalar "
                 "kernels + sparse RBF, none tensor-core-eligible), so this JAX-only TF32 "
                 "precision cost is why TF32 throughput is never compared head-to-head with "
                 "SOMA-X."),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["vertex_error_mm"]), f"relative_mean={result['relative_mean']}",
          f"body_radius_mm={result['body_radius_mm']}")


if __name__ == "__main__":
    main()
