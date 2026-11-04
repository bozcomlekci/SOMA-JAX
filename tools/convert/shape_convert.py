"""Convert identity coefficients between identity models.

Fits the *target* model's identity coefficients so that its body reproduces
the *source* body. Both bodies are compared the way SOMA compares identities:
in SOMA topology, after the per-identity skeleton fit and the repose to SOMA's
shared bind pose (``SOMALayer.prepare_identity(repose_to_bind_pose=True)``),
so the models' different rest poses — MHR's, SMPL's T-pose, SOMA's own — do not
leak into the fitted shape. The fit is damped Gauss–Newton through the target
layer's differentiable ``prepare_identity``; for a PCA backend it converges in
a few steps. A source that the target's shape space cannot represent leaves a
residual (reported per sample); ``--prior`` regularises such fits.

Supported backends: ``soma`` (128-coefficient PCA), ``smpl`` / ``smplx`` /
``smplh`` (betas; need the registration-only model files), ``mhr`` and
``garment_measurement`` (identity packs from
``tools/pipeline/build_identity_packs.py``), ``anny``.

Usage::

    # GarmentMeasurement coefficients -> SOMA PCA coefficients
    python tools/convert/shape_convert.py \\
        --src-model garment_measurement --src-asset assets/identity/identity_garment.npz \\
        --src-coeffs garment.npy --tgt-model soma --output soma.npy

    # SOMA -> MHR
    python tools/convert/shape_convert.py \\
        --src-model soma --src-coeffs soma.npy \\
        --tgt-model mhr --tgt-asset assets/identity/identity_mhr.npz --output mhr.npy

``--soma-asset`` is optional: without it the rig is built from the resolved
upstream assets (NVIDIA's ``SOMA_neutral.npz`` or a SOMA-JAX archive also work).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKENDS = ["soma", "smpl", "smplx", "smplh", "mhr", "anny", "garment_measurement"]
JACOBIAN_CHUNK = 16     # tangent columns per compiled forward-mode pass


def parse_args():
    p = argparse.ArgumentParser(description="Convert identity coefficients between models")
    p.add_argument("--src-coeffs", required=True, help="source coefficients (.npy), (K,) or (B, K)")
    p.add_argument("--src-model", required=True, choices=BACKENDS)
    p.add_argument("--tgt-model", required=True, choices=BACKENDS)
    p.add_argument("--soma-asset", default=None,
                   help="SOMA asset: NVIDIA's SOMA_neutral.npz or a SOMA-JAX archive "
                        "(default: build from the resolved upstream assets)")
    p.add_argument("--src-asset", default=None, help="identity data for the source backend")
    p.add_argument("--tgt-asset", default=None, help="identity data for the target backend")
    p.add_argument("--num-coeffs", type=int, default=None,
                   help="target coefficients to fit (default: all the target model has)")
    p.add_argument("--iters", type=int, default=8, help="max Gauss-Newton iterations")
    p.add_argument("--prior", type=float, default=0.0,
                   help="optional Tikhonov weight pulling coefficients toward zero, relative "
                        "to the mean curvature of the fit. Plain least squares by default "
                        "(as upstream's converter); a target shape space that cannot "
                        "represent the source well can drift to large coefficients — the "
                        "printed max |coeff| shows it — and a small prior (1e-3) curbs that")
    p.add_argument("--no-repose", action="store_true",
                   help="compare raw rest shapes instead of bind-pose ones")
    p.add_argument("--output", required=True, help="output (.npy) for the converted coefficients")
    return p.parse_args()


def forward_jacobian(fn, num_coeffs: int, chunk: int = JACOBIAN_CHUNK):
    """``c -> d fn(c) / dc`` by forward mode, ``chunk`` tangent columns at a time.

    One ``jax.jacfwd`` over the whole skeleton fit is a single very large
    batched program; the XLA in the oldest supported JAX (0.4.26) segfaults
    compiling it. Fixed-size chunks (the last one zero-padded) compile once and
    cost the same arithmetic.
    """
    import jax
    import jax.numpy as jnp

    jvps = jax.jit(lambda c, tangents: jax.vmap(lambda t: jax.jvp(fn, (c,), (t,))[1])(tangents))
    n_chunks = -(-num_coeffs // chunk)
    basis = jnp.zeros((n_chunks * chunk, num_coeffs), jnp.float32)
    basis = basis.at[jnp.arange(num_coeffs), jnp.arange(num_coeffs)].set(1.0)

    def jacobian(c):
        cols = [jvps(c, basis[i * chunk:(i + 1) * chunk]) for i in range(n_chunks)]
        return jnp.concatenate(cols, axis=0)[:num_coeffs].T          # (M, K)
    return jacobian


def fit_coefficients(body_fn, target, num_coeffs: int, iters: int = 8, prior: float = 0.0):
    """Damped Gauss-Newton: ``argmin_c ||body_fn(c) - target||^2 + mu ||c||^2``.

    Args:
        body_fn: ``(K,) -> (V, 3)`` differentiable body of the target model.
        target: ``(V, 3)`` body to reproduce.
        num_coeffs: ``K``.
        iters: maximum iterations.
        prior: ``mu`` relative to the mean diagonal of ``J^T J`` at the start.

    Returns:
        ``(coeffs (K,), rms_error)`` with the RMS vertex error of the fit.
    """
    import jax
    import jax.numpy as jnp

    target = jnp.asarray(target)
    residual = jax.jit(lambda c: (body_fn(c) - target).reshape(-1))
    jacobian = forward_jacobian(lambda c: body_fn(c).reshape(-1), num_coeffs)

    c = jnp.zeros((num_coeffs,), jnp.float32)
    eye = jnp.eye(num_coeffs, dtype=jnp.float32)
    mu = None
    r = residual(c)
    cost = float(r @ r)
    lam = 1e-6
    for _ in range(iters):
        J = jacobian(c)
        JtJ = J.T @ J
        scale = float(jnp.trace(JtJ)) / num_coeffs + 1e-12
        if mu is None:
            mu = prior * scale
        g = J.T @ r + mu * c
        step = jnp.linalg.solve(JtJ + (mu + lam * scale) * eye, -g)
        r_new = residual(c + step)
        cost_new = float(r_new @ r_new)
        total, total_new = cost + mu * float(c @ c), cost_new + mu * float((c + step) @ (c + step))
        if total_new < total:
            converged = total - total_new < 1e-10 * max(total, 1e-30)
            c, r, cost = c + step, r_new, cost_new
            lam = max(lam / 10.0, 1e-9)
            if converged:
                break
        else:
            lam *= 10.0
    return np.asarray(c), float(np.sqrt(cost / target.shape[0]))


def main():
    args = parse_args()
    import jax.numpy as jnp
    from _common import load_layer

    src_coeffs = np.load(args.src_coeffs).astype(np.float32)
    if src_coeffs.ndim == 1:
        src_coeffs = src_coeffs[None]

    src = load_layer(args.soma_asset, identity_model_type=args.src_model,
                     identity_model_path=args.src_asset)
    tgt = load_layer(args.soma_asset, identity_model_type=args.tgt_model,
                     identity_model_path=args.tgt_asset)
    K = int(tgt.identity_model.n_betas if args.num_coeffs is None else args.num_coeffs)
    if K <= 0:
        sys.exit(f"target model {args.tgt_model!r} has no identity coefficients to fit")
    repose = not args.no_repose

    def body(layer, coeffs):
        return layer.prepare_identity(coeffs, repose_to_bind_pose=repose)[0]

    sources = np.asarray(body(src, jnp.asarray(src_coeffs)))          # (B, V, 3)
    out = np.zeros((src_coeffs.shape[0], K), np.float32)
    for i, target in enumerate(sources):
        out[i], rms = fit_coefficients(lambda c: body(tgt, c[None])[0], target, K,
                                       args.iters, args.prior)
        start = float(np.sqrt(np.mean(np.sum(
            (np.asarray(body(tgt, jnp.zeros((1, K))))[0] - target) ** 2, axis=-1))))
        print(f"  [{i}] RMS vertex distance to the source body: "
              f"{start * 1e3:.2f} mm (target at zero) -> {rms * 1e3:.2f} mm (fitted), "
              f"max |coeff| {np.abs(out[i]).max():.2f}")

    np.save(args.output, out)
    print(f"Saved converted coefficients to: {args.output}")
    print(f"  Source: {args.src_model} {src_coeffs.shape}")
    print(f"  Target: {args.tgt_model} {out.shape}")


if __name__ == "__main__":
    main()
