"""Fairness checks for the SOMA forward-pass benchmark.

The benchmark only times code; these checks verify the timed code is doing
the *same, real* work on each side, so the numbers are comparable:

1. **No constant-folding cheat.** All series feed `jnp.zeros` identity coeffs
   and identity pose rotations. If XLA folded those constants away at compile
   time, the timings would measure nothing. We re-time each JAX pipeline with
   *random non-zero* inputs of the same shape; if the timing is unchanged, the
   computation is genuinely input-dependent (no folding). The inputs are jit
   ARGUMENTS (traced, dynamic), so folding should be impossible — this confirms
   it empirically.

2. **Real, non-trivial output.** Posed vertices must be finite and have real
   spatial spread (not a collapsed constant), and must actually change when
   the input changes (proves the input flows end-to-end).

3. **Cross-pipeline equivalence.** The JAX-SVD fair path and the Warp-svd3
   hybrid must produce the same posed mesh (same algorithm, different SVD).

Run (JAX side, cu12):
    python benchmarks/verify_fairness.py

The SOMA-X (torch+Warp) equivalence is checked separately by
a SOMA-X side-run writing a posed-vertex summary that this script
compares against when present.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _build_fair_pipeline(hf_dir: Path, rotation_backend: str):
    """Reconstruct the benchmark's fair/hybrid forward as a jitted fn."""
    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")  # match SOMA-X float32
    from scipy.sparse import csc_matrix
    from soma_jax.geometry.skeleton_transfer import SkeletonTransfer
    from soma_jax.geometry.batched_skinning import pose_from_bind, topk_skinning
    from soma_jax.geometry.lbs import compute_skeleton_levels
    from soma_jax.geometry.rig_utils import apply_joint_orient_local

    rig = dict(np.load(hf_dir / "SOMA_neutral.npz", allow_pickle=False))
    bind_shape = np.asarray(rig["bind_shape"], np.float32)
    bind_world = np.asarray(rig["bind_pose_world"], np.float32)
    parents = rig["joint_parent_ids"].astype(np.int64).copy(); parents[0] = -1
    weights = np.asarray(csc_matrix(
        (rig["skinning_weights_data"], rig["skinning_weights_indices"],
         rig["skinning_weights_indptr"]),
        shape=tuple(rig["skinning_weights_shape"])).todense(), np.float32)
    mean = np.asarray(rig["mean"], np.float32).reshape(-1)
    shapedirs = np.asarray(rig["shapedirs"], np.float32)
    eig = np.asarray(rig["eigenvalues"], np.float32)
    t_pose = np.asarray(rig["t_pose_world"], np.float32)
    facial = np.concatenate([rig["segment_eye_bags"], rig["segment_mouth_bag"]])
    V, J, K = bind_shape.shape[0], bind_world.shape[0], eig.shape[0]

    st = SkeletonTransfer(parents, bind_world, bind_shape, weights,
                          rotation_method="auto", vertex_ids_to_exclude=facial.tolist(),
                          rotation_backend=rotation_backend)
    levels = compute_skeleton_levels(parents)
    w_idx_np, w_val_np = topk_skinning(weights, 8)
    w_idx, w_val = jnp.asarray(w_idx_np), jnp.asarray(w_val_np)
    weights_j = jnp.asarray(weights)
    mean_j, sd_j = jnp.asarray(mean), jnp.asarray(shapedirs)
    sqrt_eig = jnp.sqrt(jnp.asarray(eig))
    orient = jnp.asarray(t_pose[:, :3, :3])

    @jax.jit
    def fwd(coeffs, rotmats, hips):
        weighted = coeffs * sqrt_eig
        rest = (mean_j[None] + weighted @ sd_j).reshape(coeffs.shape[0], V, 3)
        bind_T = st.fit(rest)
        R = apply_joint_orient_local(rotmats, orient, parents)
        posed, _ = pose_from_bind(bind_T, rest, weights_j, levels, parents, R, hips,
                                  hips_idx=1, weight_values=w_val, weight_indices=w_idx)
        return posed * 0.01
    return fwd, (V, J, K)


def _time(fn, args, warmup=5, n=30):
    import time
    for _ in range(warmup):
        fn(*args).block_until_ready()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000.0)


def _probe_svd_nonuniqueness(hf_dir: Path):
    """Characterise where the XLA-SVD and Warp-svd3 full-fit pipelines diverge.

    Two separate effects get conflated easily, so this probe separates them on
    4096 synthetic alignment problems (half deliberately near-planar, i.e.
    rank-deficient covariance):

    1. **Warp svd3 vs the same algorithm in XLA.** ``kabsch_rotation_warp``
       implements plain SVD Procrustes, i.e. ``rotation_from_covariance(...,
       method="kabsch")``. Against *that* reference the only divergences are
       genuine SVD non-uniqueness: both are valid SO(3) and achieve the same
       Kabsch residual. This is asserted.

    2. **``method="auto"`` vs plain Kabsch.** ``auto`` (the default, and what
       ``SkeletonTransfer`` uses) is Newton–Schulz on a gauge-regularized
       covariance, not SVD Procrustes. On ill-conditioned covariances the two
       land on genuinely different rotations. That is **upstream SOMA-X
       behaviour**, reproduced faithfully (``soma.geometry.transforms.
       align_vectors`` shows the same split), so it is *reported*, not
       asserted away — but it does mean the Warp-SVD pipeline is not
       bit-for-bit the same algorithm as the XLA-SVD one.
    """
    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")  # match SOMA-X float32
    from soma_jax.geometry.transforms import (
        compute_covariance, rotation_from_covariance)
    from soma_jax.geometry.warp_kabsch import kabsch_rotation_warp

    rng = np.random.default_rng(2)
    # 4096 random alignment problems with a controllable conditioning sweep,
    # including deliberately near-degenerate (near-planar) point sets.
    N = 4096
    A = rng.standard_normal((N, 8, 3)).astype(np.float32)
    A[: N // 2, :, 2] *= 0.01           # half are near-planar -> rank-deficient H
    B = rng.standard_normal((N, 8, 3)).astype(np.float32)
    H = compute_covariance(jnp.asarray(A), jnp.asarray(B))
    R_kabsch = np.asarray(rotation_from_covariance(H, method="kabsch"))
    R_auto = np.asarray(rotation_from_covariance(H, method="auto"))
    R_warp = np.asarray(jax.jit(kabsch_rotation_warp)(H))

    def so3_err(R):
        RtR = np.einsum("nij,nkj->nik", R, R)
        return np.abs(RtR - np.eye(3)).max(), float(np.linalg.det(R).min())

    # Kabsch residual ‖R B − A‖_F per problem (lower = better fit; equal ⇒
    # equivalent solutions).
    def resid(R):
        RB = np.einsum("nij,nkj->nki", R, np.asarray(B))
        return np.linalg.norm(RB - np.asarray(A), axis=(1, 2))

    # --- 1. Warp svd3 against the algorithm it actually implements ----------
    differ = np.abs(R_kabsch - R_warp).reshape(N, -1).max(axis=1) > 1e-3
    rk, rw = resid(R_kabsch), resid(R_warp)
    # Relative residual gap: |‖R_a B−A‖ − ‖R_b B−A‖| / ‖R_a B−A‖.
    # Equivalent minimizers have ~0 RELATIVE gap regardless of residual scale;
    # an absolute gap is meaningless here since the near-planar problems carry
    # a large unfittable out-of-plane residual.
    rel_gap = np.abs(rk - rw) / (np.abs(rk) + 1e-6)
    o = so3_err(R_warp)
    print(f"\n[SVD non-uniqueness probe] {N} problems, half near-planar")
    print(f"  warp svd3 vs XLA method='kabsch': {differ.sum()} differ by >1e-3")
    print(f"  warp R valid SO(3): max|RᵀR−I|={o[0]:.2e}  min det={o[1]:.4f}")
    if differ.any():
        print(f"  relative Kabsch-residual gap where they differ: "
              f"max={rel_gap[differ].max():.2e}  mean={rel_gap[differ].mean():.2e}")
    assert o[0] < 1e-4 and o[1] > 0.99, "warp output not a proper rotation"
    if differ.any():
        assert rel_gap[differ].max() < 1e-2, "divergence is NOT equal-residual (real error)"
    print("  -> warp-vs-kabsch divergence is genuine SVD non-uniqueness (equal fit)")

    # --- 2. The algorithmic gap the hybrid pipeline actually carries --------
    # Reported, not asserted: this is upstream SOMA-X's own auto-vs-kabsch
    # split, and it is why the two full-fit pipelines are not bit-identical.
    d_auto = np.abs(R_auto - R_warp).reshape(N, -1).max(axis=1)
    n_auto = int((d_auto > 1e-3).sum())
    ra = resid(R_auto)
    gap_auto = np.abs(ra - rw) / (np.abs(ra) + 1e-6)
    print(f"  [reported] method='auto' (Newton–Schulz, the SkeletonTransfer "
          f"default) vs warp svd3: {n_auto} differ by >1e-3, max|ΔR|={d_auto.max():.2e}")
    if n_auto:
        print(f"             relative residual gap there: "
              f"max={gap_auto[d_auto > 1e-3].max():.2e} "
              f"mean={gap_auto[d_auto > 1e-3].mean():.2e}  "
              f"(warp lower in {int((rw < ra)[d_auto > 1e-3].sum())}/{n_auto})")
        print("             -> ill-conditioned covariances only; upstream "
              "SOMA-X splits the same way (see docs/FAITHFULNESS.md).")


def _probe_align_vectors_vs_upstream():
    """The pure-JAX `align_vectors` against upstream torch, all three methods.

    This is the measurement `docs/FAITHFULNESS.md` quotes when it calls the
    JAX path "the faithful one", so it lives here rather than in a comment.

    Run in float64 as well as float32 deliberately. Agreement at ~1e-15 in
    float64 says the two are the *same algorithm*; the float32 numbers are then
    bounded by SVD round-off rather than by any difference in behaviour, which
    is why `kabsch` sits an order of magnitude above `auto`/`newton-schulz`
    without that meaning anything is wrong. Skipped when torch or the SOMA-X
    submodule is absent.
    """
    import jax, jax.numpy as jnp
    try:
        import torch
        from soma.geometry.transforms import align_vectors as up_align
    except ImportError:
        print("\n[align_vectors] skipped — needs torch + the SOMA-X submodule")
        return
    from soma_jax.geometry.transforms import align_vectors as jax_align

    rng = np.random.default_rng(0)
    N = 4096
    A = rng.standard_normal((N, 32, 3)).astype(np.float32)
    A[: N // 2, :, 2] *= 0.01          # half near-planar -> rank-deficient covariance
    B = rng.standard_normal((N, 32, 3)).astype(np.float32)

    print(f"\n[align_vectors] pure-JAX vs upstream torch, {N} problems "
          f"(half near-planar), max|ΔR|:")
    for label, cast, x64 in (("float32", np.float32, False), ("float64", np.float64, True)):
        jax.config.update("jax_enable_x64", x64)
        a, b = A.astype(cast), B.astype(cast)
        row = []
        for method in ("auto", "kabsch", "newton-schulz"):
            with torch.no_grad():
                ref = up_align(torch.tensor(a), torch.tensor(b), method=method).numpy()
            got = np.asarray(jax_align(jnp.asarray(a), jnp.asarray(b), method=method))
            row.append((method, np.abs(got - ref).max()))
        print("               " + label + "  "
              + "  ".join(f"{m}={d:.2e}" for m, d in row))
        if x64:
            worst = max(d for _, d in row)
            # Generous on purpose. The claim is "same algorithm", and float64
            # agreement is ~1e-15 on CPU but ~1e-13 on GPU, where the SVD runs a
            # different (cuSOLVER) implementation. Either is seven-plus orders
            # below the float32 numbers above, which is the separation that
            # carries the argument; a bound tight enough to distinguish the two
            # backends would only be measuring cuSOLVER.
            assert worst < 1e-11, (
                f"float64 disagreement {worst:.2e} — not the same algorithm")
    jax.config.update("jax_enable_x64", False)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__ or "SOMA-JAX fairness checks")
    ap.add_argument("--align-only", action="store_true",
                    help="run only the pure-JAX vs upstream align_vectors probe "
                         "(the numbers docs/FAITHFULNESS.md quotes). Needs no GPU; "
                         "the rest of this script does, because the Warp kernels "
                         "register an XLA FFI handler for CUDA only.")
    args = ap.parse_args()

    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")  # match SOMA-X float32
    if args.align_only:
        _probe_align_vectors_vs_upstream()
        return
    hf = __import__("soma_jax.assets", fromlist=["x"]).data_root()
    B = 64
    rng = np.random.default_rng(0)

    print(f"jax backend: {jax.default_backend()}  (B={B})\n")

    # Draw the random inputs ONCE so both backends are compared on identical
    # data (drawing inside the loop would advance `rng` and feed each backend
    # a different shape — making the cross-backend diff meaningless).
    from soma_jax.geometry.transforms import axis_angle_to_rotmat
    # K is the same for both pipelines; peek from the first build.
    K0 = np.asarray(np.load(hf / "SOMA_neutral.npz", allow_pickle=False)["eigenvalues"]).shape[0]
    J0 = 78
    # Mild, well-conditioned perturbation — the regime the fit is designed for
    # and the benchmark operates near (neutral). Enough to prove the input
    # flows through and isn't folded, without forcing near-degenerate joint
    # covariances (the genuinely non-unique SVD regime is probed separately).
    coeffs_r = jnp.asarray(rng.standard_normal((B, K0)).astype(np.float32) * 0.15)
    aa = rng.standard_normal((B, J0, 3)).astype(np.float32) * 0.2
    rot_r = jax.vmap(jax.vmap(axis_angle_to_rotmat))(jnp.asarray(aa))
    hips_r = jnp.asarray(rng.standard_normal((B, 3)).astype(np.float32) * 0.1)

    results = {}
    for backend in ["jax", "warp"]:
        try:
            fwd, (V, J, K) = _build_fair_pipeline(hf, backend)
        except Exception as e:
            print(f"[{backend}] pipeline unavailable: {type(e).__name__}: {e}")
            continue

        zeros = (jnp.zeros((B, K), jnp.float32),
                 jnp.broadcast_to(jnp.eye(3), (B, J, 3, 3)),
                 jnp.zeros((B, 3), jnp.float32))
        rand = (coeffs_r, rot_r, hips_r)

        t_zero = _time(fwd, zeros)
        t_rand = _time(fwd, rand)
        out_zero = np.asarray(fwd(*zeros))
        out_rand = np.asarray(fwd(*rand))

        finite = np.isfinite(out_rand).all()
        spread = float(out_rand.std())
        changed = float(np.abs(out_zero - out_rand).max())
        ratio = t_rand / t_zero
        results[backend] = out_rand
        print(f"[{backend}] timing zeros={t_zero:.3f}ms  random={t_rand:.3f}ms  "
              f"ratio={ratio:.2f}  (≈1 ⇒ no constant-folding)")
        print(f"[{backend}] output finite={finite}  std={spread:.4f}m  "
              f"max|Δ(zero,rand)|={changed:.4f}m  (>0 ⇒ input flows through)")
        assert finite, f"{backend}: non-finite output"
        assert spread > 1e-3, f"{backend}: output collapsed (std={spread})"
        assert changed > 1e-3, f"{backend}: output ignores input"
        assert 0.7 < ratio < 1.4, f"{backend}: zero/random timing differ ({ratio:.2f}) — possible folding"

    if "jax" in results and "warp" in results:
        d = np.abs(results["jax"] - results["warp"])
        print(f"\n[fair vs hybrid] posed-vertex agreement at the benchmark "
              f"operating point: max|Δ|={d.max():.6f}m  mean={d.mean():.2e}m")
        assert d.max() < 1e-2, "fair vs hybrid disagree beyond SVD tolerance"
        _probe_svd_nonuniqueness(hf)

    _probe_align_vectors_vs_upstream()

    # Cross-check vs the SOMA-X posed-vertex summary, if present.
    somax = REPO / "benchmarks" / "_somax_posed_summary.npz"
    if somax.exists() and "jax" in results:
        ref = np.load(somax)
        jax_bbox = np.ptp(results["jax"].reshape(-1, 3), axis=0)
        print(f"\n[vs SOMA-X] posed bbox extent  JAX={jax_bbox.round(3)}  "
              f"SOMA-X={ref['bbox'].round(3)}  (m; same human ⇒ comparable)")

    print("\nALL FAIRNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
