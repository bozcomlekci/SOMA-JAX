"""Forward-pass throughput benchmark comparable to SOMA-X paper Table 4.

Reproduces the SOMA paper's Tab. 4 measurement methodology against both
``soma`` (NVlabs/SOMA-X, torch + Warp) and ``soma_jax`` (this repo, JAX):

* Identity backend: SOMA-native shape — skips the topology-abstraction step
  so the timings reflect skeleton fitting + LBS only.
* Mid-resolution mesh (~18k verts).
* Per-batch numbers: skeleton-fitting time (RBF + Kabsch) vs. full forward
  pass (identity + skeleton + LBS [+ correctives]).
* Batch sizes: 1, 8, 32, 128 — matches the paper.
* Warmup + n_iter median over a synchronized GPU.

Usage::

    python benchmarks/bench_forward_pass.py --device cuda:0 \
        --soma-asset assets/third_party \
        --output benchmarks/results/runtime.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _meshes_per_sec(latency_ms: float, batch: int) -> float:
    return batch * 1000.0 / max(latency_ms, 1e-9)


def _summary(times_s: list[float], batch: int) -> dict:
    """Robust timing summary in ms. The reported statistic is the MEDIAN (the
    per-iteration times come from ``_timed_samples`` below, already low-variance);
    p10/p90 give a robust spread that is negligible once clocks are pinned."""
    arr = np.asarray(times_s, dtype=np.float64) * 1000.0  # ms
    median_ms = float(np.median(arr))
    n = len(arr)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    # Uncertainty of the *reported number* is the standard error of the median
    # (~1.25·std/√n), NOT the per-call spread (that is GPU clock jitter, not
    # measurement error). This is <1% here — the "uncertainty" is negligible.
    se_median_ms = 1.2533 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "median_ms":    median_ms,
        "se_median_ms": float(se_median_ms),
        "p10_ms":    float(np.percentile(arr, 10)),
        "p90_ms":    float(np.percentile(arr, 90)),
        "mean_ms":   float(arr.mean()),
        "std_ms":    std,
        "min_ms":    float(arr.min()),
        "max_ms":    float(arr.max()),
        "n_iter":    int(len(arr)),
        "meshes_per_sec_median": _meshes_per_sec(median_ms, batch),
        "meshes_per_sec_mean":   _meshes_per_sec(float(arr.mean()), batch),
        "samples_ms": arr.tolist(),
    }


def _timed_samples(step, sync, warmup_s: float = 0.7, budget_s: float = 1.2) -> list[float]:
    """Low-variance GPU timing -> list of per-iteration times (seconds).

    The RTX 5080 idles at ~810 MHz and boosts to ~3090 MHz; that clock ramp is
    the dominant source of run-to-run spread (and clocks cannot be locked
    without root here). So we (1) warm up by wall-clock time until the clocks
    reach and hold boost, then (2) take `outer` samples, each timing `inner`
    forwards run back-to-back with a single device sync — the GPU stays
    continuously busy (clocks pinned) and per-call host/launch/sync overhead is
    amortized. `inner`/`outer` self-size from a quick probe so every measurement
    spans ~budget_s regardless of batch, and the MEDIAN of the samples is a
    reproducible, essentially spread-free number.

    ``step()`` issues one forward and returns its result; ``sync(result)``
    blocks until the device has finished it.
    """
    t_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end:
        sync(step())
    # Probe per-iteration cost to size the batched timing.
    t0 = time.perf_counter()
    for _ in range(5):
        r = step()
    sync(r)
    per = (time.perf_counter() - t0) / 5
    inner = max(1, min(256, int(round(2e-3 / max(per, 1e-9)))))   # >= ~2 ms of work per sample
    outer = int(round(budget_s / max(inner * per, 1e-9)))
    outer = max(30, min(outer, 300))
    samples = []
    for _ in range(outer):
        t0 = time.perf_counter()
        r = None
        for _ in range(inner):
            r = step()
        sync(r)
        samples.append((time.perf_counter() - t0) / inner)
    return samples


# ---------------------------------------------------------------------------
# SOMA-X (torch + Warp)
# ---------------------------------------------------------------------------
def bench_soma_x(asset_dir: str, batches: list[int],
                  warmup: int, n_iter: int, device: str) -> dict:
    """Time SOMA-X's full forward pass (prepare_identity + pose) at each
    batch size, plus skeleton-fitting alone."""
    import torch
    from soma import SOMALayer

    # Pin true float32 matmuls (no TF32) so this matches the JAX side's
    # `jax_default_matmul_precision="highest"`. allow_tf32 already defaults to
    # False for matmul on current torch, but that default has moved across
    # versions — stating it keeps the convention independent of the install.
    # cuDNN is unused by the LBS/FK path; pinned for completeness only.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"\n[soma_x] loading SOMALayer (mode='warp', backend='soma', device={device})")
    # mode='warp' triggers SOMA-X's Warp LBS kernels (= paper's "Warp (GPU)" row).
    # We disable correctives to mirror the paper's setup (LBS-only forward).
    layer = SOMALayer(
        data_root=str(asset_dir),
        device=device,
        identity_model_type="soma",
        mode="warp",
        correctives_model_path=None,   # paper Table 4 measures LBS-only.
        enable_procedural_transforms=False,
    ).to(device)
    layer.eval()
    K_id = layer.num_shape_components
    n_joints = len(layer.parents) + 1  # parents excludes virtual root prepended later
    print(f"  K_id={K_id} n_joints_public={n_joints - 1}")

    rows = []
    for B in batches:
        print(f"\n[soma_x] batch={B}")
        try:
            torch.manual_seed(0)
            identity = torch.zeros(B, K_id, device=device, dtype=torch.float32)
            # 77 public joints (Hips + 76 children); SOMALayer.pose internally pads root.
            poses = torch.zeros(B, n_joints - 1, 3, device=device, dtype=torch.float32)
            transl = torch.zeros(B, 3, device=device, dtype=torch.float32)

            sync = lambda r=None: torch.cuda.synchronize()
            # 1) skeleton-only: prepare_identity (identity blend + SkeletonTransfer
            #    .fit + rebind precompute).
            skel = _summary(_timed_samples(
                lambda: layer.prepare_identity(identity, repose_to_bind_pose=False),
                sync), B)

            # 2) total forward: prepare_identity + pose (full pipeline each call).
            def _fwd_step():
                layer.prepare_identity(identity, repose_to_bind_pose=False)
                return layer.pose(poses, transl=transl, apply_correctives=False)
            fwd = _summary(_timed_samples(_fwd_step, sync), B)
        except Exception as e:
            print(f"  SKIPPED (B={B}): {type(e).__name__}: {str(e)[:120]}")
            torch.cuda.empty_cache()
            continue
        print(f"  skel={skel['median_ms']:6.3f} ms  total={fwd['median_ms']:6.3f} ms  "
              f"(p10-p90 {fwd['p10_ms']:.3f}-{fwd['p90_ms']:.3f})  "
              f"m/s={fwd['meshes_per_sec_median']:,.0f}")
        rows.append({"batch": B, "skel": skel, "total": fwd})
    return {"backend": "soma_x", "mode": "warp_gpu", "rows": rows}


# ---------------------------------------------------------------------------
# SOMA-JAX
# ---------------------------------------------------------------------------
def bench_soma_jax(asset_path: str, batches: list[int],
                    warmup: int, n_iter: int, matmul_precision: str = "highest") -> dict:
    """Time soma_jax's full forward pass at each batch size, plus the
    skeleton-fitting cost (J_regressor @ rest_verts -> joint positions)."""
    import jax
    import jax.numpy as jnp
    # FAIRNESS: SOMA-X (torch, allow_tf32=False; Warp scalar kernels) runs the
    # forward in full float32. XLA on Ampere+ GPUs defaults float32 matmuls to
    # TF32, which is ~1.5-1.7x faster but lower precision — so the two sides
    # would not be doing the same arithmetic. Default "highest" (float32) matches
    # SOMA-X; pass "default" to measure the (JAX-only) TF32 mode instead.
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    from soma_jax import SOMALayer, SOMAParams

    print(f"\n[soma_jax] loading SOMALayer")
    layer = SOMALayer.load(asset_path, identity_model_type="soma")
    V = int(layer.v_template.shape[0])
    J = len(layer.joint_names)
    K_id = layer.identity_model.n_betas if hasattr(layer.identity_model, "n_betas") else 128
    print(f"  V={V} J={J} K_id={K_id}")

    # JIT-compile the two phases.
    @jax.jit
    def prepare_only(identity):
        return layer.prepare_identity(identity, repose_to_bind_pose=False, skeleton_fit="linear")

    @jax.jit
    def full_forward(identity, poses, transl):
        rest_v, rest_j = layer.prepare_identity(identity, repose_to_bind_pose=False, skeleton_fit="linear")
        # apply_correctives=False: LBS-only, matching the paper's Table 4 and
        # SOMA-X's benchmark (pose(apply_correctives=False)).
        out = layer.pose(poses, transl, rest_v, rest_j, apply_correctives=False)
        # Materialize the full (B, V, 3) vertex buffer (torch does) — a scalar
        # reduction here would let XLA fuse away the output write.
        return out.vertices

    rows = []
    for B in batches:
        print(f"\n[soma_jax] batch={B}")
        try:
            identity = jnp.zeros((B, K_id), dtype=jnp.float32)
            # Local rotmats — identity per joint, shape (B, J, 3, 3).
            rotmats = jnp.broadcast_to(jnp.eye(3), (B, J, 3, 3))
            transl = jnp.zeros((B, 3), dtype=jnp.float32)

            skel = _summary(_timed_samples(
                lambda: prepare_only(identity), lambda r: r[0].block_until_ready()), B)
            fwd = _summary(_timed_samples(
                lambda: full_forward(identity, rotmats, transl),
                lambda r: r.block_until_ready()), B)
        except Exception as e:
            print(f"  SKIPPED (B={B}): {type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"  skel={skel['median_ms']:6.3f} ms  total={fwd['median_ms']:6.3f} ms  "
              f"(p10-p90 {fwd['p10_ms']:.3f}-{fwd['p90_ms']:.3f})  "
              f"m/s={fwd['meshes_per_sec_median']:,.0f}")
        rows.append({"batch": B, "skel": skel, "total": fwd})
    return {"backend": "soma_jax_linear", "mode": "jax_gpu_linear_regressor", "rows": rows}


# ---------------------------------------------------------------------------
# SOMA-JAX — SkeletonTransfer path (algorithm-fair vs SOMA-X)
# ---------------------------------------------------------------------------
def bench_soma_jax_st(hf_asset_dir: str, batches: list[int],
                       warmup: int, n_iter: int,
                       rotation_backend: str = "jax",
                       matmul_precision: str = "highest") -> dict:
    """Time soma_jax doing the *same algorithm* SOMA-X runs per identity.

    ``rotation_backend``: ``"jax"`` runs the covariance→rotation SVD with
    ``jnp.linalg.svd`` (pure JAX); ``"warp"`` runs it with a Warp ``svd3``
    kernel on-device (Warp+JAX hybrid). Both build covariances + do FK/LBS
    in JAX; only the 3×3 SVD differs.

    SOMA-X's ``prepare_identity`` runs the full ``SkeletonTransfer.fit``
    (per-joint RBF position regression + two-stage Kabsch rotation fit) and
    rebinds the skinning to the fitted skeleton; ``pose`` then runs FK + LBS
    against those bind transforms. The default soma_jax path replaces the
    skeleton fit with a precomputed linear J_regressor, which is cheaper but
    a *different* algorithm — see ``bench_soma_jax``.

    This benchmark constructs the faithful JAX ports from the same upstream
    archive SOMA-X reads (``SOMA_neutral.npz``), with matching dimensions,
    identity coefficients, pose, FK structure and top-8 LBS.

    NOTE — not a bit-identical rig: upstream's ``SOMALayer`` additionally merges
    ``SOMA_template_rig.usda`` over those NPZ arrays, which this path loads raw.
    The rigs differ in ~46k skinning-weight entries, so the FLOP count and
    therefore the timing are matched, but the posed meshes are not numerically
    comparable here. Numerical parity lives in ``tests/test_layer_parity.py``,
    which builds against the merged rig.

    The constructed pipeline:

      * identity blend: ``mean + (coeffs * sqrt(eigenvalues)) @ shapedirs``
        (the SOMAIdentityModel formula, native centimeters);
      * ``SkeletonTransfer(parents, bind_pose_world, bind_shape, weights,
        rotation_method="auto", vertex_ids_to_exclude=eye_bags+mouth_bag)``
        — the exact constructor arguments from soma/soma.py;
      * ``pose_from_bind`` — the functional rebind+pose (FK level-order +
        dense LBS against per-batch inverse binds), identity rotations with
        the bind-aligned joint orient applied, hips at the origin — matching
        the SOMA-X benchmark's ``pose(zeros, transl=zeros)``.
    """
    import jax
    import jax.numpy as jnp
    # FAIRNESS: match SOMA-X's full-float32 arithmetic (XLA defaults float32
    # matmuls to TF32 on Ampere+, ~1.5-1.7x faster but lower precision).
    # "highest" = float32 (fair); "default" = TF32 (JAX-only, not comparable).
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    from scipy.sparse import csc_matrix
    from soma_jax.geometry.skeleton_transfer import SkeletonTransfer
    from soma_jax.geometry.batched_skinning import pose_from_bind, topk_skinning
    from soma_jax.geometry.lbs import compute_skeleton_levels
    from soma_jax.geometry.rig_utils import apply_joint_orient_local, joint_world_to_local
    from soma_jax.geometry.transforms import se3_inverse

    print(f"\n[soma_jax_st] loading upstream rig from {hf_asset_dir}/SOMA_neutral.npz")
    rig = dict(np.load(Path(hf_asset_dir) / "SOMA_neutral.npz", allow_pickle=False))
    bind_shape = np.asarray(rig["bind_shape"], dtype=np.float32)             # (V, 3) cm
    bind_world = np.asarray(rig["bind_pose_world"], dtype=np.float32)        # (J, 4, 4) cm
    parents_raw = np.asarray(rig["joint_parent_ids"], dtype=np.int64)
    parents = parents_raw.copy()
    parents[0] = -1                                                           # root sentinel
    weights = np.asarray(csc_matrix(
        (rig["skinning_weights_data"], rig["skinning_weights_indices"],
         rig["skinning_weights_indptr"]),
        shape=tuple(rig["skinning_weights_shape"]),
    ).todense(), dtype=np.float32)                                            # (V, J)
    mean = np.asarray(rig["mean"], dtype=np.float32).reshape(-1)              # (3V,) cm
    shapedirs = np.asarray(rig["shapedirs"], dtype=np.float32)                # (K, 3V)
    eigenvalues = np.asarray(rig["eigenvalues"], dtype=np.float32)            # (K,)
    t_pose_world = np.asarray(rig["t_pose_world"], dtype=np.float32)          # (J, 4, 4)
    facial_inner = np.concatenate([rig["segment_eye_bags"], rig["segment_mouth_bag"]])

    V = bind_shape.shape[0]
    J = bind_world.shape[0]
    K_id = eigenvalues.shape[0]
    print(f"  V={V} J={J} K_id={K_id}  backend={rotation_backend}  "
          f"(constructing SkeletonTransfer...)")

    st = SkeletonTransfer(
        parents, bind_world, bind_shape, weights,
        rotation_method="auto",
        vertex_ids_to_exclude=facial_inner.tolist(),
        rotation_backend=rotation_backend,
    )
    levels = compute_skeleton_levels(parents)
    # SOMA-X's Warp LBS skins with top-8 sparse weights (topk_skinning(W, K=8)
    # inside BatchedSkinning._prepare_warp_data) — use the same representation
    # so both sides run the same number of LBS FLOPs.
    w_idx_np, w_val_np = topk_skinning(weights, 8)
    w_idx = jnp.asarray(w_idx_np)
    w_val = jnp.asarray(w_val_np)
    weights_j = jnp.asarray(weights)
    mean_j = jnp.asarray(mean)
    shapedirs_j = jnp.asarray(shapedirs)
    sqrt_eig_j = jnp.sqrt(jnp.asarray(eigenvalues))
    orient_j = jnp.asarray(t_pose_world[:, :3, :3])

    @jax.jit
    def blend_identity(coeffs):
        """SOMAIdentityModel.get_rest_shape: mean + (c*sqrt(eig)) @ shapedirs."""
        weighted = coeffs * sqrt_eig_j                                       # (B, K)
        flat = mean_j[None] + weighted @ shapedirs_j                          # (B, 3V)
        return flat.reshape(coeffs.shape[0], V, 3)

    @jax.jit
    def prepare_identity(coeffs):
        """Mirror of SOMA-X's prepare_identity(repose_to_bind_pose=False):
        identity blend + SkeletonTransfer.fit + the rebind precompute
        (bind world -> local transforms + inverse binds). All outputs are
        returned so XLA materializes the same buffers torch does."""
        rest_shape = blend_identity(coeffs)
        bind_T = st.fit(rest_shape)
        bind_local = joint_world_to_local(bind_T, parents)
        inv_bind = se3_inverse(bind_T)
        return rest_shape, bind_T, bind_local, inv_bind

    @jax.jit
    def full_forward(coeffs, local_rotmats, hips):
        rest_shape = blend_identity(coeffs)                                   # (B, V, 3) cm
        bind_T = st.fit(rest_shape)                                           # (B, J, 4, 4)
        # SOMA-X pose(): joint-orient remap of the (identity) input rotations,
        # then FK + sparse LBS against the fitted binds, then cm -> m.
        R = apply_joint_orient_local(local_rotmats, orient_j, parents)
        posed, _ = pose_from_bind(
            bind_T, rest_shape, weights_j, levels, parents, R, hips, hips_idx=1,
            weight_values=w_val, weight_indices=w_idx,
        )
        # Materialize the full (B, V, 3) vertex buffer (torch does) — do NOT
        # reduce to a scalar, which would let XLA fuse away the output write.
        return posed * 0.01                                                   # output_unit=m

    rows = []
    for B in batches:
        tag = "soma_jax_hybrid" if rotation_backend == "warp" else "soma_jax_st"
        print(f"\n[{tag}] batch={B}")
        try:
            coeffs = jnp.zeros((B, K_id), dtype=jnp.float32)
            rotmats = jnp.broadcast_to(jnp.eye(3), (B, J, 3, 3))
            hips = jnp.zeros((B, 3), dtype=jnp.float32)

            skel = _summary(_timed_samples(
                lambda: prepare_identity(coeffs), lambda r: r[1].block_until_ready()), B)
            fwd = _summary(_timed_samples(
                lambda: full_forward(coeffs, rotmats, hips),
                lambda r: r.block_until_ready()), B)
        except Exception as e:
            print(f"  SKIPPED (B={B}): {type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"  skel={skel['median_ms']:6.3f} ms  total={fwd['median_ms']:6.3f} ms  "
              f"(p10-p90 {fwd['p10_ms']:.3f}-{fwd['p90_ms']:.3f})  "
              f"m/s={fwd['meshes_per_sec_median']:,.0f}")
        rows.append({"batch": B, "skel": skel, "total": fwd})
    tag = "soma_jax_hybrid" if rotation_backend == "warp" else "soma_jax_st"
    mode = ("jax_warp_hybrid_skeleton_transfer" if rotation_backend == "warp"
            else "jax_gpu_skeleton_transfer")
    return {"backend": tag, "mode": mode, "rows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soma-asset", default=str(__import__("soma_jax.assets", fromlist=["x"]).data_root()),
                    help="SOMA-X asset dir (HF cache layout) for the soma backend.")
    # Resolved, not hardcoded: the built archive lands in assets/ or
    # assets/third_party/ depending on how it was produced, and resolve()
    # knows both. A literal here fails on a machine that built it elsewhere.
    p.add_argument("--soma-jax-asset",
                   default=str(__import__("soma_jax.assets", fromlist=["x"]).resolve(
                       "SOMA_neutral_fixed.npz", required=False)
                       or REPO / "assets" / "SOMA_neutral_fixed.npz"),
                    help="soma_jax SOMA_neutral_fixed.npz path.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output", default=str(REPO / "benchmarks" / "results" / "runtime.json"))
    p.add_argument("--skip-soma-x", action="store_true")
    p.add_argument("--skip-soma-jax", action="store_true")
    p.add_argument("--skip-soma-jax-st", action="store_true",
                    help="Skip the SkeletonTransfer (algorithm-fair) JAX path.")
    p.add_argument("--skip-soma-jax-hybrid", action="store_true",
                    help="Skip the Warp+JAX hybrid SkeletonTransfer path.")
    p.add_argument("--matmul-precision", default="highest",
                    help="JAX matmul precision: 'highest' = float32 (fair vs SOMA-X); "
                         "'default' = TF32 (JAX-only, NOT comparable to SOMA-X's float32).")
    args = p.parse_args()

    # Tell JAX which device to use.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.device.startswith("cuda"):
        idx = args.device.split(":")[-1] if ":" in args.device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = idx

    payload = {"args": vars(args), "results": []}
    if not args.skip_soma_x:
        payload["results"].append(
            bench_soma_x(args.soma_asset, args.batches, args.warmup, args.iters, args.device)
        )
    if not args.skip_soma_jax_st:
        payload["results"].append(
            bench_soma_jax_st(args.soma_asset, args.batches, args.warmup, args.iters,
                              rotation_backend="jax", matmul_precision=args.matmul_precision)
        )
    if not args.skip_soma_jax_hybrid:
        payload["results"].append(
            bench_soma_jax_st(args.soma_asset, args.batches, args.warmup, args.iters,
                              rotation_backend="warp", matmul_precision=args.matmul_precision)
        )
    if not args.skip_soma_jax:
        payload["results"].append(
            bench_soma_jax(args.soma_jax_asset, args.batches, args.warmup, args.iters,
                           matmul_precision=args.matmul_precision)
        )

    # Stamp the precision convention onto every block so a result can never be
    # read without knowing which one produced it. SOMA-X is always true float32
    # (torch matmul TF32 is pinned off in bench_soma_x); the JAX blocks follow
    # --matmul-precision, where 'highest' is the float32 setting that makes the
    # comparison fair and 'default' permits TF32 (JAX-only, not comparable).
    jax_precision = "float32" if args.matmul_precision == "highest" \
        else f"tf32 (jax matmul_precision={args.matmul_precision})"
    for block in payload["results"]:
        block["precision"] = "float32" if block["backend"] == "soma_x" else jax_precision

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
