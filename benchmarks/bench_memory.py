"""Peak GPU-memory benchmark for the SOMA forward pass (RTX 5080).

Companion to ``bench_forward_pass.py`` — same four pipelines, same LBS-only
SOMA-native setup, but it measures **peak GPU memory** vs batch size.

One fresh subprocess per (method, batch) so each number is an absolute
footprint with no allocator carry-over between batch sizes.

Single metric — ``peak_mib``: the high-water mark of **live device bytes the
process actually demands**, plus the fixed CUDA-context overhead.

    peak_mib = context_mib + peak over time of (framework_live + warp_live)

Nothing here polls. Every byte is counted where it is allocated:

* **XLA** — ``memory_stats()["bytes_in_use"]`` / ``["peak_bytes_in_use"]``,
  counters the BFC allocator maintains on every alloc and free.
* **PyTorch** — ``torch.cuda.memory_allocated()`` /
  ``max_memory_allocated()``, likewise exact.
* **NVIDIA Warp** — a counting allocator installed with
  ``wp.set_device_allocator`` that *wraps and delegates to* the allocator Warp
  was already using. It changes no allocation policy (same mempool setting,
  same underlying calls); it only records sizes. This is what makes SOMA-X's
  column honest: SOMA-X runs with Warp's mempool disabled, so its Warp buffers
  are plain ``cudaMalloc`` and are invisible to ``torch.cuda`` counters — and
  they scale with batch, so ignoring them understates SOMA-X badly.
* **CUDA context** — measured once with a single ``cuMemGetInfo`` read before
  any model data is allocated. One read, not a poll.

Why not nvidia-smi: it reports what a process has *reserved* from the driver,
so a pooling allocator inflates it — and it has to be sampled, so it both
jitters and can miss short-lived peaks. Reservation is an allocator policy
artifact (XLA's BFC pool grows in coarse power-of-two blocks it never
releases), not a statement about how much memory the pipeline needs.

**Combining two allocators.** ``peak(a + b) != peak(a) + peak(b)``, so the
combined high-water is sampled at every Warp alloc/free event (reading the
framework's live-bytes counter at that instant) and once per iteration. Since
the largest transient buffers in this workload are Warp's own, the peak
coincides with a Warp event and the sampled value is tight. The per-iteration
framework-only peak is also folded in, so a torch/XLA-only maximum can never be
missed. ``peak_upper_mib`` records the pessimistic ``framework_peak +
warp_peak`` bound for comparison; when it sits close to ``peak_mib`` the
estimate is well determined.

Usage (driven by ``run_memory.sh``)::

    python bench_memory.py --method fair --batch 256 --out results/memory.jsonl
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

MIB = 1024.0 ** 2


def _device_used_mib() -> float | None:
    """Device memory in use, read from the CUDA driver.

    Used only to size the fixed CUDA-context overhead, twice per run. Two
    reads — this is not a poller.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return (total.value - free.value) / MIB
    except Exception:
        return None


def _measure_context_mib(framework_live) -> float:
    """Fixed per-process GPU overhead: CUDA context + framework runtime.

    Called once, after the backend has created its context but before any model
    data is loaded, and net of whatever the framework has already allocated. It
    does not scale with batch size; it is the y-intercept every pipeline pays
    just for being on the GPU.
    """
    used = _device_used_mib()
    if used is None:
        return 0.0
    return max(0.0, used - framework_live() / MIB)


class _WarpCounter:
    """Counting wrapper around Warp's existing allocator.

    Implements Warp's ``Allocator`` protocol by delegating to the allocator the
    device was already using, so allocation *behaviour* is untouched — only
    sizes are recorded. ``on_change`` is called after every alloc/free so the
    caller can sample the combined footprint at exactly the moments it moves.
    """

    def __init__(self, inner, on_change=None):
        self._inner = inner
        self._on_change = on_change
        self._sizes: dict[int, int] = {}
        self.live = 0
        self.peak = 0

    def allocate(self, size_in_bytes: int) -> int:
        ptr = self._inner.allocate(size_in_bytes)
        self._sizes[ptr] = size_in_bytes
        self.live += size_in_bytes
        if self.live > self.peak:
            self.peak = self.live
        if self._on_change is not None:
            self._on_change()
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        self._inner.deallocate(ptr, size_in_bytes)
        self.live -= self._sizes.pop(ptr, size_in_bytes)
        if self._on_change is not None:
            self._on_change()


def _install_warp_counter(on_change=None) -> _WarpCounter | None:
    """Wrap the current Warp allocator on cuda:0. Returns None if Warp is absent."""
    try:
        import warp as wp
    except Exception:
        return None
    try:
        counter = _WarpCounter(wp.get_device_allocator("cuda:0"), on_change)
        wp.set_device_allocator("cuda:0", counter)
        return counter
    except Exception:
        return None


class _PeakTracker:
    """Combined live-bytes high-water across a framework allocator and Warp."""

    def __init__(self, framework_live):
        self._framework_live = framework_live
        self.warp: _WarpCounter | None = None
        self.peak_bytes = 0

    def sample(self) -> None:
        warp_live = self.warp.live if self.warp is not None else 0
        total = self._framework_live() + warp_live
        if total > self.peak_bytes:
            self.peak_bytes = total

    def fold_framework_peak(self, framework_peak_bytes: int) -> None:
        """Fold in a framework-only maximum (Warp may have been idle then)."""
        warp_live = self.warp.live if self.warp is not None else 0
        total = framework_peak_bytes + warp_live
        if total > self.peak_bytes:
            self.peak_bytes = total


def _jax_inputs(B, J, K):
    import jax.numpy as jnp
    return (jnp.zeros((B, K), jnp.float32),
            jnp.broadcast_to(jnp.eye(3), (B, J, 3, 3)),
            jnp.zeros((B, 3), jnp.float32))


def _pin_torch_float32() -> None:
    """Force true float32 matmuls on the torch side (no TF32).

    ``allow_tf32`` already defaults to False for matmul on current torch, but
    that default has changed across versions — setting it explicitly keeps the
    stated convention independent of the installed torch. cuDNN is not used by
    the LBS/FK path, so pinning it is a no-op kept for completeness.
    """
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _jax_mem(field: str) -> int:
    try:
        import jax
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return 0
    return int(stats.get(field) or 0)


def _run_jax(method: str, B: int, hf: Path, soma_npz: Path,
             warmup: int, run_s: float, matmul_precision: str = "highest") -> dict:
    import jax

    # Same convention as bench_forward_pass.py: 'highest' = true float32, which
    # is what makes this comparable with SOMA-X's float32 torch/Warp path.
    # Leaving this unset lets XLA pick TF32 for float32 matmuls on Ampere+.
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    # Force backend/context creation, then size the fixed overhead.
    jax.device_put(0.0).block_until_ready()
    framework_live = lambda: _jax_mem("bytes_in_use")          # noqa: E731
    context_mib = _measure_context_mib(framework_live)
    tracker = _PeakTracker(framework_live)

    if method in ("fair", "hybrid"):
        from verify_fairness import _build_fair_pipeline
        if method == "hybrid":
            # The hybrid pipeline runs a Warp kernel, so it allocates through
            # Warp too — count it on this side as well, or the comparison tilts.
            tracker.warp = _install_warp_counter(tracker.sample)
        fwd, (V, J, K) = _build_fair_pipeline(hf, "warp" if method == "hybrid" else "jax")
    elif method == "linear":
        from soma_jax import SOMALayer
        layer = SOMALayer.load(str(soma_npz), identity_model_type="soma")
        J = len(layer.joint_names)
        K = layer.identity_model.n_betas if hasattr(layer.identity_model, "n_betas") else 128

        @jax.jit
        def fwd(coeffs, rotmats, transl):
            rv, rj = layer.prepare_identity(coeffs, repose_to_bind_pose=False, skeleton_fit="linear")
            out = layer.pose(rotmats, transl, rv, rj, apply_correctives=False)
            return out.vertices
    else:
        raise ValueError(method)

    args = _jax_inputs(B, J, K)
    for _ in range(warmup):
        fwd(*args).block_until_ready()

    t_end = time.perf_counter() + run_s
    n = 0
    while time.perf_counter() < t_end:
        fwd(*args).block_until_ready()
        tracker.sample()
        n += 1

    xla_peak = _jax_mem("peak_bytes_in_use")
    tracker.fold_framework_peak(xla_peak)
    warp_peak = tracker.warp.peak if tracker.warp is not None else 0
    return {
        "peak_mib": context_mib + tracker.peak_bytes / MIB,
        "peak_upper_mib": context_mib + (xla_peak + warp_peak) / MIB,
        "context_mib": context_mib,
        "framework_peak_mib": xla_peak / MIB,
        "warp_peak_mib": warp_peak / MIB,
        "iters": n,
    }


def _run_soma_x(B: int, hf: Path, warmup: int, run_s: float) -> dict:
    import torch
    _pin_torch_float32()

    # Force context creation, then size the fixed overhead.
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    context_mib = _measure_context_mib(torch.cuda.memory_allocated)
    tracker = _PeakTracker(torch.cuda.memory_allocated)
    # Install before SOMA-X builds anything, so every Warp buffer is counted.
    import warp  # noqa: F401  (SOMA-X imports it anyway; make ordering explicit)
    tracker.warp = _install_warp_counter(tracker.sample)

    from soma import SOMALayer
    layer = SOMALayer(data_root=str(hf), device="cuda:0",
                      identity_model_type="soma", mode="warp",
                      correctives_model_path=None,
                      enable_procedural_transforms=False).to("cuda:0")
    layer.eval()
    K = layer.num_shape_components
    nj = len(layer.parents) + 1
    identity = torch.zeros(B, K, device="cuda:0")
    poses = torch.zeros(B, nj - 1, 3, device="cuda:0")
    transl = torch.zeros(B, 3, device="cuda:0")

    for _ in range(warmup):
        layer.prepare_identity(identity, repose_to_bind_pose=False)
        layer.pose(poses, transl=transl, apply_correctives=False)
    torch.cuda.synchronize()

    torch_peak = 0
    t_end = time.perf_counter() + run_s
    n = 0
    while time.perf_counter() < t_end:
        # Per-iteration torch peak, so a torch-only maximum between Warp events
        # is never lost. Resetting a counter does not change how SOMA-X runs.
        torch.cuda.reset_peak_memory_stats()
        layer.prepare_identity(identity, repose_to_bind_pose=False)
        layer.pose(poses, transl=transl, apply_correctives=False)
        torch.cuda.synchronize()
        tracker.sample()
        iter_peak = torch.cuda.max_memory_allocated()
        tracker.fold_framework_peak(iter_peak)
        torch_peak = max(torch_peak, iter_peak)
        n += 1

    warp_peak = tracker.warp.peak if tracker.warp is not None else 0
    return {
        "peak_mib": context_mib + tracker.peak_bytes / MIB,
        "peak_upper_mib": context_mib + (torch_peak + warp_peak) / MIB,
        "context_mib": context_mib,
        "framework_peak_mib": torch_peak / MIB,
        "warp_peak_mib": warp_peak / MIB,
        "iters": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["soma_x", "fair", "hybrid", "linear"])
    p.add_argument("--batch", type=int, required=True)
    from soma_jax.assets import data_root as _data_root
    p.add_argument("--hf", default=str(_data_root()))
    # See bench_forward_pass.py: resolved rather than hardcoded.
    p.add_argument("--soma-npz",
                   default=str(__import__("soma_jax.assets", fromlist=["x"]).resolve(
                       "SOMA_neutral_fixed.npz", required=False)
                       or REPO / "assets" / "SOMA_neutral_fixed.npz"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--run-s", type=float, default=2.5)
    p.add_argument("--out", required=True)
    p.add_argument("--matmul-precision", default="highest",
                   help="JAX matmul precision: 'highest' = float32 (fair vs SOMA-X, "
                        "and the convention used by bench_forward_pass.py); "
                        "'default' = TF32 (JAX-only, NOT comparable to SOMA-X).")
    args = p.parse_args()

    # Both frameworks keep their SHIPPED allocator — overriding either would stop
    # measuring the pipeline a real user gets. Preallocation is the one thing
    # disabled: JAX's default grabs 75% of the card up front, which would make
    # every reading identical and the whole sweep meaningless.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        if args.method == "soma_x":
            res = _run_soma_x(args.batch, Path(args.hf), args.warmup, args.run_s)
        else:
            res = _run_jax(args.method, args.batch, Path(args.hf),
                           Path(args.soma_npz), args.warmup, args.run_s,
                           args.matmul_precision)
        status = "ok"
    except Exception as e:
        res = {"peak_mib": None, "iters": 0}
        status = f"{type(e).__name__}: {str(e)[:100]}"

    # Record the precision convention alongside every number, so a row can
    # never be read without knowing which one produced it.
    precision = "float32" if args.method == "soma_x" or \
        args.matmul_precision == "highest" else f"jax:{args.matmul_precision}"
    row = {"method": args.method, "batch": args.batch, "status": status,
           "precision": precision, **res}
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{args.method}] B={args.batch}  status={status}  "
          f"peak_mib={res['peak_mib']}")


if __name__ == "__main__":
    main()
