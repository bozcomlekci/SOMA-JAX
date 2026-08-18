#!/usr/bin/env bash
# Run the SOMA forward-pass benchmark for both backends on RTX 5080.
#
# Torch (SOMA-X) needs CUDA 13 NVRTC, JAX (soma_jax) ships its own CUDA 12.
# Mixing both LD_LIBRARY_PATH-es in one process breaks one side, so we
# invoke them as two subprocesses with different envs and merge results.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}                       # a python with torch + jax + warp (see docs/INSTALL.md)
SCRIPT=$REPO/benchmarks/bench_forward_pass.py

# CUDA 13 NVRTC for torch — auto-detected from the torch env's nvidia-cu13 wheel;
# override by exporting TORCH_CUDA_LIBS.
TORCH_CUDA_LIBS=${TORCH_CUDA_LIBS:-$("$PY" -c 'import os,nvidia.cu13 as c; print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)}

# The timing self-sizes for low variance: a wall-clock warmup pins boost clocks
# and inner-batching amortizes host/launch overhead; the reported value is the
# median (see bench_forward_pass._timed_samples). No iter/warmup knobs needed.
BATCHES=${BATCHES:-"1 8 32 128"}
DEVICE=${DEVICE:-cuda:0}
RESULTS=$REPO/benchmarks/results

# Pin to the RTX 5080.
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$RESULTS"

echo "==> SOMA-X (torch + Warp GPU)"
LD_LIBRARY_PATH=$TORCH_CUDA_LIBS XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PY" "$SCRIPT" --skip-soma-jax --skip-soma-jax-st --skip-soma-jax-hybrid \
    --batches $BATCHES --device "$DEVICE" \
    --output "$RESULTS/_soma_x.json"

echo
echo "==> SOMA-JAX (full fit JAX-SVD + Warp-SVD hybrid + linear-regressor)"
# Warp + JAX share the same process here (both cu12); the hybrid path runs a
# Warp svd3 kernel inside the JAX graph via XLA FFI.
env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PY" "$SCRIPT" --skip-soma-x \
    --batches $BATCHES \
    --output "$RESULTS/_soma_jax.json"

echo
echo "==> merging into benchmarks/results/runtime.json"
RESULTS="$RESULTS" "$PY" - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["RESULTS"])
merged = {"results": []}
for name in ["_soma_x.json", "_soma_jax.json"]:
    with open(out / name) as f:
        merged["results"].extend(json.load(f)["results"])
with open(out / "runtime.json", "w") as f:
    json.dump(merged, f, indent=2)
print(f"wrote {out/'runtime.json'} ({len(merged['results'])} backends)")
PY

echo "done."
