#!/usr/bin/env bash
# Run the SOMA forward-pass benchmark for both backends — on the RTX 5080 by
# default, or on the host with DEVICE=cpu.
#
# Torch (SOMA-X) needs CUDA 13 NVRTC, JAX (soma_jax) ships its own CUDA 12.
# Mixing both LD_LIBRARY_PATH-es in one process breaks one side, so we
# invoke them as two subprocesses with different envs and merge results.
#
# Knobs (environment variables; the defaults reproduce the GPU tables' run):
#   BATCHES       batch sizes                             default "1 8 32 128"
#   DEVICE        cuda:N or cpu                           default cuda:0
#   SOMA_X_MODES  SOMA-X skinning backends, one process   default "warp"
#                 each ("warp dense" on CPU: Warp's CPU kernels and
#                 upstream's dense PyTorch LBS)
#   MIN_SAMPLES   floor on timed samples per point        default 30
#   OUTPUT        merged results file                     default benchmarks/results/runtime.json
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}                       # a python with torch + jax + warp (see docs/INSTALL.md)
SCRIPT=$REPO/benchmarks/bench_forward_pass.py

# The timing self-sizes for low variance: a wall-clock warmup pins boost clocks
# and inner-batching amortizes host/launch overhead; the reported value is the
# median (see bench_forward_pass._timed_samples). No iter/warmup knobs needed.
BATCHES=${BATCHES:-"1 8 32 128"}
DEVICE=${DEVICE:-cuda:0}
SOMA_X_MODES=${SOMA_X_MODES:-warp}
MIN_SAMPLES=${MIN_SAMPLES:-30}
RESULTS=$REPO/benchmarks/results
OUTPUT=${OUTPUT:-$RESULTS/runtime.json}

if [[ $DEVICE == cuda* ]]; then
    # CUDA 13 NVRTC for torch — auto-detected from the torch env's nvidia-cu13 wheel;
    # override by exporting TORCH_CUDA_LIBS.
    TORCH_CUDA_LIBS=${TORCH_CUDA_LIBS:-$("$PY" -c 'import os,nvidia.cu13 as c; print(os.path.join(list(c.__path__)[0],"lib"))' 2>/dev/null || true)}
    # Pin to the RTX 5080.
    export CUDA_VISIBLE_DEVICES=0
    TAG=""
else
    TORCH_CUDA_LIBS=""
    export CUDA_VISIBLE_DEVICES=""     # nothing lands on a GPU, even where there is one
    TAG="_$DEVICE"
fi

mkdir -p "$RESULTS"
PARTS=()

for MODE in $SOMA_X_MODES; do
    SUFFIX=""
    [[ $MODE == warp ]] || SUFFIX="_$MODE"
    PART="$RESULTS/_soma_x$SUFFIX$TAG.json"
    echo "==> SOMA-X (torch + $MODE skinning, $DEVICE)"
    LD_LIBRARY_PATH=$TORCH_CUDA_LIBS XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PY" "$SCRIPT" --skip-soma-jax --skip-soma-jax-st --skip-soma-jax-hybrid \
        --soma-x-mode "$MODE" --batches $BATCHES --device "$DEVICE" \
        --min-samples "$MIN_SAMPLES" --output "$PART"
    PARTS+=("$PART")
    echo
done

echo "==> SOMA-JAX (full fit JAX-SVD + Warp-SVD hybrid + linear-regressor, $DEVICE)"
# Warp + JAX share the same process here (both cu12); the hybrid path runs a
# Warp svd3 kernel inside the JAX graph via XLA FFI (CUDA only: skipped on CPU).
PART="$RESULTS/_soma_jax$TAG.json"
env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PY" "$SCRIPT" --skip-soma-x \
    --batches $BATCHES --device "$DEVICE" --min-samples "$MIN_SAMPLES" \
    --output "$PART"
PARTS+=("$PART")

echo
echo "==> merging into ${OUTPUT#"$REPO"/}"
OUTPUT="$OUTPUT" "$PY" - "${PARTS[@]}" <<'PY'
import json, os, pathlib, sys
merged = {"results": [], "env": {}}
for name in sys.argv[1:]:
    with open(name) as f:
        part = json.load(f)
    merged["results"].extend(part["results"])
    key = pathlib.Path(name).stem.lstrip("_")          # e.g. soma_x_dense_cpu
    if part.get("env"):
        merged["env"][key] = part["env"]
    if part.get("skipped"):
        merged.setdefault("skipped", {}).update(part["skipped"])
out = pathlib.Path(os.environ["OUTPUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"wrote {out} ({len(merged['results'])} backends)")
PY

echo "done."
