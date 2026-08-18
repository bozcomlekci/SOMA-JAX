#!/usr/bin/env bash
# Peak GPU-memory sweep: 4 methods x batch sizes, one FRESH subprocess per
# (method, batch) so each number is an absolute footprint with no caching-
# allocator carry-over. torch (SOMA-X) needs CUDA 13 NVRTC; JAX ships its own
# CUDA 12 — keep them on separate LD_LIBRARY_PATHs, like run_runtime.sh.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}                       # a python with torch + jax + warp (see docs/INSTALL.md)
SCRIPT=$REPO/benchmarks/bench_memory.py
TORCH_CUDA_LIBS=${TORCH_CUDA_LIBS:-$("$PY" -c 'import os,nvidia.cu13 as c; print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)}
OUT=$REPO/benchmarks/results/memory.jsonl
BATCHES=${BATCHES:-"1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192"}
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$(dirname "$OUT")"
: > "$OUT"   # truncate

run_method () {
  local method="$1"; shift
  local ld="$1"; shift
  echo "==> $method"
  for B in $BATCHES; do
    if [ -n "$ld" ]; then
      LD_LIBRARY_PATH="$ld" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PY" "$SCRIPT" --method "$method" --batch "$B" --out "$OUT" || true
    else
      env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PY" "$SCRIPT" --method "$method" --batch "$B" --out "$OUT" || true
    fi
  done
}

run_method soma_x "$TORCH_CUDA_LIBS"
run_method fair   ""
run_method hybrid ""
run_method linear ""

echo "done -> $OUT"