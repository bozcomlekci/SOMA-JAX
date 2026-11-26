#!/usr/bin/env bash
# SOMA-X vs SOMA-JAX comparison GIF and the float32 -> TF32 teaser. Same rig,
# same motion, same settings (LBS-only, mid-LOD); the speed ratios drawn come
# from benchmarks/results at batch 2048. torch (SOMA-X) needs CUDA 13 NVRTC; JAX
# ships CUDA 12 — keep them on separate LD_LIBRARY_PATHs and run as subprocesses.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python}                       # a python with torch + jax + warp (see docs/INSTALL.md)
CR=$REPO/tools/compare_render
TORCH_CUDA_LIBS=${TORCH_CUDA_LIBS:-$("$PY" -c 'import os,nvidia.cu13 as c; print(os.path.join(list(c.__path__)[0],"lib"))' 2>/dev/null || true)}
OUT=$REPO/demo_renders/compare
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"

# Motion: a SOMA-skeleton BVH clip when one is reachable (BVH=... or
# BVH_ROOT=...), else the clip SOMA-X itself ships for its demo
# (third_party/SOMA-X/assets/example_animation.npy).
BVH_ROOT=${BVH_ROOT:-/path/to/bvh/clips}
BVH=${BVH:-$BVH_ROOT/walk/clip_001.bvh}
if [ -f "$BVH" ]; then
    MOTION_SRC=(--bvh "$BVH")
else
    MOTION_SRC=(--npy "$REPO/third_party/SOMA-X/assets/example_animation.npy")
fi

echo "==> generate shared motion (${MOTION_SRC[1]##*/})"
env -u LD_LIBRARY_PATH "$PY" "$CR/gen_motion.py" \
    "${MOTION_SRC[@]}" --seconds "${SECONDS_WIN:-6}" --play-fps "${PLAY_FPS:-25}" \
    --out "$OUT/motion.npz"

echo "==> pose with SOMA-X (torch + Warp)"
LD_LIBRARY_PATH="$TORCH_CUDA_LIBS" "$PY" "$CR/pose_somax.py" \
    --motion "$OUT/motion.npz" --out "$OUT/somax.npz"

echo "==> pose with SOMA-JAX"
env -u LD_LIBRARY_PATH "$PY" "$CR/pose_somajax.py" \
    --motion "$OUT/motion.npz" --out "$OUT/somajax.npz"

echo "==> render comparison GIF"
env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl "$PY" "$CR/render_compare.py" \
    --somax "$OUT/somax.npz" --somajax "$OUT/somajax.npz" \
    --gif "$REPO/assets/media/soma_x_vs_soma_jax.gif"

echo "==> render float32 -> TF32 teaser GIF"
env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl "$PY" "$CR/render_tf32_teaser.py" \
    --somax "$OUT/somax.npz" --somajax "$OUT/somajax.npz" \
    --gif "$REPO/assets/media/soma_jax_tf32_teaser.gif"
echo "done -> assets/media/soma_x_vs_soma_jax.gif, assets/media/soma_jax_tf32_teaser.gif"
