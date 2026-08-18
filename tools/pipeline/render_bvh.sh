#!/usr/bin/env bash
# Render a side-by-side comparison (SOMA / MHR / Anny / Garment / SMPL-X /
# SMPL) for each BVH clip listed in $CLIPS — one motion per row, in real time,
# grounded on a common floor.
#
#   bash tools/render_bvh.sh
#   bash tools/render_bvh.sh path/to/clip_list.txt
#   bash tools/render_bvh.sh - <<EOF                        # read clips from stdin
#   <label> <relative/path/inside/BVH_ROOT>
#   ...
#   EOF
#
# Each line of the clip list is "<label> <bvh_relative_path>". Lines starting
# with '#' and blank lines are skipped. An example clip list ships at
# tools/pipeline/render_bvh.clips.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}               # a python with jax + warp (see docs/INSTALL.md)
BVH_ROOT=${BVH_ROOT:-/path/to/bvh/clips}
SUBJ_NPZ=${SUBJ_NPZ:-/path/to/subject_shape.npz}
OUT=${OUT:-$REPO/demo_renders/bvh}

CLIPS_FILE=${1:-$REPO/tools/pipeline/render_bvh.clips}

mkdir -p "$OUT"
cd "$REPO"

while IFS= read -r line || [ -n "$line" ]; do
  # Skip blanks + comments.
  [[ -z "${line// }" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  label=$(awk '{print $1}' <<<"$line")
  rel=$(awk '{print $2}' <<<"$line")
  bvh="$BVH_ROOT/$rel"
  if [ ! -f "$bvh" ]; then
    echo "[$label] skip — file missing: $bvh"
    continue
  fi
  echo "================================================================"
  echo "[$label]  $bvh"
  echo "================================================================"
  dest="$OUT/$label"
  mkdir -p "$dest"
  env -u LD_LIBRARY_PATH \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PYTHON" tools/pipeline/demo_soma_vis.py \
    --soma-model assets/SOMA_neutral_fixed.npz \
    --smpl-model data/smpl/SMPL_NEUTRAL.npz \
    --smplx-model data/smplx/SMPLX_NEUTRAL.npz \
    --bvh-motion "$bvh" \
    --mhr-subject "$SUBJ_NPZ" \
    --mhr-identity assets/identity/identity_mhr.npz \
    --anny-identity assets/identity/identity_anny.npz \
    --garment-identity assets/identity/identity_garment.npz \
    --side-by-side --soma-skeleton-overlay \
    --target-fps 20 \
    --num-frames 200 \
    --ground-lock \
    --width 384 --height 384 \
    --output-dir "$dest" \
    --gif "$dest/demo.gif"
done < <(if [ "$CLIPS_FILE" = "-" ]; then cat; else cat "$CLIPS_FILE"; fi)

echo "Done. Outputs under $OUT/"
