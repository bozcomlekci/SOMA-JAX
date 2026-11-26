#!/usr/bin/env bash
# Render a side-by-side comparison (SOMA / MHR / Anny / Garment / SMPL-X /
# SMPL) for each BVH clip listed in $CLIPS — one motion per row, in real time,
# grounded on a common floor.
#
#   bash tools/pipeline/render_bvh.sh
#   bash tools/pipeline/render_bvh.sh path/to/clip_list.txt
#   bash tools/pipeline/render_bvh.sh - <<EOF               # read clips from stdin
#   <label> <relative/path/inside/BVH_ROOT>
#   ...
#   EOF
#
# Each line of the clip list is "<label> <bvh_relative_path>". Lines starting
# with '#' and blank lines are skipped. An example clip list ships at
# tools/pipeline/render_bvh.clips.
#
# Inputs (environment): BVH_ROOT (clip directory), SUBJ_NPZ (MHR subject for
# the MHR column), SOMA_MODEL (runtime archive from
# tools/pipeline/build_soma_rig.py), IDENTITY_DIR (packs from
# tools/pipeline/build_identity_packs.py), SMPL_MODEL / SMPLX_MODEL.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}               # a python with jax + warp (see docs/INSTALL.md)
BVH_ROOT=${BVH_ROOT:-/path/to/bvh/clips}
SUBJ_NPZ=${SUBJ_NPZ:-/path/to/subject_shape.npz}
SOMA_MODEL=${SOMA_MODEL:-assets/SOMA_neutral_fixed.npz}
IDENTITY_DIR=${IDENTITY_DIR:-assets/identity}
SMPL_MODEL=${SMPL_MODEL:-data/smpl/SMPL_NEUTRAL.npz}
SMPLX_MODEL=${SMPLX_MODEL:-data/smplx/SMPLX_NEUTRAL.npz}
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
    --soma-model "$SOMA_MODEL" \
    --smpl-model "$SMPL_MODEL" \
    --smplx-model "$SMPLX_MODEL" \
    --bvh-motion "$bvh" \
    --mhr-subject "$SUBJ_NPZ" \
    --mhr-identity "$IDENTITY_DIR/identity_mhr.npz" \
    --anny-identity "$IDENTITY_DIR/identity_anny.npz" \
    --garment-identity "$IDENTITY_DIR/identity_garment.npz" \
    --side-by-side --soma-skeleton-overlay \
    --target-fps 20 \
    --num-frames 200 \
    --ground-lock \
    --width 384 --height 384 \
    --output-dir "$dest" \
    --gif "$dest/demo.gif"
done < <(if [ "$CLIPS_FILE" = "-" ]; then cat; else cat "$CLIPS_FILE"; fi)

echo "Done. Outputs under $OUT/"
