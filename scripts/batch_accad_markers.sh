#!/bin/bash
# Retarget ACCAD marker mocap onto K1 for the walking + kicking categories.
set -u
PY=/home/xcj/miniconda3/envs/env_isaaclab/bin/python
SRC=/data/rl_robot/accad/ACCAD
OUT=${1:-/data/rl_robot/BoosterRobotics/booster_assets/motions/K1/accad_markers}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/walk" "$OUT/kick"
ok=0; fail=0
run() {  # $1=category $2=kind $3=extra args
  for f in "$SRC/$1"/*_stageii.npz; do
    b=$(basename "$f" _stageii.npz)
    # Subject calibration trials are not motion data (they include Male1_Cal)
    case "$b" in *_Cal*|*_Calibration*|*Subject_Cal*) continue ;; esac
    name=$(echo "$b" | tr -c 'A-Za-z0-9._-' '_' | sed 's/__*/_/g;s/_$//')
    dst="$OUT/$2/$name.npz"
    [ -f "$dst" ] && continue
    if timeout 900 "$PY" "$here/accad_markers_to_k1.py" --clip "$f" --out "$dst" ${3:-} \
         > "/tmp/accad_$name.log" 2>&1; then
      ok=$((ok+1))
    else
      fail=$((fail+1)); echo "FAIL $1/$b"; tail -3 "/tmp/accad_$name.log" | sed 's/^/     /'
    fi
  done
}
run Female1Walking_c3d      walk --anchor_walk
run Male1Walking_c3d        walk --anchor_walk
run Male2Walking_c3d        walk --anchor_walk
run MartialArtsWalksTurns_c3d walk --anchor_walk
run Male2MartialArtsKicks_c3d kick
echo "done: ok=$ok fail=$fail -> $OUT"
