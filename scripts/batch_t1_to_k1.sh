#!/bin/bash
# Retarget the paper's T1 trajectories onto K1 by IK (replaces joint copying).
set -u
PY=/home/xcj/miniconda3/envs/env_isaaclab/bin/python
SRC=/data/rl_robot/BoosterRobotics/code/data
OUT=${1:-/data/rl_robot/BoosterRobotics/booster_assets/motions/K1/amp_paper_t1ik}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/walk" "$OUT/kick"
ok=0; fail=0
for f in "$SRC"/*.csv; do
  b=$(basename "$f" .csv)
  case "$b" in *kick*) kind=kick; extra="";; *) kind=walk; extra="--anchor_walk";; esac
  dst="$OUT/$kind/$b.npz"
  [ -f "$dst" ] && continue
  if timeout 900 "$PY" "$here/t1_traj_to_k1.py" --csv "$f" --out "$dst" $extra \
       > "/tmp/t1fix_$b.log" 2>&1; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "FAIL $b"; tail -3 "/tmp/t1fix_$b.log" | sed 's/^/   /'
  fi
done
echo "done: ok=$ok fail=$fail -> $OUT"
